// pipeline.rs —— 四阶段流水线
//
// 对应 Python 版的四个脚本：
//   a_提取音频.py    → 0pre 里的 mp4/mkv 转成 flac
//   b_分离人声.py    → 读 0pre，分离出人声写 1audio
//   c_人声检测.py    → 读 1audio，VAD 切片写 2vad
//   d_语音识别.py    → 读 2vad，翻译写 3asr
//
// ⚠️ 为什么把四个阶段拆成四次 spawn_background，而不是一次跑完：
//
//   Reactor 的 LocalSender 是 !Send 的，不能 move 进后台闭包；
//   而 spawn_background 的闭包【只能返回一个消息】。
//   所以后台没法"边跑边发进度"。
//
//   拆成四次之后，每跑完一个阶段就回一条消息，update 里收到后
//   再 spawn 下一个阶段。这样每个阶段结束界面都能刷新一次，
//   而且能精确知道"现在卡在哪一步"。
//
//   代价是阶段之间的输出会分别到达（而不是连续的流），但这四个脚本
//   本来就是串行的，间隔只在一阶段结束时，可以接受。

use std::sync::{Arc, Mutex};

use windows_reactor::CancellationToken;

use crate::backend;

/// 每个阶段跑完回给界面的一条消息
pub struct StageResult {
    /// 阶段号 1..=4
    pub stage: usize,
    /// 子进程退出码；-1 表示我们自己判定失败（比如启动都失败了）
    pub code: i32,
    /// 这个阶段收集到的全部输出行
    pub lines: Vec<String>,
}

/// 四个脚本的文件名与阶段标题，顺序不能变。
///
/// 路径里带 `app/` —— 脚本装在子目录里（见 config::SCRIPTS_SUBDIR）。
/// 注意这里是【相对于工作目录】的路径，run_script 会把它和 work_dir 拼起来；
/// 而 Python 脚本内部的 "0pre" 之类仍然是相对工作目录（不是脚本目录），
/// 所以放子目录不影响它们读写数据。
pub const STAGES: [(usize, &str, &str); 4] = [
    (1, "app/a_提取音频.py", "第1步 提取音频"),
    (2, "app/c_人声检测.py", "第2步 人声检测(VAD)"),
    (3, "app/b_分离人声.py", "第3步 分离人声"),
    (4, "app/d_语音识别.py", "第4步 语音识别翻译"),
];

/// 运行模式（v1.20）：全量 / 精简 / 兼容。
///
/// 判定条件（只在启动时检测一次）：
///   · Full：N 卡显存 ≥ 8 GB
///   · Lite：N 卡显存 6~8 GB（int8 精简，仍走全量流水线）
///   · Compat：其余所有情况（无 N 卡 / 显存不足 6 GB）
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunMode {
    Full,
    Lite,
    Compat,
}

/// 兼容模式的流水线：CPU 转写 + 内置 llama.cpp 翻译。
///
/// 4 个阶段编号和全量模式一致，进度体系不用改：
///   1 提取音频（复用 a 脚本）
///   2 语音转写（Silero VAD + SenseVoice）
///   3 翻译（内置 llama.cpp + Qwen2.5-1.5B）
///   4 字幕整理（复用 e 脚本：时长上限 + 词表）
pub const STAGES_COMPAT: [(usize, &str, &str); 4] = [
    (1, "app/a_提取音频.py", "第1步 提取音频"),
    (2, "app/light_asr.py", "第2步 语音转写"),
    (3, "app/light_translate.py", "第3步 翻译"),
    (4, "rust:tidy", "第4步 字幕整理"),
];

/// 按模式取阶段表。
pub fn stages(mode: RunMode) -> &'static [(usize, &'static str, &'static str)] {
    match mode {
        RunMode::Compat => &STAGES_COMPAT,
        _ => &STAGES,
    }
}
// v1.7 调整了第 2、3 步的顺序：VAD 提前到分离之前。
// 理由：VAD 不需要人声就能做（在原始音频上检测语音活动），
// 而分离耗时和音频长度严格成正比。先圈出语音段再只分离那些段，
// 实测 12 分钟视频分离 65s→9.45s（省 85%），30 分钟音声 122s→52s（省 57%）。

use crate::config;
use crate::srt;

/// 枚举本机的显示适配器名称（AMD / Intel / NVIDIA）。
///
/// ★ 刻意【不调用 llama-server】：用户明确要求，llama 和 1.5B 模型
///   只在「勾选兼容模式 + 点开始处理」时才启动。启动程序时调
///   `llama-server --list-devices` 会闪一个黑框，也违背这个要求。
///
/// 这里直接用 Win32 的 EnumDisplayDevicesW 枚举显卡名，纯 API 调用，
/// 不起进程、不加载模型。设备名的用途只有两个：状态栏显示、
/// 判断「有没有非 N 卡设备」。
pub fn detect_vulkan_devices() -> Vec<String> {
    use windows_sys::Win32::Graphics::Gdi::{
        EnumDisplayDevicesW, DISPLAY_DEVICE_ACTIVE, DISPLAY_DEVICEW,
    };

    let mut devices = Vec::new();
    let mut idx: u32 = 0;
    loop {
        let mut dd: DISPLAY_DEVICEW = unsafe { std::mem::zeroed() };
        dd.cb = std::mem::size_of::<DISPLAY_DEVICEW>() as u32;
        let ok = unsafe { EnumDisplayDevicesW(std::ptr::null(), idx, &mut dd, 0) };
        if ok == 0 {
            break;
        }
        idx += 1;

        // 千问评审：不要用 ATTACHED_TO_DESKTOP 过滤（无头/远程/多屏时会漏），
        // 用 ACTIVE 判断设备是否启用即可。
        if dd.StateFlags & DISPLAY_DEVICE_ACTIVE == 0 {
            continue;
        }
        let name = String::from_utf16_lossy(&dd.DeviceString)
            .trim()
            .to_string();
        if name.is_empty() {
            continue;
        }
        // 虚拟/基本显示适配器不是真实 GPU，排掉。
        let low = name.to_lowercase();
        if low.contains("basic display") || low.contains("remote") || low.contains("virtual") {
            continue;
        }
        if !devices.contains(&name) {
            devices.push(name);
        }
    }
    devices
}

/// 检测这台机器能不能用 CUDA。
///
/// 返回 (显示文本, 是不是 GPU, 显存 GB, 有没有 Vulkan, CUDA 卡名, Vulkan 设备名)。
///
/// 最后两项是给副标题用的，【两个都要留着】：
///   · 用户在界面里手动切到兼容模式时，副标题要写「<型号> Vulkan 加速中」，
///     可这时 CUDA 是好的 —— 光知道「走没走 CUDA」不够，得两套名字都有。
///   兼容模式下第一个字段被压成「兼容模式」四个字，型号不在里面，抠不出来。
///
/// 对应 Python 版的 _detect_device。它跑一小段 Python 去问 torch，
/// 因为「有没有 CUDA」只有装在这些脚本用的那个环境里才问得准 ——
/// 在本进程里猜、或者查注册表，都不如直接问 torch。
///
/// 注意这是【阻塞】的：torch 首次导入要 2~3 秒。所以调用方要放后台线程，
/// 不能直接在界面线程里调。
///
/// 模式判定（v1.20）：
///   · CUDA 且显存 ≥ 8.1 GB      → 全量模式（float16）
///   · CUDA 且显存 6~8.1 GB      → 精简模式（int8，仍是全量流水线）
///   · CUDA 但显存 < 6 GB        → 兼容模式
///   · 没有 CUDA 但有 Vulkan     → 兼容模式（走 llama.cpp）
///   · 什么都没有                → 兼容模式（纯 CPU）
pub fn detect_device() -> (String, bool, f64, bool, String, String) {
    // 这段 Python 的缩进是语法的一部分，改的时候注意别动前面的空格。
    // 最外层不能有缩进。
    let code = r#"
import torch
try:
    available = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if available else ''
    if available:
        try:
            gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            gb = 0.0
        print('GPU:' + name + '|' + ('%.1f' % gb))
    else:
        print('CPU')
except Exception:
    print('ERROR')
"#;

    // Vulkan 设备枚举不依赖 torch，先做（llama-server 不在就返回空）。
    let vulkan = detect_vulkan_devices();
    let has_vulkan = !vulkan.is_empty();

    match backend::run_python_inline(&config::python_exe(), code) {
        Ok(s) if s.starts_with("GPU:") => {
            // 格式：GPU:<名字>|<显存GB>
            let rest = s[4..].trim();
            let (name, vram) = match rest.split_once('|') {
                Some((n, v)) => (n.trim(), v.trim().parse::<f64>().unwrap_or(0.0)),
                None => (rest, 0.0),
            };
            // 阈值和 settings.py 的 _NEED_F16_MIB（8289 MiB）保持一致：
            // float16 峰值 4265 + 基线 3000 + 预留 1024 = 8289 MiB ≈ 8.1 GB
            let lite = vram > 0.0 && vram < 8.1;
            let text = if vram > 0.0 && vram < 6.0 {
                "兼容模式".to_string()
            } else if name.is_empty() {
                "GPU（已启用 CUDA）".to_string()
            } else if lite {
                format!("GPU：{name}（{vram:.1} GB 显存 · 精简模式）")
            } else {
                format!("GPU：{name}（已启用 CUDA）")
            };
            let vulkan_name = vulkan.join(" / ");
            (text, true, vram, has_vulkan, name.to_string(), vulkan_name)
        }
        // 没有 CUDA：不再说「CPU 慢 50 倍」，而是走兼容模式。
        Ok(s) if s == "CPU" => {
            // 设备状态栏只写「兼容模式」四个字（用户明确要求）。
            //   原来写的是「兼容模式（Vulkan：NVIDIA GeForce RTX 5070）」
            //   这种带型号的长句 —— 型号在副标题里，状态栏只需说明
            //   "现在用的是哪个模式"。
            let text = "兼容模式".to_string();
            // 兼容模式副标题要写「<显卡/核显型号> Vulkan 加速中」，
            // 型号从显示适配器枚举里拿；纯 CPU 就留空，副标题不显示这段。
            let vulkan_name = vulkan.join(" / ");
            (text, false, 0.0, has_vulkan, String::new(), vulkan_name)
        }
        Ok(other) => (
            format!("未知（检测失败：{other}）"),
            false,
            0.0,
            has_vulkan,
            String::new(),
            vulkan.join(" / "),
        ),
        Err(e) => {
            // 常见情况：解释器路径不对、torch 没装。
            // 截断一下，别让一整条 traceback 把状态栏撑爆。
            let mut msg = e.replace(['\r', '\n'], " ");
            if msg.chars().count() > 80 {
                msg = msg.chars().take(80).collect::<String>() + "…";
            }
            (
                format!("未知（{msg}）"),
                false,
                0.0,
                has_vulkan,
                String::new(),
                vulkan.join(" / "),
            )
        }
    }
}

/// 第4步之后的纯文本整理：调用 app/e_字幕整理.py。
///
/// ★ 刻意【不占阶段编号、不进进度体系】。
///   现有进度条 / ETA 是按 4 个阶段 + 实测权重 + 语音时长成本模型
///   一点点校准出来的（见 ui.rs 里那一大段注释），多一个阶段就要
///   重算所有权重和 stage_ratio，风险高而收益为零 —— 整理只是几百
///   毫秒的文本处理。
///   所以它作为 finalize 之前的一个后台小任务存在：
///   第4步跑完（进度 100%）→ 整理字幕 → 改名归位。
///
/// 失败也【不阻断收尾】：字幕本身是好的，只是没整理。宁可报告
/// 「整理失败」让字幕原样落地，也不要因此丢掉已经跑了几十分钟的结果。
pub fn tidy_subtitles() -> Vec<String> {
    // v1.21：字幕整理改为纯 Rust（srt.rs），不再 spawn Python。
    // 逻辑与 e_字幕整理.py 一致：时长上限 + 不压下一句 + 最少 0.4 秒 + 词表。
    let app_dir = std::path::Path::new(config::work_dir()).join("app");
    let asr_dir = config::dir_asr();
    let mut lines = srt::tidy_dir(std::path::Path::new(&asr_dir), &app_dir);
    lines.push("[整理] 字幕时间轴整理完成（Rust）".to_string());
    lines
}

/// 字幕收尾：把 3asr\ 里生成的字幕改名成 `<原名>.chs.srt`，
/// 并且按用户选择决定放到视频旁边还是留在 3asr\。返回日志行。
///
/// 对应 Python 版的 _finalize_subtitles。几个关键点必须保持一致：
///
///   1. 【`.chs.srt` 这个扩展名不能改】。PotPlayer 这类播放器靠语言代码
///      识别外挂字幕，`_中文字幕.srt` 这种名字它不认，不会自动加载。
///      这是实际踩过的坑：字幕生成了但播放器不自动显示。
///
///   2. 3asr\ 里生成的文件名可能带着 `_中文字幕` 或 `_(Vocals)...` 后缀，
///      要剥掉才能和视频文件名对上。对不上就找不到该把字幕放哪儿。
///
///   3. 找不到对应视频的字幕【留在 srt_dir】但同样改名 —— 不能丢。
///
/// `dest_dir` 是【不勾选「放视频旁边」时】字幕的归宿。它来自界面上那个
/// 输出目录，用户可以改（比如改到别的盘）。留空则用默认的 3asr\。
pub fn finalize_subtitles(
    files: &[String],
    copy_to_video: bool,
    dest_dir: &str,
) -> Vec<String> {
    // 源目录永远是 3asr\（四个脚本把字幕生成在那儿，改不了）；
    // 目标目录才是用户可以改的那个。
    let src = config::dir_asr();
    let dst = if dest_dir.trim().is_empty() {
        src.clone()
    } else {
        dest_dir.trim().to_string()
    };
    finalize_subtitles_in(&src, &dst, files, copy_to_video)
}

/// 实际的收尾逻辑。
///
///   srt_dir  —— 【源】：四个脚本把字幕生成在这儿（永远是 3asr\），从这里扫。
///   dest_dir —— 【目标】：不勾选「放视频旁边」时字幕的去处。用户可以改。
///
/// 两个都作为参数传进来是为了能写单元测试 —— 测的时候给临时目录，
/// 不会碰到真实的 3asr\。
fn finalize_subtitles_in(
    srt_dir: &str,
    dest_dir: &str,
    files: &[String],
    copy_to_video: bool,
) -> Vec<String> {
    let mut logs = Vec::new();

    if !std::path::Path::new(srt_dir).is_dir() {
        logs.push(format!("[字幕] 找不到目录 {srt_dir}"));
        return logs;
    }

    // 文件名去扩展名 -> 完整路径。用来把字幕和视频对上。
    let mut video_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for f in files {
        let base = std::path::Path::new(f)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let stem = std::path::Path::new(&base)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        video_map.insert(stem, f.clone());
    }

    let entries = match std::fs::read_dir(&srt_dir) {
        Ok(e) => e,
        Err(e) => {
            logs.push(format!("[字幕] 读不了 {srt_dir}: {e}"));
            return logs;
        }
    };

    let mut moved = 0usize;
    let mut kept = 0usize;
    let mut failed = 0usize;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let fname = match path.file_name() {
            Some(n) => n.to_string_lossy().to_string(),
            None => continue,
        };
        if !fname.to_lowercase().ends_with(".srt") {
            continue;
        }

        // 剥掉 _中文字幕 / _(Vocals)... 后缀
        let mut stem = fname[..fname.len() - 4].to_string();
        if let Some(rest) = stem.strip_suffix("_中文字幕") {
            stem = rest.to_string();
        }
        if let Some(pos) = stem.find("_(Vocals)") {
            stem.truncate(pos);
        }

        // 决定目标位置
        //
        //   勾选「放视频旁边」 → 视频所在目录
        //   不勾选             → dest_dir（界面上那个输出目录，用户可以改）
        //
        // 注意不勾选时【不要】用 srt_dir —— 那是源目录（3asr），
        // 用户改输出目录就是为了把字幕放到别处去。
        let (dst_dir, matched) = match video_map.get(&stem) {
            Some(video_path) => {
                if copy_to_video {
                    let d = std::path::Path::new(video_path)
                        .parent()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|| srt_dir.to_string());
                    (d, true)
                } else {
                    (dest_dir.to_string(), true)
                }
            }
            // 找不到对应视频 -> 同样放 dest_dir，不要丢在源目录里
            None => (dest_dir.to_string(), false),
        };

        let dst = format!(r"{dst_dir}\{stem}.chs.srt");

        // 目标已存在就先删掉 —— rename 在 Windows 上不会覆盖
        if dst != path.to_string_lossy() && std::path::Path::new(&dst).exists() {
            let _ = std::fs::remove_file(&dst);
        }
        if dst == path.to_string_lossy() {
            continue; // 已经是目标名了
        }

        match std::fs::rename(&path, &dst) {
            Ok(_) => {
                if matched {
                    moved += 1;
                    logs.push(format!(
                        "[字幕] {fname} → {}{}",
                        if copy_to_video { "视频目录\\" } else { "3asr\\" },
                        std::path::Path::new(&dst)
                            .file_name()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default()
                    ));
                } else {
                    kept += 1;
                    logs.push(format!("[字幕] {fname} → {stem}.chs.srt（无对应视频，放进输出目录）"));
                }
            }
            Err(e) => {
                // rename 跨卷会失败，退回复制 + 删除
                match std::fs::copy(&path, &dst) {
                    Ok(_) => {
                        let _ = std::fs::remove_file(&path);
                        moved += 1;
                        logs.push(format!("[字幕] {fname} → {stem}.chs.srt（跨卷复制）"));
                    }
                    Err(e2) => {
                        failed += 1;
                        logs.push(format!("[字幕] {fname} 处理失败: {e} / {e2}"));
                    }
                }
            }
        }
    }

    logs.push(format!(
        "[字幕] 完成：移到视频旁 {moved} 个，放进输出目录 {kept} 个，失败 {failed} 个"
    ));
    logs
}

/// 把用户选的文件准备到 0pre\ 里，返回 (就绪数, 日志行)。
///
/// 对应 Python 版的 _prepare_0pre。两个要点：
///
///   1. 【优先用硬链接】。Python 版用的是 os.link，失败才 shutil.copy2。
///      这是为了不复制几 GB 的视频文件 —— 硬链接在同一卷上是瞬间完成的，
///      而且不占额外空间。跨卷（比如视频在 H: 而工作目录在 D:）时会失败，
///      那就回退到真复制。
///
///   2. 0pre 里的旧文件必须先清掉，否则上一轮残留的会混进来一起处理。
///      Python 版是在主线程先清的，这里一并做掉。
/// 不关心取消的调用走这里（单元测试用）。
/// 界面实际调的是下面那个可以取消的版本。
#[allow(dead_code)]
pub fn prepare_0pre(files: &[String]) -> Result<(usize, Vec<String>), String> {
    prepare_0pre_cancellable(files, &|| false)
}

/// 登记要处理的源文件，不复制视频。
///
/// ★ 这里【不再复制文件】，只把源路径写进工作目录下的 `_input.txt`，
///   然后由 `a_提取音频.py` 直接读那些源文件、把 flac 写进 0pre。
///
/// 为什么改：
///   原来是"优先硬链接、不行就真复制"。硬链接只能在同一卷内用，而视频
///   常常在别的盘（H:\迅雷下载），程序却在 D:\ —— 跨盘必然退化成真复制，
///   一次几 GB、要好几分钟。而这些视频本来就在硬盘上，再复制一份纯属浪费。
///
///   现在视频留在原地，只有 flac（本来就要生成的中间产物）落在 0pre。
///
/// 为什么清单文件放在【工作目录根】而不是 0pre 里：
///   `b_分离人声.py` 用的是 `separator.separate(0pre目录)` —— 整个目录扫，
///   不看扩展名。清单要是放进 0pre，会被它当成音频文件去处理。
///
/// `cancelled` 参数保留是为了不动调用方。现在这一步只剩写一个小文本文件，
/// 快到来不及取消（真正耗时的复制已经没有了）。
pub fn prepare_0pre_cancellable(
    files: &[String],
    cancelled: &dyn Fn() -> bool,
) -> Result<(usize, Vec<String>), String> {
    let pre = config::dir_pre();
    std::fs::create_dir_all(&pre).map_err(|e| format!("建不出 {pre}: {e}"))?;

    let mut logs = Vec::new();

    // 清空 0pre：里面有上一轮留下的 flac
    let mut cleaned = 0usize;
    if let Ok(rd) = std::fs::read_dir(&pre) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_file() && std::fs::remove_file(&p).is_ok() {
                cleaned += 1;
            }
        }
    }
    if cleaned > 0 {
        logs.push(format!("  [准备] 清掉了 0pre 里 {cleaned} 个旧文件"));
    }

    if cancelled() {
        logs.push("  [准备] 已取消".to_string());
        return Ok((0, logs));
    }

    // 写源文件清单（每行一个绝对路径，UTF-8 —— 路径里会有中文）
    let list_path = format!(r"{}\_input.txt", config::work_dir());
    let mut content = String::new();
    for f in files {
        content.push_str(f);
        content.push('\n');
    }

    if let Err(e) = std::fs::write(&list_path, &content) {
        return Err(format!("写不了 {list_path}: {e}"));
    }

    let mut registered = 0usize;
    for f in files {
        let name = std::path::Path::new(f)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| f.clone());
        logs.push(format!("  [准备] {name}"));
        registered += 1;
    }
    logs.push(format!("  [准备] {registered} 个文件已就绪"));

    Ok((registered, logs))
}

/// 分块复制，每块之后检查一次是否被取消。
///
/// 被取消时会把写到一半的目标文件删掉 —— 留个半截文件在 0pre 里
/// 会被下一步当成正常输入，处理出莫名其妙的结果。
#[allow(dead_code)]
fn copy_file_cancellable(
    src: &str,
    dst: &str,
    cancelled: &dyn Fn() -> bool,
) -> std::io::Result<()> {
    use std::io::{Read, Write};

    let mut reader = std::fs::File::open(src)?;
    let mut writer = std::fs::File::create(dst)?;

    // 4 MB 一块：够大（系统调用次数不至于太密），
    // 又够小（13 GB 也就 3300 次检查，取消的延迟感觉不出来）
    let mut buf = vec![0u8; 4 * 1024 * 1024];

    loop {
        if cancelled() {
            // 先把句柄放掉再删，否则 Windows 上文件还被占着删不掉
            drop(writer);
            let _ = std::fs::remove_file(dst);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "复制被取消",
            ));
        }

        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
    }

    writer.flush()?;
    Ok(())
}

/// 这一行值不值得显示在界面的「当前动作」里。
///
/// 过滤掉三类噪音（都是实测里真的出现过、把界面搞乱的）：
///   · torch / python 的警告：`W1016 18:59:55.484000 31704 site-packages\tor...`
///   · 第三方库的日志：`2026-10-16 18:59:48,310 - INFO - separator - xxx`
///   · 只有盘符路径、没有说明的碎行
fn is_meaningful_line(line: &str) -> bool {
    let t = line.trim();
    if t.chars().count() < 2 {
        return false;
    }

    // 时间戳开头的库日志
    if t.starts_with("20") && t.matches(" - ").count() >= 2 {
        return false;
    }

    // torch / warnings 的级别前缀（W1016、E1016、I1016……）
    let head: String = t.chars().take(2).collect();
    if head == "W1" || head == "E1" || head == "I1" {
        return false;
    }

    // 栈帧 / 库路径
    if t.contains("site-packages") || t.contains("File \"") {
        return false;
    }

    // 光秃秃一个路径（第二个字符是冒号、且整行没有空格）
    let chars: Vec<char> = t.chars().collect();
    if chars.len() > 3 && chars[1] == ':' && !t.contains(' ') {
        return false;
    }

    true
}

/// 跑一个阶段。stage 从 1 开始。
///
/// 这个函数会被塞进 spawn_background，所以签名必须是
/// FnOnce(CancellationToken) -> 某个消息的形态；这里为了能被多次调用，
/// 做成一个接受 stage 的构造函数。
/// 各阶段"真的开始吃 GPU 了"的标志行。
///
/// ── 为什么不能用脚本自己的"开始处理"那几行 ──
///
///   第一版把标志设成脚本打印的
///       b → "开始批量处理目录 ..."
///       d → "开始推理: xxx"
///   结果踩了坑：脚本打完这行之后，还要过几十秒 GPU 才有动静。
///   用户截图反馈「显示已加载，但是显卡没干活呀」——
///   因为 separator.load_model() 只把权重读进【内存】，
///   ONNX Runtime 的推理会话是【惰性创建】的：真正第一次推理时才去
///   加载 CUDA provider、编译 kernel。这段时间 GPU 占用是 0%，
///   脚本却已经宣称"开始批量处理"了。
///
/// ── 现在用什么判断 ──
///
///   用 tqdm 进度条。separator / faster-whisper 真正开始推理时会输出
///      0%|          | 0/706 [00:00<?, ?it/s]
///   这行的 `it/s]` 尾巴是 tqdm 的固定格式，只有真在算才会出现。
///   它出现 = GPU 已经在干活，这时候说"已加载、正在处理"才准确。
///
///   c 阶段（pyannote VAD）没有 tqdm，保留它自己的标志。
pub fn ready_marker(mode: RunMode, stage: usize) -> &'static str {
    match mode {
        RunMode::Compat => match stage {
            2 => "处理:",
            3 => "翻译:",
            _ => "",
        },
        _ => match stage {
            2 => "处理音频:",
            3 => "it/s]",
            4 => "it/s]",
            _ => "",
        },
    }
}

pub fn run_stage(
    mode: RunMode,
    stage: usize,
    _token: CancellationToken,
    latest: Arc<Mutex<String>>,
    ready: Arc<std::sync::atomic::AtomicBool>,
    // tqdm 进度的独立通道（0.0~1.0）。和 latest 分开，避免互相覆盖。
    frac: Arc<Mutex<f64>>,
    // tqdm 自己算的剩余秒数（0 表示没有）。同样独立，避免和上面两个抢。
    eta: Arc<Mutex<f64>>,
    // 最后一条字幕的结束时间（秒，原始音频时间轴）。
    // 第4步用它算真实进度，见 ui 侧的 recompute_progress。
    sub_end: Arc<Mutex<f64>>,
) -> StageResult {
    let (_, script, _) = stages(mode)[stage - 1];
    let marker = ready_marker(mode, stage);

    // ★ 纯 Rust 阶段（不 spawn Python）：v1.21 起字幕整理走这里。
    //   日志行直接产出，进度/ETA 不参与（整理只要几十毫秒）。
    if let Some(kind) = script.strip_prefix("rust:") {
        let lines = match kind {
            "tidy" => {
                let app_dir = std::path::Path::new(config::work_dir()).join("app");
                let asr_dir = config::dir_asr();
                srt::tidy_dir(std::path::Path::new(&asr_dir), &app_dir)
            }
            _ => vec![format!("[阶段] 未知的 Rust 阶段：{kind}")],
        };
        if let Some(last) = lines.last() {
            if let Ok(mut l) = latest.lock() {
                *l = last.clone();
            }
        }
        return StageResult {
            stage,
            code: 0,
            lines,
        };
    }
    let ready2 = Arc::clone(&ready);

    // on_line 是 Fn（不是 FnMut），用 Arc<Mutex<..>> 收集
    let sink: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink2 = Arc::clone(&sink);
    let latest2 = Arc::clone(&latest);
    let frac2 = Arc::clone(&frac);
    let eta2 = Arc::clone(&eta);
    let sub2 = Arc::clone(&sub_end);

    let code = backend::run_script(
        &config::python_exe(),
        config::work_dir(),
        script,
        move |line| {
            // ★ 结构化事件（v1.17，协议见 app/events.py）：
            //   脚本用 "@@EV@@ " 前缀输出机器可读行。
            //   这里【只更新与下面正则完全相同的变量】（sub_end），
            //   不引入新的进度/ETA 算法 —— 事件只是同一份数据更可靠的
            //   来源，正则继续兜底。事件行不进日志、不当"当前动作"。
            if let Some(ev) = line.strip_prefix("@@EV@@ ") {
                let mut parts = ev.splitn(4, '|');
                if parts.next() == Some("sub") {
                    let _start = parts.next();
                    if let Some(end_s) = parts.next() {
                        if let Ok(end) = end_s.trim().parse::<f64>() {
                            if end > 0.0 {
                                if let Ok(mut v) = sub2.lock() {
                                    *v = end;
                                }
                            }
                        }
                    }
                }
                // progress 事件预留：当前 tqdm 通道已经够用，
                // 等事件在几个版本里验证稳定后再接。
                return;
            }

            // ★ 字幕时间戳的解析【刻意放在 is_meaningful_line 外面】。
            //
            //   字幕行 [6347.13s -> 6358.25s]: 好高兴 是第4步进度的
            //   唯一数据源。那个时间戳就是【原始音频时间轴】
            //   （分离结果被填回了原长度），所以"最后一条字幕到哪"
            //   就等于"处理到音频的第几秒"。
            //
            //   放在过滤外面是为了防止将来 is_meaningful_line 加了新规则
            //   把字幕行误判掉 —— 那样进度会永远到不了 100%，
            //   而且不会有任何报错，属于最难查的静默失效。
            if let Some(c) = config::translation_re().captures(line) {
                if let Ok(end) = c[2].trim().parse::<f64>() {
                    if end > 0.0 {
                        if let Ok(mut v) = sub2.lock() {
                            *v = end;
                        }
                    }
                }
            }

            // 顺手把"最新一行"也记下来。
            //
            // 界面用 Msg::Tick（每秒一次）去读它，于是脚本正在干什么
            // （"正在加载模型"、"提取中: xxx"、whisper 的进片段进度）
            // 能实时显示在进度卡片上 —— 不用等整个阶段跑完。
            //
            // ★ 但只记【有信息量】的行。脚本的原始输出里混着大量噪音：
            //   torch 的 W1016 警告、第三方库的 INFO 日志、纯路径行。
            //   照单全收的话，「当前动作」那栏显示的就是一串看不懂的东西
            //   （实测出现过 `W1016 18:59:55.484000 31704 site-packages\tor...`）。
            // 模型加载完成的标志行 —— 置位之后界面会把提示改成
            // 「模型已加载」。见 ready_marker()。
            if !marker.is_empty() && line.contains(marker) {
                ready2.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            if is_meaningful_line(line) {
                // ★ tqdm 进度行要【优先保留】，不能被「[等待]」心跳行挤掉。
                //
                //   第3步分离期间，两路输出交替刷新：
                //     · tqdm 每 0.45 秒一行：  3%|▎  | 5/173 [00:02<01:23, 2.02it/s]
                //     · 心跳每 3 秒一行：      [等待] 人声分离 xxx 已用 3 秒 ...
                //
                //   原来是谁最后写谁生效，于是 live_line 大部分时间停在
                //   「[等待]」上。界面每 100ms 去读一次，几乎总会读到心跳行，
                //   解析不到 tqdm → stage_frac 一直是 0 → 进度条钉在 14% 不动，
                //   直到第3步跑完才跳到 50%（实测用户反馈"突然从14到50"）。
                //
                //   规则改成：
                //     · tqdm 行 → 总是覆盖
                //     · 非 tqdm 行 → 只在当前不是 tqdm 行时才覆盖
                //   （阶段跑完后 live_line 会停在 100% 那行，但那时
                //     StageDone 已经把 stage_frac 置 1.0，不影响。）
                // ★ live_line 恢复成单纯的"最新一行"。
                //
                //   曾经为了修「进度条不动」让它优先保留 tqdm 行，结果
                //   字幕行（非 tqdm）被条件排除、根本写不进去 ——
                //   实时预览整段空白，直到这一步跑完才一次性冒出来。
                //
                //   现在进度改用独立通道 live_frac（下面几行），
                //   live_line 只负责给「当前动作」和「实时预览」用。
                if let Ok(mut l) = latest2.lock() {
                    *l = line.to_string();
                }

                // ★ 独立通道：把 tqdm 进度写进 live_frac。
                //
                //   为什么必须分开：第4步识别期间，tqdm 每 3 秒一行，
                //   字幕行每 0.3 秒一条，两者共用 live_line 必然互相覆盖 ——
                //   界面 100ms 轮询时，要么读到字幕（丢进度），
                //   要么读到 tqdm（丢字幕）。实测两种症状都出现过。
                if let Some(c) = config::tqdm_re().captures(line) {
                    let cur: f64 = c[1].parse().unwrap_or(0.0);
                    let tot: f64 = c[2].parse().unwrap_or(0.0);
                    if tot > 0.0 {
                        if let Ok(mut f) = frac2.lock() {
                            *f = (cur / tot).clamp(0.0, 1.0);
                        }
                    }
                }

                // ★ 顺手把 tqdm 自己算的剩余时间也抓下来。
                //
                //   tqdm 行里有 [00:01<01:12, 18.53seconds/s]，
                //   尖括号后面就是它算的剩余。这是【按真实处理速度】
                //   实时更新出来的，比我们拿成本模型估准得多 ——
                //   而且从第一帧就有了。
                if let Some(c) = config::tqdm_eta_re().captures(line) {
                    if let Some(rem) = config::parse_hms(&c[2]) {
                        if let Ok(mut r) = eta2.lock() {
                            // 注意：rem 可以是 0（tqdm 说"没剩了"），
                            // 这和"还没拿到 ETA"是两回事。
                            // 用 -1 表示后者，见 ui 侧的初始化。
                            *r = rem;
                        }
                    }
                }
            }
            if let Ok(mut v) = sink2.lock() {
                v.push(line.to_string());
            }
        },
    )
    .unwrap_or(-1);

    let lines = match Arc::try_unwrap(sink) {
        Ok(m) => m.into_inner().unwrap_or_default(),
        Err(e) => e.lock().map(|v| v.clone()).unwrap_or_default(),
    };

    StageResult { stage, code, lines }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 在临时目录下造一个测试场景，返回 (工作根, srt 目录, 视频目录)
    fn make_scene(name: &str) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("asmr_test_{name}"));
        let _ = std::fs::remove_dir_all(&root);
        let srt = root.join("3asr");
        let vid = root.join("videos");
        std::fs::create_dir_all(&srt).unwrap();
        std::fs::create_dir_all(&vid).unwrap();
        (root, srt, vid)
    }

    // 核心行为：3asr\ 里带 _中文字幕 / _(Vocals) 后缀的文件，
    // 都要剥掉后缀、改成 .chs.srt，然后按勾选决定去向。
    //
    // .chs.srt 这个扩展名是关键：PotPlayer 靠语言代码识别外挂字幕，
    // 叫 xxx_中文字幕.srt 它不认，不会自动加载 —— 这是真实踩过的坑。
    #[test]
    fn strip_suffix_and_move_to_video_dir() {
        let (root, srt, vid) = make_scene("move");

        std::fs::write(srt.join("第一集_中文字幕.srt"), "字幕内容").unwrap();
        std::fs::write(srt.join("第二集_(Vocals)_model_x.srt"), "字幕内容").unwrap();
        std::fs::write(srt.join("孤儿.srt"), "没有对应视频").unwrap();

        let files = vec![
            vid.join("第一集.mp4").to_string_lossy().to_string(),
            vid.join("第二集.mkv").to_string_lossy().to_string(),
        ];

        let logs = finalize_subtitles_in(&srt.to_string_lossy(), &srt.to_string_lossy(), &files, true);

        assert!(vid.join("第一集.chs.srt").is_file(), "第一集应移到视频目录并改名");
        assert!(vid.join("第二集.chs.srt").is_file(), "第二集应移到视频目录并改名");
        assert!(!srt.join("第一集_中文字幕.srt").exists(), "原文件应已移走");
        assert!(srt.join("孤儿.chs.srt").is_file(), "孤儿字幕应放进输出目录并改名");

        let joined = logs.join("\n");
        assert!(joined.contains("移到视频旁 2 个"), "日志应统计出 2 个: {joined}");
        assert!(joined.contains("放进输出目录"), "日志应有放进输出目录的计数: {joined}");

        let _ = std::fs::remove_dir_all(&root);
    }

    // 不勾选时，有对应视频的字幕也应该【留在 3asr\】，
    // 但仍然要改名成 .chs.srt（否则播放器还是不认识）。
    #[test]
    fn keep_in_place_but_still_rename() {
        let (root, srt, vid) = make_scene("keep");

        std::fs::write(srt.join("第三集_中文字幕.srt"), "字幕内容").unwrap();
        let files = vec![vid.join("第三集.mp4").to_string_lossy().to_string()];

        finalize_subtitles_in(&srt.to_string_lossy(), &srt.to_string_lossy(), &files, false);

        assert!(srt.join("第三集.chs.srt").is_file(), "应留在 3asr 并改名");
        assert!(!vid.join("第三集.chs.srt").exists(), "不该跑到视频目录去");

        let _ = std::fs::remove_dir_all(&root);
    }

    // 目标文件已存在时必须能覆盖 —— Windows 的 rename 不会自动覆盖，
    // 不先删掉就会失败，然后字幕就丢在原地了。
    #[test]
    fn overwrite_existing_target() {
        let (root, srt, vid) = make_scene("overwrite");

        std::fs::write(srt.join("第四集_中文字幕.srt"), "新字幕").unwrap();
        std::fs::write(vid.join("第四集.chs.srt"), "旧字幕").unwrap();

        let files = vec![vid.join("第四集.mp4").to_string_lossy().to_string()];
        // 源 = srt（字幕生成的地方），目标 = 视频所在目录（copy_to_video=true 时会用它）
        finalize_subtitles_in(&srt.to_string_lossy(), &vid.to_string_lossy(), &files, true);

        let content = std::fs::read_to_string(vid.join("第四集.chs.srt")).unwrap();
        assert_eq!(content, "新字幕", "旧文件应被新字幕覆盖");

        let _ = std::fs::remove_dir_all(&root);
    }

    // 目录不存在时不能 panic，返回一条提示就行
    #[test]
    fn missing_dir_does_not_panic() {
        let logs = finalize_subtitles_in(
            r"Z:\definitely\not\here",
            r"Z:\definitely\not\here",
            &[],
            true,
        );
        assert!(logs[0].contains("找不到目录"), "应给出提示: {logs:?}");
    }
}

/// 彻底删除所有中间过程文件，只保留字幕。返回 (删掉个数, 失败个数)。
///
/// 清理范围（按流程顺序）：
///     0pre\    —— 用户音视频的副本
///     1audio\  —— 人声分离出的 WAV（可能有几百 MB）
///     2vad\    —— VAD 切出来的片段 srt
/// 保留：
///     3asr\    —— 最终字幕，绝对不碰
///
/// ★ 用 std::fs::remove_file —— 这是【永久删除】：不进回收站、无法撤销。
///   这是用户明确要求的行为：字幕一生成完就全删干净，中途点停止也立刻删，
///   不弹窗问、也不留残骸。
///
/// ⚠️ 刚杀过子进程时文件句柄可能还没释放，所以失败要重试几轮。
///    Python 版是重试 4 次、每次 sleep 0.3s，这里保持一致。
///
/// ⚠️ 只删文件，不删目录本身 —— 四个脚本都假定这些目录存在。
///
/// `include_pre` 控制要不要连 0pre\ 一起清：
///    · 用户选了文件 → 清（马上会重新放新的进去）
///    · 用户没选文件、要靠 0pre\ 里现成的东西跑 → 【不清】，
///      否则就把他手动放进去的文件删掉了，反而没东西可处理。
pub fn purge_process_files(include_pre: bool) -> (usize, usize) {
    let mut deleted = 0usize;
    let mut failed = 0usize;

    let mut dirs: Vec<String> = vec![config::dir_audio(), config::dir_vad()];
    if include_pre {
        dirs.insert(0, config::dir_pre());
    }

    // 源文件清单（_input.txt）也要清掉。
    // 它是 prepare 阶段写在【工作目录根】的那份路径列表，
    // 留着的话下次 a_提取音频.py 会读到过期内容。
    if include_pre {
        let list_path = format!(r"{}\_input.txt", config::work_dir());
        if std::fs::remove_file(&list_path).is_ok() {
            deleted += 1;
        }
    }

    for d in dirs {
        if !std::path::Path::new(&d).is_dir() {
            continue;
        }

        let mut last_round_failed = 0usize;
        for attempt in 0..4u32 {
            let files: Vec<std::path::PathBuf> = match std::fs::read_dir(&d) {
                Ok(rd) => rd
                    .flatten()
                    .filter(|e| e.path().is_file())
                    .map(|e| e.path())
                    .collect(),
                Err(_) => break,
            };
            if files.is_empty() {
                break;
            }

            // 每轮重新算，最后只用最后一轮的结果报告失败数 ——
            // 不用累加，否则同一个文件在第一轮失败、第二轮成功会被记两次。
            let mut round_failed = 0usize;
            for f in &files {
                match std::fs::remove_file(f) {
                    Ok(_) => deleted += 1,
                    Err(_) => round_failed += 1,
                }
            }
            last_round_failed = round_failed;

            if round_failed == 0 || attempt == 3 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
        failed += last_round_failed;
    }

    (deleted, failed)
}
