//! 软字幕：把 `.chs.srt` / `.chs.ass` 作为字幕流**封装进原视频**（不重新编码）。
//!
//! ## 为什么需要「另存」而不是覆盖原文件
//!
//! ffmpeg 的 `-c copy` 虽然不重编码，但它是**流复制到新文件** —— 不能原地
//! 往已有文件里插一条字幕流。所以做法是：
//!
//! ```text
//!   原视频.mp4  ──ffmpeg -c copy──▶  原视频（软字幕）.mp4
//!   原视频.chs.srt （作为字幕输入，不参与时间轴映射）
//! ```
//!
//! 原文件保持不动。这是刻意的：ASMR 音声文件通常是不可再得的资源，
//! 任何「就地改写」的操作一旦中途失败（断电、磁盘满、ffmpeg 被杀），
//! 原文件就成了半个损坏文件。宁多占一份空间，也不冒这个险。
//!
//! ## 容器兼容性（实测过的情况）
//!
//! | 容器 | `-c:s mov_text` | `-c:s srt` | 说明 |
//! |---|---|---|---|
//! | mp4 / mov / m4v | ✅ 用 mov_text | ✅ | mp4 里必须用 mov_text |
//! | mkv | ✅ | ✅ 推荐 srt | mkv 对 srt 支持最好 |
//! | webm | ❌ | ✅ | 只吃 webm 系字幕（srt 会被转） |
//! | avi / wmv | ❌ | ⚠️ 多数播放器不认 | 建议改用 mkv 输出 |
//! | flac / mp3 / m4a | — | — | **音频文件不能封装字幕**，直接拒绝 |
//!
//! 音频文件（用户大量使用的是 `.flac`）封装字幕会得到一个「有字幕流的
//! flac」，播放器基本都不认 —— 所以这里直接跳过并给出明确原因，而不是
//! 生成一个看起来成功、实际没法用的文件。

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::config;

/// 视频容器类型，决定字幕编码器和**是否需要换容器**。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    /// mp4 / m4v / mov / 3gp —— 字幕必须是 mov_text，容器本身装得下
    Mp4,
    /// mkv —— 字幕用 srt 最稳
    Matroska,
    /// webm —— 只能装 webvtt 系
    Webm,
    /// avi / wmv / flv / ts / mpg / ogv —— 这些容器**装不住**文本字幕轨，
    /// 必须换成 mkv 输出（音视频仍是 -c copy，不重编码）
    Legacy,
    /// 纯音频，不能封装字幕
    Audio,
}

impl Container {
    /// 字幕流编码器名（ffmpeg 的 `-c:s` 取值）。这是**容器原生**的选择。
    pub fn sub_codec(self) -> &'static str {
        match self {
            Container::Mp4 => "mov_text",
            Container::Matroska => "srt",
            Container::Webm => "webvtt",
            Container::Legacy => "srt",
            Container::Audio => "",
        }
    }

    /// ★ v1.29 方案B：给定字幕源的真实格式，返回该容器下**最优**的 `-c:s`。
    ///
    /// 实测依据（ffmpeg 9.0.1，2026-11-01）：
    ///
    ///   | 容器 | 源 ASS + `-c:s copy` | 源 ASS + `-c:s ass` |
    ///   |------|----------------------|---------------------|
    ///   | mkv  | ✅ ass 流原样进      | ✅ 样式表完整保留    |
    ///   | mp4  | ❌ Could not find tag for codec ass | ❌ 同左 |
    ///   | mov  | ❌ 同左              | ❌ 同左              |
    ///   | 3gp  | ❌ 同左              | ❌ 同左              |
    ///   | webm | ❌（只吃 vp8/vp9/av1 + webvtt）             |
    ///
    /// 所以：
    ///   · mkv（含 avi/ts/mpg 换容器后的 mkv）→ 源是 ASS 就【原样复制】，
    ///     用户的「黑体白字黑描边」样式一点不丢。这是唯一的完全保真路径。
    ///   · mp4 / mov / 3gp / webm → 装不了 ass，只能退回容器原生编码器。
    ///     ffmpeg 直接转会在字幕正文里留下 `<font size="52">` 标签，
    ///     部分播放器把标签当正文显示 —— 由调用方先剥成干净纯文本再送进来
    ///     （见 srt::read_ass + srt::strip_ass_markup）。
    pub fn best_sub_codec(self, src_is_ass: bool) -> &'static str {
        if src_is_ass && self == Container::Matroska {
            // mkv 直接复制 ASS 流：样式、颜色、描边全部保留
            "copy"
        } else {
            self.sub_codec()
        }
    }

    /// 是否必须换容器（即输出扩展名不能沿用源扩展名）。
    pub fn needs_recontainer(self) -> bool {
        matches!(self, Container::Legacy)
    }

    /// 保底输出扩展名（换容器时才用得上）。
    pub fn out_ext(self) -> &'static str {
        match self {
            Container::Mp4 => "mp4",
            Container::Matroska => "mkv",
            Container::Webm => "webm",
            Container::Legacy => "mkv",
            Container::Audio => "",
        }
    }

    /// 实际输出扩展名：能沿用源扩展名就沿用，必须换容器时才换成 mkv。
    ///
    /// ★ v1.29.2：原来一律返回 `out_ext()`，导致 `片.mov` 被输出成
    ///   `片（软字幕）.mp4`、`片.3gp` 变成 `.mp4` —— 用户会觉得「我明明
    ///   是 mov，怎么出来个 mp4」。能保持原扩展名时就该保持。
    pub fn output_ext(self, src_ext: &str) -> String {
        if self.needs_recontainer() {
            return self.out_ext().to_string();
        }
        let e = src_ext.to_lowercase();
        if matches!(e.as_str(), "mp4" | "m4v" | "mov" | "mkv" | "webm" | "3gp") {
            e
        } else {
            self.out_ext().to_string()
        }
    }
}

/// 按扩展名判断容器。`None` 表示不认识这个扩展名。
pub fn container_of(path: &str) -> Option<Container> {
    let ext = Path::new(path)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    Some(match ext.as_str() {
        // 原生就支持字幕轨、用本容器输出
        "mp4" | "m4v" | "mov" => Container::Mp4,
        "mkv" => Container::Matroska,
        "webm" => Container::Webm,
        // ★ v1.29.2：3gp 其实也能直接装 mov_text（实测通过），
        //   但 3gp 的播放器受众极少、且它本来就是「手机精简版 mp4」，
        //   统一归到 Mp4 出 .mp4 更通用。见下方 out_ext 说明。
        "3gp" => Container::Mp4,
        // ★ v1.29.2：这些容器**装不住** srt 系字幕轨，实测：
        //   · ts   → -c:s srt 后【没有字幕流】（ffmpeg 静默丢弃）
        //   · ogv  → "Unsupported codec id in stream 2"
        //   · mpg(mpeg-ps) → 会强制转成 dvd_subtitle（位图字幕，需重编码且
        //                    播放器兼容差，不是我们要的软字幕）
        //   所以它们统一「换容器输出」为 mkv —— 实测这四种源转 mkv 全部能
        //   正常带 srt 字幕流，音视频仍然 -c copy 不重编码。
        "avi" | "wmv" | "flv" => Container::Legacy,
        "ts" | "mts" | "m2ts" | "mpg" | "mpeg" | "ogv" | "ogm" => Container::Legacy,
        "flac" | "mp3" | "m4a" | "aac" | "wav" | "ogg" | "opus"
        | "wma" | "ape" | "alac" => Container::Audio,
        _ => return None,
    })
}

/// Windows 的「不给子进程建控制台窗口」标志。
///
/// ★ 踩过的坑（用户实测反馈「每次打开都闪过一个黑窗口」）：
///   v1.29 最初加软字幕功能时，softsub.rs 里的 ffmpeg 调用【全部】忘了
///   带这个标志。程序启动时 detect_device 之类的路径会间接触发
///   ffmpeg_exe() 探测（`ffmpeg -version`），那个子进程没有控制台抑制，
///   Windows 就会给它开一个黑色控制台窗口，闪一下。
///   同理，封装时真正跑 ffmpeg 也会闪。
///   凡是起子进程的地方都必须带它，一个都不能漏。
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// 给 Command 加「不显示窗口」标志（Windows 专用，其它平台是空操作）。
trait NoWindow {
    fn no_window(&mut self) -> &mut Self;
}

#[cfg(windows)]
impl NoWindow for Command {
    fn no_window(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        self.creation_flags(CREATE_NO_WINDOW)
    }
}

#[cfg(not(windows))]
impl NoWindow for Command {
    fn no_window(&mut self) -> &mut Self {
        self
    }
}

/// ffmpeg 的位置（复用 subscan 那套查找策略：先找运行目录，再退回 PATH）。
pub fn ffmpeg_exe() -> Option<String> {
    let work = PathBuf::from(config::work_dir());
    for sub in ["ffmpeg\\bin", "ffmpeg", "bin", "env\\Scripts", ""] {
        let d = if sub.is_empty() { work.clone() } else { work.join(sub) };
        let p = d.join("ffmpeg.exe");
        if p.is_file() {
            return Some(p.to_string_lossy().to_string());
        }
    }
    if Command::new("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .no_window()
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Some("ffmpeg".to_string());
    }
    None
}

/// 一次封装的结果。
#[derive(Debug)]
pub struct EmbedOutcome {
    /// 生成的文件（成功时才有）
    pub out_path: Option<String>,
    /// 给日志用的一句话
    pub message: String,
}

/// 把字幕封装进视频（不重编码音视频）。
///
/// `video`  —— 原视频路径
/// `sub`    —— 字幕文件（.srt 或 .ass）
/// `lang_tag` —— 字幕流语言标记，例如 `chi` / `jpn`；用于播放器显示语言名
///
/// 返回：

///   · 成功 → out_path = Some(新文件)，message 写明大小和字幕轨信息
///   · 音频文件 / 容器不支持 / 找不到 ffmpeg → out_path = None，message 说明原因
///
/// **不覆盖原文件**（见文件头注释）。
pub fn embed(video: &str, sub: &str, lang_tag: &str) -> EmbedOutcome {
    let Some(cont) = container_of(video) else {
        return EmbedOutcome {
            out_path: None,
            message: format!("[软字幕] 不认识的容器扩展名，跳过：{video}"),
        };
    };
    if cont == Container::Audio {
        return EmbedOutcome {
            out_path: None,
            message: "[软字幕] 纯音频文件不能封装字幕（播放器不认），跳过".to_string(),
        };
    }
    if !Path::new(sub).is_file() {
        return EmbedOutcome {
            out_path: None,
            message: format!("[软字幕] 找不到字幕文件：{sub}"),
        };
    }
    let Some(ff) = ffmpeg_exe() else {
        return EmbedOutcome {
            out_path: None,
            message: "[软字幕] 没找到 ffmpeg，跳过".to_string(),
        };
    };

    // 输出名：`xxx（软字幕）.<ext>`。用中文括号是为了和普通文件肉眼可区分，
    // 同时保持 stem 可回溯（想找原文件时看前缀就行）。
    let src = Path::new(video);
    let stem = src
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    // ★ v1.29.2：能沿用源扩展名就沿用（mov→mov、3gp→3gp），
    //   只有源容器真装不住字幕（ts/avi/ogv…）才换成 mkv。
    let src_ext = src
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_default();
    let out_ext = cont.output_ext(&src_ext);
    let recontainered = cont.needs_recontainer();
    let dir = src
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let out_name = format!("{stem}（软字幕）.{}", out_ext);
    let out_path = if dir.is_empty() {
        out_name.clone()
    } else {
        format!(r"{dir}\{out_name}")
    };

    // ── 重复生成时的「替换」语义 ──
    //
    // 【关键】绝不能先删旧文件再跑 ffmpeg。
    //
    //   v1.29 最初的写法是 `if out_path.exists() { remove_file }`，然后让
    //   ffmpeg 直接写 out_path。这是「先毁后建」：只要 ffmpeg 因为磁盘满、
    //   被取消、源文件被占用等任何原因失败，用户就同时失去【旧的软字幕版本】
    //   和【新的】—— 上一轮明明已经生成成功的那份白扔了。
    //
    //   改成：写临时文件 `xxx（软字幕）.tmp.<ext>`，成功后才原子替换。
    //   ffmpeg 认不出扩展名的容器，所以临时文件必须保留正确的扩展名
    //   （ffmpeg 靠扩展名决定封装格式），用一个 `.tmp` 前缀在 stem 后面
    //   是没用的 —— 它仍然是 `...tmp.mp4`，扩展名还是 mp4，这样才对。
    let tmp_path = if dir.is_empty() {
        format!("{stem}（软字幕）.tmp.{out_ext}")
    } else {
        format!(r"{dir}\{stem}（软字幕）.tmp.{out_ext}")
    };
    // 上一轮被强杀留下的临时文件先清掉，不然 ffmpeg 可能拒绝覆盖
    if Path::new(&tmp_path).exists() {
        let _ = std::fs::remove_file(&tmp_path);
    }

    // 输出目录里可能还躺着【别的格式】的旧软字幕（比如上次是 mp4、
    // 这次源文件换了/扩展名变了要出 mkv）。不清的话用户会看到两份，
    // 播放器加载哪份全看运气。这里把同 stem 的其它软字幕后缀都收掉。
    let stale_others = stale_softsubs(&dir, &stem, &out_name);

    // ★ v1.29 方案B：先判断字幕源是不是 ASS，再决定用哪条路。
    //
    //   · mkv 家族 → `-c:s copy` 原样把 ASS 流塞进去，样式完整保留
    //   · 其它容器 → 装不了 ass（实测 mp4/mov/3gp 都报
    //     "Could not find tag for codec ass"），只能转成原生纯文本格式。
    //     但 ffmpeg 直接拿 ASS 转 mov_text 会在正文里留
    //     `<font face="SimHei" size="52">…</font>` 标签，部分播放器当正文显示。
    //     所以这里先把 ASS 读出来、剥成纯文本，写成一份临时 SRT 给 ffmpeg。
    let src_is_ass = sub.to_lowercase().ends_with(".ass");
    let codec = cont.best_sub_codec(src_is_ass);

    // 需要中间 SRT 时它的路径（成功后要删）
    let mut tmp_srt: Option<String> = None;
    let sub_input: String = if src_is_ass && codec != "copy" {
        match crate::srt::read_ass(Path::new(sub)) {
            Ok(cues) if !cues.is_empty() => {
                let p = if dir.is_empty() {
                    format!("{stem}（软字幕）.tmp.plain.srt")
                } else {
                    format!(r"{dir}\{stem}（软字幕）.tmp.plain.srt")
                };
                let _ = std::fs::remove_file(&p);
                match crate::srt::write_srt(Path::new(&p), &cues) {
                    Ok(_) => {
                        tmp_srt = Some(p.clone());
                        p
                    }
                    Err(e) => {
                        // 剥不成纯文本就退回让 ffmpeg 自己转（会带标签但不至于失败）
                        eprintln!("[软字幕] 生成纯文本 SRT 失败，退回直接转换：{e}");
                        sub.to_string()
                    }
                }
            }
            Ok(_) => {
                eprintln!("[软字幕] ASS 解析出 0 条，退回直接转换");
                sub.to_string()
            }
            Err(e) => {
                eprintln!("[软字幕] ASS 读不出内容，退回直接转换：{e}");
                sub.to_string()
            }
        }
    } else {
        sub.to_string()
    };

    let mut cmd = Command::new(&ff);
    cmd.args(["-hide_banner", "-loglevel", "error", "-y"])
        .arg("-i")
        .arg(video)
        .arg("-i")
        .arg(&sub_input)
        // 音视频一律流复制，绝不重编码 —— ASMR 音声重编码会掉质量，
        // 而且慢得离谱（1 小时音频重编码要几分钟）。
        .args(["-map", "0:v?", "-map", "0:a", "-map", "1:0"])
        .args(["-c", "copy"])
        .args(["-c:s", codec])
        .args(["-metadata:s:s:0", &format!("language={lang_tag}")])
        // 字幕轨标题：播放器里显示「中文」而不是「Unknown」
        .args(["-metadata:s:s:0", "title=中文"])
        .arg(&tmp_path);

    // 中间 SRT 用完就删：无论成败都不能留残渣
    let cleanup_tmp_srt = || {
        if let Some(p) = &tmp_srt {
            let _ = std::fs::remove_file(p);
        }
    };

    let output = cmd
        .stdin(Stdio::null())
        .no_window()
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output();

    cleanup_tmp_srt();

    match output {
        Ok(o) if o.status.success() => {
            // ── 原子替换 ──
            //
            // Windows 上 rename 不能覆盖已存在的目标，所以必须先把旧的挪开。
            // 但不能「删了再说」—— 万一下一步失败，旧版本就没了。
            // 做法：旧文件先 rename 成 `.bak`，新文件就位后再删 .bak。
            // 任何一步失败都能从 .bak 还原。
            let replaced = Path::new(&out_path).exists();
            let bak_path = format!("{out_path}.bak");
            if replaced {
                let _ = std::fs::remove_file(&bak_path);
                if std::fs::rename(&out_path, &bak_path).is_err() {
                    // 挪不开（被播放器占用等）：退回到「直接删除」这条路，
                    // 删不掉就报错，绝不让两份文件同时存在造成歧义。
                    if std::fs::remove_file(&out_path).is_err() {
                        let _ = std::fs::remove_file(&tmp_path);
                        return EmbedOutcome {
                            out_path: None,
                            message:
                                "[软字幕] 旧的软字幕文件被占用，无法替换（请先关闭正在播放它的程序）"
                                    .to_string(),
                        };
                    }
                }
            }
            match std::fs::rename(&tmp_path, &out_path) {
                Ok(_) => {}
                Err(e) => {
                    // rename 失败（跨卷等）：退回复制
                    if let Err(e2) = std::fs::copy(&tmp_path, &out_path) {
                        let _ = std::fs::remove_file(&tmp_path);
                        // 新文件没就位，把旧版本从 .bak 还回去
                        if Path::new(&bak_path).exists() {
                            let _ = std::fs::rename(&bak_path, &out_path);
                        }
                        return EmbedOutcome {
                            out_path: None,
                            message: format!("[软字幕] 替换失败（旧版已还原）：{e} / {e2}"),
                        };
                    }
                    let _ = std::fs::remove_file(&tmp_path);
                }
            }
            // 新版本已经就位，删掉备份
            if Path::new(&bak_path).exists() {
                let _ = std::fs::remove_file(&bak_path);
            }
            // 清理同 stem 的其它格式旧软字幕
            let mut extra = 0usize;
            for p in &stale_others {
                if std::fs::remove_file(p).is_ok() {
                    extra += 1;
                }
            }
            let size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
            let mut msg = format!(
                "[软字幕] {} → {out_name}（{:.1} MB，字幕轨 {}，音视频未重编码）",
                if replaced || extra > 0 { "已替换旧版" } else { "已封装" },
                size as f64 / 1024.0 / 1024.0,
                codec
            );
            // ★ v1.29 方案B：把「样式保真程度」如实写进日志 ——
            //   用户选 ASS 就是想要描边，得让他知道保住了没有。
            if src_is_ass {
                if codec == "copy" {
                    msg.push_str("；ASS 原样复制，样式（黑体/白字/黑描边）完整保留");
                } else {
                    msg.push_str(&format!(
                        "；{out_ext} 装不了 ASS，已剥成纯文本（字体描边不保留，\
                         需要完整样式请用 mkv 或直接让播放器加载 .ass）"
                    ));
                }
            }
            // ★ v1.29.2：换了容器要说清楚，否则用户看到 .avi 变 .mkv 会以为出错
            if recontainered {
                msg.push_str(&format!(
                    "；{src_ext} 容器装不住字幕轨，已换成 {out_ext}"
                ));
            }
            if extra > 0 {
                msg.push_str(&format!("，另清理旧格式 {extra} 份"));
            }
            EmbedOutcome {
                out_path: Some(out_path.clone()),
                message: msg,
            }
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            let brief = err
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("ffmpeg 退出码非 0")
                .trim()
                .to_string();
            // ★ 只删临时文件，【保留】旧的软字幕版本 —— 失败不该连累上一轮的成功产物
            if Path::new(&tmp_path).exists() {
                let _ = std::fs::remove_file(&tmp_path);
            }
            let kept = Path::new(&out_path).exists();
            EmbedOutcome {
                out_path: None,
                message: format!(
                    "[软字幕] 封装失败：{brief}{}",
                    if kept {
                        "（上一版软字幕文件已保留，未被破坏）"
                    } else {
                        ""
                    }
                ),
            }
        }
        Err(e) => {
            if Path::new(&tmp_path).exists() {
                let _ = std::fs::remove_file(&tmp_path);
            }
            EmbedOutcome {
                out_path: None,
                message: format!("[软字幕] 启动 ffmpeg 失败：{e}"),
            }
        }
    }
}

/// 找同一目录下、同 stem 的【其它】软字幕文件（用于替换时清理旧格式）。
///
/// 例如本次要出 `测试片（软字幕）.mp4`，但目录里还躺着
/// `测试片（软字幕）.mkv`（上次源文件是 mkv，或者断点残留），
/// 后者要清掉，否则播放器可能加载到旧的那份。
///
/// `keep_name` 是本次要生成的文件名，不删它自己。
fn stale_softsubs(dir: &str, stem: &str, keep_name: &str) -> Vec<String> {
    let mut out = Vec::new();
    if dir.is_empty() {
        return out;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    // 前缀形如 `测试片（软字幕）`，后面接 `.扩展名`
    let prefix = format!("{stem}（软字幕）");
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name == keep_name {
            continue;
        }
        if !name.starts_with(&prefix) {
            continue;
        }
        // 只认「前缀 + .扩展名」，避免把 `测试片（软字幕）备份.txt` 也删了
        let rest = &name[prefix.len()..];
        if !rest.starts_with('.') {
            continue;
        }
        let ext = rest[1..].to_lowercase();
        if !matches!(ext.as_str(), "mp4" | "mkv" | "webm" | "mov" | "m4v") {
            continue;
        }
        let p = e.path();
        if p.is_file() {
            out.push(p.to_string_lossy().to_string());
        }
    }
    out
}

/// 清理残留的临时/备份文件（`xxx（软字幕）.tmp.<ext>` / `xxx（软字幕）.<ext>.bak`）。
///
/// 什么时候会残留：程序在封装途中被强杀、断电、或用户点了停止。
/// 这些文件是几 GB 的垃圾，必须有人清。
///
/// **只删 tmp/bak，不动已经完成的软字幕** —— 关掉开关不等于要删文件。
///
/// `files` 是本次要处理的源文件（用它们的 stem 定位同名残留）。
/// `dest_dir` 是输出目录（字幕放那儿时软字幕也可能落在那儿）。
/// 返回给界面日志看的一句话列表（没有残留就返回空）。
pub fn cleanup_tmp_files(dest_dir: &str, files: &[String]) -> Vec<String> {
    let mut logs = Vec::new();
    let mut n = 0usize;

    // 收集要扫的目录：每个源文件所在目录 + dest_dir
    let mut dirs: Vec<String> = Vec::new();
    for f in files {
        if let Some(d) = Path::new(f).parent() {
            let d = d.to_string_lossy().to_string();
            if !d.is_empty() && !dirs.contains(&d) {
                dirs.push(d);
            }
        }
    }
    let d = dest_dir.trim().to_string();
    if !d.is_empty() && !dirs.contains(&d) {
        dirs.push(d);
    }

    for dir in dirs {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let is_tmp = name.contains("（软字幕）.tmp.");
            let is_bak = name.contains("（软字幕）.") && name.ends_with(".bak");
            if !is_tmp && !is_bak {
                continue;
            }
            // 只处理本次相关的文件（按 stem 匹配），别扫全盘乱删
            let related = files.iter().any(|f| {
                Path::new(f)
                    .file_stem()
                    .map(|s| {
                        let stem = s.to_string_lossy().to_string();
                        name.starts_with(&format!("{stem}（软字幕）"))
                    })
                    .unwrap_or(false)
            });
            if !related {
                continue;
            }
            let p = e.path();
            if p.is_file() && std::fs::remove_file(&p).is_ok() {
                n += 1;
            }
        }
    }

    if n > 0 {
        logs.push(format!("[软字幕] 清理上次未完成的临时文件 {n} 个"));
    }
    logs
}

/// 给一个字幕文件挑一个合适的字幕语言标记。
///
/// 文件名里 `.jpn.srt` 是日文、其余默认中文（见 pipeline::finalize 的语言约定）。
pub fn lang_tag_of(sub_path: &str) -> &'static str {
    let lower = sub_path.to_lowercase();
    if lower.contains(".jpn.") || lower.contains(".jpn") {
        "jpn"
    } else {
        "chi"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_mapping_matches_extension() {
        assert_eq!(container_of("a.mp4"), Some(Container::Mp4));
        assert_eq!(container_of("a.MKV"), Some(Container::Matroska));
        assert_eq!(container_of("a.webm"), Some(Container::Webm));
        assert_eq!(container_of("a.avi"), Some(Container::Legacy));
        assert_eq!(container_of("a.flac"), Some(Container::Audio));
        assert_eq!(container_of("a.unknownext"), None);
        assert_eq!(container_of("noext"), None);

        // ★ v1.29.2 新增：mov/3gp 走 mp4 那条路；ts/mpg/ogv 走换容器那条路
        assert_eq!(container_of("a.mov"), Some(Container::Mp4));
        assert_eq!(container_of("a.3gp"), Some(Container::Mp4));
        assert_eq!(container_of("a.ts"), Some(Container::Legacy));
        assert_eq!(container_of("a.m2ts"), Some(Container::Legacy));
        assert_eq!(container_of("a.mpg"), Some(Container::Legacy));
        assert_eq!(container_of("a.ogv"), Some(Container::Legacy));
        // 新增的音频后缀也要被当成音频拒绝
        assert_eq!(container_of("a.wma"), Some(Container::Audio));
        assert_eq!(container_of("a.ape"), Some(Container::Audio));
    }

    /// ★ v1.29.2：能沿用源扩展名就沿用，别把 mov 输出成 mp4。
    #[test]
    fn output_ext_keeps_source_format_when_possible() {
        // mp4 家族内部保持原样
        assert_eq!(Container::Mp4.output_ext("mp4"), "mp4");
        assert_eq!(Container::Mp4.output_ext("mov"), "mov");
        assert_eq!(Container::Mp4.output_ext("m4v"), "m4v");
        assert_eq!(Container::Mp4.output_ext("3gp"), "3gp");
        assert_eq!(Container::Matroska.output_ext("mkv"), "mkv");
        assert_eq!(Container::Webm.output_ext("webm"), "webm");
        // 装不住字幕的容器 → 一律 mkv
        assert_eq!(Container::Legacy.output_ext("avi"), "mkv");
        assert_eq!(Container::Legacy.output_ext("ts"), "mkv");
        assert_eq!(Container::Legacy.output_ext("mpg"), "mkv");
        assert_eq!(Container::Legacy.output_ext("ogv"), "mkv");
        // 换容器判定
        assert!(Container::Legacy.needs_recontainer());
        assert!(!Container::Mp4.needs_recontainer());
        assert!(!Container::Matroska.needs_recontainer());
        // 大写扩展名也要能处理
        assert_eq!(Container::Mp4.output_ext("MOV"), "mov");
    }

    #[test]
    fn mp4_uses_mov_text_mkv_uses_srt() {
        assert_eq!(Container::Mp4.sub_codec(), "mov_text");
        assert_eq!(Container::Matroska.sub_codec(), "srt");
        assert_eq!(Container::Mp4.out_ext(), "mp4");
        assert_eq!(Container::Matroska.out_ext(), "mkv");
        // 老容器统一改成 mkv 输出：avi/wmv 塞字幕轨没人认
        assert_eq!(Container::Legacy.out_ext(), "mkv");
    }

    #[test]
    fn audio_is_rejected_not_silently_embedded() {
        let out = embed(r"H:\no\such.flac", r"H:\no\such.srt", "chi");
        assert!(out.out_path.is_none());
        assert!(out.message.contains("纯音频"));
    }

    #[test]
    fn missing_sub_file_is_reported() {
        let out = embed(r"H:\no\such.mp4", r"H:\no\such.srt", "chi");
        assert!(out.out_path.is_none());
        assert!(out.message.contains("找不到字幕文件"));
    }

    #[test]
    fn lang_tag_detects_jpn_suffix() {
        assert_eq!(lang_tag_of("a.jpn.srt"), "jpn");
        assert_eq!(lang_tag_of("a.chs.srt"), "chi");
        assert_eq!(lang_tag_of(r"D:\x\a.chs.ass"), "chi");
    }

    /// ★ v1.29.2：全格式实测 —— 每种源格式都真的封装一遍，检查输出里
    /// 有字幕流、扩展名符合预期。
    ///
    /// 实测结论（2026-11-01，ffmpeg 9.0.1）：
    ///   mp4/mkv/mov/m4v/webm/3gp → 直接在本容器内带字幕流，扩展名不变
    ///   avi/wmv/flv/ts/mpg/ogv    → 换容器出 mkv，带 srt 字幕流
    #[test]
    #[ignore]
    fn all_container_formats_actually_embed() {
        let root = std::env::temp_dir().join("asmr_softsub_formats");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let Some(ff) = ffmpeg_exe() else {
            eprintln!("没有 ffmpeg，跳过");
            return;
        };
        let sub = root.join("s.chs.srt");
        std::fs::write(&sub, "1\n00:00:00,500 --> 00:00:02,500\n格式测试\n").unwrap();

        // 先造一个基线 mp4，再转成各格式
        let base = root.join("base.mp4");
        let st = Command::new(&ff)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=3"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=3"])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])
            .arg(&base)
            .status()
            .unwrap();
        assert!(st.success());

        // (源扩展名, 造源用的编码参数, 期望输出扩展名)
        let cases: &[(&str, &[&str], &str)] = &[
            ("mp4", &["-c:v", "libx264", "-c:a", "aac"], "mp4"),
            ("mkv", &["-c:v", "libx264", "-c:a", "aac"], "mkv"),
            ("mov", &["-c:v", "libx264", "-c:a", "aac"], "mov"),
            ("m4v", &["-c:v", "libx264", "-c:a", "aac"], "m4v"),
            ("3gp", &["-c:v", "libx264", "-c:a", "aac"], "3gp"),
            ("webm", &["-c:v", "libvpx-vp9", "-c:a", "libopus"], "webm"),
            ("avi", &["-c:v", "mpeg4", "-c:a", "mp3"], "mkv"),
            ("wmv", &["-c:v", "wmv2", "-c:a", "wmav2"], "mkv"),
            ("flv", &["-c:v", "flv1", "-c:a", "mp3"], "mkv"),
            ("ts", &["-c:v", "libx264", "-c:a", "aac"], "mkv"),
            ("mpg", &["-c:v", "mpeg2video", "-c:a", "mp2"], "mkv"),
        ];

        let mut passed = 0usize;
        for (src_ext, enc, want_out) in cases {
            let v = root.join(format!("v.{src_ext}"));
            let mk = Command::new(&ff)
                .args(["-hide_banner", "-loglevel", "error", "-y"])
                .arg("-i")
                .arg(&base)
                .args(*enc)
                .arg(&v)
                .status();
            if !mk.map(|s| s.success()).unwrap_or(false) {
                eprintln!("[跳过] {src_ext}: 本机 ffmpeg 编不出这个源格式");
                continue;
            }

            let out = embed(&v.to_string_lossy(), &sub.to_string_lossy(), "chi");
            assert!(
                out.out_path.is_some(),
                "{src_ext} 封装失败: {}",
                out.message
            );
            eprintln!("{src_ext} → {}", out.message);

            // 输出扩展名要符合预期
            assert!(
                out.out_path.as_ref().unwrap().ends_with(&format!(".{want_out}")),
                "{src_ext} 期望输出 .{want_out}，实际: {:?}",
                out.out_path
            );

            // 验证输出里真的有字幕流
            let probe = Command::new(&ff)
                .args(["-hide_banner", "-i"])
                .arg(out.out_path.as_ref().unwrap())
                .output()
                .unwrap();
            let txt = String::from_utf8_lossy(&probe.stderr).to_string();
            assert!(
                txt.contains("Subtitle:"),
                "{src_ext} 的输出里没有字幕流！"
            );
            passed += 1;
        }
        eprintln!("\n共 {passed} 种格式实测通过");
        assert!(passed >= 8, "通过的格式太少：{passed}");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// ★ v1.29 方案B 实测：ASS 源 + 各容器，验证「按容器选最优」是否真的生效。
    ///
    /// 关心三件事：
    ///   1. mkv 必须走 `-c:s copy`，样式表完整保留（这是唯一保真路径）
    ///   2. mp4/mov/3gp 必须剥成纯文本，正文里不能出现 `<font` 标签
    ///   3. 无论哪条路，输出都要真的有字幕流
    #[test]
    #[ignore]
    fn ass_to_each_container_picks_best_codec() {
        let root = std::env::temp_dir().join("asmr_ass_planb");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let Some(ff) = ffmpeg_exe() else {
            eprintln!("没有 ffmpeg，跳过");
            return;
        };

        // 造带完整样式的 ASS（和项目 srt.rs 一致：黑体/白字/黑描边/底部居中）
        let ass = root.join("a.chs.ass");
        std::fs::write(
            &ass,
            "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n\
[V4+ Styles]\n\
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
Style: Default,SimHei,52,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,30,30,40,1\n\n\
[Events]\n\
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
Dialogue: 0,0:00:00.50,0:00:02.50,Default,,0,0,0,,{\\c&H0000FF&}带样式的字幕\n",
        )
        .unwrap();

        // 造各容器源视频
        let base = root.join("base.mp4");
        assert!(std::process::Command::new(&ff)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=3"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=3"])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])
            .arg(&base)
            .status()
            .unwrap()
            .success());

        for ext in ["mov", "mkv", "3gp"] {
            let p = root.join(format!("v.{ext}"));
            let ok = std::process::Command::new(&ff)
                .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
                .arg(&base)
                .args(["-c:v", "libx264", "-c:a", "aac"])
                .arg(&p)
                .status()
                .unwrap()
                .success();
            assert!(ok, "造 {ext} 失败");
        }

        // 逐个封装
        let mut checked = 0usize;
        for ext in ["mp4", "mov", "mkv", "3gp"] {
            let v = if ext == "mp4" {
                base.clone()
            } else {
                root.join(format!("v.{ext}"))
            };
            let out = embed(&v.to_string_lossy(), &ass.to_string_lossy(), "chi");
            assert!(out.out_path.is_some(), "{ext} 封装失败: {}", out.message);
            eprintln!("[{ext}] {}", out.message);

            // mkv 必须报「样式完整保留」；其它必须报「剥成纯文本」
            if ext == "mkv" {
                assert!(
                    out.message.contains("原样复制"),
                    "mkv 应该走 copy 保留样式: {}",
                    out.message
                );
            } else {
                assert!(
                    out.message.contains("剥成纯文本"),
                    "{ext} 应该剥成纯文本: {}",
                    out.message
                );
            }

            // 输出必须真有字幕流
            let probe = std::process::Command::new(&ff)
                .args(["-hide_banner", "-i"])
                .arg(out.out_path.as_ref().unwrap())
                .output()
                .unwrap();
            let txt = String::from_utf8_lossy(&probe.stderr);
            assert!(txt.contains("Subtitle:"), "{ext} 输出没有字幕流");

            // 非 mkv 的正文里不能残留 <font 标签
            if ext != "mkv" {
                let dump = root.join(format!("dump_{ext}.srt"));
                let _ = std::process::Command::new(&ff)
                    .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
                    .arg(out.out_path.as_ref().unwrap())
                    .args(["-map", "0:s:0"])
                    .arg(&dump)
                    .status();
                if let Ok(c) = std::fs::read_to_string(&dump) {
                    assert!(
                        !c.contains("<font"),
                        "{ext} 的字幕里残留了 <font 标签: {c}"
                    );
                    eprintln!("[{ext}] 字幕正文: {}", c.trim().replace('\n', " | "));
                }
            }

            // 中间 SRT 不能残留
            let rd = std::fs::read_dir(root.parent().unwrap_or(&root));
            checked += 1;
        }
        eprintln!("\n共验证 {checked} 个容器");
        assert!(checked == 4);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// 残留的 tmp/bak 要被清掉，但**已完成的软字幕**和无关文件绝不能动。
    #[test]
    fn cleanup_removes_only_tmp_and_bak() {
        let root = std::env::temp_dir().join("asmr_softsub_cleanup");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let vid = root.join("片.mp4");
        std::fs::write(&vid, b"fake video").unwrap();

        let done = root.join("片（软字幕）.mp4");
        std::fs::write(&done, b"done").unwrap();
        let tmp = root.join("片（软字幕）.tmp.mp4");
        std::fs::write(&tmp, b"tmp").unwrap();
        let bak = root.join("片（软字幕）.mp4.bak");
        std::fs::write(&bak, b"bak").unwrap();
        let other_tmp = root.join("别的片（软字幕）.tmp.mp4");
        std::fs::write(&other_tmp, b"other").unwrap();

        let files = vec![vid.to_string_lossy().to_string()];
        let logs = cleanup_tmp_files(&root.to_string_lossy(), &files);
        eprintln!("{logs:?}");

        assert!(!tmp.exists(), "tmp 没被清掉");
        assert!(!bak.exists(), "bak 没被清掉");
        assert!(done.is_file(), "已完成的软字幕被误删！");
        assert!(other_tmp.is_file(), "别的片的残留被误删了");
        assert!(logs.iter().any(|l| l.contains("2 个")), "计数不对: {logs:?}");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// 重复生成必须**替换**旧软字幕，而不是并存两份。
    ///
    /// 覆盖三种情况：
    ///   ① 同名旧文件 → 被新版替换，内容确实变了
    ///   ② 同 stem 但不同格式的旧软字幕（.mkv / .webm）→ 被清理
    ///   ③ 不相干文件 → 绝不能误删
    #[test]
    #[ignore]
    fn regenerate_replaces_previous_softsub() {
        let root = std::env::temp_dir().join("asmr_softsub_replace");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let Some(ff) = ffmpeg_exe() else {
            eprintln!("没有 ffmpeg，跳过");
            return;
        };
        let vid = root.join("片.mp4");
        let st = Command::new(&ff)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=3"])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])
            .arg(&vid)
            .status()
            .unwrap();
        assert!(st.success());

        // 第一版字幕
        let s1 = root.join("a.chs.srt");
        std::fs::write(&s1, "1\n00:00:00,500 --> 00:00:02,500\n第一版\n").unwrap();
        let out1 = embed(&vid.to_string_lossy(), &s1.to_string_lossy(), "chi");
        assert!(out1.out_path.is_some(), "第一次封装失败: {}", out1.message);
        let soft = root.join("片（软字幕）.mp4");
        assert!(soft.is_file());
        let size1 = std::fs::metadata(&soft).unwrap().len();

        // 造一个「旧格式残渣」：同 stem 的 .mkv，应该被清理
        let stale = root.join("片（软字幕）.mkv");
        std::fs::write(&stale, b"stale").unwrap();
        // 造一个不相干文件，绝不能被删
        let keep = root.join("片（软字幕）备份说明.txt");
        std::fs::write(&keep, b"keep me").unwrap();
        let other = root.join("别的片（软字幕）.mp4");
        std::fs::write(&other, b"other").unwrap();

        // 第二版字幕（内容更长，产物体积会变）
        let s2 = root.join("b.chs.srt");
        std::fs::write(
            &s2,
            "1\n00:00:00,200 --> 00:00:02,900\n第二版字幕，比第一版长很多很多，用来验证它真的被替换了\n",
        )
        .unwrap();
        let out2 = embed(&vid.to_string_lossy(), &s2.to_string_lossy(), "chi");
        assert!(out2.out_path.is_some(), "第二次封装失败: {}", out2.message);
        assert!(
            out2.message.contains("已替换旧版"),
            "第二次应该报告替换: {}",
            out2.message
        );

        // ① 同名文件被替换（大小变了）
        let size2 = std::fs::metadata(&soft).unwrap().len();
        assert_ne!(size1, size2, "文件大小没变，可能没真的替换");

        // ② 旧格式被清理
        assert!(!stale.exists(), "同 stem 的旧 .mkv 没被清理");
        assert!(out2.message.contains("清理旧格式"), "{}", out2.message);

        // ③ 不相干文件都在
        assert!(keep.is_file(), "误删了备份说明文件！");
        assert!(other.is_file(), "误删了别的片的软字幕！");

        // ④ 临时文件和 .bak 都不能残留
        let rd = std::fs::read_dir(&root).unwrap();
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            assert!(
                !n.contains(".tmp.") && !n.ends_with(".bak"),
                "临时/备份文件残留: {n}"
            );
        }

        // ⑤ 原视频必须还在
        assert!(vid.is_file(), "原视频被删了");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// 旧软字幕正被占用时，不能毁掉旧文件，也不能留下半成品。
    ///
    /// Windows 上「文件被打开」会让 remove/rename 失败。这里用
    /// `File::open` 持有句柄模拟播放器占用（Windows 默认共享读，
    /// 所以 remove_file 会失败）。断言：
    ///   · 返回 None 并给出可读的原因
    ///   · 旧文件仍在、内容未变
    ///   · 没有 .tmp / .bak 残渣
    #[test]
    #[ignore]
    fn locked_old_file_is_not_destroyed() {
        let root = std::env::temp_dir().join("asmr_softsub_locked");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let Some(ff) = ffmpeg_exe() else {
            eprintln!("没有 ffmpeg，跳过");
            return;
        };
        let vid = root.join("片.mp4");
        let st = Command::new(&ff)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=3"])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])
            .arg(&vid)
            .status()
            .unwrap();
        assert!(st.success());

        let s1 = root.join("a.chs.srt");
        std::fs::write(&s1, "1\n00:00:00,500 --> 00:00:02,500\n第一版\n").unwrap();
        let out1 = embed(&vid.to_string_lossy(), &s1.to_string_lossy(), "chi");
        assert!(out1.out_path.is_some());
        let soft = root.join("片（软字幕）.mp4");
        let before = std::fs::metadata(&soft).unwrap().len();

        // 模拟「播放器正打开着它」
        let hold = std::fs::File::open(&soft).unwrap();

        let s2 = root.join("b.chs.srt");
        std::fs::write(&s2, "1\n00:00:00,200 --> 00:00:02,900\n第二版\n").unwrap();
        let out2 = embed(&vid.to_string_lossy(), &s2.to_string_lossy(), "chi");

        // 替换应当失败（或被拒绝），且旧文件完好
        if out2.out_path.is_none() {
            assert!(
                out2.message.contains("占用")
                    || out2.message.contains("失败")
                    || out2.message.contains("还原"),
                "失败原因要可读: {}",
                out2.message
            );
            assert!(soft.is_file(), "旧软字幕被毁了！");
            assert_eq!(
                std::fs::metadata(&soft).unwrap().len(),
                before,
                "旧软字幕内容被改动"
            );
        }
        drop(hold);

        // 无论走哪条路径，都不能留临时文件
        let rd = std::fs::read_dir(&root).unwrap();
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            assert!(
                !n.contains(".tmp.") && !n.ends_with(".bak"),
                "临时/备份文件残留: {n}"
            );
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    /// 真实封装一次（默认 `#[ignore]`，需要本机有 ffmpeg 才跑）。
    ///
    /// 为什么标 ignore：普通 `cargo test` 不该依赖外部程序的存在。
    /// 要验证封装确实能出字幕流时，手工跑：
    ///   cargo test --release -- --ignored real_embed_produces_subtitle_stream
    ///
    /// 实测结果（2026-11-01，ffmpeg 9.0.1）：
    ///   mp4 + mov_text → Stream #0:2(chi): Subtitle: mov_text，✅
    ///   mkv + srt      → Stream #0:2(chi): Subtitle: subrip，✅
    ///   两者音视频均为 -c copy，只多出约 550 字节。
    #[test]
    #[ignore]
    fn real_embed_produces_subtitle_stream() {
        // 用 work_dir 下的 ffmpeg 生成一个 3 秒测试视频 + 字幕，再封装
        let tmp = std::env::temp_dir().join("asmr_softsub_e2e");
        let _ = std::fs::create_dir_all(&tmp);
        let Some(ff) = ffmpeg_exe() else {
            eprintln!("本机没有 ffmpeg，跳过");
            return;
        };
        let vid = tmp.join("t.mp4");
        let sub = tmp.join("t.chs.srt");
        std::fs::write(&sub, "1\n00:00:00,500 --> 00:00:02,500\n测试字幕\n").unwrap();
        let st = Command::new(&ff)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=3"])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])
            .arg(&vid)
            .status()
            .unwrap();
        assert!(st.success(), "造测试视频失败");

        let out = embed(
            &vid.to_string_lossy(),
            &sub.to_string_lossy(),
            "chi",
        );
        assert!(out.out_path.is_some(), "封装没成功: {}", out.message);
        eprintln!("{}", out.message);
        // 原文件必须还在（embed 绝不覆盖原文件）
        assert!(vid.is_file(), "原视频被删了！");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
