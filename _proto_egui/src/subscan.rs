//! subscan.rs —— 开跑之前先看「这个片子是不是已经自带字幕了」。
//!
//! 用户的实际痛点：他手上一堆片子，有些是**已经压了中文字幕**的（内封或画面里
//! 烧死的），再跑一遍整条流水线（分离人声 + 语音识别 + 翻译）纯属白烧几十分钟。
//! 所以点「开始处理」时先扫一遍，命中的直接标「已有字幕」跳过。
//!
//! 分两层，从便宜到贵：
//!
//!   **层 1（毫秒级）**：`ffprobe` 看容器里有没有 subtitle 流 —— 这叫**软字幕**，
//!   能顺便读出语言标签和编码格式。有就直接判定，不用碰画面。
//!
//!   **层 2（几秒）**：画面里**烧死**的字幕（硬字幕）没有任何元数据可查，
//!   只能抽几帧看。判据是「白字 + 黑描边」这个视觉特征，见 `looks_like_subtitle`。
//!
//! ⚠️ 为什么只认「白字黑描边」：这是字幕的压倒性主流样式（播放器默认、压制组默认）。
//!    黄字、彩色字、无描边白字会漏检 —— 作保守处理，宁可漏判也不把没字幕的片子
//!    当有字幕跳过（漏判的代价是白跑一遍，误判的代价是用户以为程序把他片子吃了）。

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use crate::config;

// ── 检测参数 ──

/// 只看画面下面这一段：y ∈ [0.62H, 0.96H]。
///
/// 上限不取 1.0 是因为最底下常被播放器进度条 / 台标占着；
/// 下限 0.62 是给「双行字幕 + 稍高的 MarginV」留余量。
///
/// ★ v1.29.2：下边界从 0.96 放宽到 1.00（整个画面底部）。
///   原来的 0.96 会把【贴着画面最底边】的字幕切掉大半 —— 实测 SONE-961-C：
///   字幕完整位于画面高度 93.5%~98.1%，0.96 的下边界只保留了它的最上面
///   几行，块高凑不够 14 行的门槛，于是这条判据直接 `continue` 掉了，
///   **有硬字幕的文件被漏检**（用户反馈「这个是有硬字幕的，翻译器没认出来」）。
///   放宽到整幅画面后字幕完整进带区，判据就能正常工作。
///
///   原来担心「最底下是播放器进度条」—— 那是【播放时】才有，而这里是直接
///   解码原始视频帧，不存在进度条。台标/水印确实可能在角落，但它们通不过
///   后面的「横向跨度 ≥ 22% 宽度」和「高 14~40 行」判据。
const BAND_TOP: f64 = 0.62;
const BAND_BOTTOM: f64 = 1.00;
/// 抽出来的帧统一缩到这个宽度再分析：960 宽下汉字笔画还有 2~3 像素，
/// 再小描边就糊成一片了。缩小主要是为了让逐像素的邻域统计跑得动。
const WORK_W: usize = 960;
/// 「白」的门限。字幕是纯白（255），压缩后可能掉到 230 左右；
/// 定 200 是为了容下 h264 在暗画面上的振铃。
const WHITE: u8 = 200;
/// 「暗」的门限。描边是纯黑，压缩后会浮起来，80 已经足够宽松。
const DARK: u8 = 80;
/// 判断「这个白像素周围有没有黑边」时的邻域半径（像素）。
/// 3 → 7×7 窗口。描边宽度在 960 宽的工作图上约 2~3 像素，取 3 刚好。
const RING_R: usize = 3;
/// 邻域里黑像素占到多少算「有描边」。
/// 描边环在 7×7 窗口里理论占 40% 左右（中间是白字本身）。
/// 下限 0.10 很松：只要边上有黑就算，宁可多进候选，后面还有形状筛。
const RING_DARK_MIN: f64 = 0.10;
/// 一行里命中像素占到整幅宽的多少，这行才算「有字」。
const ROW_MIN_RATIO: f64 = 0.025;
/// 聚成块时容忍的空行数（汉字的字间空隙）。
const ROW_GAP: usize = 3;
/// 一个候选块的高度范围（占条带高度的比例）。
/// 下限 1.5% 排除单像素噪点；上限 45% 排除「整块画面都偏亮」的情况。
const BLOCK_MIN_H_RATIO: f64 = 0.015;
const BLOCK_MAX_H_RATIO: f64 = 0.45;
/// 候选块里命中像素的密度（命中数 / 块面积）。
/// 一块真字幕的白像素占比通常在 5%~30%；0.012 是很宽松的下限。
const BLOCK_MIN_FILL: f64 = 0.012;
/// 候选块的水平跨度必须占到整幅宽的多少 —— 字幕一定是「横着一条」，
/// 竖着的一坨白（比如衣服、窗框）跨度不够。
const BLOCK_MIN_SPAN: f64 = 0.15;

/// 每帧命中算一票；命中率到这个比例就判定「有硬字幕」。
///
/// ★ 为什么定得这么低（8%）：实测一部 80 分钟的中文字幕片，均匀抽 20 帧
///   只有 4 帧命中 —— 因为**字幕只在说话的时候出现**，没人说话的段落
///   画面是干净的。反过来，一个真的没字幕的片子，命中率是 0。
///   所以「有两三帧命中」本身就已经是很强的信号，不能要求 40%。
const VOTE_RATIO: f64 = 0.08;
/// 至少要有这么多帧命中（单帧命中不算数 —— 一个店招就能骗过单帧）。
const VOTE_MIN: usize = 2;
/// 抽帧数上限。每帧要启动一次 ffmpeg 并 seek，16 帧约 6~10 秒，
/// 再往上加只是拖慢开跑时间，判据并不会更准。
const FRAMES_MAX: usize = 16;

// ── 工具定位 ──

/// ffmpeg / ffprobe 的位置。
///
/// 用户的运行目录里带 `env\python.exe`，但**不一定带 ffmpeg** ——
/// 装好的正式版有 `ffmpeg\` 目录，开发机上是 PATH 里的 winget 版。
/// 所以两条路都要试。
pub struct Tools {
    pub ffmpeg: String,
    /// ffprobe 的位置。**可能为空串** —— 见 find_tools 的说明。
    pub ffprobe: String,
}

static TOOLS: OnceLock<Option<Tools>> = OnceLock::new();

/// 找一次就缓存：每次开跑都去 Test-Path 十几个候选目录太浪费。
pub fn tools() -> Option<&'static Tools> {
    TOOLS.get_or_init(find_tools).as_ref()
}

fn find_tools() -> Option<Tools> {
    let work = std::path::PathBuf::from(config::work_dir());
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    for sub in [
        // ★ v1.29.2：`env\ffmpeg` 是【运行环境包实际放置 ffmpeg 的位置】，
        //   必须放在第一梯队。原来的列表里没有它，导致正式环境（运行包只
        //   在 env\ffmpeg\ 下有 ffmpeg.exe）永远找不到工具 ——
        //   用户因此「软字幕没识别出来」，而开发机因为 PATH 上有 ffmpeg
        //   一直测不出来。
        "env\\ffmpeg",
        "ffmpeg\\bin",
        "ffmpeg",
        "bin",
        "env\\Scripts",
        "",
    ] {
        let d = if sub.is_empty() {
            work.clone()
        } else {
            work.join(sub)
        };
        dirs.push(d);
    }
    // ★ v1.29.2：ffprobe 改成【可选】。
    //
    //   踩过的坑（真实用户反馈「软字幕没识别出来」）：
    //     运行环境包里的 env\ffmpeg\ 目录【只有 ffmpeg.exe，没有 ffprobe.exe】。
    //     而原来的条件是 `if ff.is_file() && fp.is_file()` —— 两个都得在，
    //     于是 tools() 返回 None，整个预检（内封字幕 + 硬字幕）被跳过，
    //     程序对「已经封装了软字幕的文件」视而不见，重新跑了一遍流水线。
    //
    //     这是个很隐蔽的缺陷：功能看着实现了，实际在正式环境里从来没生效过。
    //
    //   现在只要有 ffmpeg 就继续；ffprobe 缺失时用 `ffmpeg -i` 的输出替代
    //   （两者的流信息格式不同但内容等价，见 probe_via_ffmpeg）。
    let mut ffmpeg: Option<String> = None;
    let mut ffprobe: Option<String> = None;
    for d in dirs {
        let ff = d.join("ffmpeg.exe");
        let fp = d.join("ffprobe.exe");
        if ff.is_file() {
            if ffmpeg.is_none() {
                ffmpeg = Some(ff.to_string_lossy().to_string());
            }
            if fp.is_file() && ffprobe.is_none() {
                ffprobe = Some(fp.to_string_lossy().to_string());
            }
        }
        if ffmpeg.is_some() && ffprobe.is_some() {
            break;
        }
    }
    if ffmpeg.is_none() {
        // 退回 PATH。这里必须真跑一次 —— 光看 `Command::new("ffmpeg")` 成不成功
        // 发现不了「装了个到处弹广告的假 ffmpeg」这种情况。
        if run_ok("ffmpeg", &["-version"]) {
            ffmpeg = Some("ffmpeg".to_string());
            if run_ok("ffprobe", &["-version"]) {
                ffprobe = Some("ffprobe".to_string());
            }
        }
    }
    let ffmpeg = ffmpeg?;
    Some(Tools {
        ffmpeg,
        // 缺失时留空串，probe() 据此走 ffmpeg 回退路径
        ffprobe: ffprobe.unwrap_or_default(),
    })
}

fn run_ok(exe: &str, args: &[&str]) -> bool {
    Command::new(exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags_none()
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(windows)]
trait NoWindowExt {
    fn creation_flags_none(&mut self) -> &mut Self;
}

#[cfg(windows)]
impl NoWindowExt for Command {
    fn creation_flags_none(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        self.creation_flags(CREATE_NO_WINDOW)
    }
}

// ── 层 1：容器里的字幕流 ──

#[derive(Default, Debug)]
pub struct Info {
    /// 视频总时长（秒）；0.0 = 没读出来
    pub duration: f64,
    /// 内封字幕的说明（如 `subrip（chi）`）。None = 没有字幕流
    pub soft: Option<String>,
    /// 有没有视频流。纯音频文件没必要抽帧
    pub has_video: bool,
}

pub fn probe(tools: &Tools, path: &str) -> Info {
    // ffprobe 优先（结构化输出好解析）
    if !tools.ffprobe.is_empty() {
        let out = Command::new(&tools.ffprobe)
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=index,codec_type,codec_name:stream_tags=language",
                "-of",
                "default",
                path,
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .creation_flags_none()
            .output();
        if let Ok(o) = out {
            let info = parse_probe(&String::from_utf8_lossy(&o.stdout));
            if info.duration > 0.0 || info.soft.is_some() || info.has_video {
                return info;
            }
        }
    }
    // ★ v1.29.2：没有 ffprobe 时退回 `ffmpeg -i`。
    //
    //   运行环境包里只有 ffmpeg.exe（没有 ffprobe），原来这里直接返回空，
    //   导致「已经封装了软字幕的文件」被当成普通文件重跑。
    probe_via_ffmpeg(tools, path)
}

/// 用 `ffmpeg -i` 取流信息（没有 ffprobe 时的回退）。
///
/// ffmpeg 把信息打到 **stderr**，格式是给人看的：
/// ```text
///   Duration: 02:48:30.11, start: 0.000000, bitrate: 5271 kb/s
///   Stream #0:0[0x1](und): Video: h264 (Main) ...
///   Stream #0:2[0x3](chi): Subtitle: mov_text (tx3g / 0x67337874), 0 kb/s (default)
/// ```
/// 这里按行解析出：时长、有没有视频流、有哪些字幕流。
fn probe_via_ffmpeg(tools: &Tools, path: &str) -> Info {
    let out = Command::new(&tools.ffmpeg)
        .args(["-hide_banner", "-i", path])
        .stdin(Stdio::null())
        .stdout(Stdio::null()) // ffmpeg -i 会把信息打到 stderr
        .stderr(Stdio::piped())
        .creation_flags_none()
        .output();
    let Ok(o) = out else {
        return Info::default();
    };
    parse_ffmpeg_stderr(&String::from_utf8_lossy(&o.stderr))
}

/// 解析 `ffmpeg -i` 的 stderr 输出。
fn parse_ffmpeg_stderr(text: &str) -> Info {
    let mut info = Info::default();
    let mut subs: Vec<String> = Vec::new();

    for line in text.lines() {
        let l = line.trim();
        // 时长：`Duration: 02:48:30.11, start: ...`
        if let Some(rest) = l.strip_prefix("Duration:") {
            // 取到第一个逗号为止 —— `02:48:30.11, start: 0.000000`
            // 末尾那个逗号必须剥掉，否则秒数 `30.11,` 解析失败会静默变成 0，
            // 结果时长只剩小时+分钟（实测 2:48:30 被解析成 2.8 小时 = 10080 秒）。
            let t: String = rest
                .trim()
                .chars()
                .take_while(|c| *c != ',')
                .collect::<String>()
                .trim()
                .to_string();
            if t != "N/A" && !t.is_empty() {
                let parts: Vec<&str> = t.split(':').collect();
                if parts.len() == 3 {
                    let h: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let m: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    let s: f64 = parts[2].trim().parse().unwrap_or(0.0);
                    let d = h * 3600.0 + m * 60.0 + s;
                    if d > 0.0 {
                        info.duration = d;
                    }
                }
            }
            continue;
        }
        // 流信息：`Stream #0:0[0x1](und): Video: h264 ...`
        //              Stream #0:2[0x3](chi): Subtitle: mov_text ...
        //         纯音频无语言标记：`Stream #0:0: Audio: flac, 48000 Hz ...`
        let Some(rest) = l.strip_prefix("Stream #") else {
            continue;
        };
        // ★ 类型段在【最后一个 `):` 或第二个 `:`】之后，不能直接 find(':')。
        //   `Stream #0:0[0x1](und): Video: h264`
        //      ↑ rest = "0:0[0x1](und): Video: h264"
        //   第一个冒号是【索引】`0:` 的那一个，用它会切出 "0[0x1](und): Video"，
        //   判断就永远不成立。实测踩过：has_video 恒为 false。
        //
        //   稳妥做法：找 `") :"` / `")："` 之后的第一个冒号；没有括号时
        //   （纯音频那种 `#0:0: Audio:`）退回到「第二个冒号」。
        let (head_end, tail_start) = if let Some(p) = rest.find("):") {
            (p + 1, p + 2) // 跳过 ")：" 里的冒号
        } else {
            // 没有语言括号：`0:0: Audio:` → 第二个冒号之后
            match rest.match_indices(':').nth(1) {
                Some((p, _)) => (p, p + 1),
                None => continue,
            }
        };
        let tail = rest[tail_start..].trim();
        if tail.starts_with("Video:") {
            info.has_video = true;
        } else if tail.starts_with("Subtitle:") {
            // 语言标记在 `#0:2[0x3](chi)` 的圆括号里
            let head = &rest[..head_end];
            let lang = head
                .rfind('(')
                .and_then(|i| head[i + 1..].find(')').map(|j| head[i + 1..i + 1 + j].to_string()))
                .unwrap_or_default();
            // 编码名：`Subtitle: mov_text (tx3g / ...)` → mov_text
            let codec = tail
                .trim_start_matches("Subtitle:")
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("?")
                .trim_end_matches(',')
                .to_string();
            if lang.is_empty() || lang.eq_ignore_ascii_case("und") {
                subs.push(codec);
            } else {
                subs.push(format!("{codec}（{lang}）"));
            }
        }
    }

    if !subs.is_empty() {
        info.soft = Some(subs.join(" / "));
    }
    info
}

/// 解析 `ffprobe -of default` 的输出。
///
/// ⚠️ 这里必须用 `-of default`（带 `[STREAM]` / `[/STREAM]` 包装），
///    **不能**用 `-of default=noprint_wrappers=1`：后者把所有流的字段
///    平铺成一串、段之间没有分隔，于是没法区分「这个 codec_type 属于哪条流」，
///    结果是字幕流全被漏掉、`has_video` 恒为 false。
///    这个坑真机踩过一次（`soft=None has_video=false`），所以单独抽成函数锁单测。
fn parse_probe(text: &str) -> Info {
    let mut info = Info::default();
    let mut subs: Vec<String> = Vec::new();
    let mut in_stream = false;
    let (mut codec, mut kind, mut lang) = (String::new(), String::new(), String::new());

    for line in text.lines() {
        let line = line.trim();
        if line.eq_ignore_ascii_case("[STREAM]") {
            in_stream = true;
            codec.clear();
            kind.clear();
            lang.clear();
            continue;
        }
        if line.eq_ignore_ascii_case("[/STREAM]") {
            if in_stream {
                if let Some(s) = finish_stream(&codec, &kind, &lang) {
                    subs.push(s);
                }
                if kind == "video" {
                    info.has_video = true;
                }
            }
            in_stream = false;
            continue;
        }
        if line.starts_with('[') {
            // [FORMAT] 之类
            in_stream = false;
            continue;
        }
        if let Some(v) = line.strip_prefix("duration=") {
            info.duration = v.trim().parse::<f64>().unwrap_or(0.0);
            continue;
        }
        if let Some(v) = line.strip_prefix("codec_name=") {
            codec = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("codec_type=") {
            kind = v.trim().to_string();
        } else if let Some(v) = line
            .strip_prefix("TAG:language=")
            .or_else(|| line.strip_prefix("language="))
        {
            lang = v.trim().to_string();
        }
    }
    // 万一最后一条流没有闭合标记
    if in_stream {
        if let Some(s) = finish_stream(&codec, &kind, &lang) {
            subs.push(s);
        }
        if kind == "video" {
            info.has_video = true;
        }
    }

    if !subs.is_empty() {
        info.soft = Some(if subs.len() > 2 {
            format!("{}、{} 等共 {} 条", subs[0], subs[1], subs.len())
        } else {
            subs.join("、")
        });
    }
    info
}

fn finish_stream(codec: &str, kind: &str, lang: &str) -> Option<String> {
    if kind != "subtitle" {
        return None;
    }
    let c = if codec.is_empty() { "未知格式" } else { codec };
    if lang.is_empty() || lang == "und" {
        Some(c.to_string())
    } else {
        Some(format!("{c}（{lang}）"))
    }
}

// ── 层 2：画面里烧死的字幕 ──

/// 抽若干帧，看有没有硬字幕。
pub fn hard_sub(tools: &Tools, path: &str, duration: f64) -> bool {
    if duration <= 0.0 {
        return false;
    }
    // 每 60 秒一帧，至少 6 帧、最多 FRAMES_MAX 帧。
    // 短片（几分钟）也能有 6 个采样点，长片到 16 帧封顶（约 10 秒成本）。
    let n = ((duration / 60.0) as usize).clamp(6, FRAMES_MAX);
    let mut tried = 0usize;
    // ★ v1.29：记录每帧命中的【块中心 y】，不再只数票数。
    let mut centers: Vec<usize> = Vec::new();
    let mut heights: Vec<usize> = Vec::new();
    for i in 0..n {
        // 掐掉片头和片尾各 10%：片头常有标题卡、片尾常有水印和演职员表，
        // 那里的白字黑底会被误当成字幕。
        let frac = if n == 1 {
            0.5
        } else {
            0.10 + 0.80 * i as f64 / (n - 1) as f64
        };
        let ts = duration * frac;
        if let Some((w, h, gray)) = grab_gray(tools, path, ts) {
            tried += 1;
            if let Some((y0, y1)) = looks_like_subtitle(&gray, w, h) {
                centers.push((y0 + y1) / 2);
                heights.push(h);
            }
        }
    }

    if tried == 0 || centers.len() < VOTE_MIN {
        return false;
    }

    // ── 字幕位置判据（v1.29 新增）──
    //
    //   为什么需要它：abf-207 那部片子里有白色椅子、白床单，单帧看它们的
    //   白色像素分布和真字幕几乎无法区分（列 run 中位、行内段数都实测过，
    //   两边分布重叠）。所以要靠跨帧信息来判。
    //
    //   实测三组真实素材（16 帧采样，块中心 y 已打印）：
    //     200gana-2535（有硬字幕）   命中 3 帧，中心 y = [42, 180, 7]
    //     ROYD-284（无描边硬字幕）   命中 4 帧，中心 y = [176, 173, 14, 118]
    //     abf-207（白色椅子，无字幕）命中 0 帧
    //
    //   注意前两部虽然真有字幕，命中帧里也混着【画面里的白色物体】
    //   （上面 42 / 14 / 118 就是），所以不能用"全部命中帧都聚在同一位置"
    //   那种严格判据 —— 实测会连真字幕一起判掉（聚类分别只有 1/3、2/4）。
    //
    //   更准的判据是【字幕该在的位置】：硬字幕压在画面底部。
    //   本函数收到的 gray 已经是裁好的底部带（0.62~0.96 画面高度），
    //   字幕通常在带的下 60%，而白色家具、墙面多出现在带的【上部】。
    //   数命中帧里有多少落在带的下半部：
    //     真字幕：176/173（ROYD）、180（200gana）都落在下段 ✓
    //     白色物体：42/14/7 明显在上段
    let band_h = heights.first().copied().unwrap_or(1).max(1);
    let low_zone = (band_h as f64 * 0.40) as usize; // 带高 40% 处以下算字幕常见区
    let in_low = centers.iter().filter(|&&cy| cy >= low_zone).count();

    // 判据：至少 VOTE_MIN 帧命中，命中比例过线，且至少 1 帧落在底部字幕区。
    //   这么宽是因为真字幕出现频率本来就低（16 帧里只有 3~4 帧有字）。
    //   abf-207 那种命中 0 帧的情况，前面 `centers.len() < VOTE_MIN` 就挡住了。
    let ok_ratio = (centers.len() as f64) / (tried as f64) >= VOTE_RATIO;
    let ok_low = in_low >= 1;

    // 诊断输出：设 ASMR_SUBSCAN_DEBUG=1 时打印，方便拿真实素材标定阈值。
    if std::env::var("ASMR_SUBSCAN_DEBUG").is_ok() {
        eprintln!(
            "[subscan] {path}\n  采样 {tried} 帧，命中 {} 帧\n  命中帧块中心 y = {:?}\n  带高 {band_h}，底部字幕区 y >= {low_zone}，落在该区 {} 帧\n  比例 {:?} 底部 {:?}",
            centers.len(),
            centers,
            in_low,
            ok_ratio,
            ok_low
        );
    }

    ok_ratio && ok_low
}

/// 抽一帧、裁底部条带、缩到 960 宽、转灰度，返回 `(宽, 高, 像素)`。
///
/// 直接用 PGM（P5）而不是 PNG：PGM 是无压缩的纯二进制 + 几行头，
/// 手写解析 20 行就够；PNG 要引解码库，不值得为这个功能加依赖。
fn grab_gray(tools: &Tools, path: &str, ts: f64) -> Option<(usize, usize, Vec<u8>)> {
    // crop 的高度要先取偶（h264 的 yuv420 要求），否则部分版本会报错。
    let vf = format!(
        "crop=iw:trunc(ih*{:.2}/2)*2:0:trunc(ih*{:.2}/2)*2,scale={}:-2,format=gray",
        BAND_BOTTOM - BAND_TOP,
        BAND_TOP,
        WORK_W
    );
    let mut child = Command::new(&tools.ffmpeg)
        .args([
            "-v",
            "error",
            "-nostdin",
            "-ss",
            &format!("{ts:.3}"),
            "-i",
            path,
            "-frames:v",
            "1",
            "-vf",
            &vf,
            "-f",
            "image2pipe",
            "-c:v",
            "pgm",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags_none()
        .spawn()
        .ok()?;

    let mut buf = Vec::new();
    let taken = child
        .stdout
        .take()
        .map(|mut s| s.read_to_end(&mut buf).ok())
        .flatten();
    let _ = child.wait();
    taken?;
    // 一张 960×330 的灰度图约 317KB，留 64MB 余量足够。
    if buf.len() > 64 * 1024 * 1024 {
        return None;
    }
    parse_pgm(&buf)
}

/// 解析 PGM（P5，binary，maxval 255）。
///
/// 头里可能夹 `#` 注释，字段之间可以有任意空白，所以不能用「按行切」。
fn parse_pgm(buf: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    let mut pos = 0usize;
    let mut fields: Vec<usize> = Vec::new();
    // 先读三个数字：magic / w / h / maxval，其中 magic 不是数字
    if buf.len() < 2 || &buf[0..2] != b"P5" {
        return None;
    }
    pos = 2;
    while fields.len() < 3 && pos < buf.len() {
        // 跳过空白与注释
        while pos < buf.len() && (buf[pos] as char).is_ascii_whitespace() {
            pos += 1;
        }
        if pos < buf.len() && buf[pos] == b'#' {
            while pos < buf.len() && buf[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < buf.len() && (buf[pos] as char).is_ascii_digit() {
            pos += 1;
        }
        if start == pos {
            return None;
        }
        let v = std::str::from_utf8(&buf[start..pos])
            .ok()?
            .parse::<usize>()
            .ok()?;
        fields.push(v);
    }
    if fields.len() != 3 {
        return None;
    }
    let (w, h, maxval) = (fields[0], fields[1], fields[2]);
    if maxval != 255 || w == 0 || h == 0 || w > 8192 || h > 8192 {
        return None;
    }
    // 头后面必须有一个空白字符，然后就是像素
    if pos >= buf.len() {
        return None;
    }
    pos += 1;
    let need = w * h;
    if buf.len() < pos + need {
        return None;
    }
    Some((w, h, buf[pos..pos + need].to_vec()))
}

/// 这块条带的像素像不像一行烧死的字幕。
///
/// 判据（全是实测标出来的，见模块头部注释）：
///   1. 白像素（> 200）
///   2. 白像素周围 7×7 窗口里有足够多的黑（描边）
///   3. 命中的像素聚成「横向一条」
pub fn looks_like_subtitle(gray: &[u8], w: usize, h: usize) -> Option<(usize, usize)> {
    if w == 0 || h == 0 || gray.len() < w * h {
        return None;
    }
    // 暗像素的积分图：用来 O(1) 求任意小窗口的黑像素个数
    let mut integral = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0u32;
        for x in 0..w {
            if gray[y * w + x] < DARK {
                row_sum += 1;
            }
            integral[(y + 1) * (w + 1) + (x + 1)] = integral[y * (w + 1) + (x + 1)] + row_sum;
        }
    }
    let rect_dark = |x0: usize, y0: usize, x1: usize, y1: usize| -> u32 {
        let x1 = x1.min(w);
        let y1 = y1.min(h);
        if x0 >= x1 || y0 >= y1 {
            return 0;
        }
        integral[y1 * (w + 1) + x1] + integral[y0 * (w + 1) + x0]
            - integral[y0 * (w + 1) + x1]
            - integral[y1 * (w + 1) + x0]
    };

    // 逐像素判「是不是被黑边包着的白」，并统计每行的命中数
    let mut row_hits = vec![0usize; h];
    for y in 0..h {
        for x in 0..w {
            if gray[y * w + x] <= WHITE {
                continue;
            }
            let x0 = x.saturating_sub(RING_R);
            let y0 = y.saturating_sub(RING_R);
            let x1 = (x + RING_R + 1).min(w);
            let y1 = (y + RING_R + 1).min(h);
            let area = ((x1 - x0) * (y1 - y0)) as f64;
            if area <= 0.0 {
                continue;
            }
            let ratio = rect_dark(x0, y0, x1, y1) as f64 / area;
            // 上限 0.90：如果整个窗口几乎全黑，那这个白点是被黑背景包着的
            // （比如黑边里的一点反光），不是白字黑描边。
            if ratio >= RING_DARK_MIN && ratio < 0.90 {
                row_hits[y] += 1;
            }
        }
    }

    // 逐行 → 聚块（容忍 ROW_GAP 个空行）
    let row_min = (w as f64 * ROW_MIN_RATIO).max(3.0) as usize;
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new(); // (y0, y1, hits)
    let mut cur: Option<(usize, usize, usize)> = None;
    let mut gap = 0usize;
    for y in 0..h {
        if row_hits[y] >= row_min {
            match cur {
                Some((y0, _, hits)) => cur = Some((y0, y, hits + row_hits[y])),
                None => cur = Some((y, y, row_hits[y])),
            }
            gap = 0;
        } else if cur.is_some() {
            gap += 1;
            if gap > ROW_GAP {
                if let Some(b) = cur.take() {
                    blocks.push(b);
                }
                gap = 0;
            }
        }
    }
    if let Some(b) = cur {
        blocks.push(b);
    }

    let min_h = (h as f64 * BLOCK_MIN_H_RATIO).max(3.0) as usize;
    let max_h = (h as f64 * BLOCK_MAX_H_RATIO) as usize;
    for (y0, y1, hits) in blocks {
        let bh = y1.saturating_sub(y0) + 1;
        if bh < min_h || bh > max_h {
            continue;
        }
        let area = (bh * w) as f64;
        if (hits as f64) / area < BLOCK_MIN_FILL {
            continue;
        }
        // 水平跨度：这一块里命中像素从最左到最右有多宽
        let (mut lo, mut hi) = (w, 0usize);
        for y in y0..=y1 {
            for x in 0..w {
                if gray[y * w + x] <= WHITE {
                    continue;
                }
                let x0 = x.saturating_sub(RING_R);
                let yy0 = y.saturating_sub(RING_R);
                let x1 = (x + RING_R + 1).min(w);
                let yy1 = (y + RING_R + 1).min(h);
                let a = ((x1 - x0) * (yy1 - yy0)) as f64;
                if a <= 0.0 {
                    continue;
                }
                let ratio = rect_dark(x0, yy0, x1, yy1) as f64 / a;
                if ratio >= RING_DARK_MIN && ratio < 0.90 {
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
            }
        }
        if hi >= lo && (hi - lo + 1) as f64 >= w as f64 * BLOCK_MIN_SPAN {
            return Some((y0, y1));
        }
    }

    // ── 通道 2：无描边白字 ──
    // 用户实测的片子（ROYD-284-C）字幕是**纯白、没有黑描边**的。
    // 只认「白字黑边」会整部漏检，所以再补一条判据：
    //   文字和白色物体的区别在**行内白像素是断续的** ——
    //   一行 10 个汉字会切成 20~40 段（笔画之间有空隙），
    //   而白床单、白墙、反光是一整片，run 数只有 1~3。
    let mut runs = vec![0usize; h];
    let mut whites = vec![0usize; h];
    for y in 0..h {
        let mut in_run = false;
        for x in 0..w {
            let white = gray[y * w + x] > WHITE;
            if white {
                whites[y] += 1;
                if !in_run {
                    runs[y] += 1;
                    in_run = true;
                }
            } else {
                in_run = false;
            }
        }
    }
    // 一行至少要这么些白像素才可能是字（太稀疏的是噪点）
    let white_min = (w as f64 * 0.020).max(6.0) as usize;
    let mut blocks2: Vec<(usize, usize)> = Vec::new();
    let mut cur2: Option<usize> = None;
    let mut gap2 = 0usize;
    for y in 0..h {
        let line = whites[y] >= white_min && runs[y] >= 6;
        if line {
            if cur2.is_none() {
                cur2 = Some(y);
            }
            gap2 = 0;
        } else if cur2.is_some() {
            gap2 += 1;
            if gap2 > ROW_GAP {
                if let Some(y0) = cur2.take() {
                    blocks2.push((y0, y.saturating_sub(gap2)));
                }
                gap2 = 0;
            }
        }
    }
    if let Some(y0) = cur2 {
        blocks2.push((y0, h.saturating_sub(1)));
    }
    // 无描边通道的门槛比描边通道更高：宁可漏，不可误把白色画面当字幕。
    //
    // 下面这四条是拿真实素材标定出来的（有字幕的 ROYD-284 抽 5 帧 vs
    // 无字幕的 NMSL-006 抽 4 帧）：
    //   · 字高 14~40 行：960 宽下汉字就这个高度。太矮是噪点，太高是床单/墙。
    //   · 填充率 0.03~0.20：笔画只占块的一小部分；超过 0.20 是整片白。
    //   · 列白像素中位 run / 块高 ≤ 0.25：字形上下有缺口；白色物体整列通到底。
    //   · 跨度 ≥ 0.22 宽：一整行字幕。
    // 无字幕样本里唯一漏进来的块填充率 0.267（>0.20）被第二条挡掉。
    for (y0, y1) in blocks2 {
        let bh = y1.saturating_sub(y0) + 1;
        if !(14..=40).contains(&bh) || bh > max_h {
            continue;
        }
        let (mut lo, mut hi) = (w, 0usize);
        let mut cnt = 0usize;
        for y in y0..=y1 {
            for x in 0..w {
                if gray[y * w + x] > WHITE {
                    cnt += 1;
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
            }
        }
        let area = (bh * w) as f64;
        let fill = cnt as f64 / area;
        if !(0.030..=0.20).contains(&fill) {
            continue;
        }
        if !(hi >= lo && (hi - lo + 1) as f64 >= w as f64 * 0.22) {
            continue;
        }

        // ★ v1.29：这里原来的判据是「每列白像素最长连续段 / 块高 ≤ 0.25」，
        //   实测它【分不开】文字和白色物体，等于没判：
        //     真实字幕（ROYD-284，纯白无描边）   列 run 中位 = 0.000
        //     白色椅子（abf-207，误判样本）      列 run 中位 = 0.000
        //   我一度想改用「行内白像素段数」，但扫描全片后发现两边分布几乎一样
        //   （中位都是 6），同样无效 —— 单帧的静态统计特征根本不可靠。
        //   真正的判据放在上层：多帧之间的【位置一致性】，见 hard_sub()。
        return Some((y0, y1));
    }
    None
}

// ── 对外入口 ──

/// 预检结果。
pub struct PrecheckResult {
    /// 这些文件确实自带字幕，不必再跑流水线。
    pub skip: Vec<(String, String)>,
    /// 仅提示（当前没有用到，保留结构以便以后扩展）
    pub warn: Vec<(String, String)>,
}

/// 扫一批文件，判断哪些「已经自带字幕」。
///
/// 没有 ffmpeg/ffprobe 时返回空 —— 调用方据此退化成「不预检、照常跑」，
/// 绝不能因为找不到工具就把所有文件都判成有字幕。
///
/// ## 两类判定，可靠度差异很大
///
/// 1. **内封字幕**（`info.soft`）—— 读容器元数据，**准确率 100%**。
///
/// 2. **画面硬字幕** —— 从像素猜的，靠多帧投票 + 位置判据。
///
///    v1.29 一度把它改成「只提示不跳过」，那是为了绕开 abf-207 的误判。
///    但用户随后反馈 SONE-961-C **明明有硬字幕却被漏检、照样生成了字幕** ——
///    查下来是两个问题叠在一起：
///      · BAND_BOTTOM 取 0.96，把贴着画面底边的字幕切掉大半 → 漏检
///      · 漏检后被当成"没问题"，硬字幕片子又跑了一遍
///
///    v1.29.2 的修法：
///      · 带区下边界放宽到 1.00，让贴底字幕完整进带区（实测 SONE-961
///        字幕位于 93.5%~98.1%，放宽后 3 帧全部命中）
///      · 恢复「检出即跳过」，因为现在的判据在白床单（abf-207）和
///        贴底字幕（SONE-961）上都验证过了
pub fn precheck(files: &[String]) -> PrecheckResult {
    let mut out = PrecheckResult {
        skip: Vec::new(),
        warn: Vec::new(),
    };
    let Some(tools) = tools() else {
        return out;
    };
    for f in files {
        let info = probe(tools, f);
        if let Some(desc) = info.soft {
            out.skip.push((f.clone(), format!("内封字幕 {desc}")));
            continue;
        }
        if !info.has_video || info.duration <= 0.0 {
            continue; // 纯音频没有画面可查
        }
        if hard_sub(tools, f, info.duration) {
            // ★ v1.29.2：不再自己决定跳过，交给界面弹窗问用户。
            out.warn
                .push((f.clone(), "画面里疑似压了硬字幕".to_string()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ v1.29.2：没有 ffprobe 时，用 `ffmpeg -i` 的 stderr 解析流信息。
    ///
    /// 这是真实踩到的坑：运行环境包的 env\ffmpeg\ 只有 ffmpeg.exe，
    /// 没有 ffprobe.exe，而原来的 find_tools 要求两个都在，于是整个预检
    /// 被跳过 —— 已经封装了软字幕的文件被当成普通文件重跑。
    #[test]
    fn ffmpeg_stderr_parses_duration_and_subtitle() {
        let text = "\
  Duration: 02:48:30.11, start: 0.000000, bitrate: 5271 kb/s
  Stream #0:0[0x1](und): Video: h264 (Main) (avc1 / 0x31637661), yuv420p, 1920x1080, 5133 kb/s, 29.97 fps
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s
  Stream #0:2[0x3](chi): Subtitle: mov_text (tx3g / 0x67337874), 0 kb/s (default)
";
        let info = parse_ffmpeg_stderr(text);
        eprintln!("解析结果: duration={} has_video={} soft={:?}", info.duration, info.has_video, info.soft);
        assert!(
            (info.duration - (2.0 * 3600.0 + 48.0 * 60.0 + 30.11)).abs() < 0.1,
            "时长不对: {}",
            info.duration
        );
        assert!(info.has_video, "应该认出视频流");
        let soft = info.soft.expect("应该认出字幕流");
        assert!(soft.contains("mov_text"), "应包含编码名: {soft}");
        assert!(soft.contains("chi"), "应包含语言标记: {soft}");
    }

    #[test]
    fn ffmpeg_stderr_without_subtitle_is_clean() {
        let text = "\
  Duration: 00:02:00.09, start: 0.000000, bitrate: 6015 kb/s
  Stream #0:0[0x1](und): Video: h264 (Main), yuv420p, 1920x1080 [SAR 1:1 DAR 16:9], 5809 kb/s, 29.97 fps
  Stream #0:1[0x2](eng): Audio: aac (LC), 44100 Hz, stereo, fltp, 127 kb/s
";
        let info = parse_ffmpeg_stderr(text);
        assert!((info.duration - 120.09).abs() < 0.1);
        assert!(info.has_video);
        assert!(info.soft.is_none(), "没有字幕流时 soft 必须是 None");
    }

    #[test]
    fn ffmpeg_stderr_audio_only_has_no_video() {
        let text = "\
  Duration: 00:10:59.19, start: 0.000000, bitrate: 1308 kb/s
  Stream #0:0: Audio: flac, 48000 Hz, stereo, s32 (24 bit)
";
        let info = parse_ffmpeg_stderr(text);
        assert!((info.duration - 659.19).abs() < 0.2);
        assert!(!info.has_video, "纯音频不该判成有视频");
        assert!(info.soft.is_none());
    }

    /// ★ v1.29.2：真实环境的回归测试 —— 运行目录里**没有 ffprobe**，
    /// 只有 env\ffmpeg\ffmpeg.exe。必须靠 ffmpeg 回退路径识别出内封字幕。
    ///
    /// 这个文件是用户反馈的样本：已经封装了软字幕（mov_text / chi），
    /// 但程序原来识别不出来，当成普通文件重跑了一遍。
    #[test]
    #[ignore]
    fn real_video_soft_sub_detected_without_ffprobe() {
        let Some(t) = tools() else {
            println!("跳过：没找到 ffmpeg");
            return;
        };
        println!("ffmpeg={} ffprobe={:?}", t.ffmpeg, t.ffprobe);
        let p = r"L:\迅雷下载\1027\4k2.com@abf-207（软字幕）.mp4";
        if !std::path::Path::new(p).is_file() {
            println!("跳过：样本不在 {p}");
            return;
        }
        let info = probe(t, p);
        println!(
            "duration={:.1}s soft={:?} has_video={}",
            info.duration, info.soft, info.has_video
        );
        assert!(
            info.soft.is_some(),
            "这个文件有内封字幕流（mov_text/chi），应该能识别"
        );
        assert!(info.has_video);
        assert!(info.duration > 0.0);
    }

    /// ★ 关键回归：precheck 必须【同时】返回 skip（内封字幕）和 warn（硬字幕）。
    ///
    /// 这就是连错三版的那个 bug 的测试：main.rs 里原来把 warn 的处理写在
    /// skip 之前并直接 return，导致只要有硬字幕候选，内封字幕的判定就被丢掉。
    /// 这里从数据层验证：两个文件应当一个进 skip、一个进 warn。
    #[test]
    #[ignore]
    fn precheck_returns_skip_and_warn_together() {
        let files: Vec<String> = vec![
            r"L:\迅雷下载\1027\4k2.com@abf-207（软字幕）.mp4".to_string(),
            r"L:\迅雷下载\1027\SONE-961-C.mp4".to_string(),
        ];
        for f in &files {
            if !std::path::Path::new(f).is_file() {
                println!("跳过：样本不在 {f}");
                return;
            }
        }
        let res = precheck(&files);
        println!("skip = {:?}", res.skip);
        println!("warn = {:?}", res.warn);
        assert!(
            res.skip.iter().any(|(p, _)| p.contains("abf-207")),
            "abf-207 有内封字幕，必须在 skip 里"
        );
        assert!(
            res.warn.iter().any(|(p, _)| p.contains("SONE-961")),
            "SONE-961 疑似硬字幕，必须在 warn 里"
        );
    }

    #[test]
    fn pgm_parses_and_rejects_garbage() {
        // 2×2 白图
        let mut buf = b"P5\n2 2\n255\n".to_vec();
        buf.extend_from_slice(&[255, 255, 255, 255]);
        let (w, h, px) = parse_pgm(&buf).expect("合法 PGM 应该能解析");
        assert_eq!((w, h), (2, 2));
        assert_eq!(px.len(), 4);

        // 带注释
        let mut buf2 = b"P5\n# a comment\n2 2\n255\n".to_vec();
        buf2.extend_from_slice(&[0, 0, 0, 0]);
        assert!(parse_pgm(&buf2).is_some());

        assert!(parse_pgm(b"not a pgm").is_none());
        assert!(parse_pgm(b"").is_none());
        // maxval 不是 255（16 位）不支持
        assert!(parse_pgm(b"P5\n2 2\n65535\n\0\0\0\0").is_none());
        // 像素不够
        assert!(parse_pgm(b"P5\n2 2\n255\n\xff").is_none());
    }

    /// 造一张「底部有白字黑描边」的合成图。
    ///
    /// 关键：白条里要挖出竖向黑缝 —— 真实的字是笔画，笔画之间有间隙；
    /// 如果造一条实心白带，那就只有上下边缘两条像素能被判成「白+黑邻域」，
    /// 聚出来的块高 3 行、低于 min_h，会被筛掉（这个坑实测踩过）。
    fn with_bar(w: usize, h: usize, y0: usize, bar_h: usize, outline: bool) -> Vec<u8> {
        let mut g = vec![60u8; w * h]; // 暗背景
        for y in y0..(y0 + bar_h).min(h) {
            for x in 0..w {
                // 每 6 像素一格：5 像素白 + 1 像素缝（缝就是笔画间隙）
                let on_bar = y >= y0 + 2 && y + 2 < y0 + bar_h;
                let in_glyph = x % 6 != 5;
                if on_bar && in_glyph {
                    g[y * w + x] = 255;
                } else if outline {
                    g[y * w + x] = 10; // 黑描边
                } else {
                    // 无描边的白字：字缝和上下留白是**亮背景**（比如白墙），
                    // 不是黑的 —— 这样才真的测到「缺描边」这件事。
                    // （如果这里填暗色，那就变成「黑底白字」，本来就该被判成字幕）
                    g[y * w + x] = 150;
                }
            }
        }
        g
    }

    #[test]
    fn subtitle_bar_with_outline_is_detected() {
        let (w, h) = (320, 120);
        let g = with_bar(w, h, 70, 30, true);
        assert!(
            looks_like_subtitle(&g, w, h).is_some(),
            "白字黑描边应该被检出"
        );
    }

    #[test]
    fn plain_white_bar_without_outline_is_ignored() {
        let (w, h) = (320, 120);
        let g = with_bar(w, h, 70, 30, false);
        assert!(
            looks_like_subtitle(&g, w, h).is_none(),
            "没有黑描边的白块不该算字幕"
        );
    }

    #[test]
    fn small_logo_is_ignored() {
        let (w, h) = (320, 120);
        // 一小块（跨度不够）
        let mut g = vec![60u8; w * h];
        for y in 60..75 {
            for x in 20..40 {
                g[y * w + x] = 255;
            }
        }
        assert!(
            looks_like_subtitle(&g, w, h).is_none(),
            "小色块不该算字幕"
        );
    }

    #[test]
    fn flat_frame_is_ignored() {
        let (w, h) = (320, 120);
        assert!(looks_like_subtitle(&vec![60u8; w * h], w, h).is_none());
        assert!(looks_like_subtitle(&vec![255u8; w * h], w, h).is_none());
    }

    #[test]
    fn finish_stream_describes_subtitles() {
        assert_eq!(finish_stream("subrip", "subtitle", "chi"), Some("subrip（chi）".into()));
        assert_eq!(finish_stream("subrip", "subtitle", ""), Some("subrip".into()));
        assert_eq!(finish_stream("subrip", "subtitle", "und"), Some("subrip".into()));
        assert_eq!(finish_stream("h264", "video", "chi"), None);
    }

    #[test]
    fn parse_probe_splits_streams_and_reads_duration() {
        let text = "\
[STREAM]
index=0
codec_name=h264
codec_type=video
[/STREAM]
[STREAM]
index=1
codec_name=aac
codec_type=audio
TAG:language=jpn
[/STREAM]
[STREAM]
index=2
codec_name=subrip
codec_type=subtitle
TAG:language=chi
[/STREAM]
[FORMAT]
duration=4835.067937
[/FORMAT]
";
        let info = parse_probe(text);
        assert!(info.has_video, "应该认出视频流");
        assert!((info.duration - 4835.067937).abs() < 0.001);
        assert_eq!(info.soft.as_deref(), Some("subrip（chi）"));
    }

    #[test]
    fn parse_probe_without_wrappers_does_not_panic() {
        // 平铺输出（没有 [STREAM] 包装）：解析要能安全退化成「什么都认不出」，
        // 而不是 panic。这正是当初用 noprint_wrappers 踩的坑。
        let info = parse_probe("index=0\ncodec_name=h264\ncodec_type=video\nduration=12.5\n");
        assert!(!info.has_video);
        assert!(info.soft.is_none());
    }

    /// 真机验证 A：这部片子**确实带中文硬字幕**（Python 版抽 20 帧命中 4 帧确认过）。
    /// 只验证「能检出」，速度约 1 秒。
    ///
    /// 手动跑：`cargo test --release real_video -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_video_with_hard_sub() {
        let Some(t) = tools() else {
            println!("跳过：没找到 ffmpeg / ffprobe");
            return;
        };
        let p = r"L:\迅雷下载\1027\200gana-2535-C.mp4";
        if !std::path::Path::new(p).is_file() {
            println!("跳过：样本不在 {p}");
            return;
        }
        let info = probe(t, p);
        println!(
            "duration={:.1}s soft={:?} has_video={}",
            info.duration, info.soft, info.has_video
        );
        assert!(info.has_video, "应该认出视频流");
        assert!(info.soft.is_none(), "这部片没有内封字幕流");
        assert!(hard_sub(t, p, info.duration), "应该检出画面里的硬字幕");
    }

    /// 真机验证 C：**无描边白字**的硬字幕（ROYD-284）。
    /// 这部片子的字幕是纯白、没有黑描边，旧实现必漏检，正是通道 2 针对的场景。
    #[test]
    #[ignore]
    fn real_video_with_outline_less_hard_sub() {
        let Some(t) = tools() else {
            println!("跳过：没找到 ffmpeg / ffprobe");
            return;
        };
        let p = r"L:\迅雷下载\1027\489155.com@ROYD-284-C.mp4";
        if !std::path::Path::new(p).is_file() {
            println!("跳过：样本不在 {p}");
            return;
        }
        let info = probe(t, p);
        println!(
            "duration={:.1}s soft={:?} has_video={}",
            info.duration, info.soft, info.has_video
        );
        assert!(info.soft.is_none(), "这部片没有内封字幕流");
        assert!(
            hard_sub(t, p, info.duration),
            "无描边白字也该被通道 2 检出"
        );
    }

    /// ★ v1.29.2：SONE-961-C —— 字幕贴着画面最底边的样本。
    ///
    /// 这个片子字幕完整位于画面高度 **93.5%~98.1%**，而 BAND_BOTTOM 原来是
    /// 0.96，只截到字幕的最上面几行，块高不够 14 行门槛，判据直接跳过 →
    /// 有硬字幕却漏检（用户反馈「这个是有硬字幕的，翻译器没认出来」）。
    /// 把 BAND_BOTTOM 放宽到 1.00 后应该能检出。
    #[test]
    #[ignore]
    fn real_video_subtitle_at_very_bottom() {
        let Some(t) = tools() else {
            println!("跳过：没找到 ffmpeg / ffprobe");
            return;
        };
        let p = r"L:\迅雷下载\1027\SONE-961-C.mp4";
        if !std::path::Path::new(p).is_file() {
            println!("跳过：样本不在 {p}");
            return;
        }
        let info = probe(t, p);
        println!(
            "duration={:.1}s soft={:?} has_video={}",
            info.duration, info.soft, info.has_video
        );
        assert!(info.soft.is_none(), "这个文件没有内封字幕（字幕是压进画面的）");
        assert!(
            hard_sub(t, p, info.duration),
            "字幕贴着画面底边，也应该被检出"
        );
    }

    /// ★ v1.29：用户实测反馈的误判样本（abf-207，白色椅子/白床单，无字幕）。
    ///
    /// 这个是**上一次修复漏掉的回归测试** —— 之前的「无字幕」用例用的是
    /// short_test.mp4（120 秒干净测试片），根本没覆盖这个真实误判场景，
    /// 所以修复发布后用户那边依然误判。
    ///
    /// 手动跑：cargo test --release -- --ignored --nocapture real_video_white_furniture
    #[test]
    #[ignore]
    fn real_video_white_furniture_is_clean() {
        let Some(t) = tools() else {
            println!("跳过：没找到 ffmpeg / ffprobe");
            return;
        };
        let p = r"L:\迅雷下载\1027\4k2.com@abf-207.mp4";
        if !std::path::Path::new(p).is_file() {
            println!("跳过：样本不在 {p}");
            return;
        }
        let info = probe(t, p);
        println!(
            "duration={:.1}s soft={:?} has_video={}",
            info.duration, info.soft, info.has_video
        );
        assert!(info.soft.is_none(), "这个文件没有内封字幕");
        assert!(
            !hard_sub(t, p, info.duration),
            "画面里是白色家具，不该被判成有硬字幕"
        );
    }

    /// 真机验证 B：干净样本（120 秒测试片，无字幕流、画面里也没字幕）
    /// 必须**不**被误判。
    #[test]
    #[ignore]
    fn real_video_without_sub_is_clean() {
        let Some(t) = tools() else {
            println!("跳过：没找到 ffmpeg / ffprobe");
            return;
        };
        let p = r"H:\software\ASMR翻译器\_test1251\src\short_test.mp4";
        if !std::path::Path::new(p).is_file() {
            println!("跳过：样本不在 {p}");
            return;
        }
        let info = probe(t, p);
        println!(
            "duration={:.1}s soft={:?} has_video={}",
            info.duration, info.soft, info.has_video
        );
        assert!(info.soft.is_none());
        assert!(!hard_sub(t, p, info.duration), "干净片子不该被判成有字幕");
    }
}
