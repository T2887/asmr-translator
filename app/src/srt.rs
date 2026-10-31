// srt.rs —— SRT 读写 + 字幕整理（纯 Rust，替代 e_字幕整理.py 的文本部分）
//
// 为什么先搬这一块：
//   · 纯文本处理，不依赖任何模型/解释器，风险最低
//   · 逻辑固定（时长上限 + 不重叠 + 词表替换），可以写单元测试
//   · 每次运行省掉一次 Python 启动（约 1~2 秒）

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// 解析 SRT。容错：允许 BOM、CRLF、空行。
pub fn read_srt(path: &Path) -> Result<Vec<Cue>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("读不了 {}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&raw);
    let text = text.trim_start_matches('\u{feff}').replace("\r\n", "\n");

    let mut cues = Vec::new();
    for block in text.split("\n\n") {
        let mut lines = block.lines();
        let Some(_idx) = lines.next() else { continue };
        let Some(time_line) = lines.next() else { continue };
        let Some((a, b)) = time_line.split_once("-->") else { continue };
        let Some(start) = parse_time(a) else { continue };
        let Some(end) = parse_time(b) else { continue };
        let body: Vec<&str> = lines.collect();
        // 保留换行：双语字幕是「日文原文一行 + 中文一行」，合并成一行就毁了两行排版。
        let text = body.join("\n").trim().to_string();
        if !text.is_empty() {
            cues.push(Cue { start, end, text });
        }
    }
    Ok(cues)
}

fn parse_time(s: &str) -> Option<f64> {
    let s = s.trim().replace(',', ".");
    let mut parts = s.split(':');
    let h: f64 = parts.next()?.parse().ok()?;
    let m: f64 = parts.next()?.parse().ok()?;
    let sec: f64 = parts.next()?.parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + sec)
}

pub fn fmt_time(t: f64) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1000;
    let ms = ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

pub fn write_srt(path: &Path, cues: &[Cue]) -> Result<(), String> {
    let mut out = String::new();
    for (i, c) in cues.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            fmt_time(c.start),
            fmt_time(c.end),
            c.text
        ));
    }
    std::fs::write(path, out).map_err(|e| format!("写不了 {}: {e}", path.display()))
}

// ─────────────────────── 字幕输出格式（SRT / ASS） ───────────────────────

/// 输出字幕格式。三个选项互斥。
///
/// 【为什么 ASS 放在 Rust 侧做，而不是改 Python】
///   识别/翻译那条链路（`d_语音识别.py`）只产 SRT，动它要重新跑一遍
///   语音识别才知道有没有坏 —— 风险高、验证慢。
///   而 ASS 本质上就是「同一批时间轴 + 一套固定样式」，收尾时手里正好
///   有解析好的 `Cue`，读 SRT → 写 ASS 是纯文本转换，不碰任何模型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubFmt {
    /// 只出 .srt（原来唯一的行为，默认值）
    Srt,
    /// 只出 .ass
    Ass,
    /// 两个都出：播放器优先加载 ASS，SRT 留作备份/给不认 ASS 的软件
    Both,
}

impl SubFmt {
    pub fn want_srt(self) -> bool {
        matches!(self, SubFmt::Srt | SubFmt::Both)
    }

    pub fn want_ass(self) -> bool {
        matches!(self, SubFmt::Ass | SubFmt::Both)
    }

    pub fn label(self) -> &'static str {
        match self {
            SubFmt::Srt => "仅 SRT",
            SubFmt::Ass => "仅 ASS",
            SubFmt::Both => "SRT + ASS",
        }
    }
}

/// ASS 文件头（含样式表）。样式按用户要求固定：
///   **黑体（SimHei）/ 白字 / 黑描边 / 底部居中 / 字号适中**。
///
/// 逐字段对照：
///   `SimHei`        —— 黑体（Windows 自带，不依赖装字体）
///   `52`            —— 字号。52 在 1080p 坐标系下约占画面高 4.8%，
///                      双行字幕也不压画面
///   `&H00FFFFFF`    —— PrimaryColour 白字。ASS 是 BGR + 前置 alpha：
///                      00=不透明, FF FF FF = 白
///   `&H00000000`    —— OutlineColour 黑描边
///   `Outline 3`     —— 描边 3px。1px 在浅色画面上看不清，5px 会糊住笔画
///   `Shadow 0`      —— 不要阴影，否则加描边后显得脏
///   `Alignment 2`   —— 底部居中（ASS 的 2 = bottom center，
///                      这个编号跟 SSA 的 1/2/3 体系不同，别混）
///   `MarginV 40`    —— 离底边 40px，避免压到播放器进度条
///
/// ⚠️ PlayResX/Y 写死 1920×1080：播放器会按实际画面等比缩放。反过来
///    「取视频原生分辨率」需要在收尾时多跑一次 ffprobe，还得处理旋转
///    元数据（手机录的视频 width/height 是反的），为省这点缩放误差不值。
const ASS_HEADER: &str = "[Script Info]
; 由 ASMR 自动翻译工具生成
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,SimHei,52,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
";

/// ASS 的时间戳是 `h:mm:ss.cc`（小时**不补零**，秒后是**百分秒**两位），
/// 和 SRT 的 `HH:MM:SS,mmm` 完全不是一回事 —— 直接拿 fmt_time 的结果
/// 写进 Dialogue 行，播放器会显示 0 条字幕（解析失败不报错，静默丢弃）。
pub fn fmt_ass_time(t: f64) -> String {
    // NaN / ±Inf 在 SRT 解析里本不该出现，但真漏进来时 `as u64` 会给出
    // 一个天文数字（不是 panic），屏幕上就是一条横跨几小时的字幕。
    // 统一按 0 处理：至少看得见、能定位。
    let t = if t.is_finite() { t.max(0.0) } else { 0.0 };
    let cs = (t * 100.0).round() as u64;
    // 百分秒先到整数再拆字段，所以 59.995 → cs=6000 → `0:01:00.00`，
    // 不会出现 `0:00:60.00` 这种非法值（s / m 都带 % 约束在 0-59）。
    let h = cs / 360_000;
    let m = (cs % 360_000) / 6_000;
    let s = (cs % 6_000) / 100;
    let cs = cs % 100;
    // 小时按 ASS 惯例不补零（Aegisub 导出就是 `0:00:01.00`）。
    format!("{h}:{m:02}:{s:02}.{cs:02}")
}

/// 把一条字幕正文转成 ASS 的 `Text` 字段。
///
/// 两处必须处理，否则不是「难看」而是「内容丢失」：
///   ① 换行 → `\N`（硬换行）。双语字幕是「日文原文 + 中文」两行；
///      直接把 \n 塞进 Dialogue 行，ASS 会当成一行，两行字挤成一坨。
///   ② 半角花括号 → 全角。ASS 的 `{...}` 是样式覆盖语法（`{\an8}` `{\fs80}`），
///      正文里出现半角花括号会被当标签整个吞掉，屏幕上直接少字。
///
/// 反斜杠本身不处理：正文里出现 `\N` `\h` 这类字面量在实践中不存在，
/// 而按逃逸规则重写反而会破坏 ASMR 拟声词里偶尔出现的反斜杠字符。
fn ass_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '\n' => out.push_str("\\N"),
            '\r' => {}
            '{' => out.push('｛'),
            '}' => out.push('｝'),
            _ => out.push(ch),
        }
    }
    out
}

/// 写 ASS（Advanced SubStation Alpha）字幕。
///
/// 与 `write_srt` 平行：同一批 `Cue`，两套方言。
pub fn write_ass(path: &Path, cues: &[Cue]) -> Result<(), String> {
    let mut out = String::with_capacity(ASS_HEADER.len() + cues.len() * 96);
    out.push_str(ASS_HEADER);
    for c in cues {
        out.push_str(&format!(
            "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
            fmt_ass_time(c.start),
            fmt_ass_time(c.end),
            ass_escape(&c.text)
        ));
    }
    // ASS 传统用 CRLF。libass / mpv / ffmpeg 吃 LF 没问题，但个别国产
    // 播放器（PotPlayer 老版本）对纯 LF 的样式段解析会出岔子 —— 转一下，
    // 代价是一次字符串拷贝，收益是省掉一整类「某个播放器不显示字幕」的报障。
    let text = out.replace('\n', "\r\n");
    // 先写 `.ass.tmp` 再改名（同目录 rename 是原子的）：
    // 直接写目标文件的话，中途失败（磁盘满、被杀进程）会留下**半截 .ass**，
    // 播放器照样加载它，用户看到的是「字幕只到第 3 分钟就没了」——
    // 比没有字幕更像 bug，还很难往源头上想。
    let tmp = path.with_extension("ass.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("写不了 {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("写不了 {}: {e}", path.display())
    })
}

/// 读全局词表 + 按作品词表（app\词表.txt 和 app\词表_<stem>.txt）。
pub fn load_wordlist(app_dir: &Path, stem: Option<&str>) -> Vec<(String, String)> {
    let mut files: Vec<PathBuf> = vec![app_dir.join("词表.txt")];
    if let Some(s) = stem {
        files.push(app_dir.join(format!("词表_{s}.txt")));
    }
    let mut pairs = Vec::new();
    for f in files {
        let Ok(raw) = std::fs::read(&f) else { continue };
        let text = String::from_utf8_lossy(&raw);
        let text = text.trim_start_matches('\u{feff}');
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((a, b)) = line.split_once("=>") else { continue };
            let (a, b) = (a.trim(), b.trim());
            if !a.is_empty() && !b.is_empty() && a != b {
                pairs.push((a.to_string(), b.to_string()));
            }
        }
    }
    // 长词优先
    pairs.sort_by(|x, y| y.0.chars().count().cmp(&x.0.chars().count()));
    pairs
}

pub fn apply_words(text: &str, words: &[(String, String)]) -> String {
    let mut out = text.to_string();
    for (a, b) in words {
        if out.contains(a) {
            out = out.replace(a, b);
        }
    }
    out
}

/// 整理一条字幕的正文（词表替换 + 折叠重复）。
///
/// 双语字幕（兼容模式勾了「生成日文字幕」但没勾「日文单独成文件」）是
/// 「日文原文一行 + 中文一行」：**词表只能作用在最后一行（译文）上**。
/// 词表是按中文写的（`卑处 => 秘书`），套到日文原文行上会把原文改坏，
/// 而原文正是这份双语字幕存在的意义。
pub fn tidy_text(text: &str, words: &[(String, String)]) -> String {
    match text.rsplit_once('\n') {
        None => collapse_repeats(&apply_words(text, words)),
        Some((head, last)) => format!(
            "{}\n{}",
            head,
            collapse_repeats(&apply_words(last, words))
        ),
    }
}

/// 这些重复单元在 ASMR 里是正常的拟声/叠词，**不折叠**（千问评审指出）。
///
/// 例：`咕嘟咕嘟`、`啵啵啵啵`、`哈哈哈哈` 都是「ABAB」形状，正是音声的核心
/// 内容；不列白名单会被折成 `咕嘟` / `啵啵` / `哈哈`，语义就丢了。
/// 注意：这里放的是**重复单元**（如 `咕嘟`），不是整串。
const REPEAT_WHITELIST: [&str; 26] = [
    "咕嘟", "咕噜", "扑通", "叮咚", "吧唧", "哗啦", "滴答", "呼噜", "啪嗒", "咔嚓",
    "咕叽", "啾啾", "啵啵", "噗噗", "嗡嗡", "沙沙", "簌簌", "潺潺", "呼呼", "咚咚",
    "砰砰", "啪啪", "嗒嗒", "嘶嘶", "哼哼", "哈哈",
];

/// 折叠相邻重复片段：`欢迎欢迎` → `欢迎`（v1.24.3）
///
/// 为什么需要：第 4 步开着 `initial_prompt` / `hotwords` 时，whisper 会把
/// 碎段合并成更流畅的长句（22 条 → 11 条，字幕质量明显更好），代价是在
/// 块拼接边界重复吐出同一段词。实测：
///   「哎呀 欢迎欢迎 欢迎来到我的保健室」← 日语只说了一次「欢迎」
/// 分别只开 prompt 或只开 hotwords 都能触发，所以不能靠关掉它们来修
/// —— 关掉字幕会碎成一条一条的。
///
/// 规则：
///   · 只折叠**重复单元长度 ≥2** 的（`谢谢`/`看看`/`嗯嗯` 这种单字重复单元
///     根本进不了循环，天然不动）
///   · 单元长度上限 4（原 8；长句重复罕见，折了反而丢内容）
///   · 白名单里的拟声叠词跳过（见 REPEAT_WHITELIST）
///   · 连续重复多次（`欢迎欢迎欢迎`）一次折成一份
pub fn collapse_repeats(text: &str) -> String {
    let cs: Vec<char> = text.chars().collect();   // 用字符向量，避免 &str 切片的 UTF-8 边界 panic
    let n = cs.len();
    let mut out: Vec<char> = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let mut hit = false;
        let mut l = ((n - i) / 2).min(4);
        while l >= 2 {
            if cs[i..i + l] == cs[i + l..i + 2 * l] {
                let piece: String = cs[i..i + l].iter().collect();
                if REPEAT_WHITELIST.contains(&piece.as_str()) {
                    break;   // 正常拟声叠词，整串保留
                }
                let mut k = 2usize;
                while i + (k + 1) * l <= n && cs[i..i + l] == cs[i + k * l..i + (k + 1) * l] {
                    k += 1;
                }
                // 重复之后紧跟「的 / 地 / 得 / 着」的是正常叠词：
                //   「一抖一抖的」「大口大口地」「biubiu地」「摩擦摩擦的」
                // 而重复幻觉后面跟的是新内容或句尾：
                //   「欢迎欢迎 欢迎来到…」「享受了 享受了 无论…」「细致地 细致地」
                if matches!(cs.get(i + k * l), Some('的') | Some('地') | Some('得') | Some('着')) {
                    break;
                }
                out.extend_from_slice(&cs[i..i + l]);
                i += k * l;
                hit = true;
                break;
            }
            l -= 1;
        }
        if !hit {
            out.push(cs[i]);
            i += 1;
        }
    }
    out.into_iter().collect()
}

/// 字幕整理：时长上限 + 不压到下一句 + 最少 0.4 秒 + 词表替换 + 折叠重复。
/// 返回 (整理后的 cues, 被收缩的条数)。与 e_字幕整理.py 的规则一致。
pub fn tidy(cues: &[Cue], words: &[(String, String)]) -> (Vec<Cue>, usize) {
    let mut out = Vec::with_capacity(cues.len());
    let mut shrunk = 0usize;
    for (i, c) in cues.iter().enumerate() {
        let a = c.start;
        let n = c.text.chars().filter(|ch| !ch.is_whitespace()).count() as f64;
        // 常数项 2.0（原 0.8）：第 4 步 d_语音识别.py 会按 VAD 段末把 end
        // 补长最多 1.2 秒；这里若仍用 0.8，短句会被再切回去，补长等于白做。
        let cap = a + (0.9 * n + 2.0).max(1.0);
        let mut b = c.end.min(cap);
        if let Some(next) = cues.get(i + 1) {
            if next.start > a {
                b = b.min(next.start);
            }
        }
        b = b.max(a + 0.4);
        if b < c.end - 0.01 {
            shrunk += 1;
        }
        out.push(Cue {
            start: a,
            end: b,
            text: tidy_text(&c.text, words),
        });
    }
    (out, shrunk)
}

/// 整理一个 3asr 目录下的所有 SRT，返回日志行。
/// stem 用于找按作品的词表（文件名里的 `_(Vocals)` 后缀会被剥掉）。
pub fn tidy_dir(srt_dir: &Path, app_dir: &Path) -> Vec<String> {
    let mut logs = Vec::new();
    let Ok(rd) = std::fs::read_dir(srt_dir) else {
        logs.push(format!("[整理] 读不了目录 {}", srt_dir.display()));
        return logs;
    };
    let mut paths: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().map(|x| x.eq_ignore_ascii_case("srt")).unwrap_or(false))
        .collect();
    paths.sort();
    if paths.is_empty() {
        logs.push("[整理] 没有找到字幕文件".to_string());
        return logs;
    }

    let mut total_shrunk = 0usize;
    let mut total = 0usize;
    for p in &paths {
        let name = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        // 日文字幕（兼容模式勾了「额外生成日文字幕」的产物）不参与中文整理：
        //   · 词表是「中文错译修正表」，套到日文上只会改坏；
        //   · 时间轴本来就与中文字幕逐条一致，不需要再收一遍。
        if name.to_lowercase().ends_with(".jpn.srt") {
            logs.push(format!("[整理] 跳过日文字幕：{name}"));
            continue;
        }
        let stem = Path::new(&name)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let base = stem.split("_(").next().unwrap_or(&stem);
        let words = load_wordlist(app_dir, Some(base));
        match read_srt(p) {
            Ok(cues) if !cues.is_empty() => {
                let (out, shrunk) = tidy(&cues, &words);
                if write_srt(p, &out).is_err() {
                    logs.push(format!("[整理] 写回失败：{name}"));
                    continue;
                }
                total += cues.len();
                total_shrunk += shrunk;
                let max_before = cues.iter().map(|c| c.end - c.start).fold(0.0, f64::max);
                let max_after = out.iter().map(|c| c.end - c.start).fold(0.0, f64::max);
                logs.push(format!(
                    "[整理] {name}：{} 条，收缩 {shrunk} 条，最长 {max_before:.1}s → {max_after:.1}s",
                    cues.len()
                ));
            }
            Ok(_) => logs.push(format!("[整理] 跳过（空字幕）：{name}")),
            Err(e) => logs.push(format!("[整理] {e}")),
        }
    }
    logs.push(format!(
        "[整理] 完成：{} 个文件 / {total} 条，收缩 {total_shrunk} 条",
        paths.len()
    ));
    logs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_format_roundtrip() {
        let cues = vec![Cue { start: 2.0, end: 7.45, text: "测试".into() }];
        assert_eq!(fmt_time(cues[0].start), "00:00:02,000");
        assert_eq!(fmt_time(cues[0].end), "00:00:07,450");
    }

    #[test]
    fn tidy_caps_long_cue() {
        let cues = vec![
            Cue { start: 0.0, end: 220.0, text: "好的".into() },
            Cue { start: 230.0, end: 232.0, text: "下一句".into() },
        ];
        let (out, shrunk) = tidy(&cues, &[]);
        assert_eq!(shrunk, 1);
        assert!(out[0].end <= 0.0 + 0.9 * 2.0 + 2.0 + 0.001);
    }

    #[test]
    fn wordlist_applies() {
        let (out, _) = tidy(
            &[Cue { start: 0.0, end: 1.0, text: "卑处".into() }],
            &[("卑处".to_string(), "秘书".to_string())],
        );
        assert_eq!(out[0].text, "秘书");
    }

    // 「额外生成日文字幕」的产物不参与中文整理：
    //   · 词表是中文错译修正表，套到日文上只会改坏；
    //   · 它的时间轴本来就与中文字幕逐条一致，不需要再收一遍。
    #[test]
    fn tidy_dir_skips_jpn_subtitle() {
        let dir = std::env::temp_dir().join("asmr_srt_jpn_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let jp = dir.join("第一集.jpn.srt");
        let jp_body = "1\n00:00:01,000 --> 00:00:02,000\n卑处だよ\n";
        std::fs::write(&jp, jp_body).unwrap();
        let zh = dir.join("第一集.srt");
        std::fs::write(&zh, "1\n00:00:01,000 --> 00:00:02,000\n卑处\n").unwrap();
        std::fs::write(dir.join("词表.txt"), "卑处 => 秘书\n").unwrap();

        let logs = tidy_dir(&dir, &dir);

        assert_eq!(
            std::fs::read_to_string(&jp).unwrap(),
            jp_body,
            "日文字幕必须原样保留（不能被中文词表改）"
        );
        assert!(
            std::fs::read_to_string(&zh).unwrap().contains("秘书"),
            "中文字幕仍要按词表整理"
        );
        assert!(
            logs.iter().any(|l| l.contains("跳过日文字幕")),
            "日志里应说明跳过了日文字幕：{logs:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // 行内双语（兼容模式勾了「生成日文字幕」但没勾「日文单独成文件」）：
    // 一条字幕两行，日文原文在上、中文在下。
    //   词表是中文错译修正表，只能动最后一行（译文）；
    //   把原文行也替换掉，这份双语字幕就没有存在意义了。
    #[test]
    fn tidy_text_only_touches_last_line() {
        let cues = vec![Cue {
            start: 0.0,
            end: 2.0,
            text: "卑处です\n我是卑处".into(),
        }];
        let (out, _) = tidy(&cues, &[("卑处".to_string(), "秘书".to_string())]);
        assert_eq!(out[0].text, "卑处です\n我是秘书");
    }

    // 双语字幕的两行排版不能在读文件时被合并成一行（曾经是 join(" ")）。
    #[test]
    fn read_srt_keeps_inner_newline() {
        let dir = std::env::temp_dir().join("asmr_srt_bilingual_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.srt");
        std::fs::write(&p, "1\n00:00:01,000 --> 00:00:02,000\nお姉さん\n姐姐\n").unwrap();

        let cues = read_srt(&p).unwrap();

        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "お姉さん\n姐姐");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn time_edges() {
        assert_eq!(parse_time("00:00:00,000"), Some(0.0));
        assert_eq!(parse_time("23:59:59,999"), Some(23.0 * 3600.0 + 59.0 * 60.0 + 59.999));
        assert_eq!(fmt_time(0.0), "00:00:00,000");
        assert_eq!(fmt_time(86399.999), "23:59:59,999");
    }

    #[test]
    fn collapse_repeats_folds_doubled_word() {
        assert_eq!(
            collapse_repeats("哎呀 欢迎欢迎 欢迎来到我的保健室"),
            "哎呀 欢迎 欢迎来到我的保健室"
        );
        assert_eq!(collapse_repeats("欢迎欢迎欢迎"), "欢迎");
        assert_eq!(collapse_repeats("我知道了我知道了"), "我知道了");
    }

    #[test]
    fn collapse_repeats_keeps_normal_reduplication() {
        // 单字重复单元（谢谢 / 看看 / 慢慢 / 嗯嗯）不是重复幻觉，不能动
        assert_eq!(collapse_repeats("谢谢 看看 慢慢来"), "谢谢 看看 慢慢来");
        assert_eq!(collapse_repeats("嗯嗯 痒痒的"), "嗯嗯 痒痒的");
        assert_eq!(collapse_repeats("高高兴兴"), "高高兴兴");
        // 「ABAB + 的/地」是正常叠词，必须原样（这是降低误伤的关键判据）
        assert_eq!(collapse_repeats("喂 别一抖一抖的"), "喂 别一抖一抖的");
        assert_eq!(collapse_repeats("大口大口地吞下去了"), "大口大口地吞下去了");
        assert_eq!(collapse_repeats("biubiu地尽情地射精吧"), "biubiu地尽情地射精吧");
        assert_eq!(collapse_repeats("我会好好地挖挖耳朵 摩擦摩擦的"),
                   "我会好好地挖挖耳朵 摩擦摩擦的");
        assert_eq!(collapse_repeats(""), "");
        assert_eq!(collapse_repeats("普通的一句话"), "普通的一句话");
    }

    #[test]
    fn collapse_repeats_keeps_whitelisted_onomatopoeia() {
        // ASMR 的拟声叠词是「ABAB」形状，必须原样保留（千问评审）
        assert_eq!(collapse_repeats("咕嘟咕嘟"), "咕嘟咕嘟");
        assert_eq!(collapse_repeats("啵啵啵啵"), "啵啵啵啵");
        assert_eq!(collapse_repeats("哈哈哈哈"), "哈哈哈哈");
        assert_eq!(collapse_repeats("扑通扑通地跳"), "扑通扑通地跳");
    }

    #[test]
    fn collapse_repeats_ignores_long_units() {
        // 重复单元超过 4 字不再折叠（长句被说两遍罕见，折了会丢内容）
        assert_eq!(collapse_repeats("今天我们很开心今天我们很开心"),
                   "今天我们很开心今天我们很开心");
    }

    #[test]
    fn tidy_collapses_repeats() {
        let (out, _) = tidy(
            &[Cue { start: 0.0, end: 5.0, text: "哎呀 欢迎欢迎 欢迎来到我的保健室".into() }],
            &[],
        );
        assert_eq!(out[0].text, "哎呀 欢迎 欢迎来到我的保健室");
    }

    #[test]
    fn longer_words_first() {
        let words = vec![
            ("你好".to_string(), "A".to_string()),
            ("你好吗".to_string(), "B".to_string()),
        ];
        let (out, _) = tidy(
            &[Cue { start: 0.0, end: 2.0, text: "你好吗".into() }],
            &words,
        );
        // 词表来自 load_wordlist 时已按长度降序；这里手动排序模拟
        let mut sorted = words.clone();
        sorted.sort_by(|x, y| y.0.chars().count().cmp(&x.0.chars().count()));
        let (out2, _) = tidy(
            &[Cue { start: 0.0, end: 2.0, text: "你好吗".into() }],
            &sorted,
        );
        assert_eq!(out2[0].text, "B");
        // 未排序时短词先替换，会产生 "A吗" —— 仅用于确认排序的重要性
        assert_eq!(out[0].text, "A吗");
    }

    // ASS 的时间戳是 h:mm:ss.cc，不是 SRT 的 HH:MM:SS,mmm。
    // 这两者混用是「字幕整份不显示」最典型的成因，所以单独立一条。
    #[test]
    fn ass_time_format() {
        assert_eq!(fmt_ass_time(0.0), "0:00:00.00");
        assert_eq!(fmt_ass_time(1.0), "0:00:01.00");
        assert_eq!(fmt_ass_time(3723.456), "1:02:03.46");
        // 负时间（识别出来的时间轴有极小负值）必须夹到 0，不能出现 "-1:00:00.00"
        assert_eq!(fmt_ass_time(-5.0), "0:00:00.00");
        // 百分秒进位：59.995s 四舍五入成 6000cs，必须整体进位成 1 分 0 秒，
        // 而不是输出非法的 "0:00:60.00"（严格解析器会丢整条字幕）。
        assert_eq!(fmt_ass_time(59.995), "0:01:00.00");
        assert_eq!(fmt_ass_time(3599.999), "1:00:00.00");
        // NaN / Inf 兜底成 0，别让一条字幕横跨几小时
        assert_eq!(fmt_ass_time(f64::NAN), "0:00:00.00");
        assert_eq!(fmt_ass_time(f64::INFINITY), "0:00:00.00");
    }

    // 样式是用户点名要的：黑体、白字、黑描边、底部居中、字号适中。
    // 错一个字段播放器不会报错，只会「看着不对」，所以锁死字符串。
    #[test]
    fn ass_style_matches_spec() {
        let dir = std::env::temp_dir().join("asmr_ass_style_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.ass");
        write_ass(&p, &[Cue { start: 1.0, end: 2.5, text: "お姉さん\n姐姐".into() }]).unwrap();
        let t = std::fs::read_to_string(&p).unwrap();

        assert!(t.contains("ScriptType: v4.00+"), "必须是 ASS v4.00+：{t}");
        assert!(t.contains("PlayResX: 1920") && t.contains("PlayResY: 1080"), "{t}");
        // Fontname=SimHei 黑体 / Fontsize=52 / 主色白 / 描边色黑
        assert!(
            t.contains("Style: Default,SimHei,52,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"),
            "字体颜色不对：{t}"
        );
        // BorderStyle=1 / Outline=3 / Shadow=0 / Alignment=2(底部居中) / MarginV=40
        assert!(t.contains(",0,0,1,3,0,2,40,40,40,1"), "描边/对齐/MarginV 不对：{t}");
        // 双语两行必须变成 \N，否则挤成一行
        assert!(
            t.contains("Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,お姉さん\\N姐姐"),
            "Dialogue 行不对：{t}"
        );
        // 正文里的花括号会被当样式标签吞掉，应转全角
        let p2 = dir.join("b.ass");
        write_ass(&p2, &[Cue { start: 0.0, end: 1.0, text: "{测试}".into() }]).unwrap();
        assert!(std::fs::read_to_string(&p2).unwrap().contains("｛测试｝"), "花括号未转义");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // 三个格式开关的互斥关系（「两个都要」时 srt 和 ass 都是 true）。
    #[test]
    fn sub_fmt_switches() {
        assert!(SubFmt::Srt.want_srt() && !SubFmt::Srt.want_ass());
        assert!(!SubFmt::Ass.want_srt() && SubFmt::Ass.want_ass());
        assert!(SubFmt::Both.want_srt() && SubFmt::Both.want_ass());
    }
}
