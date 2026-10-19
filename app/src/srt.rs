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
        let text = body.join(" ").trim().to_string();
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

/// 字幕整理：时长上限 + 不压到下一句 + 最少 0.4 秒 + 词表替换。
/// 返回 (整理后的 cues, 被收缩的条数)。与 e_字幕整理.py 的规则一致。
pub fn tidy(cues: &[Cue], words: &[(String, String)]) -> (Vec<Cue>, usize) {
    let mut out = Vec::with_capacity(cues.len());
    let mut shrunk = 0usize;
    for (i, c) in cues.iter().enumerate() {
        let a = c.start;
        let n = c.text.chars().filter(|ch| !ch.is_whitespace()).count() as f64;
        let cap = a + (0.9 * n + 0.8).max(1.0);
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
            text: apply_words(&c.text, words),
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
        assert!(out[0].end <= 0.0 + 0.9 * 2.0 + 0.8 + 0.001);
    }

    #[test]
    fn wordlist_applies() {
        let (out, _) = tidy(
            &[Cue { start: 0.0, end: 1.0, text: "卑处".into() }],
            &[("卑处".to_string(), "秘书".to_string())],
        );
        assert_eq!(out[0].text, "秘书");
    }

    #[test]
    fn time_edges() {
        assert_eq!(parse_time("00:00:00,000"), Some(0.0));
        assert_eq!(parse_time("23:59:59,999"), Some(23.0 * 3600.0 + 59.0 * 60.0 + 59.999));
        assert_eq!(fmt_time(0.0), "00:00:00,000");
        assert_eq!(fmt_time(86399.999), "23:59:59,999");
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
}
