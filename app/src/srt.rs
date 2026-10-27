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
}
