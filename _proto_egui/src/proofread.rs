// proofread.rs —— 字幕校对：数据层 + 音频试听
//
// 用户点「✎ 校对字幕」后，把这一批产出里已经写好的 `.chs.srt` 读进来，
// 逐句列出，允许：
//   · 双击改中文（兼容模式还能改日文原文）
//   · 点最右侧 × 删掉整句（可撤销）
//   · 点时间戳试听那一段（rodio）
//   · 「保存并写回字幕」直接覆盖原 `.chs.srt`（先备份 `.bak`）
//
// 为什么日文原文只有兼容模式有：
//   全量/精简模式走 whisper 的 `task="translate"`，日语音频一步出中文，
//   中间没有日文文本层 —— 和「全量模式吃不到 LoRA」是同一个结构性原因。
//   兼容模式是 `light_asr.py`（Haas-ja）先转写日文到 `2vad\<stem>.json`，
//   再 `light_translate.py` 翻译，所以日文原文现成。

use std::fs;
use std::path::{Path, PathBuf};

use crate::srt;

/// 校对表里的一行。
#[derive(Clone, Debug)]
pub struct ProofRow {
    pub start: f64,
    pub end: f64,
    /// 日文原文。全量/精简模式恒为空串。
    pub ja: String,
    /// 中文台词。
    pub zh: String,
    /// 用户点了 ×（还没保存，只是标记）。保存时被剔除。
    pub deleted: bool,
    /// 相对载入时的原始文本改过没有（用来标「已改」）。
    pub edited: bool,
    /// 低置信行（v1.40.1）：Haas-ja 独立复核时听出的内容偏少，
    /// 说明这句可能是吞字、半截音效或幻觉。界面标橙提示用户重点听。
    pub lowconf: bool,
}

impl ProofRow {
    /// 时间戳文本，和 SRT 里一致（`00:02:38,870`）。
    pub fn time_label(&self) -> String {
        srt::fmt_time(self.start)
    }
}

/// 载入结果。
pub struct Loaded {
    pub rows: Vec<ProofRow>,
    /// srt 文件路径。
    pub srt_path: PathBuf,
    /// 有日文原文（兼容模式）时为 true。
    pub has_ja: bool,
    /// 试听用的音频路径（视频/音频原文件）。
    pub media: PathBuf,
}

/// 从 srt 路径 + 可选 2vad json 载入。
///
/// `vad_json` 只在兼容模式下有；不存在时 `has_ja = false`，
/// 界面就不画「识别原文」那一列 —— 和全量模式的表现一致。
pub fn load(srt_path: &Path, vad_json: Option<&Path>, media: &Path,
            lowconf_json: Option<&Path>) -> Result<Loaded, String> {
    let cues = srt::read_srt(srt_path)?;
    if cues.is_empty() {
        return Err(format!("{} 里没有字幕条目", srt_path.display()));
    }

    // 日文原文：`[{start, end, text}, ...]`，按时间窗和 cue 对齐。
    // 对齐口径和流水线其它地方一致：取时间中点落在 cue 区间内的那段。
    // ★ 必须排序：`light_asr.py` 是按识别顺序写 JSON 的（分块并行），
    //   数组顺序不保证时间单调。不排序直接拼，同一句 cue 会对出乱序日文。
    let mut ja_rows = match vad_json {
        Some(p) if p.is_file() => parse_vad_json(p).unwrap_or_default(),
        _ => Vec::new(),
    };
    ja_rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let has_ja = !ja_rows.is_empty();

    // 低置信段（v1.40.1）：d_语音识别.py 写出的 [{start,end,text,heard}]，
    // 内容是「逃过了人声闸、但 Haas-ja 复核只听到很少几种字」的段。
    // 判据和日文对齐一致：cue 的时间中点落在低置信区间里就算。
    let lowconf_spans: Vec<(f64, f64)> = match lowconf_json {
        Some(p) if p.is_file() => parse_vad_json(p)
            .map(|v| v.into_iter().map(|(a, b, _)| (a, b)).collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    };

    let rows = cues
        .into_iter()
        .map(|c| ProofRow {
            start: c.start,
            end: c.end,
            lowconf: {
                let mid = (c.start + c.end) * 0.5;
                lowconf_spans.iter().any(|(a, b)| mid >= *a && mid <= *b)
            },
            ja: if has_ja {
                pick_ja(&ja_rows, c.start, c.end)
            } else {
                String::new()
            },
            zh: c.text,
            deleted: false,
            edited: false,
        })
        .collect();

    Ok(Loaded {
        rows,
        srt_path: srt_path.to_path_buf(),
        has_ja,
        media: media.to_path_buf(),
    })
}

/// 极简 JSON 解析：只认 `[{"start":..,"end":..,"text":".."}, ...]`。
///
/// 为什么不用 serde：整个工程一个 JSON 依赖都没加（`Cargo.toml` 只有
/// eframe/image/regex/rfd/windows-sys），为一个只读的小文件拉一个
/// serde_json 进来不值当。这个格式是 `light_asr.py` 自己写的，可控。
fn parse_vad_json(path: &Path) -> Result<Vec<(f64, f64, String)>, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("读不了 {}: {e}", path.display()))?;
    let cs: Vec<char> = raw.chars().collect();
    let n = cs.len();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < n {
        // 找下一个「字符串外」的对象起点
        while i < n && cs[i] != '{' {
            i += 1;
        }
        if i >= n {
            break;
        }
        let body_start = i + 1;

        // 找配对的 '}' —— 必须知道自己在不在字符串里，
        // 否则日文台词里的全角 ｝ / 半角 } 会把对象提前截断。
        let mut depth = 1i32;
        let mut in_str = false;
        let mut esc = false;
        let mut j = body_start;
        while j < n {
            let c = cs[j];
            if in_str {
                if esc {
                    esc = false;
                } else if c == '\\' {
                    esc = true;
                } else if c == '"' {
                    in_str = false;
                }
            } else {
                match c {
                    '"' => in_str = true,
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
            }
            j += 1;
        }
        if j >= n {
            break;
        }

        let body: String = cs[body_start..j].iter().collect();
        // start / end 只在 "text" 之前找 —— 台词里出现 "start" 这几个字符时不会误取。
        let head_end = body.find("\"text\"").unwrap_or(body.len());
        let (head, tail) = body.split_at(head_end);
        if let (Some(a), Some(b)) = (num_field(head, "start"), num_field(head, "end")) {
            if let Some(t) = str_field(tail, "text") {
                if !t.is_empty() {
                    out.push((a, b, t));
                }
            }
        }
        i = j + 1;
    }

    Ok(out)
}

/// 取 `"key": 123.45` 里的数字。key 传不带引号的字面名。
fn num_field(body: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let i = body.find(&pat)?;
    let rest = &body[i + pat.len()..];
    let rest = rest.trim_start().strip_prefix(':')?;
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
        })
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// 取 `"key": "字符串"` 里的字符串（认 \" 与 \uXXXX 转义）。
fn str_field(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = body.find(&pat)?;
    let rest = &body[i + pat.len()..];
    let rest = rest.trim_start().strip_prefix(':')?;
    let rest = rest.trim_start().strip_prefix('"')?;

    let mut raw = String::new();
    let mut esc = false;
    for ch in rest.chars() {
        if esc {
            raw.push('\\');
            raw.push(ch);
            esc = false;
            continue;
        }
        if ch == '\\' {
            esc = true;
            continue;
        }
        if ch == '"' {
            return Some(unescape(&raw));
        }
        raw.push(ch);
    }
    None
}

fn unescape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            o.push(c);
            continue;
        }
        match it.next() {
            Some('n') => o.push('\n'),
            Some('t') => o.push('\t'),
            Some('r') => o.push('\r'),
            Some('"') => o.push('"'),
            Some('\\') => o.push('\\'),
            Some('u') => {
                let hex: String = it.by_ref().take(4).collect();
                if let Ok(n) = u32::from_str_radix(&hex, 16) {
                    if let Some(ch) = char::from_u32(n) {
                        o.push(ch);
                    }
                }
            }
            Some(other) => o.push(other),
            None => break,
        }
    }
    o
}

/// 从日文段里挑出「时间上属于这条 cue」的那段。
///
/// 判定用中点（和 `cut_all.py` / `cut_v3_sent.py` 的中心判定同源）：
/// 重叠就带的写法会让一个 4 秒 cue 吃进前后三句，日文比中文多一倍。
fn pick_ja(rows: &[(f64, f64, String)], a: f64, b: f64) -> String {
    let lo = a + (b - a) * 0.15;
    let hi = b - (b - a) * 0.15;
    let mut hit = String::new();
    for (s, e, t) in rows {
        let mid = (s + e) / 2.0;
        if mid >= lo && mid <= hi {
            if !hit.is_empty() {
                hit.push(' ');
            }
            hit.push_str(t);
        }
    }
    if !hit.is_empty() {
        return hit;
    }
    // 区间内没有中点，退回「起点最近的那一句」，但限 2 秒内，
    // 否则会把几秒外完全不相干的句子贴上来。
    let mut best: Option<(f64, &str)> = None;
    for (s, _, t) in rows {
        let d = (s - a).abs();
        if d <= 2.0 && best.map_or(true, |(bd, _)| d < bd) {
            best = Some((d, t));
        }
    }
    best.map(|(_, t)| t.to_string()).unwrap_or_default()
}

/// 保存：把没被删的行写回 srt（先备份 `.bak`）。
///
/// 返回 (写入条数, 备份路径)。备份已存在时不覆盖 —— 用户第一次改之前
/// 的原文才是他要后悔回去的那份。
pub fn save(loaded: &Loaded, rows: &[ProofRow]) -> Result<(usize, PathBuf), String> {
    let mut cues: Vec<srt::Cue> = rows
        .iter()
        .filter(|r| !r.deleted)
        .map(|r| srt::Cue {
            start: r.start,
            end: r.end,
            text: r.zh.clone(),
        })
        .collect();

    if cues.is_empty() {
        return Err("全部删掉了，不给保存（至少留一句）".to_string());
    }

    // SRT 要求 start < end；用户编辑不会碰时间，这里只防脏数据。
    for c in cues.iter_mut() {
        if c.end <= c.start {
            c.end = c.start + 0.5;
        }
    }

    // ★ v1.30.7：备份放到数据目录的 backup\ 下，不再扔在素材旁边。
    //
    //   原来用 `.with_extension("srt.bak")`，保存一次就在用户素材目录里
    //   多出 `xxx.chs.srt.bak`（用户实测反馈 m21123：「点了保存产生了
    //   2 个文件」）。素材目录是用户自己的地方，不该被程序塞东西。
    let bak = {
        let base = std::path::Path::new(crate::config::data_dir()).join("backup");
        let _ = fs::create_dir_all(&base);
        let name = loaded
            .srt_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "subtitle.srt".to_string());
        base.join(format!("{name}.bak"))
    };
    if !bak.exists() {
        fs::copy(&loaded.srt_path, &bak)
            .map_err(|e| format!("备份失败 {}: {e}", bak.display()))?;
    }

    srt::write_srt(&loaded.srt_path, &cues)?;
    Ok((cues.len(), bak))
}

// ─────────────────────────────────────────────────────────────
// 音频试听
// ─────────────────────────────────────────────────────────────

use std::io::BufReader;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

enum Cmd {
    Play(PathBuf, f64, f64),
    Stop,
}

/// 试听播放器。
///
/// 为什么单独开线程：`rodio::OutputStream` 在 Windows 上要在**同一个线程**
/// 里创建并持有到播放结束，跨线程移动会直接 panic（WASAPI 的线程亲和性）。
/// 所以这里用「一条常驻线程 + 命令通道」，UI 侧只发命令不碰音频对象。
pub struct Player {
    tx: Sender<Cmd>,
    /// 当前在播的 cue 序号（UI 用来画高亮）。
    pub playing: Option<usize>,
    /// 播完的时刻（UI 轮询用）。
    rx: Receiver<usize>,
}

impl Drop for Player {
    /// 窗口关掉时立刻停掉还在播的那一段。
    ///
    /// 没有这个，Player 被 drop 时只释放 tx，音频线程要等下一次
    /// `recv_timeout` 超时 + channel 断开才退出，中间 sink 继续播完
    /// （用户已经关窗了，声音还在响）。
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Stop);
    }
}

impl Player {
    pub fn new() -> Self {
        let (tx, rx_cmd) = mpsc::channel::<Cmd>();
        let (tx_done, rx) = mpsc::channel::<usize>();
        std::thread::spawn(move || audio_thread(rx_cmd, tx_done));
        Self {
            tx,
            playing: None,
            rx,
        }
    }

    pub fn play(&mut self, media: &Path, start: f64, end: f64, idx: usize) {
        self.playing = Some(idx);
        let _ = self.tx.send(Cmd::Play(media.to_path_buf(), start, end));
    }

    pub fn stop(&mut self) {
        self.playing = None;
        let _ = self.tx.send(Cmd::Stop);
    }

    /// 每帧调一次：播完了就把高亮清掉。
    pub fn poll(&mut self) {
        while let Ok(_idx) = self.rx.try_recv() {
            self.playing = None;
        }
    }
}

#[cfg(feature = "audio")]
fn audio_thread(rx: Receiver<Cmd>, done: Sender<usize>) {
    use rodio::{Decoder, OutputStream, Sink, Source};

    let mut stream: Option<(OutputStream, rodio::OutputStreamHandle)> = None;
    let mut sink: Option<Sink> = None;

    loop {
        match rx.recv_timeout(Duration::from_millis(120)) {
            Ok(Cmd::Play(path, start, end)) => {
                if sink.as_ref().map_or(false, |s| !s.empty()) {
                    if let Some(s) = sink.take() {
                        s.stop();
                    }
                }
                if stream.is_none() {
                    match OutputStream::try_default() {
                        Ok((s, h)) => stream = Some((s, h)),
                        Err(_) => continue,
                    }
                }
                let Some((_, handle)) = stream.as_ref() else {
                    continue;
                };
                let file = match fs::File::open(&path) {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                let Ok(dec) = Decoder::new(BufReader::new(file)) else {
                    continue;
                };
                let sr = dec.sample_rate();
                let ch = dec.channels() as u32;
                let skip = (start.max(0.0) * sr as f64) as u64;
                let take = (((end - start).max(0.2)) * sr as f64) as u64;
                let src = dec
                    .skip_duration(Duration::from_secs_f64(start.max(0.0)))
                    .take_duration(Duration::from_secs_f64((end - start).max(0.2)));
                let _ = (skip, take, ch);

                let Ok(s) = Sink::try_new(handle) else {
                    continue;
                };
                s.append(src);
                sink = Some(s);
            }
            Ok(Cmd::Stop) => {
                if let Some(s) = sink.take() {
                    s.stop();
                }
                let _ = done.send(usize::MAX);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(s) = sink.as_ref() {
                    if s.empty() {
                        sink = None;
                        let _ = done.send(0);
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

#[cfg(not(feature = "audio"))]
fn audio_thread(rx: Receiver<Cmd>, done: Sender<usize>) {
    // 没有 audio feature 时（不想拉 rodio 依赖的构建）退化成空实现：
    // 命令照收，不发声音。UI 不会卡。
    let _ = done;
    loop {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(_) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// 播放器构造时刻（测试用）。
pub fn now() -> Instant {
    Instant::now()
}

// ─────────────────────────────────────────────────────────────
// 界面
// ─────────────────────────────────────────────────────────────

use eframe::egui;
use egui::{Align, Color32, FontId, Layout, RichText, ScrollArea, Sense, Stroke, Vec2};

pub struct PfState {
    pub loaded: Loaded,
    pub player: Player,
    /// 正在编辑的行：(行号, 是不是日文列)
    pub edit: Option<(usize, bool)>,
    /// 底部提示
    pub msg: String,
}

impl PfState {
    pub fn new(loaded: Loaded) -> Self {
        Self {
            loaded,
            player: Player::new(),
            edit: None,
            msg: String::new(),
        }
    }
}

const INK: Color32 = Color32::from_rgb(0x2b, 0x26, 0x22);
const INK2: Color32 = Color32::from_rgb(0x5c, 0x53, 0x49);
const INK3: Color32 = Color32::from_rgb(0x9a, 0x90, 0x86);
const ACCENT: Color32 = Color32::from_rgb(0xb8, 0x48, 0x1f);
const ACCENT_DEEP: Color32 = Color32::from_rgb(0xc2, 0x53, 0x2a);
const LINE: Color32 = Color32::from_rgb(0xf4, 0xef, 0xe9);
const ROW_HL: Color32 = Color32::from_rgb(0xfd, 0xf3, 0xed);
const ROW_GONE: Color32 = Color32::from_rgb(0xfb, 0xf8, 0xf5);

/// 画整个校对窗口。返回 true 表示用户把它关了。
pub fn ui(ctx: &egui::Context, st: &mut PfState) -> bool {
    st.player.poll();

    let screen = ctx.content_rect();
    let mut closed = false;

    // 卡片尺寸跟着屏幕走：宽 84%、高 88%，两头夹住，保证小窗也能用。
    let card_w = (screen.width() * 0.84).clamp(760.0, 1500.0);
    let card_h = (screen.height() * 0.88).clamp(460.0, 1020.0);
    let x = screen.left() + (screen.width() - card_w) * 0.5;
    let y = screen.top() + (screen.height() - card_h) * 0.5;

    // 一层全屏 Area：先铺半透明遮罩（点不到后面的界面），再摆卡片。
    egui::Area::new(egui::Id::new("pf_overlay"))
        .order(egui::Order::Foreground)
        .fixed_pos(screen.left_top())
        .show(ctx, |ui| {
            ui.painter().rect_filled(
                screen,
                0.0,
                Color32::from_rgba_unmultiplied(0x1a, 0x14, 0x10, 0x55),
            );
            // 吃掉整屏的点击，避免点穿到主界面
            let _ = ui.allocate_rect(screen, Sense::click_and_drag());

            egui::Area::new(egui::Id::new("pf_card"))
                .order(egui::Order::Foreground)
                .fixed_pos(egui::pos2(x, y))
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(Color32::WHITE)
                        .corner_radius(14.0)
                        .inner_margin(egui::Margin::symmetric(18, 14))
                        .show(ui, |ui| {
                            ui.allocate_ui_with_layout(
                                Vec2::new(card_w - 36.0, card_h - 28.0),
                                Layout::top_down(Align::Min),
                                |ui| {
                                    if header(ui) {
                                        closed = true;
                                    }
                                    body(ui, st);
                                },
                            );
                        });
                });
        });

    closed
}

/// 卡片头部：橙色竖条 + 标题 + 右上角提示 / 关闭。
/// 返回 true 表示点了关闭。
fn header(ui: &mut egui::Ui) -> bool {
    let mut closed = false;
    ui.horizontal(|ui| {
        let (bar_rect, _) = ui.allocate_exact_size(Vec2::new(3.0, 17.0), Sense::hover());
        ui.painter().rect_filled(bar_rect, 1.5, ACCENT);
        ui.add_space(8.0);
        ui.add(
            egui::Label::new(RichText::new("字幕校对").size(15.5).color(INK).strong())
                .selectable(false),
        );

        let rest = ui.available_width().max(220.0);
        ui.allocate_ui_with_layout(
            Vec2::new(rest, 20.0),
            Layout::right_to_left(Align::Center),
            |ui| {
                if ui
                    .add(
                        egui::Button::new(RichText::new("✕").size(14.0).color(INK3))
                            .fill(Color32::TRANSPARENT)
                            .stroke(Stroke::NONE)
                            .corner_radius(8.0),
                    )
                    .clicked()
                {
                    closed = true;
                }
                ui.add_space(12.0);
                ui.add(
                    egui::Label::new(
                        RichText::new("点时间可播放 · 双击文本可直接修改 · 点右侧 × 删除整句")
                            .size(11.5)
                            .color(INK3),
                    )
                    .selectable(false),
                );
            },
        );
    });
    ui.add_space(8.0);
    closed
}

/// 列宽（表头、行、分隔线共用同一份，不然对不齐）。
///
/// 时间列和 × 列是固定宽：时间戳格式 `00:00:00,000` 是等宽数字，
/// 112 px 刚好；× 只需要一个字符位。
/// 剩下的按比例分给日文原文 / 中文台词 —— 兼容模式日文那列窄一点
/// （日文句子通常比中文译文短），全量模式没有日文列，中文占满。
const W_TIME: f32 = 112.0;
const W_DEL: f32 = 26.0;
const COL_GAP: f32 = 8.0;

fn cols(avail: f32, has_ja: bool) -> (f32, f32) {
    let rest = (avail - W_TIME - W_DEL - COL_GAP * 3.0).max(140.0);
    if has_ja {
        let w_ja = (rest * 0.42).max(90.0);
        let w_zh = (rest - w_ja).max(90.0);
        (w_ja, w_zh)
    } else {
        (0.0, rest)
    }
}

fn body(ui: &mut egui::Ui, st: &mut PfState) {
    let has_ja = st.loaded.has_ja;
    let n = st.loaded.rows.len();
    let n_del = st.loaded.rows.iter().filter(|r| r.deleted).count();
    let n_edit = st.loaded.rows.iter().filter(|r| r.edited).count();

    let (hdr_ja, hdr_zh) = cols(ui.available_width(), has_ja);
    ui.horizontal(|ui| {
        ui.add_sized(
            [W_TIME, 16.0],
            egui::Label::new(RichText::new("时间").size(12.0).color(INK3)),
        );
        ui.add_space(COL_GAP);
        if has_ja {
            ui.add_sized(
                [hdr_ja, 16.0],
                egui::Label::new(RichText::new("识别原文").size(12.0).color(INK3)),
            );
            ui.add_space(COL_GAP);
        }
        ui.add_sized(
            [hdr_zh, 16.0],
            egui::Label::new(RichText::new("中文台词").size(12.0).color(INK3)),
        );
    });
    ui.add_space(4.0);
    ui.separator();

    let mut do_play: Option<(usize, f64, f64)> = None;
    let mut do_del: Option<usize> = None;
    let mut do_edit: Option<(usize, bool)> = None;

    ScrollArea::vertical()
        .id_salt("pf_rows")
        .auto_shrink([false, false])
        .max_height(ui.available_height() - 46.0)
        .show(ui, |ui| {
            for i in 0..n {
                let (start, end, ja, zh, deleted, edited, playing, lowconf) = {
                    let r = &st.loaded.rows[i];
                    (
                        r.start,
                        r.end,
                        r.ja.clone(),
                        r.zh.clone(),
                        r.deleted,
                        r.edited,
                        st.player.playing == Some(i),
                        r.lowconf,
                    )
                };

                let bg = if deleted {
                    ROW_GONE
                } else if playing {
                    ROW_HL
                } else {
                    Color32::TRANSPARENT
                };

                egui::Frame::new()
                    .fill(bg)
                    .inner_margin(egui::Margin::symmetric(2, 5))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // 时间戳（可点播）
                            let t = srt::fmt_time(start);
                            let lab = ui.add_sized(
                                [W_TIME, 18.0],
                                egui::Label::new(
                                    RichText::new(format!("▶ {t}"))
                                        .size(12.0)
                                        .color(if deleted { INK3 } else { ACCENT }),
                                )
                                .sense(Sense::click()),
                            );
                            if lab.clicked() && !deleted {
                                do_play = Some((i, start, end));
                            }
                            ui.add_space(COL_GAP);

                            let (w_ja, w_zh) = cols(ui.available_width(), has_ja);
                            if has_ja {
                                cell(ui, i, true, &ja, deleted, w_ja, edited, false, &mut do_edit, st);
                                ui.add_space(COL_GAP);
                            }
                            cell(ui, i, false, &zh, deleted, w_zh, edited, lowconf, &mut do_edit, st);

                            // 最右侧 ×
                            let sym = if deleted { "↺" } else { "×" };
                            let col = if deleted {
                                Color32::from_rgb(0x8a, 0x7f, 0x75)
                            } else {
                                Color32::from_rgb(0xd6, 0xcd, 0xc3)
                            };
                            if ui
                                .add(
                                    egui::Button::new(RichText::new(sym).size(13.0).color(col))
                                        .fill(Color32::TRANSPARENT)
                                        .stroke(Stroke::NONE)
                                        .corner_radius(9.0)
                                        .min_size(Vec2::new(20.0, 20.0)),
                                )
                                .clicked()
                            {
                                do_del = Some(i);
                            }
                        });
                    });
                // ★ v1.30.7：行之间不再画横线。
                //
                //   一百多句就是一百多条线，整个表格看起来像被切成碎片
                //   （用户实测反馈 m21123：这些白色横线不要）。
                //   行距本身已经够区分了，悬停还有浅橙底色。
            }
        });

    if let Some((i, a, b)) = do_play {
        let media = st.loaded.media.clone();
        st.player.play(&media, a, b, i);
    }
    if let Some(i) = do_del {
        st.loaded.rows[i].deleted = !st.loaded.rows[i].deleted;
    }
    if let Some((i, is_ja)) = do_edit {
        st.edit = Some((i, is_ja));
    }

    ui.add_space(8.0);
    // 分隔线铺满卡片内容宽度（不用 ui.separator() —— 它只画控件那一段）
    let sep = egui::Rangef::new(
        ui.min_rect().left(),
        ui.min_rect().left() + ui.available_width(),
    );
    ui.painter()
        .hline(sep, ui.min_rect().bottom() + 2.0, Stroke::new(1.0, LINE));
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("共 {n} 句 · 已改 {n_edit} 句 · 已删 {n_del} 句"))
                .size(12.0)
                .color(INK3),
        );
        ui.add_space(10.0);
        if ui
            .add(
                egui::Button::new(RichText::new("还原全部").size(12.5).color(INK2))
                    .fill(Color32::from_rgb(0xf0, 0xea, 0xe3))
                    .stroke(Stroke::NONE)
                    .corner_radius(8.0),
            )
            .clicked()
        {
            for r in st.loaded.rows.iter_mut() {
                r.deleted = false;
                r.edited = false;
            }
            st.edit = None;
            st.msg.clear();
        }

        // ★ v1.30.7：右对齐那块必须【显式给宽度】。
        //
        //   原来写 `ui.with_layout(Layout::right_to_left(...))` 直接嵌在
        //   `horizontal` 里 —— horizontal 按内容宽度排版，不会把这块撑到
        //   行尾，它拿到的可用宽度接近 0，按钮被挤得完全看不见
        //   （用户实测：底部只剩「共 N 句」和「还原全部」）。
        let rest_w = ui.available_width().max(120.0);
        ui.allocate_ui_with_layout(
            Vec2::new(rest_w, 24.0),
            Layout::right_to_left(Align::Center),
            |ui| {
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("保存并写回字幕").size(12.5).color(Color32::WHITE),
                        )
                        .fill(ACCENT_DEEP)
                        .stroke(Stroke::NONE)
                        .corner_radius(8.0),
                    )
                    .clicked()
                {
                    match save(&st.loaded, &st.loaded.rows) {
                        Ok((n, bak)) => {
                            st.msg = format!("已写入 {n} 句（备份 {}）", bak.display())
                        }
                        Err(e) => st.msg = format!("保存失败：{e}"),
                    }
                }
            },
        );
    });
    if !st.msg.is_empty() {
        ui.add_space(4.0);
        ui.label(RichText::new(&st.msg).size(11.5).color(INK3));
    }
}

fn cell(
    ui: &mut egui::Ui,
    i: usize,
    is_ja: bool,
    text: &str,
    deleted: bool,
    width: f32,
    edited: bool,
    lowconf: bool,
    do_edit: &mut Option<(usize, bool)>,
    st: &mut PfState,
) {
    // 宽度由调用方的 cols() 算好，这里不再 clamp —— 二次收缩会让
    // 表头和行内容宽度不一致（用户实测就是「中间空一大片」）。
    let w = width.max(40.0);
    let is_editing = st.edit == Some((i, is_ja));

    if is_editing && !deleted {
        let mut buf = text.to_string();
        let resp = ui.add_sized(
            [w, 18.0],
            egui::TextEdit::singleline(&mut buf).font(FontId::proportional(13.0)),
        );
        if resp.changed() {
            if is_ja {
                st.loaded.rows[i].ja = buf;
            } else {
                st.loaded.rows[i].zh = buf;
            }
            st.loaded.rows[i].edited = true;
        }
        if resp.lost_focus() {
            st.edit = None;
        }
        return;
    }

    // 低置信行（v1.40.1）文字标橙：人声闸的独立复核只听到很少几种字，
    // 多半是吞字或半截音效。不删也不改，只是提示用户重点听这几句。
    let mut txt = RichText::new(text).size(13.0).color(if deleted {
        Color32::from_rgb(0xb8, 0xad, 0xa2)
    } else if lowconf {
        Color32::from_rgb(0xc4, 0x6a, 0x22)
    } else if is_ja {
        INK2
    } else {
        INK
    });
    if deleted {
        txt = txt.strikethrough();
    }
    let lab = ui.add_sized([w, 18.0], egui::Label::new(txt).sense(Sense::click()).truncate());
    if lab.double_clicked() && !deleted {
        *do_edit = Some((i, is_ja));
    }
    if edited && !deleted && !is_ja {
        ui.label(RichText::new("已改").size(10.0).color(ACCENT));
    }
}
