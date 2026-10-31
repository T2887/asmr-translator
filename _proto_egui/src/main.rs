//! ASMR 翻译器 —— egui 界面（对齐 app/src/ui.rs 的新版布局，已接真实流水线）
//!
//! 后端代码（backend / config / srt / pipeline）用 #[path] 直接引 app/src 下的
//! **同一份源码** —— 单一真源，不复制分叉。`windows_reactor` 是本工程里的
//! 极简替身，只为满足 pipeline.rs 里那一处 `use windows_reactor::CancellationToken;`
//! （Cargo.toml 里没有真正的 windows-reactor 依赖，所以会解析到本地模块）。
//!
//! 运行：cargo run --release

// 双击 exe 时不要弹出那个黑色控制台窗口（release 构建才隐藏；debug 保留，
// 方便看 panic 输出）。放在最前面，必须是 crate 级属性。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[path = "../../app/src/backend.rs"]
mod backend;
#[path = "../../app/src/config.rs"]
mod config;
#[path = "../../app/src/pipeline.rs"]
mod pipeline;
#[path = "../../app/src/srt.rs"]
mod srt;
mod cancel;
mod subscan;
mod updater;
mod worker;

use eframe::egui;
use pipeline::RunMode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// 标称 6/8 GB 的显卡，CUDA 可管理容量可能因驱动和固件预留略低于整档。
// 各给约 4% 容差；启动前另有 3/5 GB【当前空余】门槛负责安全兜底。
const LITE_TOTAL_MIN_MIB: f64 = 5888.0; // 5.75 GiB，兼容标称 6 GB 卡
const FULL_TOTAL_MIN_MIB: f64 = 7885.0; // 约 7.70 GiB，兼容标称 8 GB 卡
use std::os::windows::process::CommandExt;
use egui::{
    epaint::Shadow, Align, Align2, Color32, FontFamily, FontId, Frame, Label, Layout, Margin, Rect,
    RichText, ScrollArea, Sense, Stroke, Ui, Vec2,
};

// ── 色板（与 app/src/ui.rs 一致：米色纸面 + 赤陶橙）──
const BG: Color32 = Color32::from_rgb(0xf0, 0xee, 0xe6);
const CARD: Color32 = Color32::from_rgb(0xfa, 0xf9, 0xf5);
const CARD_LINE: Color32 = Color32::from_rgb(0xe3, 0xdd, 0xd0);
const SUNK: Color32 = Color32::from_rgb(0xf5, 0xf2, 0xea);
/// 文件列表里「被选中」那一行的底色。比 SUNK 再深一档 —— 直接拿 SUNK 铺在
/// 卡片底色（CARD #faf9f5）上几乎看不出差别，实测点了行也像没反应。
const SEL_BG: Color32 = Color32::from_rgb(0xeb, 0xe6, 0xd8);
const INK: Color32 = Color32::from_rgb(0x1f, 0x1e, 0x1d);
const INK2: Color32 = Color32::from_rgb(0x4a, 0x46, 0x3f);
const INK3: Color32 = Color32::from_rgb(0x6b, 0x64, 0x58);
const ACCENT: Color32 = Color32::from_rgb(0xc9, 0x64, 0x42);
const ACCENT_DEEP: Color32 = Color32::from_rgb(0xb4, 0x53, 0x2f);
const ACCENT_SOFT: Color32 = Color32::from_rgb(0xf7, 0xeb, 0xe4);
const ACCENT_INK: Color32 = Color32::from_rgb(0x7d, 0x3f, 0x26);
const OK_GREEN: Color32 = Color32::from_rgb(0x5f, 0x8a, 0x63);
/// 「失败」那一行的标记色（和 WinUI 版的 WARN 一致）。
const WARN: Color32 = Color32::from_rgb(0xa4, 0x44, 0x2c);
const DONE_INK: Color32 = Color32::from_rgb(0x8a, 0x83, 0x74);
const SCROLL: Color32 = Color32::from_rgb(0xd5, 0xcf, 0xc2);

struct FileRow {
    /// 完整路径（流水线要用）；显示时用 name
    path: String,
    name: String,
    state: String,
    /// 被跳过的原因（`内封字幕 subrip（chi）` / `画面里压了硬字幕` / `已有外挂字幕`）。
    /// 只有 `state == "已有字幕"` 时才有值，界面上鼠标悬停能看到。
    note: Option<String>,
}

struct App {
    icon: Option<egui::TextureHandle>,
    mode: RunMode,
    to_video: bool,
    force_redo: bool,
    /// 收尾时产出哪种字幕文件：仅 SRT / 仅 ASS / 两个都要。
    /// 默认「仅 SRT」—— 老用户升级后产物不变，不会突然多出一堆 .ass。
    sub_fmt: srt::SubFmt,
    /// 点「开始处理」后正在后台扫「这个片子是不是已经自带字幕」。
    /// 扫描期间按钮显示「检查字幕…」且不可点。
    prechecking: bool,
    /// 后台扫描的结果通道。Some = 还在扫，None = 没在扫。
    precheck_rx: Option<std::sync::mpsc::Receiver<Vec<(String, String)>>>,
    /// 这一批文件已经扫过了。收到结果就置 true ——
    /// 否则 start() 每次重入都会再扫一遍（队列一变就重置成 false）。
    prechecked: bool,
    /// 路径 → 跳过原因（`内封字幕 subrip（chi）` 这类）。命中的不进流水线。
    skip_note: std::collections::HashMap<String, String>,
    /// 额外生成一份日文字幕（`<视频名>.jpn.srt`）。只有兼容模式能开 ——
    /// 全量/精简模式走 Whisper 的 translate 任务，日文在模型内部就没了。
    jpn_sub: bool,
    /// 日文是单独成文件（`<名字>.jpn.srt`）还是并进中文那一份（一条字幕两行）。
    /// 只有 `jpn_sub` 打开时才在界面上出现。
    jpn_separate: bool,
    /// 首帧按显示器尺寸校准过窗口大小了吗（只做一次）
    sized_to_screen: bool,
    /// 上一帧「标题区 + 环境卡」占掉的高度（用它算卡片高度，避免同帧自我反馈）
    used_prev: f32,
    /// 卡片区总高的平滑值（拉伸窗口时高度跟着走但不跳）
    body_total_smooth: f32,
    paused: bool,
    running: bool,
    progress: f32,
    elapsed: u64,
    files: Vec<FileRow>,
    logs: Vec<String>,
    preview: Vec<String>,

    // ── 真实后端接线 ──
    worker: Option<worker::Worker>,
    #[allow(clippy::type_complexity)]
    device_rx: Option<std::sync::mpsc::Receiver<(String, bool, f64, f64, bool, String, String, f64)>>,
    device_text: String,
    full_ok: bool,
    lite_ok: bool,
    /// CUDA 卡名（torch 报的），空串 = 没测到
    cuda_name: String,
    /// Vulkan 设备名（llama 报的，已排除 N 卡），空串 = 没测到
    vulkan_name: String,
    has_vulkan: bool,
    /// 独显剩余显存 GB（nvidia-smi 的当前快照；0.0 = 读取失败）
    vram_free_gb: f64,
    /// 点「开始」前正在后台刷新剩余显存。刷新完成后再决定是否真正启动，
    /// 避免设备检测后的旧快照放过已被其他程序占满的显卡。
    vram_refresh_rx: Option<std::sync::mpsc::Receiver<f64>>,
    /// 刷新通过后下一帧真正启动。和 CPU 确认框的 pending_start 分开，
    /// 防止 start() 再次刷新形成循环。
    pending_start_after_vram: bool,
    has_gpu: bool,
    /// 显卡总显存（GB）。0.0 = 没有 GPU 或读不出来
    vram_gb: f64,
    /// 显卡计算能力（SM），如 6.1 / 7.5 / 12.0；0.0 = 未知。
    /// 低于 7.5 的老卡（Pascal / Maxwell / Volta）用自带的 torch 上 GPU
    /// 会抛 cuDNN 9.19 的 SM 不兼容错误，只能走兼容模式。
    gpu_cap: f64,
    /// 是否正在显示「只能跑在 CPU 上」的确认框
    show_cpu_warning: bool,
    /// 用户已经确认过「就用 CPU 跑」—— 确认一次后不再弹
    cpu_ok: bool,
    /// 弹窗里点了「仍然继续」→ 下一帧开头再真正开跑。
    /// 不在 ui() 里直接调 start()：egui 是即时模式，绘制过程中改一大批状态
    /// 并启动后台线程容易踩到「画到一半数据变了」，推迟到帧开头最稳。
    pending_start: bool,
    /// 上一次帧里窗口上是否悬着被拖入的文件（只用于打一行诊断日志）
    drop_hover_seen: bool,
    work_dir: String,
    data_dir: String,
    out_dir: String,
    /// 文件列表里被选中的行（下标）。可以多选 —— 点一下选中、再点一下取消。
    selected: std::collections::HashSet<usize>,
    cur_idx: usize,
    /// 上次留下的队列（启动时读出来，等用户在弹窗里选「恢复」才填进列表）
    pending_queue: Vec<String>,
    /// 上次处理到第几个（配合 pending_queue，点「恢复」后从这儿继续）
    pending_idx: usize,
    /// 是否正在显示「恢复上次的队列？」弹窗
    show_restore_dialog: bool,
    stage: usize,
    done_stages: usize,
    stage_frac: f64,
    started_at: Option<std::time::Instant>,
    current_action: String,

    // ── 进度 / 预估剩余（照抄 app/src/ui.rs 的算法，见各自的注释）──
    /// 每一步的实测耗时（按顺序 push，一个文件 4 条）—— ETA 自校准的样本
    stage_secs: Vec<u64>,
    /// 已处理完的每个文件的语音时长（秒）
    file_speech: Vec<f64>,
    /// 当前文件的语音时长（第 2 步日志 `VAD: N 段 / X.Xs` 里解析）
    cur_speech: f64,
    /// tqdm 剩余时间的指数平滑值
    tqdm_ema: f64,
    /// 上一次见到的 tqdm 原始剩余（只在它变化时才做平滑）
    last_raw_remain: f64,
    /// 本阶段是否收到过 tqdm 的 ETA
    had_eta: bool,
    /// 显示用的预估剩余（算出来之后再限速）
    eta_display: Option<u64>,
    /// 队列级三层加权的总进度（只增不减）
    progress_value: f64,
    /// 当前阶段是什么时候开始的（第 4 步前 5 秒不用 tqdm 的 ETA）
    stage_started_at: Option<std::time::Instant>,

    // ── 更新检查 ───────────────────────────────────────────────
    /// 后台检查更新的接收端（启动时挂上，拿到结果就置 None）
    update_rx: Option<std::sync::mpsc::Receiver<Result<Option<updater::UpdateInfo>, String>>>,
    /// 查到的可用更新（None = 没有新版 / 还没查）
    update_found: Option<updater::UpdateInfo>,
    /// 是否正在显示「发现新版本」弹窗
    show_update_dialog: bool,
    /// 检查是否已经跑过一次（避免每帧重复发起）
    update_checked: bool,
    /// 下载线程的接收端
    download_rx: Option<std::sync::mpsc::Receiver<Result<std::path::PathBuf, String>>>,
    /// 下载进度：(已下载字节, 总字节)
    download_progress: Arc<Mutex<(u64, Option<u64>)>>,
    /// 点「取消」时置 true，下载线程轮询到就杀掉 curl
    download_cancel: Arc<AtomicBool>,
    /// 下好的安装包路径（等用户点「立即安装」）
    downloaded: Option<std::path::PathBuf>,
    /// 更新相关的提示文字（检查失败 / 下载失败都写这儿，显示在横幅里）
    update_msg: Option<String>,
}

impl Default for App {
    fn default() -> Self {
        let work = config::work_dir().to_string();
        let data = config::data_dir().to_string();
        let logs = vec![
            "就绪。点「开始处理」会依次执行四个步骤：".into(),
            "第 1 步 提取音频 → 第 2 步 人声检测 → 第 3 步 分离人声 → 第 4 步 语音识别翻译".into(),
            "先在 [文件列表] 里添加要处理的视频/音频，再点 [开始处理]。".into(),
            format!("[目录] 数据目录: {data}"),
            "[模式] 当前使用: 全量模式".into(),
        ];

        Self {
            icon: None,
            mode: RunMode::Full,
            to_video: true,
            force_redo: false,
            sub_fmt: srt::SubFmt::Srt,
            prechecking: false,
            precheck_rx: None,
            prechecked: false,
            skip_note: std::collections::HashMap::new(),
            jpn_sub: false,
            jpn_separate: false,
            sized_to_screen: false,
            used_prev: 0.0,
            body_total_smooth: 0.0,
            paused: false,
            running: false,
            progress: 0.0,
            elapsed: 0,
            files: Vec::new(),
            logs,
            preview: Vec::new(),
            worker: None,
            device_rx: None,
            device_text: "正在检测设备…".into(),
            full_ok: true,
            lite_ok: true,
            cuda_name: String::new(),
            vulkan_name: String::new(),
            has_vulkan: false,
            vram_free_gb: 0.0,
            gpu_cap: 0.0,
            vram_refresh_rx: None,
            pending_start_after_vram: false,
            has_gpu: false,
            vram_gb: 0.0,
            show_cpu_warning: false,
            cpu_ok: false,
            pending_start: false,
            drop_hover_seen: false,
            work_dir: work,
            data_dir: data,
            // 取消勾选「放视频旁边」时这一项会显示出来，所以默认就得是真实目录
            out_dir: config::dir_asr(),
            selected: std::collections::HashSet::new(),
            cur_idx: 0,
            pending_queue: Vec::new(),
            pending_idx: 0,
            show_restore_dialog: false,
            stage: 0,
            done_stages: 0,
            stage_frac: 0.0,
            started_at: None,
            current_action: String::new(),
            stage_secs: Vec::new(),
            file_speech: Vec::new(),
            cur_speech: 0.0,
            tqdm_ema: 0.0,
            last_raw_remain: -1.0,
            had_eta: false,
            eta_display: None,
            progress_value: 0.0,
            stage_started_at: None,
            update_rx: None,
            update_found: None,
            show_update_dialog: false,
            update_checked: false,
            download_rx: None,
            download_progress: Arc::new(Mutex::new((0, None))),
            download_cancel: Arc::new(AtomicBool::new(false)),
            downloaded: None,
            update_msg: None,
        }
    }
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_fonts(&cc.egui_ctx);
        apply_theme(&cc.egui_ctx);
        let mut app = Self::default();
        app.icon = load_icon(&cc.egui_ctx);
        // 启动时读一次上次留下的队列（★ 只读一次，避免多个字段互相矛盾）。
        // 和 WinUI 版一致：不直接填进列表，先弹窗问「恢复 / 清空」。
        if let Some((q, idx)) = Self::load_queue() {
            app.pending_queue = q;
            app.pending_idx = idx;
            app.show_restore_dialog = true;
        }
        apply_round_corners(cc);
        // 设备检测要跑一小段 Python（torch 首次导入 2~3 秒）→ 放后台线程
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(pipeline::detect_device());
        });
        app.device_rx = Some(rx);
        app
    }

    // ── 列表 / 队列 ──

    fn add_path(&mut self, p: &str) {
        self.add_paths(&[p.to_string()]);
    }

    /// 批量加文件：拖进来可能有几十个路径，一次只写一遍队列文件。
    /// （逐个调 add_path 的话，拖 100 个文件就是 100 次全量重写 `_file_queue.txt`。）
    fn add_paths(&mut self, paths: &[String]) {
        let before = self.files.len();
        // 被扩展名挡下来的文件不能默默丢掉 —— 用户拖进来却「什么都没发生」
        // 是最难自己排查的一类问题，所以下面会单独打一行说明。
        let mut skipped: Vec<String> = Vec::new();
        for p in paths {
            for f in config::expand_dropped_path(p) {
                if !config::is_video_audio(&f) {
                    skipped.push(
                        std::path::Path::new(&f)
                            .file_name()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_else(|| f.clone()),
                    );
                    continue;
                }
                if self.files.iter().any(|r| r.path == f) {
                    continue;
                }
                let name = std::path::Path::new(&f)
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| f.clone());
                self.files.push(FileRow {
                    path: f,
                    name,
                    state: "排队中".into(),
                    note: None,
                });
            }
        }
        let n = self.files.len() - before;
        self.push_log(format!("[列表] 新增 {n} 个，共 {} 个文件", self.files.len()));
        if !skipped.is_empty() {
            let head: Vec<String> = skipped.iter().take(3).cloned().collect();
            let more = if skipped.len() > 3 { " …" } else { "" };
            self.push_log(format!(
                "[列表] 跳过 {} 个不支持的文件：{}{}（支持 mp4/mkv/avi/mov/wmv/flv/webm/m4v/ts/mpg/mpeg + flac/wav/mp3/m4a/aac/ogg/opus/wma）",
                skipped.len(),
                head.join("、"),
                more
            ));
        }
        if n > 0 {
            // 队列变了 → 预检结论作废，下次点开始重新扫
            self.prechecked = false;
            self.skip_note.clear();
            self.save_queue();
        }
    }

    /// 队列文件：`{数据目录}\_file_queue.txt`。
    /// 静态方法 —— `load_queue` 要在 App 还没构造出来时就能用。
    fn queue_path() -> std::path::PathBuf {
        std::path::Path::new(&config::data_dir()).join("_file_queue.txt")
    }

    /// 把队列写盘。格式和 WinUI 版【完全一致】：第 1 行 cur_idx，之后每行一个路径。
    ///
    /// 为什么必须一致：两个版本共用同一个 `_file_queue.txt`。格式一旦不同，
    /// 交替使用时会读不出对方写的队列（第一行变成路径、parse::<usize>() 直接失败）。
    ///
    /// 为什么用裸文本而不是 JSON：路径里有中文、空格、反斜杠，JSON 转义反而
    /// 更容易出岔子；路径本身不含换行，按行切是安全的。
    fn save_queue(&self) {
        let mut body = String::with_capacity(64);
        body.push_str(&self.cur_idx.to_string());
        body.push('\n');
        for f in &self.files {
            body.push_str(&f.path);
            body.push('\n');
        }
        let _ = std::fs::write(Self::queue_path(), body);
    }

    /// 读回上次的队列。返回 (文件列表, 处理到第几个)。逻辑照抄 WinUI 版：
    ///   去 BOM → 第一行 parse 成 idx → 过滤空行与已不存在的路径 →
    ///   空则 None → idx 已经到头（上轮跑完了）就删掉队列文件并返回 None。
    fn load_queue() -> Option<(Vec<String>, usize)> {
        let text = std::fs::read_to_string(Self::queue_path()).ok()?;
        // 去掉可能的 UTF-8 BOM：程序自己写的文件没有 BOM，但用户手改过、
        // 或用带 BOM 的编辑器（PowerShell WriteAllText）保存过时会有 ——
        // 那样第一行会变成 "\u{FEFF}3"，parse 失败 → 整个队列读不出来。
        let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
        let mut lines = text.lines();
        let first = lines.next()?.trim().to_string();
        // 第一行应该是 cur_idx。但它也可能是【本原型旧版本】写的队列
        // （那时只写路径、没有 idx 行）—— 那种情况下第一行就是第一个文件
        // 的路径，parse::<usize>() 会失败。这时回退成「从第 1 个开始」，
        // 否则格式升级一次就把用户攒下来的队列整个丢掉。
        let (idx, rest): (usize, Vec<String>) = match first.parse::<usize>() {
            Ok(n) => (n, lines.map(|l| l.trim().to_string()).collect()),
            Err(_) => (
                0,
                std::iter::once(first)
                    .chain(lines.map(|l| l.trim().to_string()))
                    .collect(),
            ),
        };
        let files: Vec<String> = rest
            .into_iter()
            .filter(|l| !l.is_empty())
            .filter(|l| std::path::Path::new(l).is_file())
            .collect();
        if files.is_empty() {
            return None;
        }
        // 索引已经到头 = 上一轮全部处理完了。不做这个判断的话，跑完一趟
        // 重启照样弹「恢复上次的队列？（85 个文件，处理到第 86 个）」。
        //
        // 还有一种同样该闭嘴的情况：列表里每个文件都已经有字幕了。
        //   cur_idx 停在 len-1（最后一个文件刚做完就退出）时上面的判断不成立，
        //   但用户眼里「都处理完了」，再问一句「要不要恢复」只会让他以为漏了活。
        let all_done = files.iter().all(|p| {
            let path = std::path::Path::new(p);
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let dir = path.parent().unwrap_or(std::path::Path::new("."));
            dir.join(format!("{stem}.chs.srt")).is_file()
        });
        if idx >= files.len() || all_done {
            let _ = std::fs::remove_file(Self::queue_path());
            return None;
        }
        Some((files, idx))
    }

    /// 删掉队列文件（用户选了「清空」）
    fn clear_queue_file() {
        let _ = std::fs::remove_file(Self::queue_path());
    }

    /// 用户在弹窗里选了「恢复」：把文件列表填回去，从上次的位置继续。
    fn restore_queue(&mut self) {
        let n = self.pending_queue.len();
        let idx = self.pending_idx.min(n);
        for p in std::mem::take(&mut self.pending_queue) {
            let name = std::path::Path::new(&p)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| p.clone());
            let state = if self.has_subtitle(&p) { "完成" } else { "排队中" };
            self.files.push(FileRow {
                path: p,
                name,
                state: state.to_string(),
                note: None,
            });
        }
        self.cur_idx = idx;
        self.pending_idx = 0;
        self.show_restore_dialog = false;
        // 恢复后重存一次：load_queue 可能过滤掉了已删除的文件，
        // 现在的列表才是准确的。
        self.save_queue();
        self.push_log(format!("[队列] 已恢复 {n} 个文件（从第 {} 个继续）", idx + 1));
        if let Some(next) = self.files.get(idx) {
            self.push_log(format!("[队列] 下一个: {}", next.name));
        }
        if idx >= n {
            self.push_log("[队列] 上次已经把列表跑完了，可以直接重新开始");
        }
    }

    /// 用户在弹窗里选了「清空」：删掉持久化文件，队列归零。
    fn clear_queue_all(&mut self) {
        Self::clear_queue_file();
        self.files.clear();
        self.selected.clear();
        self.preview.clear();
        self.cur_idx = 0;
        self.pending_queue.clear();
        self.pending_idx = 0;
        self.show_restore_dialog = false;
        self.save_queue();
        self.push_log("[队列] 已清空");
    }

    /// 这个文件算不算已经做完了。
    ///
    /// 按**当前选的字幕格式**判断：选了「仅 ASS」而磁盘上只有 .srt 时，
    /// 这个文件其实还没做（用户要的 .ass 不存在），不能算完成 ——
    /// 否则他改完格式再点开始会被整批跳过，还以为程序坏了。
    /// 另外如果这次勾了「日文单独成文件」，日文那份也得在。
    fn has_subtitle(&self, path: &str) -> bool {
        let p = std::path::Path::new(path);
        let stem = p
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let dir = p.parent().unwrap_or(std::path::Path::new("."));
        if self.sub_fmt.want_srt() && !dir.join(format!("{stem}.chs.srt")).is_file() {
            return false;
        }
        if self.sub_fmt.want_ass() && !dir.join(format!("{stem}.chs.ass")).is_file() {
            return false;
        }
        let want_jpn = self.jpn_sub && self.jpn_separate && matches!(self.mode, RunMode::Compat);
        if want_jpn && !dir.join(format!("{stem}.jpn.srt")).is_file() {
            return false;
        }
        true
    }

    // ── 运行控制 ──

    /// 「强制重新处理」时，先把本工具以前生成的字幕删掉。返回 `(删掉几个, 失败几个)`。
    ///
    /// 只删**自己产物**：视频旁 `<名字>.chs.srt/.ass`、`<名字>.jpn.srt/.ass`，
    /// 以及 3asr 下同名的中间件。用户自己放的别的字幕（比如不带语言后缀的
    /// `xxx.ass`）一根手指都不碰 —— 那是他的文件，不是我们生成的。
    ///
    /// 删不掉（被播放器占用 / 只读）时只记数不中断：继续跑下去，第 4 步脚本
    /// 会看到目标还在而跳过，用户至少能拿到「字幕还是旧的」这个结果，
    /// 比整个任务不开始要好。
    fn purge_old_subs(&self) -> (usize, usize) {
        let asr = std::path::PathBuf::from(config::dir_asr());
        let mut n = 0usize;
        let mut bad = 0usize;
        for f in &self.files {
            let p = std::path::PathBuf::from(&f.path);
            let stem = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            if stem.is_empty() {
                continue;
            }
            let dir = p
                .parent()
                .map(|d| d.to_path_buf())
                .unwrap_or_else(|| std::path::PathBuf::from("."));
            let mut targets: Vec<std::path::PathBuf> = Vec::new();
            for ext in ["srt", "ass"] {
                targets.push(dir.join(format!("{stem}.chs.{ext}")));
                targets.push(dir.join(format!("{stem}.jpn.{ext}")));
                targets.push(asr.join(format!("{stem}.{ext}")));
            }
            for t in targets {
                if !t.is_file() {
                    continue;
                }
                match std::fs::remove_file(&t) {
                    Ok(_) => n += 1,
                    Err(_) => bad += 1,
                }
            }
        }
        (n, bad)
    }

    fn n_stages(&self) -> usize {
        pipeline::stages(self.mode).len()
    }

    fn mode_name(&self) -> &'static str {
        match self.mode {
            RunMode::Full => "全量模式",
            RunMode::Lite => "精简模式",
            RunMode::Compat => "兼容模式",
        }
    }

    fn stage_title(&self) -> String {
        if self.stage == 0 {
            return "——".to_string();
        }
        pipeline::stages(self.mode)
            .get(self.stage - 1)
            .map(|s| s.2.to_string())
            .unwrap_or_else(|| "——".to_string())
    }

    /// 已知剩余显存时，当前模式能否安全启动。
    /// 总显存门槛（全量 8.0 / 精简 6.0 GB）判断硬件能力；这里判断当前余量：
    /// 全量峰值约 4 GB + 1 GB 安全余量，精简约 2 GB + 1 GB安全余量。
    /// 兼容模式不走 CUDA 主流水线，不受这个检查限制。
    fn selected_mode_has_free_vram(&self) -> bool {
        if self.vram_free_gb <= 0.0 {
            return true; // 读取失败时不误杀；后端仍会给出真实 OOM 错误
        }
        match self.mode {
            RunMode::Full => self.vram_free_gb >= 5.0,
            RunMode::Lite => self.vram_free_gb >= 3.0,
            RunMode::Compat => true,
        }
    }

    /// 轻量刷新 NVIDIA 当前空余显存（不导入 torch，不阻塞 UI）。
    fn refresh_free_vram_before_start(&mut self) {
        if !self.has_gpu || matches!(self.mode, RunMode::Compat) {
            self.pending_start_after_vram = true;
            return;
        }
        if self.vram_refresh_rx.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.vram_refresh_rx = Some(rx);
        self.push_log("[显存] 正在刷新当前空余显存…");
        std::thread::spawn(move || {
            let free = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
                .creation_flags(0x08000000)
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.lines().next()?.trim().parse::<f64>().ok())
                .map(|mib| mib / 1024.0)
                .unwrap_or(0.0);
            let _ = tx.send(free);
        });
    }

    fn start(&mut self) {
        if self.running {
            return;
        }
        if self.files.is_empty() {
            self.push_log("[提示] 文件列表是空的，先用「添加文件」选几个");
            return;
        }
        // 运行环境（env\python.exe）是整条流水线的前提。既没有内置环境、也没通过
        // ASMR_PYTHON 指定时，config::python_exe() 会退回裸名字 "python" 交给 PATH
        // 解析 —— 在没装 Python 的机器上会命中微软商店的占位 stub，子进程退出码
        // 9009（cmd 的「命令找不到」），用户完全看不懂。这里提前把原因说清楚。
        // 实测来源：朋友覆盖安装时把主程序装到了新目录，运行环境还在旧目录。
        // 注意只在「最终会退回裸 python」时拦：env 存在或设了 ASMR_PYTHON 都放行，
        // 免得误杀自己配了 Python 的用户。
        if config::python_exe() == "python" {
            let py = std::path::Path::new(&config::work_dir()).join(r"env\python.exe");
            self.push_log(format!("[环境缺失] 找不到运行环境：{}", py.display()));
            self.push_log(format!(
                "[环境缺失] 请把「2_运行环境_安装包」装到同一个目录：{}",
                config::work_dir()
            ));
            self.push_log(
                "[环境缺失] 如果你自己装了 Python，可设置环境变量 ASMR_PYTHON 指向 python.exe 后重开程序",
            );
            return;
        }
        // 点击开始时先轻量刷新一次 nvidia-smi；设备检测时的快照可能早已过期。
        // 刷新完成后 pump() 会把 pending_start_after_vram 置上，再进来继续。
        if self.has_gpu && !matches!(self.mode, RunMode::Compat) && !self.pending_start_after_vram {
            self.refresh_free_vram_before_start();
            return;
        }
        self.pending_start_after_vram = false;
        if !self.selected_mode_has_free_vram() {
            let need = if matches!(self.mode, RunMode::Full) { 5.0 } else { 3.0 };
            self.push_log(format!(
                "[显存不足] {}需要至少约 {need:.1} GB 空余显存，当前只有 {:.1} GB，已阻止启动。请关闭占用显存的程序，或改用兼容模式。",
                self.mode_name(), self.vram_free_gb
            ));
            return;
        }
        // 无 N 卡、显存不足 6 GB、而且没有 Vulkan 设备 → 只能跑 CPU，先弹窗确认一次
        // （照 app/src/ui.rs:1892-1901）。放在任何状态被改动之前：用户点「取消」
        // 后界面和没点过一样，不会停在半启动状态。
        let no_cuda = !self.has_gpu
            || (self.vram_gb > 0.0 && self.vram_gb * 1024.0 < LITE_TOTAL_MIN_MIB);
        if no_cuda && !self.has_vulkan && !self.cpu_ok {
            self.show_cpu_warning = true;
            return;
        }
        // ── 开跑前先看这些片子是不是已经自带字幕 ──
        //
        //   分两层：内封的**软字幕**（ffprobe，毫秒级）+ 画面里烧死的**硬字幕**
        //   （抽几帧看白字黑描边）。命中就标「已有字幕」跳过 —— 用户手上有一堆
        //   已经压好中文字幕的片子，再跑一遍人声分离 + 识别 + 翻译纯属白烧
        //   几十分钟（一部 80 分钟的片要跑小半小时）。
        //
        //   只扫一次：扫完之后 `prechecked` 才置位（表示「结果已就绪」），
        //   千万不能在发起线程时就置位 —— 那样 `try_recv` 万一没被轮询到，
        //   下次点开始就会直接跳过检测。
        //   队列一变（增删文件）就把 `prechecked` 清掉重来。
        //
        //   ⚠️ 勾了「强制重新处理」就不扫 —— 用户的规则是「强制处理 = 我就要重做，
        //      别拦我」。
        if !self.force_redo && !self.files.is_empty() {
            // 还在扫：等它，别抢先开跑（否则等于没检测）
            if self.precheck_rx.is_some() {
                self.push_log("[预检] 还在检查字幕，请稍候…");
                return;
            }
            if !self.prechecked {
                if subscan::tools().is_some() {
                    let paths: Vec<String> = self.files.iter().map(|f| f.path.clone()).collect();
                    let n = paths.len();
                    let (tx, rx) = std::sync::mpsc::channel();
                    std::thread::spawn(move || {
                        let _ = tx.send(subscan::precheck(&paths));
                    });
                    self.precheck_rx = Some(rx);
                    self.prechecking = true;
                    self.push_log(format!(
                        "[预检] 检查 {n} 个文件是否自带字幕（内封 / 画面内嵌）…"
                    ));
                    return;
                }
                self.push_log("[预检] 没找到 ffmpeg / ffprobe，跳过自带字幕检测");
                self.prechecked = true;
            }
        }
        // 勾了「强制重新处理」：先把本工具以前生成的字幕删掉。
        //
        // 不删的话，第 4 步脚本会看到目标已存在而跳过，用户勾了重做却发现
        // 字幕还是老的。只删本工具自己的产物（`.chs.*` / `.jpn.*` 和 3asr 下的中间件），
        // 不碰用户自己放的其他文件。
        if self.force_redo {
            let (n, bad) = self.purge_old_subs();
            if n > 0 {
                self.push_log(format!("[强制] 已删除 {n} 个旧字幕文件"));
            }
            if bad > 0 {
                self.push_log(format!(
                    "[强制] 有 {bad} 个旧字幕删不掉（可能被播放器占用），这次可能沿用旧字幕"
                ));
            }
        }
        // 先在不可变借用里把「谁已经有字幕」算出来，再进 iter_mut 循环 ——
        // 否则循环里调 self.has_subtitle() 会和 iter_mut 的可变借用打架（E0502）。
        let force = self.force_redo;
        let already: Vec<bool> = self
            .files
            .iter()
            .map(|f| self.has_subtitle(&f.path))
            .collect();
        // skip_note 克隆一份：循环里 self.files 已被可变借用，不能再用 self.skip_note
        let skip_note = self.skip_note.clone();
        let mut jobs: Vec<(usize, String)> = Vec::new();
        let mut newly_skipped: Vec<String> = Vec::new();
        for (i, f) in self.files.iter_mut().enumerate() {
            if !force && already[i] {
                f.state = "完成".into();
                f.note = None;
            } else if !force && skip_note.contains_key(&f.path) {
                let reason = skip_note[&f.path].clone();
                f.state = "已有字幕".into();
                f.note = Some(reason.clone());
                newly_skipped.push(format!("        · {} —— {reason}", f.name));
            } else {
                f.state = "排队中".into();
                f.note = None;
                jobs.push((i, f.path.clone()));
            }
        }
        if !newly_skipped.is_empty() {
            let total = self.files.len();
            self.push_log(format!(
                "[跳过] {} 个文件已经自带字幕，不再重复处理",
                newly_skipped.len()
            ));
            for line in newly_skipped {
                self.push_log(line);
            }
            self.push_log(format!(
                "[队列] {} 个待处理 / 共 {total} 个",
                jobs.len()
            ));
        }
        if jobs.is_empty() {
            self.push_log("[列表] 所有文件都已有字幕（勾上「强制重新处理」可重做）");
            return;
        }
        self.push_log(format!(
            "[开始] {} 个文件，模式：{}",
            jobs.len(),
            self.mode_name()
        ));
        self.preview.clear();
        self.running = true;
        self.paused = false;
        self.progress = 0.0;
        self.progress_value = 0.0;
        self.stage = 0;
        self.done_stages = 0;
        self.stage_frac = 0.0;
        self.elapsed = 0;
        self.started_at = Some(std::time::Instant::now());
        // 进度 / ETA 的样本全部作废，重新开始积累
        self.stage_secs.clear();
        self.file_speech.clear();
        self.cur_speech = 0.0;
        self.tqdm_ema = 0.0;
        self.last_raw_remain = -1.0;
        self.had_eta = false;
        self.eta_display = None;
        self.stage_started_at = None;
        let out = self.out_dir.clone();
        // 日文字幕只有兼容模式能产出（别的模式没有日文原文），
        // 界面里非兼容模式下这个框本来就是灰的，这里再兜一层。
        backend::set_jpn_sub(self.jpn_sub && matches!(self.mode, RunMode::Compat));
        backend::set_jpn_separate(
            self.jpn_sub
                && self.jpn_separate
                && matches!(self.mode, RunMode::Compat),
        );
        self.worker = Some(worker::spawn(
            jobs,
            self.mode,
            self.to_video,
            out,
            self.sub_fmt,
        ));
    }

    fn stop(&mut self) {
        if let Some(w) = &self.worker {
            w.cancel.cancel();
        }
        worker::kill();
        self.running = false;
        self.paused = false;
        self.push_log("[停止] 已请求停止，并强杀当前子进程");
    }

    fn toggle_pause(&mut self) {
        if !self.running {
            return;
        }
        if self.paused {
            worker::resume();
            self.paused = false;
            self.push_log("[继续] 已恢复进程");
        } else {
            worker::suspend();
            self.paused = true;
            self.push_log("[暂停] 已挂起进程（模型停在原地）");
        }
    }

    /// 「设备状态」那一行 —— 跟**当前模式**走，不是跟检测结果走（和 WinUI 版一致）：
    ///   全量 / 精简 → `GPU：NVIDIA GeForce RTX 5070（已启用 CUDA）`
    ///   兼容模式    → `GPU：AMD Radeon Graphics（已启用 Vulkan）`
    ///   兼容但没 Vulkan 设备 → 显存够就写 `CPU：未找到可用的 Vulkan 设备`，
    ///                           显存不够（<4 GB）说明多半是显存被别的程序占了
    fn device_line(&self) -> String {
        // 老卡（计算能力 < 7.5）优先说清楚原因 —— 否则会退到下面「兼容但没
        // Vulkan」那句，写成「CPU：未找到可用的 Vulkan 设备」，把「显卡太老」
        // 这个真正的原因藏掉了。
        let arch_old = self.gpu_cap > 0.0 && self.gpu_cap < 7.5;
        if arch_old {
            let who = plain_device_name(&self.cuda_name);
            return if who.is_empty() {
                "显卡太老（计算能力低于 7.5），请用兼容模式".to_string()
            } else {
                format!("显卡太老（{who}），请用兼容模式")
            };
        }
        let (name, accel) = self.accel();
        if !name.is_empty() && !accel.is_empty() {
            return format!("GPU：{}（已启用 {accel}）", plain_device_name(name));
        }
        if self.mode == RunMode::Compat {
            if self.has_gpu && self.vram_free_gb > 0.0 && self.vram_free_gb < 4.0 {
                let who = plain_device_name(&self.cuda_name);
                if who.is_empty() {
                    format!(
                        "GPU（剩余显存 {:.1} GB，低于兼容模式要求，已切换至 CPU 推理）",
                        self.vram_free_gb
                    )
                } else {
                    format!(
                        "GPU：{who}（剩余显存 {:.1} GB，低于兼容模式要求，已切换至 CPU 推理）",
                        self.vram_free_gb
                    )
                }
            } else {
                "CPU：未找到可用的 Vulkan 设备".to_string()
            }
        } else {
            self.device_text.clone()
        }
    }

    /// 当前该用哪块设备、哪个后端（副标题和「设备状态」共用这一份）
    fn accel(&self) -> (&str, &'static str) {
        if self.mode == RunMode::Compat {
            (
                self.vulkan_name.as_str(),
                if self.has_vulkan { "Vulkan" } else { "" },
            )
        } else {
            (
                self.cuda_name.as_str(),
                if self.cuda_name.is_empty() { "" } else { "CUDA" },
            )
        }
    }

    /// 副标题：`本地转写 · 离线翻译 · RTX 5070 CUDA 加速中 · 第 1 / 3 个文件`
    fn subtitle_text(&self) -> String {
        let (name, accel) = self.accel();
        let mut parts: Vec<String> = vec!["本地转写".into(), "离线翻译".into()];
        if !name.is_empty() && !accel.is_empty() {
            parts.push(format!("{} {accel} 加速中", short_device(name)));
        }
        let total = self.files.len();
        if total > 0 {
            let idx = if self.running {
                (self.cur_idx + 1).min(total)
            } else {
                total
            };
            parts.push(format!("第 {idx} / {total} 个文件"));
        }
        parts.join(" · ")
    }

    // ══════════════════════════════════════════════════════════
    // 进度 与 预估剩余
    //   这套口径是从 app/src/ui.rs 原样搬过来的，别在这里「简化」——
    //   WinUI 版为了这两个数字踩了十几次坑（见 config.rs 的版本说明），
    //   下面每一条注释都对应一次真实误报。
    // ══════════════════════════════════════════════════════════

    /// 数一个目录里的文件个数。
    fn count_files(dir: &str) -> usize {
        std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().is_file())
                    .count()
            })
            .unwrap_or(0)
    }

    /// 当前阶段走到哪了（0.0~1.0）：文件级 + 文件内两层。
    fn stage_ratio(&self) -> f64 {
        let total_files = self.files.len().max(1) as f64;
        let files_done = self.cur_idx.min(self.files.len()) as f64;
        let file_frac = self.stage_frac.clamp(0.0, 1.0);
        let two_level = ((files_done + file_frac) / total_files).clamp(0.0, 1.0);
        if self.running && !self.files.is_empty() && two_level > 0.0 {
            return two_level;
        }
        if self.stage_frac > 0.0 {
            return file_frac;
        }
        // 拿不到 tqdm 时退回「数输出目录文件数」
        let (from, to) = match self.stage {
            2 => (config::dir_pre(), config::dir_audio()),
            3 => (config::dir_audio(), config::dir_vad()),
            4 => (config::dir_vad(), config::dir_asr()),
            _ => return 0.0,
        };
        let total = Self::count_files(&from);
        if total == 0 {
            return 0.0;
        }
        (Self::count_files(&to) as f64 / total as f64).clamp(0.0, 1.0)
    }

    /// 从已完成文件的实测数据反推「每秒语音要多少秒处理」。
    /// 返回 (分离成本率, 识别成本率)；没有样本时 None（用经验默认值）。
    fn cost_rates(&self) -> (Option<f64>, Option<f64>) {
        let n_files = self.stage_secs.len() / 4;
        if n_files == 0 || self.file_speech.len() < n_files {
            return (None, None);
        }
        let (mut sep, mut asr, mut sp_sum) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n_files {
            let sp = self.file_speech[i];
            if sp <= 0.0 {
                continue;
            }
            sep += self.stage_secs[i * 4 + 2] as f64;
            asr += self.stage_secs[i * 4 + 3] as f64;
            sp_sum += sp;
        }
        if sp_sum <= 0.0 {
            return (None, None);
        }
        (Some(sep / sp_sum), Some(asr / sp_sum))
    }

    /// 重算总进度：队列级三层加权，只增不减（往回退会让人以为程序出错）。
    ///   总进度 = (已完成文件数 + 当前文件加权进度) / 总文件数 × 100
    fn recompute_progress(&mut self) {
        if self.files.is_empty() || self.stage == 0 {
            return;
        }
        // 四步的实测时间占比（14 个文件的日志反推，不是均分的 25%）
        const W: [f64; 4] = [0.04, 0.10, 0.36, 0.50];
        let n_all = self.files.len() as f64;
        let n_done = self.cur_idx as f64;
        let st = self.stage.min(4);
        let mut file_prog: f64 = W[..(st - 1)].iter().sum::<f64>();

        // 本步的内部进度：有 tqdm 就用，没有就用「已用时间 / 预计耗时」冒充。
        // 第 1、2 步没有 tqdm 输出，不这么做进度条会死钉在上一档不动。
        let all_done: f64 = self.stage_secs.iter().map(|v| *v as f64).sum();
        let stage_used = (self.elapsed as f64 - all_done).max(0.0);
        let sp = self.cur_speech.max(0.0);
        let frac_eff = if self.stage_frac > 0.0 {
            self.stage_frac.clamp(0.0, 1.0)
        } else {
            // expected 留 20% 余量；上限 0.95 —— 宁可到不了头，也别提前冲到 100%
            let expected = match st {
                1 => 25.0,
                2 => 45.0,
                3 if sp > 0.0 => (0.062 * sp * 1.2).max(15.0),
                4 if sp > 0.0 => (0.110 * sp * 1.2).max(15.0),
                _ => 60.0,
            };
            (stage_used / expected).clamp(0.0, 0.95)
        };
        file_prog += W[st - 1] * frac_eff;
        let p = ((n_done + file_prog) / n_all * 100.0).clamp(0.0, 100.0);
        if p > self.progress_value {
            self.progress_value = p;
        }
    }

    /// 预估【当前这个文件】还剩多少秒。None = 样本不够，界面显示「—」。
    fn compute_eta(&self) -> Option<u64> {
        if !self.running {
            return None;
        }
        let elapsed = self.elapsed as f64;
        if elapsed < 5.0 {
            return None; // 刚开始几秒的数据不可信
        }
        let sp = self.cur_speech.max(0.0);
        // 第 1、2 步还不知道语音时长。硬算会得到「约 10 秒」这种骗人的数，
        // 宁可显示「—」。
        if sp <= 0.0 {
            return None;
        }
        let all_done: f64 = self.stage_secs.iter().map(|v| *v as f64).sum();
        let stage_used = (elapsed - all_done).max(1.0);
        let (sep_opt, asr_opt) = self.cost_rates();
        let sep_est = sep_opt.unwrap_or(0.0620) * sp;
        let asr_est = asr_opt.unwrap_or(0.1100) * sp;
        let fixed_first_two = {
            let n = self.stage_secs.len() / 4;
            if n > 0 {
                (0..n)
                    .map(|i| (self.stage_secs[i * 4] + self.stage_secs[i * 4 + 1]) as f64)
                    .sum::<f64>()
                    / n as f64
            } else {
                53.0 // 15.9（提取）+ 37.1（VAD）
            }
        };
        const TAIL: f64 = 10.0; // 字幕改名、清理
        let frac = self.stage_frac.clamp(0.0, 1.0);
        let frac_ok = frac > 0.001;
        let remain = match self.stage {
            1 => {
                let left = (fixed_first_two * 0.4 - stage_used).max(0.0);
                left + fixed_first_two * 0.6 + sep_est + asr_est + TAIL
            }
            2 => {
                let left = (fixed_first_two * 0.6 - stage_used).max(0.0);
                left + sep_est + asr_est + TAIL
            }
            // 优先 tqdm 自带的 ETA > 按本阶段速度外推 > 成本模型
            3 => {
                let left = if self.had_eta {
                    self.tqdm_ema
                } else if frac_ok {
                    (stage_used * (1.0 - frac) / frac).max(0.0)
                } else if sep_est > stage_used {
                    sep_est - stage_used
                } else {
                    stage_used * 0.5
                };
                left + asr_est + TAIL
            }
            4 => {
                // 前 5 秒不用 tqdm 的 ETA：Whisper 还在 warmup，前几帧乱跳
                let warmed = self
                    .stage_started_at
                    .map(|t| t.elapsed().as_secs() >= 5)
                    .unwrap_or(true);
                let left = if self.had_eta && warmed {
                    self.tqdm_ema
                } else if frac_ok {
                    (stage_used * (1.0 - frac) / frac).max(0.0)
                } else if asr_est > stage_used {
                    asr_est - stage_used
                } else {
                    // 模型低估了（本步已经跑得比预测久）。
                    // 返回 0 会让界面卡在「约 10 秒」不动，所以按当前速度再估一半。
                    stage_used * 0.5
                };
                left + TAIL
            }
            _ => return None,
        };
        Some(remain.max(0.0) as u64)
    }

    /// 「当前文件预估剩余」这一格显示什么。
    fn eta_text(&self) -> String {
        match self.eta_display {
            Some(s) => {
                let m = s / 60;
                if m >= 60 {
                    format!("约 {} 小时 {} 分", m / 60, m % 60)
                } else if m > 0 {
                    format!("约 {m} 分 {:02} 秒", s % 60)
                } else {
                    format!("约 {s} 秒")
                }
            }
            None => {
                if self.running {
                    // 还在跑但估不出来（第 1、2 步还没拿到语音时长）
                    "—".to_string()
                } else if self.progress_value >= 99.5 {
                    "0 秒".to_string() // 跑完了就明确归零
                } else {
                    "—".to_string()
                }
            }
        }
    }

    fn push_log(&mut self, s: impl Into<String>) {
        let s = s.into();
        // 实时落盘：崩溃/被结束时，界面上那些行会一起消失，
        // 只有文件里还能看到「最后走到哪一步」。
        append_run_log(&self.data_dir, &s);
        self.logs.push(s);
        if self.logs.len() > 5000 {
            self.logs.drain(0..1500);
        }
    }

    // ── 每帧收事件 ──

    fn pump(&mut self, ctx: &egui::Context) {
        // 点开始前的轻量显存刷新结果（nvidia-smi，不导入 torch）。
        if self.vram_refresh_rx.is_some() {
            let got = self.vram_refresh_rx.as_ref().map(|rx| rx.try_recv());
            match got {
                Some(Ok(free)) => {
                    self.vram_refresh_rx = None;
                    if free > 0.0 {
                        self.vram_free_gb = free;
                        let vram_mib = self.vram_gb * 1024.0;
                        self.full_ok = self.has_gpu
                            && (self.vram_gb <= 0.0 || vram_mib >= FULL_TOTAL_MIN_MIB)
                            && free >= 5.0;
                        self.lite_ok = self.has_gpu
                            && (self.vram_gb <= 0.0 || vram_mib >= LITE_TOTAL_MIN_MIB)
                            && free >= 3.0;
                        self.push_log(format!("[显存] 当前空余显存：{free:.1} GB"));
                    } else {
                        self.push_log("[显存] 无法刷新空余显存，将按设备检测结果继续");
                    }
                    // 自动延续刚才那次「开始」，用户不必再点第二次。
                    self.pending_start_after_vram = true;
                    self.pending_start = true;
                }
                Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    self.vram_refresh_rx = None;
                    self.push_log("[显存] 刷新线程异常结束，将按设备检测结果继续");
                    self.pending_start_after_vram = true;
                    self.pending_start = true;
                }
                _ => {}
            }
        }
        // 设备检测结果（后台线程）
        if self.device_rx.is_some() {
            let got = self.device_rx.as_ref().map(|rx| rx.try_recv());
            match got {
                Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    self.device_rx = None;
                    self.device_text = "设备检测失败（可直接用兼容模式）".into();
                    self.push_log("[设备] 检测线程结束但没有回值");
                }
                Some(Ok((text, is_gpu, vram, free, vulkan, cuda_name, vk_name, cap))) => {
                self.device_rx = None;
                self.device_text = text.clone();
                // 阈值与 Python settings.py 共用同一口径：标称 6/8 GB 卡各给约 4%
                // 容差，避免驱动/固件预留后显示 6.0/8.0 GB 却被后台误判。当前空余
                // 显存仍单独要求精简 3 GB、全量 5 GB，负责真正的 OOM 安全兜底。
                let vram_mib = vram * 1024.0;
                let free_unknown = free <= 0.0;
                // 计算能力 < 7.5 的老卡（Pascal 6.x / Maxwell 5.x / Volta 7.0）：
                // 自带 torch 2.11+cu128 的 cuDNN 9.19 不再为这些架构提供 kernel，
                // 一上 GPU 加载模型就抛
                //   RuntimeError: cuDNN version 91900 is not compatible with
                //                 devices with SM < 7.5.
                // 界面上表现为「第2步退出码 1」，用户看不出是显卡太老。所以这里
                // 直接把全量/精简判为不可用，并把模式锁到兼容模式（纯 CPU 设计，
                // 速度比「全量 + CPU」快得多）。
                let arch_old = cap > 0.0 && cap < 7.5;
                self.gpu_cap = cap;
                if arch_old {
                    self.push_log(format!(
                        "[显卡] 检测到 {cuda_name}（计算能力 {cap:.1}），低于 7.5"
                    ));
                    self.push_log(
                        "[显卡] 新版 PyTorch 的 cuDNN 已不再支持这么老的显卡，全量/精简模式在 GPU 上会直接报错",
                    );
                    self.push_log(
                        "[显卡] 已锁定为兼容模式：它同样是纯 CPU 运行，但速度快得多，也没有显存要求",
                    );
                }
                // 总显存判断硬件资格；当前空余显存判断此刻是否安全启动。
                // 全量约 4 GB 峰值 + 1 GB 余量，精简约 2 GB + 1 GB 余量。
                self.full_ok = !arch_old
                    && is_gpu
                    && (vram <= 0.0 || vram_mib >= FULL_TOTAL_MIN_MIB)
                    && (free_unknown || free >= 5.0);
                self.lite_ok = !arch_old
                    && is_gpu
                    && (vram <= 0.0 || vram_mib >= LITE_TOTAL_MIN_MIB)
                    && (free_unknown || free >= 3.0);
                self.cuda_name = cuda_name.clone();
                self.vulkan_name = vk_name.clone();
                self.has_vulkan = vulkan;
                self.vram_free_gb = free;
                self.vram_gb = vram;
                self.has_gpu = is_gpu;
                // 显存 < 8.1 GB → 自动进精简模式（照 app/src/ui.rs:2354-2366）。
                // 8.1 GB 是 settings.py 算出来的门槛：识别 float16 峰值 4.2 GB
                // + 系统基线 2.9 GB + 安全余量 1 GB。低于它时 Python 侧会把
                // compute_type 换成 int8_float16，这里负责把「为什么」讲清楚。
                if is_gpu && vram > 0.0 && vram < 8.1 {
                    self.push_log(format!(
                        "[精简模式] 当前显卡只有 {vram:.1} GB 显存，为保证你的其他程序正常运行，识别环节将工作在精简模式。"
                    ));
                    self.push_log(
                        "[精简模式] 具体做法：把语音识别的计算精度从 float16 降到 int8，显存占用减半（约 4.0 GB → 2.0 GB）。",
                    );
                    self.push_log("[精简模式] 代价：识别速度慢约 20%，个别词可能不够准确。");
                }
                // 当前模式不安全：优先降到仍可运行的精简模式，否则兼容模式。
                // 老卡（arch_old）全量/精简都不可用，直接锁兼容模式，且不再拿
                // 「空余显存」解释原因（那样会误导用户去关程序腾显存）。
                if arch_old {
                    self.mode = RunMode::Compat;
                } else if self.mode == RunMode::Full && !self.full_ok {
                    if self.lite_ok {
                        self.mode = RunMode::Lite;
                        self.push_log(format!(
                            "[模式] 当前空余显存只有 {free:.1} GB（全量需约 5.0 GB），已自动切到精简模式"
                        ));
                    } else {
                        self.mode = RunMode::Compat;
                        self.push_log(format!(
                            "[模式] 当前空余显存只有 {free:.1} GB（全量需约 5.0 GB、精简需约 3.0 GB），已自动切到兼容模式"
                        ));
                    }
                } else if self.mode == RunMode::Lite && !self.lite_ok {
                    self.mode = RunMode::Compat;
                    self.push_log(format!(
                        "[模式] 当前空余显存只有 {free:.1} GB（精简需约 3.0 GB），已自动切到兼容模式"
                    ));
                }
                self.push_log(format!("[设备] {text}"));
                self.push_log(format!(
                    "[设备] CUDA：{}   Vulkan：{}",
                    if cuda_name.is_empty() { "无" } else { cuda_name.as_str() },
                    if vk_name.is_empty() { "无" } else { vk_name.as_str() }
                ));
                }
                _ => {}
            }
        }

        // 流水线事件
        let mut evs: Vec<worker::Ev> = Vec::new();
        if let Some(w) = &self.worker {
            while let Ok(ev) = w.rx.try_recv() {
                evs.push(ev);
            }
        }
        let mut finished = false;
        for ev in evs {
            match ev {
                worker::Ev::Log(l) => {
                    // 第 2 步（人声检测）的日志里有本文件的语音时长：
                    //   `VAD段已导出: xxx.vad.json（4 段 / 22.5 秒）`
                    // 分离和识别的耗时都正比于【语音时长】而不是音频总长，
                    // 这是 ETA 最关键的输入。
                    //
                    // ★ 必须在这里（逐行日志）解析，不能只在 Tick 的 latest 里解析：
                    //   latest 是「当前最新一行」的滚动快照，VAD 那一行出现后
                    //   250ms 内就可能被下一行顶掉，实测一条都抓不到，ETA 永远是「—」。
                    //   WinUI 版也是在日志行里解析的（ui.rs 的 handle_log_line）。
                    if let Some(c) = config::vad_speech_re().captures(&l) {
                        if let Ok(v) = c[1].parse::<f64>() {
                            if v > 0.0 {
                                self.cur_speech = v;
                            }
                        }
                    }
                    self.push_log(l)
                }
                worker::Ev::StageStart(stage) => {
                    self.stage = stage;
                    self.stage_frac = 0.0;
                    self.done_stages = stage - 1;
                    self.current_action.clear();
                    // 换阶段 = 换了一套耗时模型，ETA 的锚点作废
                    self.eta_display = None;
                    self.stage_started_at = Some(std::time::Instant::now());
                    self.had_eta = false;
                    self.tqdm_ema = 0.0;
                    self.last_raw_remain = -1.0;
                }
                worker::Ev::Tick { frac, eta, latest } => {
                    self.stage_frac = frac;
                    if !latest.is_empty() {
                        self.current_action = latest.clone();
                    }
                    // 第 2 步的日志里有本文件的语音时长（`VAD: 72 段 / 201.3s（占 67.1%）`）。
                    // 分离和识别的耗时都正比于【语音时长】而不是音频总长，
                    // 这是 ETA 最关键的输入。
                    if self.stage == 2 {
                        if let Some(c) = config::vad_speech_re().captures(&latest) {
                            if let Ok(v) = c[1].parse::<f64>() {
                                if v > 0.0 {
                                    self.cur_speech = v;
                                }
                            }
                        }
                    }
                    // tqdm 的 ETA 做指数平滑。
                    //   只在它真的变化时才算一次 —— 心跳 120ms 一次、tqdm 3 秒刷新一次，
                    //   同一个值会被反复平滑几十遍，表现就是「ETA 冻住」。
                    //   系数 0.95/0.05 是按时间常数定的（tau ≈ 2 秒）：
                    //   tqdm 相邻帧最大跳变 336 秒，这样能把跳变压到几秒。
                    // （worker 的契约：eta = 脚本自报的剩余秒数，0 = 这一帧没有）
                    if eta > 0.0 {
                        self.had_eta = true;
                        if (eta - self.last_raw_remain).abs() > 0.01 {
                            self.tqdm_ema = if self.tqdm_ema <= 0.0 {
                                eta
                            } else {
                                self.tqdm_ema * 0.95 + eta * 0.05
                            };
                            self.last_raw_remain = eta;
                        }
                    }
                }
                worker::Ev::StageDone { stage, code, secs } => {
                    self.done_stages = stage;
                    self.stage_frac = 0.0;
                    // 本步实测耗时 —— ETA 自校准的样本（每 4 条一个文件）
                    self.stage_secs.push(secs.max(0.0).round() as u64);
                    self.eta_display = None;
                    self.push_log(format!(
                        "[完成] 第{stage}步，退出码 {code}，本步用时 {:.1} 秒",
                        secs
                    ));
                }
                worker::Ev::Preview(lines) => {
                    // 累积而不是替换：一个文件可能分几步吐字幕（兼容模式第 3 步、
                    // 全量模式第 4 步），替换会把前一批冲掉。
                    // 去重：实时抓的那批和阶段结束时补的那批会重叠。
                    for s in lines {
                        if !self.preview.contains(&s) {
                            self.preview.push(s);
                        }
                    }
                }
                worker::Ev::FileStart { idx, total, name } => {
                    self.cur_idx = idx;
                    // 换文件 = 换一批字幕：预览区先清空。
                    //   不清的话上一个文件的字幕会一直挂在那儿，看着像新文件的内容。
                    self.preview.clear();
                    // 上一个文件的语音时长归档，作为成本率样本
                    if self.cur_speech > 0.0 {
                        self.file_speech.push(self.cur_speech);
                    }
                    self.cur_speech = 0.0;
                    self.eta_display = None; // 新文件：ETA 重新开始
                    if let Some(f) = self.files.get_mut(idx) {
                        f.state = "处理中".into();
                    }
                    self.push_log(format!("--- 文件（{}/{}）{name} ---", idx + 1, total));
                }
                worker::Ev::FileDone { idx } => {
                    if let Some(f) = self.files.get_mut(idx) {
                        f.state = "完成".into();
                    }
                    // 与 WinUI 版一致：cur_idx 表示「下一个要处理的」，做完一个就 +1。
                    //   跑完最后一个时它等于列表长度 —— load_queue 看到 idx >= len
                    //   就知道上一轮已经跑完，不会再多问一句「要不要恢复队列」。
                    self.cur_idx = idx + 1;
                    self.save_queue();
                }
                worker::Ev::FileAborted { idx, cancelled } => {
                    // 这一行没跑完就中止了。不处理的话它会永久停在「处理中」——
                    // 用户明明已经终止了任务，界面还写着处理中。
                    if let Some(f) = self.files.get_mut(idx) {
                        f.state = if cancelled { "已停止".into() } else { "失败".into() };
                    }
                    self.save_queue();
                }
                worker::Ev::AllDone => finished = true,
            }
        }
        if finished {
            self.worker = None;
            // 兜底：任何还挂着「处理中」的行都收尾。
            //   正常流程里不会有剩（每个文件要么 FileDone 要么 FileAborted），
            //   但万一事件顺序出意外，也别让界面留着假的「处理中」。
            for f in self.files.iter_mut() {
                if f.state == "处理中" {
                    f.state = "已停止".into();
                }
            }
            let was_running = self.running;
            self.running = false;
            self.paused = false;
            self.stage = 0;
            self.done_stages = 0;
            self.stage_frac = 0.0;
            self.current_action.clear();
            self.progress_value = 100.0;
            self.eta_display = None;
            // 整批都做完了 → 把游标推到队尾，队列文件里记成「已经到 len」，
            // 下次启动 load_queue 的 idx >= len 判断就会删掉队列文件、不弹窗。
            // 中途停止 / 有文件失败时不推：那种情况确实该问「要不要接着上次继续」。
            if !self.files.is_empty() && self.files.iter().all(|f| f.state == "完成") {
                self.cur_idx = self.files.len();
            }
            self.save_queue();
            if was_running {
                self.progress = 1.0;
                self.push_log("=== 队列处理结束 ===");
            }
        }

        if self.running {
            if let Some(t0) = self.started_at {
                self.elapsed = t0.elapsed().as_secs();
            }
            // 总进度：队列级三层加权（每帧重算，只增不减）
            self.recompute_progress();
            // 预估剩余：算出来之后按【每秒】限速再显示。
            //   用绝对增量而不是百分比 —— ETA 越大百分比放得越宽，
            //   233 秒时一次就允许跳 28 秒，越到后面越跳。
            //   涨 8 秒/秒、降 25 秒/秒（心跳 120ms → 涨 0.96、降 3.0）。
            match self.compute_eta() {
                Some(raw) => {
                    let smoothed = match self.eta_display {
                        Some(prev) => {
                            let up = prev as f64 + 0.96;
                            let down = (prev as f64 - 3.0).max(0.0);
                            raw.clamp(down as u64, up as u64)
                        }
                        None => raw,
                    };
                    self.eta_display = Some(smoothed);
                }
                None => self.eta_display = None,
            }
            self.progress = (self.progress_value / 100.0) as f32;
        } else {
            // 没在跑：清掉 ETA，别让界面一直挂着结束前最后一帧的数字
            self.eta_display = None;
        }

        // ★ egui 是即时模式：没有输入就不会重新跑 ui()，也就不会调 pump()。
        //   只要还有后台活儿（设备检测 / 流水线事件），就必须主动请求重绘，
        //   否则结果永远收不到，界面就卡在「正在检测设备…」。
        if self.running || self.device_rx.is_some() || self.worker.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));

        }
    }
}

/// 读 assets/icon.png（和 WinUI 版同一个文件）。
/// 无边框窗口的拖边热区。
///
/// `with_decorations(false)` 之后系统边框没了，鼠标拖窗口边缘不再改变大小 ——
/// 这里自己判断指针是否落在边缘 6 个逻辑点内：命中就换光标形状，
/// 按下时让 winit 接管这次拖拽（`ViewportCommand::BeginResize`）。
///
/// 用 `ctx.input` 直接判断而不是铺 Area：Area 会挡住它下面控件的点击，
/// 而边缘热区本身不需要参与 egui 的交互系统。
fn resize_hotspots(ctx: &egui::Context) {
    use egui::{CursorIcon, ResizeDirection};

    // 正拖着文件往窗口里来的时候，绝对不能在边缘抢「拖边改大小」——
    // OLE 拖放和系统 resize 的模态循环会互相打断，表现就是
    // 「拖文件到窗口边缘，窗口开始变大小，文件没放进去」。
    if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
        return;
    }

    const B: f32 = 6.0; // 热区厚度（逻辑点）
    let (r, mut p) = ctx.input(|i| (i.viewport_rect(), i.pointer.hover_pos()));
    let pressed = ctx.input(|i| i.pointer.primary_pressed());
    let Some(p) = p.take() else { return };
    if !r.contains(p) {
        return;
    }

    let (l, t, rt, b) = (r.left(), r.top(), r.right(), r.bottom());
    let on_l = p.x - l <= B;
    let on_r = rt - p.x <= B;
    let on_t = p.y - t <= B;
    let on_b = b - p.y <= B;

    let (dir, cursor) = match (on_l, on_r, on_t, on_b) {
        (true, _, true, _) => (ResizeDirection::NorthWest, CursorIcon::ResizeNwSe),
        (_, true, true, _) => (ResizeDirection::NorthEast, CursorIcon::ResizeNeSw),
        (true, _, _, true) => (ResizeDirection::SouthWest, CursorIcon::ResizeNeSw),
        (_, true, _, true) => (ResizeDirection::SouthEast, CursorIcon::ResizeNwSe),
        (true, _, _, _) => (ResizeDirection::West, CursorIcon::ResizeHorizontal),
        (_, true, _, _) => (ResizeDirection::East, CursorIcon::ResizeHorizontal),
        (_, _, true, _) => (ResizeDirection::North, CursorIcon::ResizeVertical),
        (_, _, _, true) => (ResizeDirection::South, CursorIcon::ResizeVertical),
        _ => return,
    };

    ctx.set_cursor_icon(cursor);
    if pressed {
        ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(dir));
    }
}

fn icon_candidates() -> Vec<std::path::PathBuf> {
    let mut cands: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            cands.push(dir.join("assets").join("icon.png"));
            cands.push(dir.join("icon.png"));
        }
    }
    cands.push(std::path::PathBuf::from(
        "D:/ASMR翻译器/app/assets/icon.png",
    ));
    cands.push(std::path::PathBuf::from("../app/assets/icon.png"));
    cands.push(std::path::PathBuf::from("assets/icon.png"));
    cands
}

/// 解码成 RGBA 像素（窗口图标和标题图标共用）。
fn icon_rgba() -> Option<(Vec<u8>, u32, u32)> {
    for p in icon_candidates() {
        let Ok(bytes) = std::fs::read(&p) else { continue };
        let Ok(img) = image::load_from_memory(&bytes) else {
            continue;
        };
        let rgba = img.to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());
        return Some((rgba.into_raw(), w, h));
    }
    None
}

/// 标题左侧的图标纹理。
fn load_icon(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let (raw, w, h) = icon_rgba()?;
    let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &raw);
    Some(ctx.load_texture("app-icon", color, egui::TextureOptions::LINEAR))
}

/// 给无边框窗口加 Win11 的系统圆角。
///
/// 关掉 `with_decorations` 之后系统不再负责窗口外形，四角变成直角；
/// DWM 的 `DWMWA_WINDOW_CORNER_PREFERENCE` 能让它重新裁成圆角。
/// Win10 上没有这个属性，调用会失败 —— 返回值忽略即可，没有副作用
/// （所以这份代码在 Win10 上跑也安全，只是不圆）。
fn apply_round_corners(cc: &eframe::CreationContext<'_>) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let Ok(handle) = cc.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(w) = handle.as_raw() else {
        return;
    };
    let pref = windows_sys::Win32::Graphics::Dwm::DWMWCP_ROUND;
    unsafe {
        windows_sys::Win32::Graphics::Dwm::DwmSetWindowAttribute(
            w.hwnd.get() as *mut core::ffi::c_void,
            windows_sys::Win32::Graphics::Dwm::DWMWA_WINDOW_CORNER_PREFERENCE as u32,
            &pref as *const _ as *const core::ffi::c_void,
            std::mem::size_of_val(&pref) as u32,
        );
    }
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    let mut load = |key: &str, candidates: &[&str]| -> bool {
        for path in candidates {
            if let Ok(bytes) = std::fs::read(path) {
                fonts.font_data.insert(
                    key.to_owned(),
                    std::sync::Arc::new(egui::FontData::from_owned(bytes)),
                );
                return true;
            }
        }
        false
    };

    // 常规体（雅黑）
    if load("zh", &["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]) {
        fonts
            .families
            .get_mut(&FontFamily::Proportional)
            .unwrap()
            .insert(0, "zh".to_owned());
        fonts
            .families
            .get_mut(&FontFamily::Monospace)
            .unwrap()
            .push("zh".to_owned());
    }
    // 粗体（雅黑 Bold）—— 用于标题，WinUI 的标题是 SemiBold
    if load("zhb", &["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyhbd.ttf"]) {
        fonts
            .families
            .entry(FontFamily::Name("bold".into()))
            .or_default()
            .push("zhb".to_owned());
        fonts
            .families
            .entry(FontFamily::Name("bold".into()))
            .or_default()
            .push("zh".to_owned());
    }

    // 符号回退：雅黑里没有 ✓（U+2713）这类符号字形，会渲染成方框。
    // WinUI 版靠系统字体链接自动回退，egui 得手动挂一个符号字体。
    if load(
        "sym",
        &[
            "C:/Windows/Fonts/seguisym.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ],
    ) {
        fonts
            .families
            .get_mut(&FontFamily::Proportional)
            .unwrap()
            .push("sym".to_owned());
        fonts
            .families
            .get_mut(&FontFamily::Monospace)
            .unwrap()
            .push("sym".to_owned());
    }

    ctx.set_fonts(fonts);
}

fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style_of(egui::Theme::Light)).clone();
    style.spacing.item_spacing = Vec2::new(8.0, 8.0);
    style.spacing.button_padding = Vec2::new(14.0, 7.0);
    style.spacing.scroll.bar_width = 6.0;
    style.spacing.scroll.bar_inner_margin = 3.0;
    style.spacing.scroll.floating = true;

    style.visuals.window_fill = BG;
    style.visuals.panel_fill = BG;
    style.visuals.extreme_bg_color = SUNK;
    style.visuals.override_text_color = Some(INK);
    style.visuals.selection.bg_fill = ACCENT_SOFT;
    style.visuals.selection.stroke = Stroke::new(1.0, ACCENT);
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, CARD_LINE);
    style.visuals.widgets.inactive.weak_bg_fill = CARD;
    style.visuals.widgets.inactive.bg_fill = SCROLL;
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, CARD_LINE);
    style.visuals.widgets.hovered.weak_bg_fill = SUNK;
    style.visuals.widgets.hovered.bg_fill = ACCENT;
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, CARD_LINE);
    style.visuals.widgets.active.weak_bg_fill = ACCENT_SOFT;
    style.visuals.widgets.active.bg_fill = ACCENT;
    style.text_styles = [
        (
            egui::TextStyle::Heading,
            FontId::new(26.0, FontFamily::Name("bold".into())),
        ),
        (egui::TextStyle::Body, FontId::proportional(13.5)),
        (egui::TextStyle::Button, FontId::proportional(13.0)),
        (egui::TextStyle::Small, FontId::proportional(12.0)),
        (egui::TextStyle::Monospace, FontId::monospace(12.5)),
    ]
    .into();
    ctx.set_theme(egui::Theme::Light);
    ctx.set_style_of(egui::Theme::Light, style.clone());
    ctx.set_style_of(egui::Theme::Dark, style);
}

// ── 自定义控件 ──

/// 处理模式的「胶囊按钮」（对应 WinUI 的 mode_chip）。
/// 圆角 15、内边距 13×6、前面 9px 圆点。
/// 两个颜色之间插值（用于 hover / 选中的过渡）。
fn lerp_c(a: Color32, b: Color32, t: f32) -> Color32 {
    let f = |x: u8, y: u8| {
        (x as f32 + (y as f32 - x as f32) * t.clamp(0.0, 1.0))
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
}

fn chip(ui: &mut Ui, label: &str, selected: bool, enabled: bool) -> egui::Response {
    let font = FontId::proportional(13.0);
    let text_w = ui
        .painter()
        .layout_no_wrap(label.to_owned(), font.clone(), INK2)
        .size()
        .x;
    let w = 13.0 + 9.0 + 7.0 + text_w + 13.0;
    let h = 6.0 * 2.0 + 13.0 + 4.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, h), Sense::click());

    // 选中 / 悬停过渡（0.13 秒）
    let ts = ui.ctx().animate_bool_with_time(
        ui.id().with(("chip_sel", label)),
        selected && enabled,
        0.13,
    );
    let th = ui.ctx().animate_bool_with_time(
        ui.id().with(("chip_hov", label)),
        resp.hovered() && enabled && !selected,
        0.13,
    );

    let mut bg = lerp_c(CARD, SUNK, th);
    bg = lerp_c(bg, ACCENT_SOFT, ts);
    let line = lerp_c(lerp_c(CARD_LINE, ACCENT, th * 0.5), ACCENT, ts);
    let dot = lerp_c(CARD_LINE, ACCENT, ts);
    let fg = if !enabled {
        INK3
    } else {
        lerp_c(INK2, ACCENT_INK, ts)
    };

    let p = ui.painter();
    p.rect_filled(rect, 15.0, bg);
    p.rect_stroke(rect, 15.0, Stroke::new(1.0, line), egui::StrokeKind::Inside);
    let cy = rect.center().y;
    let cx = rect.left() + 13.0 + 4.5;
    p.circle_filled(egui::pos2(cx, cy), 4.5, dot);
    p.text(
        egui::pos2(cx + 4.5 + 7.0, cy),
        Align2::LEFT_CENTER,
        label,
        font,
        fg,
    );
    resp
}

/// 复选框（对应 WinUI 的 check_row）：16px 方框，选中橙底白勾。
fn wcheck(ui: &mut Ui, checked: bool, text: &str) -> egui::Response {
    wcheck_enabled(ui, checked, text, true)
}

/// 带禁用态的复选框。禁用时整行不响应点击、文字变浅 ——
/// 用来表达「这个开关在当前模式下没有意义」（例：日文字幕只有兼容模式有）。
fn wcheck_enabled(ui: &mut Ui, checked: bool, text: &str, enabled: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), 22.0),
        if enabled { Sense::click() } else { Sense::hover() },
    );
    let p = ui.painter();
    let box_rect = Rect::from_center_size(
        egui::pos2(rect.left() + 8.0, rect.center().y),
        Vec2::splat(15.0),
    );
    // 勾选 / 悬停过渡
    let t = ui
        .ctx()
        .animate_bool_with_time(ui.id().with(("chk", text)), checked, 0.13);
    let th = ui.ctx().animate_bool_with_time(
        ui.id().with(("chk_hov", text)),
        resp.hovered(),
        0.13,
    );
    p.rect_filled(box_rect, 3.0, lerp_c(Color32::WHITE, ACCENT, t));
    p.rect_stroke(
        box_rect,
        3.0,
        Stroke::new(
            1.3,
            lerp_c(CARD_LINE, ACCENT, if checked { 1.0 } else { th }),
        ),
        egui::StrokeKind::Inside,
    );
    if t > 0.01 {
        let c = box_rect.center();
        let alpha = (255.0 * t) as u8;
        let mark = Color32::from_rgba_unmultiplied(255, 255, 255, alpha);
        p.line_segment(
            [c + Vec2::new(-3.8, 0.3), c + Vec2::new(-1.1, 3.0)],
            Stroke::new(1.8, mark),
        );
        p.line_segment(
            [c + Vec2::new(-1.1, 3.0), c + Vec2::new(4.0, -3.2)],
            Stroke::new(1.8, mark),
        );
    }
    p.text(
        egui::pos2(box_rect.right() + 8.0, rect.center().y),
        Align2::LEFT_CENTER,
        text,
        FontId::proportional(13.0),
        if enabled { INK } else { INK3 },
    );
    resp
}

/// 设备状态前面的小绿点。
fn dot_ok(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, OK_GREEN);
}

// ── 小组件 ──

fn card<R>(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    card_badge(ui, title, None, add)
}

fn card_badge<R>(
    ui: &mut Ui,
    title: &str,
    badge: Option<&str>,
    add: impl FnOnce(&mut Ui) -> R,
) -> R {
    // 两级阴影：外层大而淡的投影 + 卡片自身小而实的投影
    Frame::new()
        .fill(Color32::TRANSPARENT)
        .corner_radius(14.0)
        .inner_margin(Margin::same(0))
        .shadow(Shadow {
            offset: [0, 8],
            blur: 26,
            spread: 0,
            color: Color32::from_rgba_unmultiplied(0x1f, 0x1e, 0x1d, 14),
        })
        .show(ui, |ui| card_body(ui, title, badge, add))
        .inner
}

fn card_body<R>(
    ui: &mut Ui,
    title: &str,
    badge: Option<&str>,
    add: impl FnOnce(&mut Ui) -> R,
) -> R {
    Frame::new()
        .fill(CARD)
        .corner_radius(12.0)
        .inner_margin(Margin::same(14))
        .stroke(Stroke::new(1.0, CARD_LINE))
        .shadow(Shadow {
            offset: [0, 2],
            blur: 8,
            spread: 0,
            color: Color32::from_rgba_unmultiplied(0x1f, 0x1e, 0x1d, 20),
        })
        .show(ui, |ui| {
            // Frame 默认按内容收缩宽度；环境卡片内容不足整宽会在右边留一大块空白
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(3.0, 15.0), Sense::hover());
                ui.painter().rect_filled(rect, 1.5, ACCENT);
                ui.add_space(4.0);
                ui.label(
                    RichText::new(title)
                        .font(FontId::new(15.0, FontFamily::Name("bold".into())))
                        .color(INK),
                );
                if let Some(t) = badge {
                    badge_pill(ui, t);
                }
            });
            ui.add_space(10.0);
            add(ui)
        })
        .inner
}

/// 标题右侧的小胶囊（例如文件数量）。
fn badge_pill(ui: &mut Ui, text: &str) {
    let font = FontId::proportional(11.0);
    let tw = ui.painter().layout_no_wrap(text.to_owned(), font.clone(), INK3).size().x;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(tw + 14.0, 18.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 9.0, SUNK);
    p.text(rect.center(), Align2::CENTER_CENTER, text, font, INK3);
}

fn inner_panel<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(SUNK)
        .corner_radius(7.0)
        .inner_margin(Margin::same(10))
        .stroke(Stroke::new(1.0, CARD_LINE))
        .show(ui, add)
        .inner
}

fn field(ui: &mut Ui, label: &str, value: &str) {
    ui.label(RichText::new(label).size(11.0).color(INK3));
    ui.add(Label::new(RichText::new(value).size(14.0).color(INK)).truncate());
}

fn progress_bar(ui: &mut Ui, value: f32, width: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 6.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 3.0, Color32::from_rgb(0xe8, 0xe4, 0xda));
    if value > 0.0 {
        let w = (rect.width() * value.clamp(0.0, 1.0)).max(6.0);
        p.rect_filled(
            Rect::from_min_size(rect.min, Vec2::new(w, rect.height())),
            3.0,
            ACCENT,
        );
    }
}

/// 普通胶囊按钮（白底 + 细描边），用于「添加文件」「停止」这类。
fn pill(ui: &mut Ui, text: &str, bg: Color32, line: Color32, fg: Color32, enabled: bool) -> egui::Response {
    pill_icon(ui, text, 0, bg, line, fg, enabled)
}

/// 带图标的胶囊按钮。`icon`：0 = 不带，1 = 播放三角，2 = 暂停双竖条。
///
/// 为什么图标要自绘、不能用 "▶" / "‖" 这种字符：雅黑里没有这些符号的字形，
/// egui 找不到就会画成方框（WinUI 版靠系统字体链接自动回退，egui 不会）。
/// 自绘最稳，且和 WinUI 版的几何形状一致（三角 + 两条圆角竖条）。
#[allow(clippy::too_many_arguments)]
fn pill_icon(
    ui: &mut Ui,
    text: &str,
    icon: u8,
    bg: Color32,
    line: Color32,
    fg: Color32,
    enabled: bool,
) -> egui::Response {
    let font = FontId::proportional(13.0);
    let tw = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font.clone(), fg)
        .size()
        .x;
    let iw = if icon == 0 { 0.0 } else { 16.0 }; // 10 图标 + 6 间距
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(tw + iw + 30.0, 30.0), Sense::click());
    let th = ui.ctx().animate_bool_with_time(
        ui.id().with(("pill_hov", text)),
        resp.hovered() && enabled,
        0.13,
    );
    let is_primary = bg == ACCENT_SOFT;
    let hot = if bg == CARD { SUNK } else { ACCENT_DEEP };
    let bg = lerp_c(bg, hot, th);
    let line = lerp_c(line, ACCENT, th * if is_primary { 0.6 } else { 0.35 });
    let p = ui.painter();
    p.rect_filled(rect, 15.0, bg);
    p.rect_stroke(rect, 15.0, Stroke::new(1.0, line), egui::StrokeKind::Inside);

    let cy = rect.center().y;
    let x0 = rect.center().x - (tw + iw) / 2.0; // 内容（图标+文字）左边界
    if icon != 0 {
        let icx = x0 + 5.0;
        if icon == 1 {
            p.add(egui::Shape::convex_polygon(
                vec![
                    egui::pos2(icx - 3.5, cy - 5.0),
                    egui::pos2(icx - 3.5, cy + 5.0),
                    egui::pos2(icx + 4.0, cy),
                ],
                fg,
                Stroke::NONE,
            ));
        } else {
            for dx in [-3.0_f32, 3.0] {
                p.rect_filled(
                    Rect::from_center_size(egui::pos2(icx + dx, cy), Vec2::new(3.0, 11.0)),
                    1.2,
                    fg,
                );
            }
        }
    }
    p.text(
        egui::pos2(x0 + iw + tw / 2.0, cy),
        Align2::CENTER_CENTER,
        text,
        font,
        fg,
    );
    resp
}

/// 设备全名：只去商标符号（(TM) / (R)），保留厂商和型号。
fn plain_device_name(name: &str) -> String {
    name.replace("(TM)", "")
        .replace("(tm)", "")
        .replace("(R)", "")
        .replace("(r)", "")
        .trim()
        .to_string()
}

/// 副标题里的短设备名："AMD Radeon(TM) Graphics" → "Radeon Graphics"，
/// "NVIDIA GeForce RTX 5070（已启用 CUDA）" → "RTX 5070"。
/// 只认全角括号和逗号作分隔 —— 半角括号是型号的一部分（Intel(R) Iris(R) Xe）。
fn short_device(name: &str) -> String {
    let cleaned = name.trim().replace("(TM)", "").replace("(tm)", "");
    let cleaned = cleaned.replace("(R)", "").replace("(r)", "");
    let name = cleaned.trim();
    let mut s = name.trim();
    for prefix in [
        "NVIDIA GeForce ",
        "NVIDIA ",
        "AMD Radeon ",
        "AMD ",
        "Intel(R) ",
        "Intel ",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.trim();
        }
    }
    let end = s.find(|c| matches!(c, '（' | '）' | '，')).unwrap_or(s.len());
    let s = s[..end].trim();
    if s == "Graphics" {
        "Radeon Graphics".to_string()
    } else {
        s.to_string()
    }
}

/// 自绘的小对勾（文件列表「完成」标记）。
fn check_mark(ui: &mut Ui, color: Color32) {
    let (r, _) = ui.allocate_exact_size(Vec2::splat(14.0), Sense::hover());
    let p = ui.painter();
    let c = r.center();
    let s = Stroke::new(1.7, color);
    p.line_segment(
        [egui::pos2(c.x - 4.0, c.y + 0.3), egui::pos2(c.x - 1.3, c.y + 3.4)],
        s,
    );
    p.line_segment(
        [egui::pos2(c.x - 1.3, c.y + 3.4), egui::pos2(c.x + 4.3, c.y - 3.4)],
        s,
    );
}

fn ghost_button(ui: &mut Ui, text: &str) -> egui::Response {
    ghost_button_enabled(ui, text, true)
}

/// 和 `ghost_button` 一样，但可以置灰。
/// 处理中要禁掉列表编辑：worker 按「下标 + 路径」认行，列表一动它就对不上号。
fn ghost_button_enabled(ui: &mut Ui, text: &str, enabled: bool) -> egui::Response {
    let fg = if enabled { INK2 } else { INK3 };
    pill(ui, text, CARD, CARD_LINE, fg, enabled)
}

/// 兜底图标（找不到 assets/icon.png 时用）：橙色圆角方块 +
/// 三条圆头竖条（声波）+ 底部横条，与 assets/icon.png 同形。
fn logo(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(40.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 10.0, ACCENT);
    let cx = rect.center().x;
    let top = rect.top();
    let bar = |cxx: f32, h: f32, y0: f32| {
        p.rect_filled(
            Rect::from_min_size(egui::pos2(cxx - 1.6, y0), Vec2::new(3.2, h)),
            1.6,
            Color32::WHITE,
        );
    };
    bar(cx, 15.0, top + 8.0);
    bar(cx - 4.8, 10.0, top + 10.5);
    bar(cx + 4.8, 10.0, top + 10.5);
    p.rect_filled(
        Rect::from_center_size(egui::pos2(cx, rect.bottom() - 9.0), Vec2::new(17.0, 3.2)),
        1.6,
        Color32::WHITE,
    );
}


impl App {
    // ── 更新检查 ──────────────────────────────────────────────────

    /// 首帧发起后台检查，之后每帧收一次结果。**不阻塞**。
    fn poll_update(&mut self, ctx: &egui::Context) {
        // ① 首帧：挂上检查线程
        if !self.update_checked {
            self.update_checked = true;
            self.update_rx = Some(updater::check_async(config::APP_VERSION.to_string()));
        }

        // ② 收检查结果
        if self.update_rx.is_some() {
            let got = self.update_rx.as_ref().map(|rx| rx.try_recv());
            match got {
                Some(Ok(Ok(Some(info)))) => {
                    self.update_rx = None;
                    self.push_log(format!(
                        "[更新] 发现新版本 v{}（当前 v{}）",
                        info.version,
                        config::APP_VERSION
                    ));
                    self.update_found = Some(info);
                    self.show_update_dialog = true;
                }
                Some(Ok(Ok(None))) => {
                    self.update_rx = None;
                    self.push_log("[更新] 已是最新版本");
                }
                Some(Ok(Err(e))) => {
                    self.update_rx = None;
                    // 网络不通是常态（国内连不上），不打进日志免得误导用户以为出错
                    self.update_msg = Some(format!("检查更新失败：{e}"));
                }
                Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    self.update_rx = None;
                }
                _ => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(500));
                }
            }
        }

        // ③ 收下载结果
        if self.download_rx.is_some() {
            let got = self.download_rx.as_ref().map(|rx| rx.try_recv());
            match got {
                Some(Ok(Ok(path))) => {
                    self.download_rx = None;
                    *self.download_progress.lock().unwrap() = (0, None);
                    self.push_log(format!("[更新] 下载完成：{}", path.display()));
                    self.downloaded = Some(path);
                    // 下完直接把弹窗拉回前台，让用户点「立即安装」
                    self.show_update_dialog = true;
                    ctx.request_repaint();
                }
                Some(Ok(Err(e))) => {
                    self.download_rx = None;
                    *self.download_progress.lock().unwrap() = (0, None);
                    self.push_log(format!("[更新] 下载失败：{e}"));
                    self.update_msg = Some(format!(
                        "下载失败：{e}\n可以点下面的网盘链接手动下载。"
                    ));
                }
                Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    self.download_rx = None;
                }
                _ => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(200));
                }
            }
        }
    }

    /// 下载是否正在进行
    fn downloading(&self) -> bool {
        self.download_rx.is_some()
    }

    /// 启动后台下载
    fn start_download(&mut self) {
        let Some(info) = self.update_found.clone() else {
            return;
        };
        if self.downloading() {
            return;
        }
        if info.download_url.trim().is_empty() {
            self.update_msg = Some("这一个版本没有配置直链下载，请用下面的网盘链接。".into());
            return;
        }
        self.update_msg = None;
        self.downloaded = None;
        self.download_cancel.store(false, Ordering::SeqCst);
        *self.download_progress.lock().unwrap() = (0, None);

        let dest = updater::download_dir(&self.data_dir)
            .join(format!("ASMR-Translator-Setup_v{}.exe", info.version));
        let prog = self.download_progress.clone();
        let cancel = self.download_cancel.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.download_rx = Some(rx);

        let url = info.download_url.clone();
        std::thread::spawn(move || {
            let p2 = prog.clone();
            let res = updater::download_blocking(
                &url,
                &dest,
                move |cur, total| {
                    *p2.lock().unwrap() = (cur, total);
                },
                &|| cancel.load(Ordering::SeqCst),
            );
            let _ = tx.send(res.map(|_| dest));
        });
        self.push_log("[更新] 开始下载安装包…");
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.pump(&ctx);
        self.poll_update(&ctx);

        // 后台「自带字幕」扫描有结果了吗
        //
        //  先 take 出结果再改 self（`if let Some(rx) = &self.precheck_rx` 会和
        //  后面 self.skip_note.insert 的可变借用打架 —— 和 vram_refresh_rx 同一个坑）。
        if self.precheck_rx.is_some() {
            let got = self.precheck_rx.as_ref().map(|rx| rx.try_recv());
            match got {
                Some(Ok(hits)) => {
                    self.precheck_rx = None;
                    self.prechecking = false;
                    self.prechecked = true;
                    self.skip_note.clear();
                    if hits.is_empty() {
                        self.push_log("[预检] 没有发现自带字幕的文件");
                    } else {
                        let n = hits.len();
                        for (p, reason) in hits {
                            self.skip_note.insert(p, reason);
                        }
                        self.push_log(format!(
                            "[预检] {n} 个文件自带字幕，下面开始跳过的部分是它们"
                        ));
                    }
                    // 结果已就绪、start() 里也不会阻塞 → 直接开跑。
                    //
                    // 原来这里是置 `pending_start` 等下一帧，但那样中间隔了一帧，
                    // 用户这一帧里改了队列 / 点了停止的话，start() 会拿着旧的
                    // `skip_note` 和 `already[]` 去分派 —— 索引和路径都可能对不上。
                    ctx.request_repaint();
                    self.start();
                }
                Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    self.precheck_rx = None;
                    self.prechecking = false;
                    // 线程崩了：置位以免每次点开始都重试同一个会崩的扫描
                    self.prechecked = true;
                    self.push_log("[预检] 扫描线程意外结束，跳过检测直接开跑");
                    self.start();
                }
                _ => {
                    // 还在扫：保证界面继续重绘，结果一到就能被取走
                    ctx.request_repaint_after(std::time::Duration::from_millis(200));
                }
            }
        }

        // 「只能用 CPU」弹窗确认过了 → 现在才真正开跑
        // （推迟到帧开头，避免在绘制途中改一大批状态并启动后台线程）
        if self.pending_start {
            self.pending_start = false;
            self.start();
        }

        // 首帧按显示器尺寸校准窗口大小。
        //
        // 口径与 WinUI 版 preferred_window_size() 完全一致：
        //   基准 1280x1000【逻辑点】—— 与 WinUI 版 preferred_window_size() 一致
        //   （宽 min(1280, 屏宽*0.95) 且不低于 900；高 min(1000, 屏高*0.95) 且不低于 600）。
        //
        // ★ 不写成"屏幕的百分之几" —— 那样 4K 屏会开出一个占满屏幕的巨窗，
        //   而 1080P 上又变小，观感反而不一致。逻辑点基准在任意 DPI 下观感相同。
        if !self.sized_to_screen {
            if let Some(m) = ctx.input(|i| i.viewport().monitor_size) {
                self.sized_to_screen = true;
                let w = 1280.0_f32.min(m.x * 0.95).max(900.0);
                let h = 1000.0_f32.min(m.y * 0.95).max(600.0);
                if (w - 1280.0).abs() > 1.0 || (h - 1000.0).abs() > 1.0 {
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(w, h)));
                    self.push_log(format!(
                        "[窗口] 屏幕装不下默认尺寸（{:.0}x{:.0}），窗口收缩到 {:.0}x{:.0}",
                        m.x, m.y, w, h
                    ));
                }
            }
        }

        // ── 拖放进来的文件 ──
        // egui 0.37 起 DroppedFile 是个 trait（取路径用 .path() 而不是字段），
        // 而且是 Arc<dyn DroppedFile> —— 事件只在落下的那一帧出现。
        //
        // 下面这行「悬停」诊断是给排障用的：Windows 的 OLE 拖放分两步
        // （DragEnter 悬停 → Drop 落下），如果拖进来毫无反应，先看日志里
        // 有没有「鼠标正带着 N 个文件悬停」——
        // 有 → 事件到了窗口，问题在本程序的落点处理；
        // 没有 → 拖放事件根本没进窗口（系统/权限/窗口句柄层面的问题）。
        let hover_n = ctx.input(|i| i.raw.hovered_files.len());
        if hover_n > 0 && !self.drop_hover_seen {
            self.drop_hover_seen = true;
            self.push_log(format!("[拖放] 鼠标正带着 {hover_n} 个文件悬停在窗口上"));
        } else if hover_n == 0 && self.drop_hover_seen {
            self.drop_hover_seen = false;
        }
        let dropped: Vec<String> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path().to_str().map(String::from))
                .collect()
        });
        if !dropped.is_empty() {
            self.push_log(format!("[拖放] 收到 {} 个路径", dropped.len()));
            if self.running {
                // worker 拿的是「下标 + 路径」快照，运行中改列表会让它认错行；
                // 所以处理期间不收新文件（停下再加）。
                self.push_log("[拖入] 正在处理中，等这一批跑完或先点「停止」再加文件");
            } else {
                self.add_paths(&dropped); // 一次写盘，而不是每个文件写一遍
            }
        }

        // ── 自绘标题栏（窗口已去掉系统装饰）──
        egui::Panel::top("titlebar")
            .show_separator_line(false)
            .frame(
                Frame::new()
                    .fill(BG)
                    .inner_margin(Margin {
                        left: 8,
                        right: 8,
                        top: 6,
                        bottom: 0,
                    }),
            )
            .show(ui, |ui| {
                let (bar, bar_resp) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 32.0), Sense::click_and_drag());
                if bar_resp.dragged() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
                }
                if bar_resp.double_clicked() {
                    let max = ui.ctx().input(|i| i.viewport().maximized.unwrap_or(false));
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::Maximized(!max));
                }

                // 左侧：小图标 + 标题
                let mut tx = bar.left() + 8.0;
                if let Some(tex) = &self.icon {
                    let ir = Rect::from_min_size(
                        egui::pos2(tx, bar.center().y - 8.0),
                        Vec2::splat(16.0),
                    );
                    ui.painter().image(
                        tex.id(),
                        ir,
                        Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                    tx += 22.0;
                }
                ui.painter().text(
                    egui::pos2(tx, bar.center().y),
                    Align2::LEFT_CENTER,
                    "ASMR 日语语音 → 中文翻译",
                    FontId::proportional(12.0),
                    INK3,
                );

                // 右侧：最小化 / 最大化 / 关闭
                let mk = |ui: &mut Ui, id: &str, kind: u8, x: f32| -> bool {
                    let r = Rect::from_min_size(
                        egui::pos2(x, bar.top() + 1.0),
                        Vec2::new(44.0, 30.0),
                    );
                    let resp = ui.interact(r, ui.id().with(id), Sense::click());
                    let p = ui.painter();
                    if resp.hovered() {
                        p.rect_filled(
                            r,
                            5.0,
                            if kind == 2 {
                                Color32::from_rgb(0xdd, 0x5a, 0x48)
                            } else {
                                Color32::from_rgb(0xe6, 0xe2, 0xd8)
                            },
                        );
                    }
                    let c = r.center();
                    let col = if resp.hovered() && kind == 2 {
                        Color32::WHITE
                    } else {
                        INK2
                    };
                    match kind {
                        0 => {
                            p.line_segment(
                                [c + Vec2::new(-5.0, 0.0), c + Vec2::new(5.0, 0.0)],
                                Stroke::new(1.2, col),
                            );
                        }
                        1 => {
                            p.rect_stroke(
                                Rect::from_center_size(c, Vec2::splat(9.0)),
                                1.0,
                                Stroke::new(1.2, col),
                                egui::StrokeKind::Inside,
                            );
                        }
                        _ => {
                            p.line_segment(
                                [c + Vec2::new(-4.5, -4.5), c + Vec2::new(4.5, 4.5)],
                                Stroke::new(1.3, col),
                            );
                            p.line_segment(
                                [c + Vec2::new(4.5, -4.5), c + Vec2::new(-4.5, 4.5)],
                                Stroke::new(1.3, col),
                            );
                        }
                    }
                    resp.clicked()
                };
                let mut bx = bar.right() - 44.0;
                let hit_close = mk(ui, "wm_close", 2, bx);
                bx -= 44.0;
                let hit_max = mk(ui, "wm_max", 1, bx);
                bx -= 44.0;
                let hit_min = mk(ui, "wm_min", 0, bx);
                if hit_min {
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                }
                if hit_max {
                    let max = ui.ctx().input(|i| i.viewport().maximized.unwrap_or(false));
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::Maximized(!max));
                }
                if hit_close {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

        // ── 底部动作条：固定在窗口底部，窗口再小也始终在 ──
        egui::Panel::bottom("actions")
            .show_separator_line(false)
            .frame(
                Frame::new()
                    .fill(BG)
                    .inner_margin(Margin {
                        left: 20,
                        right: 20,
                        top: 8,
                        bottom: 10,
                    }),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if self.prechecking {
                        // 正在扫自带字幕：按钮变成只读的进度提示，
                        // 免得用户连点几次、每次都排一个扫描线程
                        pill(ui, "检查字幕…", ACCENT_SOFT, CARD_LINE, INK3, false);
                    } else if self.running {
                        pill(ui, "处理中…", ACCENT_SOFT, CARD_LINE, INK3, false);
                    } else if pill(ui, "开始处理 →", ACCENT_SOFT, ACCENT, ACCENT_INK, true).clicked() {
                        self.start();
                    }
                    if pill(ui, "停止", CARD, CARD_LINE, INK2, self.running).clicked() {
                        self.stop();
                    }
                    // 图标自绘：雅黑没有 "▶" / "‖" 的字形，用字符会渲染成方框
                    let (t, ic) = if self.paused { ("继续", 1) } else { ("暂停", 2) };
                    if pill_icon(ui, t, ic, CARD, CARD_LINE, INK2, self.running).clicked() {
                        self.toggle_pause();
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(
                Frame::new()
                    .fill(BG)
                    .inner_margin(Margin {
                        left: 20,
                        right: 20,
                        top: 6,
                        bottom: 8,
                    }),
            )
            .show(ui, |ui| {
                // 外层滚动：窗口被拖到缩放不同的显示器上时逻辑高度会骤减
                // （副屏 100% → 主屏 150%，物理尺寸没变、逻辑高度少了三分之一），
                // 光靠把卡片压扁未必塞得下 —— 那时至少还能滚。
                //   view_h 必须在 ScrollArea【之前】取：进了滚动区之后
                //   available_height() 是无限的，卡片高度会算到天上去。
                let view_h = ui.available_height();
                let y0 = ui.cursor().top();
                egui::ScrollArea::vertical()
                    .id_salt("page")
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_height(view_h);
                    // ── 标题区：图标 + 标题 + 副标题 ──
                    ui.horizontal(|ui| {
                        match &self.icon {
                            Some(tex) => {
                                ui.add(
                                    egui::Image::new((tex.id(), Vec2::splat(40.0)))
                                        .corner_radius(10.0),
                                );
                            }
                            None => logo(ui),
                        }
                        ui.add_space(12.0);
                        ui.vertical(|ui| {
                            ui.spacing_mut().item_spacing.y = 2.0;
                            ui.label(
                                RichText::new(format!("ASMR 日语语音 → 中文翻译  v{}", config::APP_VERSION))
                                    .font(FontId::new(26.0, FontFamily::Name("bold".into())))
                                    .color(INK),
                            );
                            ui.label(RichText::new(self.subtitle_text()).size(12.0).color(INK2));
                        });
                    });
                    ui.add_space(12.0);

                    // ── 更新横幅：只在查到新版本时出现 ──
                    if let Some(info) = self.update_found.clone() {
                        ui.add_space(2.0);
                        Frame::new()
                            .fill(ACCENT_SOFT)
                            .corner_radius(10.0)
                            .stroke(Stroke::new(1.0, ACCENT))
                            .inner_margin(Margin { left: 14, right: 14, top: 10, bottom: 10 })
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(format!("发现新版本  v{}", info.version))
                                            .size(13.5)
                                            .color(ACCENT_INK)
                                            .family(FontFamily::Name("bold".into())),
                                    );
                                    ui.label(
                                        RichText::new(format!("（当前 v{}）", config::APP_VERSION))
                                            .size(12.0)
                                            .color(INK3),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if pill(ui, "忽略", CARD, CARD_LINE, INK3, true)
                                                .clicked()
                                            {
                                                self.update_found = None;
                                            }
                                            if pill(ui, "查看更新", ACCENT_DEEP, ACCENT_DEEP, Color32::WHITE, true)
                                                .clicked()
                                            {
                                                self.show_update_dialog = true;
                                            }
                                        },
                                    );
                                });
                            });
                        ui.add_space(8.0);
                    }

                    // ── 卡片 1：环境 ──
                    card(ui, "环境", |ui| {
                        // 窄窗口时两列按比例收，别把右列挤出窗口
                        let total_w = ui.available_width();
                        let left_w = (total_w * 0.42).clamp(300.0, 520.0);
                        let col_gap = (total_w * 0.055).clamp(20.0, 72.0);
                        ui.horizontal_top(|ui| {
                            // 左列：路径 + 两个勾选项（宽窗口时约 520，对应 WinUI 的 width(520)）
                            ui.vertical(|ui| {
                                ui.set_min_width(left_w);
                                ui.set_max_width(left_w);
                                ui.spacing_mut().item_spacing.y = 12.0;
                                field(ui, "工作目录", &self.work_dir);
                                // 输出目录那一行和 WinUI 版一致：
                                //   勾了「放视频旁边」—— 位置自动决定，不给改；
                                //   取消勾选      —— 显示真实目录 + 「更改…」让用户挑。
                                ui.vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 2.0;
                                    ui.label(RichText::new("输出目录").size(11.0).color(INK2));
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 8.0;
                                        let txt = if self.to_video {
                                            "跟随视频所在目录（自动）".to_string()
                                        } else {
                                            self.out_dir.clone()
                                        };
                                        ui.label(RichText::new(txt).size(14.0).color(INK));
                                        if !self.to_video && ghost_button(ui, "更改…").clicked() {
                                            if let Some(dir) = rfd::FileDialog::new()
                                                .set_title("选择输出目录")
                                                .pick_folder()
                                            {
                                                self.out_dir = dir.to_string_lossy().to_string();
                                                self.push_log(format!(
                                                    "[输出目录] 已改为 {}",
                                                    self.out_dir
                                                ));
                                            }
                                        }
                                    });
                                });
                                ui.vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 6.0;
                                    if wcheck(ui, self.to_video, "字幕直接放到视频旁边（播放器可自动加载）")
                                        .clicked()
                                    {
                                        self.to_video = !self.to_video;
                                    }
                                    if wcheck(ui, self.force_redo, "强制重新处理（删除已有字幕）").clicked()
                                    {
                                        self.force_redo = !self.force_redo;
                                    }
                                    // 输出字幕格式：ASS 是收尾时由 Rust 从 SRT 转的，
                                    // 样式写死（黑体 / 白字 / 黑描边 / 底部居中 / 字号 52）——
                                    // 要做可调样式就得配一整套实时预览，收益不值。
                                    ui.add_space(4.0);
                                    ui.label(RichText::new("输出字幕格式").size(11.0).color(INK2));
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 8.0;
                                        for (f, txt) in [
                                            (srt::SubFmt::Srt, "仅 SRT"),
                                            (srt::SubFmt::Ass, "仅 ASS"),
                                            (srt::SubFmt::Both, "两个都要"),
                                        ] {
                                            if chip(ui, txt, self.sub_fmt == f, true).clicked()
                                                && self.sub_fmt != f
                                            {
                                                self.sub_fmt = f;
                                                self.push_log(format!(
                                                    "[设置] 输出字幕格式：{}",
                                                    f.label()
                                                ));
                                            }
                                        }
                                    });
                                    ui.label(
                                        RichText::new(
                                            "ASS：黑体白字黑描边、底部居中；选「两个都要」会同时出 .srt 和 .ass",
                                        )
                                        .size(11.0)
                                        .color(INK3),
                                    );
                                    // 日文字幕只有兼容模式有原文可用：
                                    //   全量/精简走 Whisper 的 translate 任务（日语直接进、中文直接出），
                                    //   日文在模型内部就丢掉了 —— 想拿日文得再跑一次识别，
                                    //   那是整批 +50% 的代价，不做。
                                    let can_jpn = matches!(self.mode, RunMode::Compat);
                                    if wcheck_enabled(
                                        ui,
                                        self.jpn_sub && can_jpn,
                                        "生成日文字幕（仅兼容模式开启）",
                                        can_jpn,
                                    )
                                    .clicked()
                                        && can_jpn
                                    {
                                        self.jpn_sub = !self.jpn_sub;
                                        if !self.jpn_sub {
                                            // 主开关关掉，子选项跟着复位
                                            self.jpn_separate = false;
                                        }
                                    }
                                    // 勾了主开关才出现这一条：日文放哪儿
                                    if self.jpn_sub && can_jpn {
                                        ui.add_space(3.0);
                                        if wcheck(
                                            ui,
                                            self.jpn_separate,
                                            "日文单独成文件（否则中日两行对照）",
                                        )
                                        .clicked()
                                        {
                                            self.jpn_separate = !self.jpn_separate;
                                        }
                                    }
                                });
                            });
                            ui.add_space(col_gap);
                            // 右列：设备状态 + 处理模式胶囊 + 说明
                            ui.vertical(|ui| {
                                ui.spacing_mut().item_spacing.y = 12.0;
                                ui.vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 3.0;
                                    ui.label(RichText::new("设备状态").size(11.0).color(INK3));
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 6.0;
                                        dot_ok(ui);
                                        ui.label(RichText::new(self.device_line()).size(14.0).color(INK));
                                    });
                                });
                                ui.vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 8.0;
                                    ui.label(RichText::new("处理模式").size(11.0).color(INK3));
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 8.0;
                                        if chip(ui, "全量模式", self.mode == RunMode::Full, self.full_ok)
                                            .clicked()
                                            && self.mode != RunMode::Full
                                        {
                                            self.mode = RunMode::Full;
                                            self.push_log("[模式] 当前使用: 全量模式");
                                        }
                                        if chip(ui, "精简模式", self.mode == RunMode::Lite, self.lite_ok)
                                            .clicked()
                                            && self.mode != RunMode::Lite
                                        {
                                            self.mode = RunMode::Lite;
                                            self.push_log("[模式] 当前使用: 精简模式");
                                        }
                                        if chip(ui, "兼容模式", self.mode == RunMode::Compat, true).clicked()
                                            && self.mode != RunMode::Compat
                                        {
                                            self.mode = RunMode::Compat;
                                            self.push_log("[模式] 当前使用: 兼容模式");
                                        }
                                    });
                                    ui.label(
                                        RichText::new(
                                            "全量模式需 N 卡显存 ≥ 8 GB；精简模式 ≥ 6 GB；兼容模式用于低于 6 GB / 非 N 独立显卡 / 核显",
                                        )
                                        .size(11.0)
                                        .color(INK3),
                                    );
                                });
                            });
                        });
                    });

                    ui.add_space(12.0);

                    // 两行卡片吃掉剩余高度（随窗口高度变化），相当于 WinUI Grid 的星号行
                    let gap = 12.0;
                    let card_chrome = 53.0; // 标题行 25 + 卡片内边距 28
                    // 已用高度用光标位置量（滚动区里 available_height 是无限的，不能用）
                    // 用上一帧量到的用量算可用高度：同一帧里「环境卡高度 ← 卡片高度 ← 环境卡高度」
                    // 会互相影响，窗口拉伸时表现为整页闪烁。
                    let used_now = (ui.cursor().top() - y0).max(0.0);
                    let used_ref = if self.used_prev > 0.0 { self.used_prev } else { used_now };
                    let avail_now = (view_h - used_ref).max(0.0);
                    // ★ 两条底线，别再用「压扁」去迁就小窗口：
                    //   ① body_total 不低于 410 = 行2 最小 240 + 行3 最小 170；
                    //   ② 行2/行3 各自也设最小高度。
                    //   窗口实在装不下时，多出来的高度由外层 page ScrollArea 滚动兜底 ——
                    //   压到重叠（内容溢出到相邻卡片上）比出现滚动条难看得多。
                    //   末尾那 20.0 是给卡片下圆角留的余量：内容只要比可视区高几像素，
                    //   整张卡的底部（连同 12 点圆角）就会被滚动区裁掉。
                    let bt_target = (avail_now - gap * 2.0 - card_chrome * 2.0 - 60.0).max(260.0);
                    // 平滑跟随：窗口拖大拖小时高度连得上，不会一帧一个样
                    self.body_total_smooth = if self.body_total_smooth <= 0.0 {
                        bt_target
                    } else {
                        self.body_total_smooth + (bt_target - self.body_total_smooth) * 0.4
                    };
                    let body_total = self.body_total_smooth;
                    self.used_prev = used_now;
                    //   行3（日志 / 进度）内容比行2 更吃高度，所以它拿大头
                    let body2 = (body_total * 0.535).max(100.0);
                    let body3 = (body_total - body2).max(150.0);

                    // ── 卡片 2/3：文件列表 | 实时翻译预览 ──
                    ui.columns(2, |cols| {
                        card_badge(
                            &mut cols[0],
                            "文件列表",
                            Some(&self.files.len().to_string()),
                            |ui| {
                                // 正文最小高度：左右两卡等高，且跟着窗口高度变
                                ui.set_min_height(body2);
                                if self.files.is_empty() {
                                    ui.label(
                                        RichText::new("（列表为空，点下面的按钮或把文件拖进来）")
                                            .size(12.5)
                                            .color(INK2),
                                    );
                                    ui.add_space(6.0);
                                    // 空态也保持卡片高度，和右侧预览卡对齐
                                    inner_panel(ui, |ui| {
                                        ui.set_min_height((body2 - 138.0).max(24.0));
                                        ui.label(
                                            RichText::new("（尚未添加文件）")
                                                .size(12.0)
                                                .color(DONE_INK),
                                        );
                                    });
                                } else {
                                    inner_panel(ui, |ui| {
                                        ui.set_min_height((body2 - 138.0).max(24.0));
                                        ScrollArea::vertical()
                                            .id_salt("files")
                                            .max_height((body2 - 138.0).max(24.0))
                                            .auto_shrink([false, false])
                                            .show(ui, |ui| {
                                                // 处理中点行时只记标志，出循环再打日志
                                                // （循环里还借着 self.files）
                                                let mut blocked_click = false;
                                                for (i, f) in self.files.iter().enumerate() {
                                                    let is_current = f.state == "处理中";
                                                    let is_sel = self.selected.contains(&i);
                                                    let color = match f.state.as_str() {
                                                        "完成" => OK_GREEN,
                                                        "处理中" => ACCENT,
                                                        "失败" => WARN,
                                                        // 自带字幕被跳过：用最淡的一档，
                                                        // 一眼能看出「这行不用管」
                                                        "已有字幕" => INK3,
                                                        _ => INK3,
                                                    };
                                                    let name_color = match f.state.as_str() {
                                                        "完成" => DONE_INK,
                                                        "已有字幕" => DONE_INK,
                                                        "处理中" => INK,
                                                        "失败" => INK,
                                                        _ => INK2,
                                                    };
                                                    // 两种「被标记」的状态，优先级：正在处理 > 选中
                                                    // （正在处理给浅橙，选中给更中性的浅灰，一屏里不会看混）
                                                    let bg = if is_current {
                                                        ACCENT_SOFT
                                                    } else if is_sel {
                                                        SEL_BG
                                                    } else {
                                                        Color32::TRANSPARENT
                                                    };
                                                    // 整行是一个可点区域：背景由 Frame 铺（在文字下层），
                                                    // 点击交给 ui.interact —— 行内文字不参与命中，
                                                    // 所以点空白处也算点这一行。
                                                    let row = Frame::new()
                                                        .fill(bg)
                                                        .corner_radius(7.0)
                                                        .inner_margin(Margin {
                                                            left: 8,
                                                            right: 8,
                                                            top: 2,
                                                            bottom: 2,
                                                        })
                                                        .show(ui, |ui| {
                                                            ui.set_min_width(ui.available_width());
                                                            ui.horizontal(|ui| {
                                                                // 选中行左侧配一小段橙色竖条：
                                                                //   光靠淡底在这套浅色配色里不够醒目。
                                                                //   没选中时也占同样宽度，序号才不会左右跳。
                                                                ui.spacing_mut().item_spacing.x = 5.0;
                                                                let (r, _) = ui.allocate_exact_size(
                                                                    Vec2::new(3.0, 14.0),
                                                                    Sense::hover(),
                                                                );
                                                                if is_sel {
                                                                    ui.painter().rect_filled(r, 1.5, ACCENT);
                                                                }
                                                                ui.label(
                                                                    RichText::new(format!("{}", i + 1))
                                                                        .size(11.5)
                                                                        .color(INK3),
                                                                );
                                                                ui.add_space(6.0);
                                                                let short = if f.name.chars().count() > 36 {
                                                                    let mut s: String =
                                                                        f.name.chars().take(36).collect();
                                                                    s.push('…');
                                                                    s
                                                                } else {
                                                                    f.name.clone()
                                                                };
                                                                let name_lab = ui.add(
                                                                    Label::new(
                                                                        RichText::new(short)
                                                                            .size(13.0)
                                                                            .color(name_color),
                                                                    )
                                                                    .truncate(),
                                                                );
                                                                // 被跳过的行：鼠标停在文件名上看具体原因
                                                                // （内封字幕什么格式 / 画面里烧死的）
                                                                if let Some(note) = &f.note {
                                                                    name_lab.on_hover_text(note);
                                                                }
                                                                ui.with_layout(
                                                                    Layout::right_to_left(Align::Center),
                                                                    |ui| {
                                                                        // 勾也自绘：雅黑没有 "✓"(U+2713) 的字形，
                                                                        // 靠字符就得赌机器上有没有符号字体
                                                                        if f.state == "完成" {
                                                                            check_mark(ui, color);
                                                                        }
                                                                        ui.label(
                                                                            RichText::new(&f.state)
                                                                                .size(11.5)
                                                                                .color(color),
                                                                        );
                                                                    },
                                                                );
                                                            });
                                                        });
                                                    // 整行是一个可点区域：背景由 Frame 铺（在文字下层），
                                                    // 点击交给 ui.interact —— 行内文字不参与命中，
                                                    // 所以点空白处也算点这一行。
                                                    // 运行中也注册，点了给一句日志说明为什么不理它
                                                    // （直接不注册的话，用户看到的是「点了没反应」）。
                                                    let resp = ui.interact(
                                                        row.response.rect,
                                                        ui.id().with(("file_row", i)),
                                                        Sense::click(),
                                                    );
                                                    if resp.clicked() {
                                                        if self.running {
                                                            // push_log 要 &mut self，而循环里还借着
                                                            // self.files 的不可变借用 —— 先记下来，
                                                            // 等循环结束再打日志。
                                                            blocked_click = true;
                                                        } else if is_sel {
                                                            self.selected.remove(&i);
                                                        } else {
                                                            self.selected.insert(i);
                                                        }
                                                    }
                                                }
                                                if blocked_click {
                                                    self.push_log(
                                                        "[列表] 正在处理中，等这一批跑完再改选中",
                                                    );
                                                }
                                            });
                                    });
                                }
                                // 四个按钮【始终】放在列表下方 —— WinUI 版也是这样。
                                //   以前只在「列表为空」时才画，结果跑完一批想再加文件就没入口了。
                                ui.add_space(8.0);
                                ui.horizontal(|ui| {
                                    let editable = !self.running;
                                    if ghost_button_enabled(ui, "添加文件", editable).clicked() {
                                        let mut dlg = rfd::FileDialog::new()
                                            .set_title("选择要处理的视频 / 音频");
                                        if !self.work_dir.is_empty() {
                                            dlg = dlg.set_directory(&self.work_dir);
                                        }
                                        if let Some(picked) = dlg
                                            .add_filter(
                                                "音视频",
                                                &[
                                                    "mp4", "mkv", "avi", "mov", "wmv", "flv",
                                                    "webm", "m4v", "ts", "mpg", "mpeg", "flac",
                                                    "wav", "mp3", "m4a", "aac", "ogg", "opus",
                                                    "wma",
                                                ],
                                            )
                                            .pick_files()
                                        {
                                            let paths: Vec<String> = picked
                                                .iter()
                                                .map(|p| p.to_string_lossy().to_string())
                                                .collect();
                                            self.add_paths(&paths);
                                        }
                                    }
                                    if ghost_button_enabled(ui, "添加文件夹", editable).clicked() {
                                        let mut dlg = rfd::FileDialog::new()
                                            .set_title("选择包含视频 / 音频的文件夹");
                                        if !self.work_dir.is_empty() {
                                            dlg = dlg.set_directory(&self.work_dir);
                                        }
                                        if let Some(dir) = dlg.pick_folder() {
                                            let s = dir.to_string_lossy().to_string();
                                            self.add_path(&s);
                                        }
                                    }
                                    if ghost_button_enabled(ui, "移除选中", editable).clicked() {
                                        if self.selected.is_empty() {
                                            self.push_log("[列表] 先点一下要移除的文件（再点一次取消选中）");
                                        } else {
                                            let sel = self.selected.clone();
                                            let n = sel.len();
                                            let mut kept = Vec::new();
                                            for (i, f) in std::mem::take(&mut self.files)
                                                .into_iter()
                                                .enumerate()
                                            {
                                                if !sel.contains(&i) {
                                                    kept.push(f);
                                                }
                                            }
                                            self.files = kept;
                                            self.selected.clear();
                                            self.preview.clear();
                                            // 队列变了 → 预检结论作废
                                            self.prechecked = false;
                                            self.skip_note.clear();
                                            self.save_queue();
                                            self.push_log(format!("[列表] 已移除 {n} 个选中文件"));
                                        }
                                    }
                                    if ghost_button_enabled(ui, "清空列表", editable).clicked() {
                                        self.files.clear();
                                        self.selected.clear();
                                        self.preview.clear();
                                        self.prechecked = false;
                                        self.skip_note.clear();
                                        // 下标一起归零：否则 save_queue 会把旧下标写进
                                        // 队列文件，下次启动 load_queue 直接判「跑完了」。
                                        self.cur_idx = 0;
                                        self.save_queue();
                                    }
                                });
                            },
                        );

                        card(&mut cols[1], "实时翻译预览", |ui| {
                            ui.set_min_height(body2);
                            if self.preview.is_empty() {
                                inner_panel(ui, |ui| {
                                    let size = egui::vec2(
                                        ui.available_width(),
                                        (body2 - 28.0).max(24.0),
                                    );
                                    ui.allocate_ui_with_layout(
                                        size,
                                        Layout::centered_and_justified(egui::Direction::TopDown),
                                        |ui| {
                                            ui.label(
                                                RichText::new("（等待开始）")
                                                    .size(12.0)
                                                    .color(INK3),
                                            );
                                        },
                                    );
                                });
                            } else {
                                inner_panel(ui, |ui| {
                                    ui.set_min_height((body2 - 28.0).max(24.0));
                                    ScrollArea::vertical()
                                        .id_salt("preview")
                                        .max_height((body2 - 28.0).max(24.0))
                                        .auto_shrink([false, false])
                                        .stick_to_bottom(true)
                                        .show(ui, |ui| {
                                            for s in &self.preview {
                                                ui.label(RichText::new(s).size(12.5).color(INK2));
                                                ui.add_space(2.0);
                                            }
                                        });
                                });
                            }
                        });
                    });

                    ui.add_space(12.0);

                    // ── 卡片 4/5：运行日志 | 进度 ──
                    ui.columns(2, |cols| {
                        card(&mut cols[0], "运行日志", |ui| {
                            ui.set_min_height(body3);
                            ui.horizontal(|ui| {
                                if ghost_button(ui, "导出完整日志").clicked() {
                                    let p = std::path::Path::new(&self.data_dir).join("_运行日志.txt");
                                    let body = self.logs.join("\r\n");
                                    let msg = match std::fs::write(&p, body) {
                                        Ok(_) => format!("[导出] 已写入 {}", p.display()),
                                        Err(e) => format!("[导出] 失败：{e}"),
                                    };
                                    self.push_log(msg);
                                }
                            });
                            ui.add_space(8.0);
                            inner_panel(ui, |ui| {
                                ScrollArea::vertical()
                                    .id_salt("logs")
                                    .max_height((body3 - 76.0).max(24.0))
                                    .auto_shrink([false, false])
                                    .stick_to_bottom(true)
                                    .show(ui, |ui| {
                                        for l in &self.logs {
                                            let c = if l.starts_with('[') { INK3 } else { INK2 };
                                            ui.add(
                                                Label::new(RichText::new(l).size(12.0).color(c))
                                                    .truncate(),
                                            );
                                        }
                                    });
                            });
                        });

                        card(&mut cols[1], "进度", |ui| {
                            ui.set_min_height(body3);
                            ui.columns(2, |c| {
                                c[0].vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 2.0;
                                    ui.label(RichText::new("当前阶段").size(11.0).color(INK3));
                                    ui.label(RichText::new(self.stage_title()).size(13.5).color(INK));
                                });
                                c[1].vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 2.0;
                                    ui.label(
                                        RichText::new("当前文件预估剩余").size(11.0).color(INK3),
                                    );
                                    ui.label(RichText::new(self.eta_text()).size(13.5).color(INK));
                                });
                            });
                            ui.add_space(10.0);
                            ui.columns(2, |c| {
                                c[0].vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 2.0;
                                    ui.label(RichText::new("已用时间").size(11.0).color(INK3));
                                    ui.label(
                                        RichText::new(format!(
                                            "{} 分 {} 秒",
                                            self.elapsed / 60,
                                            self.elapsed % 60
                                        ))
                                        .size(13.5)
                                        .color(INK),
                                    );
                                });
                                c[1].vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 2.0;
                                    ui.label(RichText::new("总进度").size(11.0).color(INK3));
                                    ui.label(
                                        RichText::new(format!("{:.0}%", self.progress_value))
                                            .size(13.5)
                                            .color(INK),
                                    );
                                });
                            });
                            ui.add_space(10.0);
                            let w = ui.available_width();
                            progress_bar(ui, self.progress, w);
                            ui.add_space(10.0);
                            ui.label(RichText::new("当前动作").size(11.0).color(INK3));
                            if !self.current_action.is_empty() {
                                ui.add(
                                    Label::new(
                                        RichText::new(&self.current_action).size(12.0).color(INK3),
                                    )
                                    .truncate(),
                                );
                            } else if self.running {
                                ui.label(RichText::new("启动中…").size(12.0).color(INK3));
                            }
                        });
                    });

                    });
            });

        resize_hotspots(&ctx);

        // ── 拖拽悬停提示：整窗盖一层，告诉用户松手就加进来 ──
        let hovering_files = ctx.input(|i| !i.raw.hovered_files.is_empty());
        if hovering_files {
            let p = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("drop_hint"),
            ));
            let r = ctx.input(|i| i.viewport_rect());
            p.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(0xf0, 0xee, 0xe6, 216));
            p.rect_stroke(
                r.shrink(10.0),
                14.0,
                Stroke::new(2.0, ACCENT),
                egui::StrokeKind::Inside,
            );
            p.text(
                r.center(),
                Align2::CENTER_CENTER,
                "松开鼠标即可添加",
                FontId::proportional(20.0),
                ACCENT_INK,
            );
            p.text(
                egui::pos2(r.center().x, r.center().y + 26.0),
                Align2::CENTER_CENTER,
                "支持视频 / 音频文件，也可以直接拖文件夹进来",
                FontId::proportional(13.0),
                INK3,
            );
        }

        // ── 「恢复上次的队列？」弹窗（逻辑和 WinUI 版一致）──
        //
        // 启动时读到队列文件才会出现。两个选择：
        //   恢复 → 填回列表、从上次的位置继续（已经有字幕的不会重跑）
        //   清空 → 丢弃队列，从空列表重新开始
        // ── 「发现新版本」弹窗 ──
        //
        // 方案 C：先试程序内下载（体验最好），失败就退回「打开下载页 / 网盘镜像」。
        // 国内直连 gitee.io / github.com 的成功率不稳定，所以下载失败不是异常，
        // 而是预期路径之一 —— 文案里直接把网盘链接摆出来。
        if self.show_update_dialog {
            if let Some(info) = self.update_found.clone() {
                let mut do_download = false;
                let mut do_cancel_dl = false;
                let mut do_install = false;
                let mut do_later = false;
                let mut open_page = false;
                let mut open_mirror: Option<String> = None;

                let dl_active = self.downloading();
                let (done_b, total_b) = *self.download_progress.lock().unwrap();

                egui::Modal::new(egui::Id::new("update_modal"))
                    .backdrop_color(Color32::from_rgba_unmultiplied(0x1f, 0x1e, 0x1d, 96))
                    .frame(
                        Frame::new()
                            .fill(CARD)
                            .corner_radius(12.0)
                            .inner_margin(Margin::same(20))
                            .stroke(Stroke::new(1.0, CARD_LINE))
                            .shadow(Shadow {
                                offset: [0, 10],
                                blur: 30,
                                spread: 0,
                                color: Color32::from_rgba_unmultiplied(0x1f, 0x1e, 0x1d, 40),
                            }),
                    )
                    .show(&ctx, |ui| {
                        ui.set_width(440.0);
                        ui.label(
                            RichText::new(format!("发现新版本 v{}", info.version))
                                .size(18.0)
                                .color(INK)
                                .family(FontFamily::Name("bold".into())),
                        );
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new(format!(
                                "当前版本 v{}　→　最新版本 v{}",
                                config::APP_VERSION, info.version
                            ))
                            .size(12.5)
                            .color(INK3),
                        );
                        ui.add_space(10.0);

                        if !info.notes.trim().is_empty() {
                            egui::ScrollArea::vertical()
                                .id_salt("upd_notes")
                                .max_height(140.0)
                                .show(ui, |ui| {
                                    ui.label(RichText::new(&info.notes).size(12.5).color(INK2));
                                });
                            ui.add_space(10.0);
                        }

                        // 下载进度
                        if dl_active {
                            let pct = match total_b {
                                Some(t) if t > 0 => (done_b as f32 / t as f32).clamp(0.0, 1.0),
                                _ => 0.0,
                            };
                            let txt = match total_b {
                                Some(t) if t > 0 => format!(
                                    "正在下载…  {:.1} / {:.1} MB",
                                    done_b as f64 / 1048576.0,
                                    t as f64 / 1048576.0
                                ),
                                _ => format!("正在下载…  {:.1} MB", done_b as f64 / 1048576.0),
                            };
                            ui.label(RichText::new(txt).size(12.0).color(INK3));
                            ui.add_space(4.0);
                            ui.add(
                                egui::ProgressBar::new(pct)
                                    .desired_width(400.0)
                                    .fill(ACCENT),
                            );
                            ui.add_space(10.0);
                        }

                        // 提示（失败原因等）
                        if let Some(msg) = &self.update_msg {
                            ui.label(RichText::new(msg).size(12.0).color(WARN));
                            ui.add_space(8.0);
                        }

                        // 网盘镜像（按需显示：下载失败或没有直链时才突出）
                        if !info.mirrors.is_empty() {
                            ui.label(
                                RichText::new("国内下载慢的话，可以用网盘：")
                                    .size(12.0)
                                    .color(INK3),
                            );
                            ui.add_space(4.0);
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing.x = 8.0;
                                for (name, url) in info.mirrors.clone() {
                                    if pill(ui, &name, CARD, CARD_LINE, ACCENT_INK, !dl_active)
                                        .clicked()
                                    {
                                        open_mirror = Some(url);
                                    }
                                }
                            });
                            ui.add_space(10.0);
                        }

                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 8.0;
                            if self.downloaded.is_some() {
                                if pill(ui, "立即安装", ACCENT_DEEP, ACCENT_DEEP, Color32::WHITE, true)
                                    .clicked()
                                {
                                    do_install = true;
                                }
                            } else if dl_active {
                                if pill(ui, "取消下载", CARD, CARD_LINE, WARN, true).clicked() {
                                    do_cancel_dl = true;
                                }
                            } else if !info.download_url.trim().is_empty() {
                                if pill(ui, "立即更新", ACCENT_DEEP, ACCENT_DEEP, Color32::WHITE, true)
                                    .clicked()
                                {
                                    do_download = true;
                                }
                            }
                            if pill(ui, "打开下载页", CARD, CARD_LINE, INK2, !dl_active).clicked() {
                                open_page = true;
                            }
                            if pill(ui, "稍后", CARD, CARD_LINE, INK3, !dl_active).clicked() {
                                do_later = true;
                            }
                        });
                    });

                if do_cancel_dl {
                    self.download_cancel.store(true, Ordering::SeqCst);
                    self.push_log("[更新] 已取消下载");
                }
                if do_download {
                    self.start_download();
                }
                if do_install {
                    if let Some(p) = self.downloaded.clone() {
                        match updater::run_installer(&p) {
                            Ok(()) => {
                                self.push_log("[更新] 安装程序已启动，本程序即将退出");
                                // 安装包会带 /CLOSEAPPLICATIONS 关掉我们，这里主动退也行
                                std::process::exit(0);
                            }
                            Err(e) => {
                                self.push_log(format!("[更新] 启动安装程序失败：{e}"));
                                self.update_msg = Some(format!("{e}，请手动运行：{}", p.display()));
                            }
                        }
                    }
                }
                if open_page {
                    updater::open_url(updater::RELEASES_PAGE);
                }
                if let Some(u) = open_mirror {
                    updater::open_url(&u);
                }
                if do_later {
                    self.show_update_dialog = false;
                }
            } else {
                self.show_update_dialog = false;
            }
        }

        if self.show_restore_dialog {
            let n = self.pending_queue.len();
            let idx = self.pending_idx.min(n);
            // 用 egui 自带的 Modal：遮罩由它画在弹窗【下面】，还会挡住底下的输入。
            // 之前是手画 layer_painter 遮罩 + Area 卡片 —— 同一图层内的先后不保证，
            // 实测弹窗被自己的遮罩压暗了，而且遮罩不接收点击（能点到后面的按钮）。
            let mut do_restore = false;
            let mut do_clear = false;
            egui::Modal::new(egui::Id::new("restore_modal"))
                .backdrop_color(Color32::from_rgba_unmultiplied(0x1f, 0x1e, 0x1d, 96))
                .frame(
                    Frame::new()
                        .fill(CARD)
                        .corner_radius(12.0)
                        .inner_margin(Margin::same(20))
                        .stroke(Stroke::new(1.0, CARD_LINE))
                        .shadow(Shadow {
                            offset: [0, 10],
                            blur: 30,
                            spread: 0,
                            color: Color32::from_rgba_unmultiplied(0x1f, 0x1e, 0x1d, 40),
                        }),
                )
                .show(&ctx, |ui| {
                            ui.set_width(400.0);
                            ui.label(
                                RichText::new("恢复上次的队列？")
                                    .size(18.0)
                                    .color(INK)
                                    .family(FontFamily::Name("bold".into())),
                            );
                            ui.add_space(10.0);
                            ui.label(
                                RichText::new(format!(
                                    "检测到上次留下的队列（{n} 个文件，处理到第 {} 个）。",
                                    idx + 1
                                ))
                                .size(13.0)
                                .color(INK2),
                            );
                            ui.add_space(8.0);
                            ui.label(
                                RichText::new("· 恢复：填回列表，从上次的位置继续")
                                    .size(12.5)
                                    .color(INK3),
                            );
                            ui.label(
                                RichText::new("· 清空：丢弃队列，从空列表重新开始")
                                    .size(12.5)
                                    .color(INK3),
                            );
                            ui.add_space(2.0);
                            ui.label(
                                RichText::new("（已删除的源文件会自动跳过）")
                                    .size(12.0)
                                    .color(DONE_INK),
                            );
                            ui.add_space(16.0);
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 8.0;
                                if pill(ui, "恢复", ACCENT_DEEP, ACCENT_DEEP, Color32::WHITE, true)
                                    .clicked()
                                {
                                    do_restore = true;
                                }
                                if pill(ui, "清空", CARD, CARD_LINE, INK2, true).clicked() {
                                    do_clear = true;
                                }
                            });
                });
            if do_restore {
                self.restore_queue();
            }
            if do_clear {
                self.clear_queue_all();
            }
        }

        // ── 「只能跑在 CPU 上」确认框（照 app/src/ui.rs:1892-1901）──
        //
        // 为什么必须拦一次：速度差 30~100 倍 —— GPU 上 1 小时的视频约 7 分钟，
        // CPU 可能几小时到十几小时。不明确讲清量级，用户会以为「只是慢一点」
        // 就直接开跑，然后挂一整天。确认过一次就记住（cpu_ok），不再重复弹。
        if self.show_cpu_warning {
            let mut do_go = false;
            let mut do_cancel = false;
            egui::Modal::new(egui::Id::new("cpu_modal"))
                .backdrop_color(Color32::from_rgba_unmultiplied(0x1f, 0x1e, 0x1d, 96))
                .frame(
                    Frame::new()
                        .fill(CARD)
                        .corner_radius(12.0)
                        .inner_margin(Margin::same(20))
                        .stroke(Stroke::new(1.0, CARD_LINE))
                        .shadow(Shadow {
                            offset: [0, 10],
                            blur: 30,
                            spread: 0,
                            color: Color32::from_rgba_unmultiplied(0x1f, 0x1e, 0x1d, 40),
                        }),
                )
                .show(&ctx, |ui| {
                            ui.set_width(430.0);
                            ui.label(
                                RichText::new("只能用 CPU 处理")
                                    .size(18.0)
                                    .color(WARN)
                                    .family(FontFamily::Name("bold".into())),
                            );
                            ui.add_space(10.0);
                            ui.label(
                                RichText::new(
                                    "没检测到可用的 N 卡（或显存不足 6 GB），也没有 Vulkan 设备。",
                                )
                                .size(13.0)
                                .color(INK2),
                            );
                            ui.add_space(8.0);
                            ui.label(
                                RichText::new("人声分离和语音识别只能跑在 CPU 上，速度差 30~100 倍：")
                                    .size(12.5)
                                    .color(INK2),
                            );
                            ui.label(
                                RichText::new(
                                    "· 1 小时的视频：GPU 约 7 分钟，CPU 可能要几小时到十几小时",
                                )
                                .size(12.5)
                                .color(INK3),
                            );
                            ui.add_space(16.0);
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 8.0;
                                if pill(
                                    ui,
                                    "仍然继续",
                                    ACCENT_DEEP,
                                    ACCENT_DEEP,
                                    Color32::WHITE,
                                    true,
                                )
                                .clicked()
                                {
                                    do_go = true;
                                }
                                if pill(ui, "取消", CARD, CARD_LINE, INK2, true).clicked() {
                                    do_cancel = true;
                                }
                            });
                });
            if do_go {
                self.cpu_ok = true;
                self.show_cpu_warning = false;
                self.pending_start = true; // 下一帧开头再开跑（不在这里递归进 start）
            }
            if do_cancel {
                self.show_cpu_warning = false;
            }
        }
    }
}

/// 把一行日志追加到工作目录的 `_运行日志.txt`。
///
/// 失败就静默跳过 —— 日志写不进去（目录只读、文件被占用）绝不能影响主流程。
fn append_run_log(data_dir: &str, line: &str) {
    if data_dir.is_empty() {
        return;
    }
    let p = std::path::Path::new(data_dir).join("_运行日志.txt");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
        use std::io::Write;
        let _ = writeln!(f, "{line}");
    }
}

fn main() -> eframe::Result<()> {
    // ── 崩溃现场 ──
    // GUI 进程没有控制台，Rust panic 默认只往 stderr 写一行然后退出 ——
    // 用户看到的就是「窗口突然不见了」，事后什么都查不到。
    // 这里把 payload 和「文件:行:列」落到 exe 旁边的 _崩溃日志.txt。
    {
        let crash = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("_崩溃日志.txt");
        std::panic::set_hook(Box::new(move |info| {
            use std::io::Write;
            let payload = info.payload();
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "(非字符串 panic)".to_string()
            };
            let loc = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "未知位置".to_string());
            let thread = std::thread::current()
                .name()
                .unwrap_or("未命名")
                .to_string();
            let body = format!("panic[{thread}]: {msg}\n  位置: {loc}\n");
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&crash)
            {
                let _ = f.write_all(body.as_bytes());
            }
            eprintln!("{body}");
        }));
    }

    // 原型从 target\release 跑，工作目录不在发布目录 —— 显式指过去，
    // 否则 config::work_dir() 会解析成 exe 所在目录（找不到 env\python.exe）。
    if std::env::var_os("ASMR_WORK_DIR").is_none() {
        let bundled = std::path::Path::new("D:/ASMR翻译器/package/ASMR自动翻译工具");
        if bundled.join("env/python.exe").is_file() {
            std::env::set_var("ASMR_WORK_DIR", bundled);
        }
    }
    let mut viewport = egui::ViewportBuilder::default()
        .with_decorations(false)
        .with_inner_size([1280.0, 1000.0])
        .with_min_inner_size([900.0, 600.0])
        .with_title(format!("ASMR 日语语音 → 中文翻译  v{}", config::APP_VERSION));
    // 任务栏 / 标题栏图标：和 WinUI 版同一张 assets/icon.png
    if let Some((rgba, width, height)) = icon_rgba() {
        viewport = viewport.with_icon(egui::IconData {
            rgba,
            width,
            height,
        });
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "asmr-egui-proto",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
