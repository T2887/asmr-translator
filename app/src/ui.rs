// ui.rs —— 界面 + 四阶段流水线驱动
//
// 这一版把整条流水线接起来了：点「开始处理」会依次跑
// a_提取音频.py → b_分离人声.py → c_人声检测.py → d_语音识别.py，
// 每跑完一步界面就刷新一次。
//
// ⚠️ 这个库的约束（都是踩过的）：
//
//   1. children((...)) 元组最多 16 个元素（IntoViews 只实现到 arity 16）。
//      所以日志区用固定 12 个槽位，不够的行显示成「另有 N 行未显示」。
//
//   2. window_title 是 ViewContext 的方法，只能在 view() 里调。
//      update() 拿到的 ComponentContext 没有这个方法。
//
//   3. LocalSender 是 !Send，不能 move 进后台闭包。
//      所以后台只能返回一个消息 —— 这就是流水线要拆成四次 spawn 的原因。

use windows_reactor::*;

use crate::backend;
use crate::config;
use crate::pipeline::{self, StageResult};

/// 日志区显示的槽位数。
//
//  ★ 12 行的时候整个界面比 1400x900 高出约 66px，底部的「进度」卡片和
//    「开始处理 / 停止」按钮会被挤出窗口（要滚动才能看到）。
//    减到 8 行正好省下 80px，一屏就能放全 —— 用户要的是「打开就能看到
//    所有功能」，多显示 4 行日志不值得让它少一个关键按钮。
//    （每行约 20px，改这个数要重新核对总高度。）
//
//  2026-10-20：6 → 5。Claude 暖色版把卡片内边距从 12 加到 16、行距从
//    8 加到 12，日志卡矮了约 12px，第 6 行（最新一行！）被裁掉半截。
//    与其压字号不如少一行 —— 反正「导出完整日志」还在，完整内容不会丢。
const LOG_SLOTS: usize = 5;

/// 翻译预览区显示的槽位数。
//  ★ 和 LOG_SLOTS 同理，同步减到 8 保持两栏高度接近。
const PREVIEW_SLOTS: usize = 6;

/// 界面里能显示的最大行数（超过就只留最近的）
const LOG_KEEP: usize = 500;

#[derive(Debug, Clone)]
pub enum Msg {
    /// 点了「开始处理」
    Start,
    /// 0pre 准备完毕：Ok((就绪数, 日志行)) 或 Err(原因)
    Prepared(Result<(usize, Vec<String>), String>),
    /// 某个阶段跑完了
    StageDone(usize, i32, Vec<String>),
    /// 用户点了停止
    Stop,
    /// 点了「添加文件」
    AddFiles,
    /// 点了「添加文件夹」—— 选一个目录，递归扫描里面的音视频
    AddFolder,
    /// 点了「移除选中」
    RemoveSelected,
    /// 点了「清空列表」
    ClearList,
    /// 点了文件列表里的某一行（自绘列表，序号从 0 起）
    SelectFile(usize),
    /// 勾选框变了：字幕是否直接放到视频旁边
    SetCopyToVideo(bool),
    /// 「强制重新处理」勾选框变化：勾上就要先把已有字幕删掉
    SetForceRedo(bool),
    /// 字幕时间轴整理完成（第4步之后、收尾之前的小任务）
    Tidied(Vec<String>),
    /// 字幕收尾完成（返回日志行）
    Finalized(Vec<String>),
    /// 用户从资源管理器拖了东西进来
    Dropped(DroppedData),
    /// 设备检测完成，payload 是给界面直接显示的文本
    DeviceDetected(String, bool, f64, bool),
    /// 切换运行模式
    SetModeFull,
    SetModeLite,
    SetModeCompat,
    /// 暂停 / 继续当前阶段
    TogglePause,
    /// 选择输出目录（只在没勾选「放视频旁边」时有意义）
    BrowseOutputDir,
    /// 窗口尺寸变了。payload 是新的客户区【宽度】（DIP）
    WindowSized(f64, f64),
    /// 关闭「没有可处理的文件」警告框
    DismissNoFileWarning,
    /// 「恢复上次队列」弹窗：用户选了恢复
    RestoreQueue,
    /// 「恢复上次队列」弹窗：用户选了清空
    ClearQueue,
    /// 关掉「没有 NVIDIA 显卡」的警告框
    /// 用户同意用 CPU 继续跑
    AgreeCpu,
    /// 用户不同意，退出程序
    RefuseCpu,
    /// 把内存里的完整日志导出成文件。
    ///
    /// 界面上的日志区只有固定几个槽位（Reactor 的 children 是元组，
    /// 上限 16 个，没法动态增长），历史行会被顶掉而且【翻不上去】。
    /// 用户想知道"第2步到底花了多久"却看不到前面几行 —— 实测卡在这里。
    ExportLog,
    /// 每秒一次的界面心跳（由 main.rs 里的 Win32 定时器发来）。
    /// 只为刷新「已用时间」—— 加载模型时脚本几十秒不输出，
    /// 界面上有个数字在跳，用户才不会以为程序卡死了。
    Tick,
}

pub struct AppUi {
    running: bool,
    /// 当前执行到第几阶段（0 表示还没开始）
    stage: usize,
    /// 已经完成的阶段数，用来算总进度
    done_stages: usize,
    /// 待处理文件列表（完整路径）
    files: Vec<String>,
    logs: Vec<String>,
    /// 实时翻译预览：从第4步输出里抽出来的字幕行（已经格式化好）
    preview: Vec<String>,
    /// 当前正在处理第几个文件（0-based，索引到 self.files）。
    ///
    /// 【为什么要逐个文件跑】
    ///   原来是"按步骤批量"：25 个文件一起走第1步、再一起走第2步…
    ///   这样模型只加载一次（省时间），但【矩阵的中间文件会同时堆在磁盘上】：
    ///     每步产出 flac 700MB + wav 1.5GB，25 个就是 55 GB 起步
    ///   实测把 D 盘 200 GB 直接塞满，导致 13 次 "No space left on device"、
    ///   部分 wav 写成半截，最后第3步读到坏文件才崩。
    ///
    ///   改成"逐个文件走完四步"之后：
    ///     · 磁盘峰值 = 单个文件的中间产物 ≈ 2.7 GB（不再是 70 GB）
    ///     · 代价 = 每个文件都要重新加载三个模型，每个文件多花约 45 秒
    ///       （25 个文件多 20 分钟左右，相对总时长可以接受）
    current_idx: usize,
    /// 启动时从磁盘读出来的待恢复队列
    pending_queue: Vec<String>,
    /// 上次处理到第几个
    pending_idx: usize,
    /// 是否显示「恢复上次队列」的弹窗
    show_restore_dialog: bool,
    /// 「有队列待恢复」但弹窗还没打开。
    ///
    /// ★ 为什么需要这个中间状态
    ///   ContentDialog 在【首次渲染】时如果 is_open 就是 true，
    ///   reactor 会立刻把它打开又立刻关掉，on_closed 收到 Primary，
    ///   于是弹窗一闪而过、用户根本没看到就自动"恢复"了。
    ///   实测：日志里直接出现「[队列] 已恢复 3 个文件」，而界面没有任何弹窗。
    ///
    ///   解法：启动时这个标志置 true、show_restore_dialog 置 false，
    ///   等第一次 Tick（100ms 后，那时对话框已经挂载好）再打开。
    pending_restore_prompt: bool,
    /// 上一次已经收进预览区的字幕行。
    ///
    /// 预览是【逐行】追加的：心跳每次读 live_line，发现是新的一行就解析、
    /// 塞进 preview。没有这个记录的话，同一行会被反复追加几十次
    /// （心跳每秒十来次，而一行字幕要在 live_line 上停留好几秒）。
    last_preview_line: String,
    /// 当前阶段的【内部】进度（0.0~1.0），从脚本的 tqdm 进度条解析。
    ///
    /// 为什么不用"数输出目录文件数"那一套：那只在【多文件】时有意义。
    /// 用户常见的场景是【一个长视频】，ratio 要么 0 要么 1，
    /// 全程都是 0 —— 等于阶段内进度完全没算上，ETA 于是退化成
    /// "用第1步的权重外推"，算出 1 小时 12 分这种荒唐值（实测踩过）。
    ///
    /// separator / faster-whisper 真正开始推理时会持续输出
    ///      12%|█▏  | 87/706 [01:23<09:53, 1.04it/s]
    /// `87/706` 就是当前文件的精确进度，比数文件细得多。
    stage_frac: f64,
    /// 累计收到的输出行数（不受 LOG_KEEP 截断影响）
    total_lines: usize,
    /// 停止标记。后台任务没法被强杀（子进程句柄在闭包内部），
    /// 所以这里是"跑完当前阶段就不再继续"的软停止。
    stop_requested: bool,
    /// 列表里当前选中的项。ListView 只暴露单个 selected_index()，
    /// 所以这里也只跟踪一个（Py 版支持 Ctrl 多选，这一版暂不做）。
    selected_index: Option<usize>,
    /// 进程内已经跑完的阶段数，用来驱动进度条（和 done_stages 同义，
    /// 但这里单独存一份 f64 方便直接喂给 ProgressBar）
    progress_value: f64,
    /// 字幕是否直接复制到视频所在目录。
    /// 默认 true —— 这是 Python 版定下来的默认值：只有放到视频旁边，
    /// PotPlayer 这类播放器才会自动加载字幕。
    copy_to_video: bool,
    /// 勾上「强制重新处理」时，开跑前会把已有的 .chs.srt / 3asr\*.srt 删掉。
    ///
    /// 为什么需要它：v1.6.1 加了「按产物跳过已完成」的逻辑（has_subtitle），
    /// 好处是中断后重跑不会重复做；但如果用户【就是想重做一遍】，
    /// 那些旧字幕会让文件被静默跳过 —— 用户看到队列 3 个、实际只跑了 1 个，
    /// 会以为程序坏了。这个勾选框把决定权交回给用户。
    force_redo: bool,
    /// 设备状态显示文本。初值是「检测中…」，
    /// 后台问完 torch 之后由 Msg::DeviceDetected 换成真实结果。
    device_status: String,
    /// 本机有没有可用的 CUDA。detect_device() 检测完置位。
    ///
    /// 用途：没有 N 卡时会回退到 CPU，而 CPU 跑人声分离要慢 30~100 倍
    /// （实测 GPU 1 小时视频 7 分钟，CPU 可能要几小时到十几小时）。
    /// 光在状态栏写一句"会很慢"不够 —— 用户会以为只是慢一点，
    /// 结果挂了一整天。所以点「开始处理」时要弹窗确认。
    has_gpu: bool,
    /// 本机有没有可用的 Vulkan 设备（AMD/Intel 核显或独显）。
    /// 没有 CUDA 时用它判断能不能走兼容模式。
    has_vulkan: bool,
    /// 当前选择的运行模式（全量 / 精简 / 兼容）。
    mode: pipeline::RunMode,
    /// 全量模式是否满足条件（N 卡显存 ≥ 8 GB）。
    mode_full_ok: bool,
    /// 精简模式是否满足条件（N 卡显存 6~8 GB）。
    mode_lite_ok: bool,
    /// 用户是否手动暂停了当前阶段。
    paused: bool,
    /// 暂停开始的时刻（恢复时用来把开始时间往后推，计时才准确）。
    pause_started_at: Option<std::time::Instant>,
    /// 显存小于 8 GB 时进入精简模式：settings.py 会把 compute_type
    /// 换到 int8_float16（识别阶段显存从 +3975 MiB 降到 +1960 MiB）。
    /// 这里只负责把状态显示出来，真正的切换在 Python 侧做。
    lite_mode: bool,
    /// 显卡显存（GB）。0.0 表示没有 GPU 或检测失败。
    vram_gb: f64,
    /// 用户是否已经【同意】用 CPU 跑。
    ///
    /// 显存 < 6 GB 或没有 N 卡时，弹窗问一次；
    /// 同意后置 true，再点「开始处理」就直接放行，不再反复弹。
    cpu_ok: bool,
    /// 是否正在显示"没有 NVIDIA 显卡"的警告框
    show_cpu_warning: bool,
    /// 用户是否已经看过"没有 N 卡"的警告。
    ///
    /// ContentDialog 的 on_closed 回调【不区分点了哪个按钮】——
    /// 所以没法靠回调参数判断"继续"还是"取消"。与其硬套两个按钮
    /// （点"取消"却照样开跑，更误导），不如只留一个「我知道了」，
    /// 用它记住"提醒过了"：第一次点开始 → 弹窗拦下；关掉之后
    /// 再点开始 → 放行。用户想要的就是这个语义。
    /// 输出目录：不勾选「字幕放视频旁边」时，字幕放这儿。用户可以改。
    /// 勾选时这个值不生效（字幕直接跟着视频走）。
    output_dir: String,
    /// 当前窗口客户区宽度（DIP）。
    /// 初值用 1400（就是启动时 client_size 设的那个），
    /// 之后由 Msg::WindowSized 更新。左右两栏的宽度按它算出来。
    window_width: f64,
    /// 客户区高度（DIP）。用来给主体 Grid 一个确定的高度 ——
    /// 外层的 StackPanel 不约束子元素高度，不给的话 Grid 的 STAR 行
    /// 等于无限高，滚动区会撑开把底部按钮顶出可视区。
    window_height: f64,
    /// 没添加文件就点「开始处理」时置 true，view() 会据此弹出警告框。
    /// 用状态而不是直接弹窗：ContentDialog 是 view 的一部分。
    show_no_file_warning: bool,
    /// 本次处理是什么时候开始的。用来算「已用时间」。
    started_at: Option<std::time::Instant>,
    /// 已用秒数。由 Msg::Tick 每秒刷新。
    elapsed_secs: u64,
    /// 已经跑完的每个阶段【实际】花了多少秒。
    ///
    /// 这是估算剩余时间的依据：用实测速度（秒 / 权重单位）外推，
    /// 而不是拿猜的权重硬算。跑完第1步之后估算就有底了，
    /// 随着阶段推进越来越准。
    stage_durations: Vec<u64>,
    /// 每个【文件】跑完四步的总耗时（秒）。
    ///
    /// 和 stage_durations 的区别：后者是每【步】一条（14 文件 × 4 步 = 56 条），
    /// 这个是每【文件】一条（14 条）。ETA 需要的是后者 —— 按文件累计，
    /// 才能算出"已完成的这些文件平均每秒语音要多少时间"。
    file_durations: Vec<f64>,
    /// 每个文件的【语音时长】（秒），从 VAD 的日志行
    /// `VAD: 72 段 / 201.3s（占 67.1%）` 解析出来。
    ///
    /// 为什么用它而不是音频总时长：分离和识别的耗时都与【语音时长】
    /// 成正比，与总时长无关。实测同一批文件里，按总时长算的变异系数
    /// 是 52%，按语音时长算就降到 1.7%（分离环节）。
    file_speech: Vec<f64>,
    /// 当前正在处理的文件的语音时长（VAD 跑完时才知道）
    cur_speech: f64,
    /// 当前阶段是什么时候开始的。
    stage_started_at: Option<std::time::Instant>,
    /// 上一步算出来的预估剩余秒数（已做平滑）。
    ///
    /// 为什么不直接把原始估算显示出来：早期估算很不稳 ——
    /// 第1步（提取音频）跑得飞快却只占 10% 权重，于是"已用时间"涨得比"加权进度"快，外推出来的剩余时间会【往上涨】。
    /// 一个会涨的倒计时没人会信。所以这里限制变化速率：
    /// 一次最多涨 10%、最多降 30%，看起来就稳当多了。
    eta_display: Option<u64>,
    /// 脚本当前正在干什么（它输出的最后一行）。
    ///
    /// 后台线程写、界面线程读，所以用 Arc<Mutex<..>>。
    /// 界面不直接监听它 —— 那样要跨线程推消息，而 LocalSender 是 !Send。
    /// 改成 Msg::Tick 每秒来读一次，简单可靠。
    live_line: std::sync::Arc<std::sync::Mutex<String>>,
    /// tqdm 进度的独立通道（0.0~1.0）。
    ///
    /// 为什么不和 live_line 共用：第4步识别期间 tqdm 每 3 秒一行、
    /// 字幕行每 0.3 秒一条，共用一个变量必然互相覆盖 ——
    /// 实测出现过两种症状：进度条不动（丢 tqdm）、预览空白（丢字幕）。
    /// 拆成两个通道后各走各的。
    live_frac: std::sync::Arc<std::sync::Mutex<f64>>,
    /// tqdm 自己算的剩余秒数（独立通道）。
    live_remain: std::sync::Arc<std::sync::Mutex<f64>>,
    /// 上一帧读到的 tqdm 剩余（0 表示没有）。compute_eta 优先用它。
    tqdm_remain: f64,
    /// tqdm ETA 的指数平滑值。
    ///
    /// 为什么不直接显示 tqdm 的原始 ETA：它每 3 秒按当时的平均速度重算，
    /// 速度一波动就跳（实测 72→45→90→30 这种）。平滑之后趋势保留、
    /// 抖动抹掉，而且随着处理推进样本变多，它自然越来越接近真值
    /// —— 这正是"越跑越准"该有的样子。
    tqdm_ema: f64,
    /// 上一次做 EMA 时的原始 tqdm ETA（用来判断是否是新样本）
    last_raw_remain: f64,
    /// 有没有拿到过 tqdm 的 ETA。用来区分「剩余 0」和「还没数据」。
    had_eta: bool,
    /// 最后一条字幕覆盖到的音频时刻（秒）。第4步用它算真实进度。
    live_sub_end: std::sync::Arc<std::sync::Mutex<f64>>,
    /// 这个文件的音频总长（秒），从第3步日志 `填回原长度: 6392.5s` 解析。
    audio_total: f64,
    /// 当前阶段的模型是否已加载完。
    ///
    /// pipeline 从脚本输出里抓标志行（见 pipeline::ready_marker）置位，
    /// 界面据此把「正在加载…模型（首次较慢）」换成「模型已加载」。
    /// 脚本在加载期间一言不发，光看日志分不出「还在加载」和「已经开工」，
    /// 用户会以为卡住了。
    model_ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// 「准备 0pre」那个后台任务的句柄。
    ///
    /// 存下来是为了让「停止」能取消它：准备阶段可能正在跨卷复制
    /// 十几 GB 的视频（要一分多钟），用户点了停止不该只能干等。
    /// 取消之后 copy 循环会立刻中断并删掉写了一半的文件。
    prepare_task: Option<ComponentTask>,
}

impl AppUi {
    /// 往日志里加一行，并维持总行数上限
    /// 真正往列表里塞一行。
    fn push_log_raw(&mut self, line: impl Into<String>) {
        let line = line.into();

        // ★ 扫这一行里有没有「语音时长」。
        //
        //   放在这里（而不是 Tick 的 100ms 轮询里）是刻意的：
        //   live_line 只保留脚本最新一行。VAD 打完
        //       VAD段已导出: xxx（2054 段 / 3699.8 秒）
        //   之后紧接着还会打「VAD 处理完毕」等几行，100ms 的轮询
        //   很可能错过那一行 —— 一旦错过，本次任务的 ETA 就全废了
        //   （cur_speech 为 0 → 成本模型失效）。
        //   这里一行到达就扫一次，不会漏。
        if let Some(v) = Self::parse_speech_line(&line) {
            self.cur_speech = v;
        }

        // 第3步分离脚本会打印 `填回原长度: 6392.5s → xxx.wav`，
        // 这个秒数就是【音频总长】，第4步用它当进度分母。
        // （字幕的时间戳是原始音频时间轴，因为分离结果被填回了原长度。）
        if line.contains("填回原长度") {
            if let Some(c) = config::audio_total_re().captures(&line) {
                if let Ok(v) = c[1].parse::<f64>() {
                    if v > 0.0 && v < 86400.0 {
                        self.audio_total = v;
                    }
                }
            }
        }

        // ★ 顺手解析 tqdm 进度条，更新 stage_frac。
        //
        //   原来这一步只在 Tick（100ms）里做：读 live_line 的最新一行，
        //   匹配到了就更新。问题是 live_line 会被覆盖 ——
        //   第3步分离期间，「[等待] 人声分离 xxx 已用 N 秒」每 3 秒一行，
        //   和 tqdm（每 0.45 秒一行）交替刷。轮询很容易全程只看到「等待」行，
        //   于是 stage_frac 一直是 0，进度条钉在 14% 不动，直到这一步
        //   跑完才跳到 50%。实测用户反馈"突然从 14 到了 50，不平滑"。
        //
        //   改成在 push_log_raw 里做：每行到达都过一遍，不会漏。
        if self.running && (3..=4).contains(&self.stage) {
            if let Some(c) = config::tqdm_re().captures(&line) {
                let done_n: f64 = c[1].parse().unwrap_or(0.0);
                let total_n: f64 = c[2].parse().unwrap_or(0.0);
                if total_n > 0.0 {
                    self.stage_frac = (done_n / total_n).clamp(0.0, 1.0);
                }
            }
        }

        // ★ 同时落一份到工作目录的 `_运行日志.txt`。
        //
        //   界面上的日志区只有固定几个槽位 —— Reactor 的 children 是
        //   元组，最多 16 个元素，没法动态增长，所以历史行会被顶掉，
        //   而且【翻不上去】。用户排查"某一步花了多久"时根本看不到
        //   前面几行（实测就卡在这里：想知道第2/3步用时，日志已经滚走了）。
        //
        //   写一份到文件之后，随时可以用记事本打开翻全部记录，
        //   也不怕界面滚动条缺失。
        Self::append_log_file(&line);

        self.logs.push(line);
        self.total_lines += 1;
        if self.logs.len() > LOG_KEEP {
            let drop_n = self.logs.len() - LOG_KEEP;
            self.logs.drain(0..drop_n);
        }
    }

    /// 从一行日志里提取「本文件的语音时长」。
    ///
    /// 匹配 `VAD段已导出: xxx.vad.json（2054 段 / 3699.8 秒）`。
    /// 做了合理性检查：必须是正数，且小于 24 小时（防止误匹配到别的数字）。
    fn parse_speech_line(line: &str) -> Option<f64> {
        if !line.contains("VAD段已导出") {
            return None;
        }
        let c = config::vad_speech_re().captures(line)?;
        let v: f64 = c.get(1)?.as_str().parse().ok()?;
        if v > 0.0 && v < 86400.0 {
            Some(v)
        } else {
            None
        }
    }

    /// 把一行日志追加到工作目录的 `_运行日志.txt`（失败就静默跳过）。
    fn append_log_file(line: &str) {
        use std::io::Write;
        let path = format!("{}\\{}", config::work_dir(), "_运行日志.txt");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{line}");
        }
    }

    /// 把内存里的完整日志导出一份（给「导出日志」按钮用）。
    fn export_log(&mut self) {
        let path = format!("{}\\{}", config::work_dir(), "_运行日志_导出.txt");
        let body = self.logs.join("\r\n");
        match std::fs::write(&path, body) {
            Ok(_) => {
                let n = self.logs.len();
                let msg = format!("[日志] 已导出 {n} 行到 {path}");
                self.push_log_raw(msg);
            }
            Err(e) => {
                let msg = format!("[日志] 导出失败: {e}");
                self.push_log_raw(msg);
            }
        }
    }

    /// 往日志里写一行。
    ///
    /// ★ 用户点了停止之后，这里会【丢弃】所有后续输出。
    ///
    ///   原因：停止是"立刻"的 —— Msg::Stop 马上写"任务已停止"，
    ///   但被 kill 掉的脚本还有输出在路上（"提取中: xxx.flac"、
    ///   "第1步退出码 = 1" 之类）。这些会【排在"已停止"后面】冒出来，
    ///   看起来像是停止之后又出了错，容易误导。
    ///   所以停止后除了明确要显示的那一句，其余一概不写。
    fn push_log(&mut self, line: impl Into<String>) {
        if self.stop_requested {
            return;
        }
        self.push_log_raw(line);
    }

    fn push_logs(&mut self, lines: Vec<String>) {
        for l in lines {
            self.push_log(l);
        }
    }

    /// 删掉一个文件对应的所有已有字幕（彻底删除，不进回收站）。
    ///
    /// 删两处：
    ///   ① 视频旁边 <名字>.chs.srt
    ///   ② 3asr\<名字>.srt（带 _(Vocals)_model_... 后缀的那个也得删）
    ///
    /// 返回删掉的文件数。
    fn delete_existing_subtitles(video_path: &str) -> usize {
        let p = std::path::Path::new(video_path);
        let Some(stem) = p.file_stem().map(|s| s.to_string_lossy().to_string()) else {
            return 0;
        };
        let mut n = 0;

        // ① 视频旁边
        if let Some(dir) = p.parent() {
            let f = dir.join(format!("{stem}.chs.srt"));
            if f.is_file() && std::fs::remove_file(&f).is_ok() {
                n += 1;
            }
        }

        // ② 3asr —— 可能有两个名字：原始的，和带 _(Vocals) 后缀的
        let asr = std::path::PathBuf::from(config::dir_asr());
        for name in [
            format!("{stem}.srt"),
            format!("{stem}_(Vocals)_model_bs_roformer_ep_317_sdr_12.srt"),
        ] {
            let f = asr.join(name);
            if f.is_file() && std::fs::remove_file(&f).is_ok() {
                n += 1;
            }
        }
        n
    }

    /// 判断一个视频是不是【已经生成过字幕】。
    ///
    /// ★ 为什么需要它
    ///   v1.6 的队列恢复只记了"处理到第几个"，但那个索引在程序被强杀时
    ///   不一定是最新的 —— 实测：DSH 更新要重启，翻译器跟着被干掉，
    ///   队列索引还停在 0，于是重开之后从头又跑一遍，把 15 个早就有字幕的
    ///   文件重新处理了一次（每个要 15~30 分钟）。
    ///
    ///   更可靠的做法是【直接看产物在不在】：
    ///     ① 视频旁边有 <名字>.chs.srt        → 已经做完了
    ///     ② 或者 3asr\<名字>.srt 存在        → 已经做完了
    ///   产物是事实，索引只是记录。记录会丢，事实不会。
    fn has_subtitle(video_path: &str) -> bool {
        let p = std::path::Path::new(video_path);
        let Some(stem) = p.file_stem().map(|s| s.to_string_lossy().to_string()) else {
            return false;
        };

        // ① 视频旁边
        if let Some(dir) = p.parent() {
            if dir.join(format!("{stem}.chs.srt")).is_file() {
                return true;
            }
        }

        // ② 3asr\ 目录
        let asr = std::path::PathBuf::from(config::dir_asr());
        if asr.join(format!("{stem}.srt")).is_file() {
            return true;
        }

        false
    }

    /// 队列持久化文件路径。
    fn queue_path() -> std::path::PathBuf {
        std::path::PathBuf::from(config::work_dir()).join("_file_queue.txt")
    }

    /// 把队列写盘。格式：第 1 行 current_idx，之后每行一个路径。
    ///
    /// 【为什么用裸文本而不是 JSON】
    ///   项目没引 serde，为这一个功能加依赖不值得。而且路径里有中文、
    ///   空格、反斜杠，JSON 转义反而更容易出岔子。路径本身不含换行，
    ///   按行切是安全的。
    ///
    /// 【为什么写失败要忽略】
    ///   队列记忆是便利功能。磁盘满、权限不足之类的问题不该让主流程报错。
    fn save_queue(&self) {
        let mut body = String::with_capacity(64);
        body.push_str(&self.current_idx.to_string());
        body.push('\n');
        for f in &self.files {
            body.push_str(f);
            body.push('\n');
        }
        let _ = std::fs::write(Self::queue_path(), body);
    }

    /// 读回上次的队列。返回 (文件列表, 处理到第几个)。
    ///
    /// 会过滤掉已经不存在的路径 —— 用户可能在两次运行之间删了源视频。
    /// 过滤后如果为空，返回 None（没什么可恢复的）。
    fn load_queue() -> Option<(Vec<String>, usize)> {
        let text = std::fs::read_to_string(Self::queue_path()).ok()?;
        // 去掉可能的 UTF-8 BOM。
        //
        // 程序自己写的文件不会有 BOM（std::fs::write 不写 BOM），
        // 但如果用户手改过、或者用过带 BOM 的编辑器保存，第一行会变成
        // "\u{FEFF}3"，parse::<usize>() 直接失败 → 整个队列读不出来、
        // 弹窗也不出现。这个坑实测踩过一次（测试脚本用 PowerShell 的
        // WriteAllText 写文件，带上了 BOM）。
        let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
        let mut lines = text.lines();
        let idx: usize = lines.next()?.trim().parse().ok()?;
        let files: Vec<String> = lines
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .filter(|l| std::path::Path::new(l).is_file())
            .map(|l| l.to_string())
            .collect();
        if files.is_empty() {
            return None;
        }
        let idx = idx.min(files.len());
        Some((files, idx))
    }

    /// 删掉队列文件（用户选了「清空」）
    fn clear_queue_file() {
        let _ = std::fs::remove_file(Self::queue_path());
    }

    /// 有文件进列表。做去重，返回 (新增数, 重复数)。
    fn add_files(&mut self, paths: Vec<String>) -> (usize, usize) {
        let mut added = 0usize;
        let mut dup = 0usize;
        for s in paths {
            if s.trim().is_empty() {
                continue;
            }
            if self.files.iter().any(|f| f == &s) {
                dup += 1;
            } else {
                self.files.push(s);
                added += 1;
            }
        }
        if added > 0 {
            self.save_queue();
        }
        (added, dup)
    }

    /// 清掉 0pre/1audio/2vad 里的过程文件，并把结果写进日志。
    ///
    /// 三个调用时机（和 Python 版一致）：
    ///   1. 开始处理前 —— 清掉上次异常退出留下的残骸
    ///   2. 任务结束（跑完 / 失败）—— 字幕已经生成，中间产物没用了
    ///   3. 用户点停止 —— 立刻清，不等
    ///
    /// ★ 是【永久删除】，不进回收站。这是明确要求的行为。
    /// ★ 3asr\ 绝对不碰。
    fn purge_process_files(&mut self, prefix: &str, include_pre: bool) {
        let (deleted, failed) = pipeline::purge_process_files(include_pre);
        if deleted == 0 && failed == 0 {
            return;
        }
        let mut line = format!(
            "{prefix}[清理] 已彻底删除 {deleted} 个过程文件\
             （0pre/1audio/2vad，未进回收站；3asr 里的字幕保留）"
        );
        if failed > 0 {
            line.push_str(&format!("，另有 {failed} 个被占用没能删掉"));
        }
        self.push_log(line);
    }

    /// 停止时专用的收尾：清掉过程文件，日志只留一行。
    ///
    /// 和上面那个的区别：这个【不报细节】。用户点了停止之后，
    /// 关心的是"停了没有"，不需要看删了几个文件、哪个被占用、
    /// 保留在哪 —— 那些是内部实现，写出来只是噪音。
    fn purge_on_stop(&mut self) {
        // 返回值不用管：这一句是固定的用户可见结果
        let _ = pipeline::purge_process_files(true);

        // 用 raw 版本绕过上面那条"停止后丢日志"的规则 ——
        // 这一句正是用户要看的，必须写进去。
        self.push_log_raw("任务已停止，过程文件已删除");
    }

    /// 把秒数写成人看的样子：95 → "1 分 35 秒"
    fn fmt_secs(s: u64) -> String {
        let m = s / 60;
        if m >= 60 {
            format!("{} 小时 {} 分", m / 60, m % 60)
        } else if m > 0 {
            format!("{m} 分 {:02} 秒", s % 60)
        } else {
            format!("{s} 秒")
        }
    }

    /// 算总进度百分比
    #[allow(dead_code)]
    fn total_pct(&self) -> f64 {
        (self.done_stages as f64 / pipeline::stages(self.mode).len() as f64) * 100.0
    }

    /// 当前阶段的名字
    fn stage_name(&self) -> String {
        if self.stage == 0 {
            String::new()
        } else if self.stage <= pipeline::stages(self.mode).len() {
            pipeline::stages(self.mode)[self.stage - 1].2.to_string()
        } else {
            "完成".to_string()
        }
    }

    /// 四个阶段各自占多少【时间】（不是各占 25%）。
    ///
    /// 进度条是按"阶段数"均分的（每步 25%），但四步的耗时天差地别：
    ///   · 提取音频 —— 只是 ffmpeg 转码，很快
    ///   · 分离人声 —— RoFormer 跑一遍，中等
    ///   · 人声检测 —— 轻量，很快
    ///   · 语音识别翻译 —— whisper 逐段翻译，最慢，通常占一半以上
    ///
    /// 所以算预估剩余时间时不能直接用进度条上的百分比，否则会在
    /// 第2步严重高估剩余、在第4步严重低估。这里用经验权重重新折算。
    ///
    /// ★ 这些是【实测】出来的，不是拍的。
    ///
    /// 拿一段 5 分钟的视频跑了完整四步，实际耗时：
    ///     第1步 提取音频      3 秒   →  3.1%
    ///     第2步 分离人声     23 秒   → 23.7%
    ///     第3步 人声检测     24 秒   → 24.7%
    ///     第4步 语音识别     47 秒   → 48.5%
    ///     合计               97 秒
    ///
    /// 原来的猜测值是 [0.10, 0.30, 0.05, 0.55]，其中第3步错得最厉害 ——
    /// 我以为 VAD 很轻量只给了 5%，实际它占了整整四分之一。
    ///
    /// 注：短视频里模型加载的固定开销占比偏大。视频越长，第2/4步
    /// 的相对占比会更高一些。这里按实测值取，长视频会略有偏差，
    /// 但比之前的猜测值准得多。
    // 各阶段的耗时占比 —— 【实测校准过的，别随手改】
//
// 这两组数据来自两次真实跑通（每 15 秒采样一次 GPU 和输出目录）：
//
//              1 小时视频   1.7 小时视频   采用值
//   第1步 提音频   0.040        0.099       0.07
//   第2步 分 离    0.750        0.496       0.62   ← 主要耗时都在这里
//   第3步 VAD     0.042        0.034       0.04
//   第4步 ASR     0.168        0.360       0.27
//
// ⚠ 之前那张表 [0.03, 0.24, 0.25, 0.48] 是拿【5 分钟视频】测的，
//   第2步只有 0.24 —— 而真实长视频里它占 50%~75%，低估了三倍。
//   所有 ETA 失真的根因就在这（曾经算出"剩余 3 小时 16 分"，
//   而实际全程十几分钟）。
//
//   教训：【不能用短视频的占比去套长视频】。模型加载、CUDA 初始化
//   这些固定开销在小样本里占比很大，视频一长就被稀释掉了。
// ⚠ v1.7 起顺序变了：第2步是 VAD，第3步才是分离。
//   而且分离现在【只处理 VAD 圈出的语音段】（实测只占音频长度的 15%~42%），
//   所以它的相对占比大幅下降 —— 不再是原来那个 0.62。
//   下面这组是按 v1.7 新流程的一次实测折算的（尚未在长视频上复校）：
//     VAD   约 3%（含模型加载）
//     分离  约 30%（含模型加载 + 拼接 + 填回）
//     识别  约 25%
//     其余  约 35%（模型重复加载、文件读写等固定开销，视频越长占比越小）
//   等跑过几个长视频之后再回来校准。
// 【v1.13 重新校准】用一次真实的 14 文件批量运行反推。
//
// 数据来源：D:\...\_运行日志.txt 第 191404~208419 行（14 个视频，全程 109 分 49 秒）
// 各步实测总耗时：
//   第1步 提取    223 s   →  4.1%
//   第2步 VAD     520 s   →  9.6%
//   第3步 分离   1925 s   → 35.7%
//   第4步 识别   2728 s   → 50.6%
//   ─────────────────────────────
//   四步合计     5396 s（另有 1193 s 是模型加载/拼接/填回等固定开销）
//
// 旧表 [0.07, 0.03, 0.55, 0.35] 的问题：
//   · 第2步 VAD 只有 0.03 —— 那是 v1.7【之前】的顺序（VAD 在分离之后、
//     而且跑在人声上，很快）。v1.7 把 VAD 提前到分离之前、改成跑原始音频后，
//     它变成平均 37 秒/文件，权重应该接近 0.10，低估了 3.2 倍。
//   · 第3步 分离给到 0.55 —— 但实测只占 35.7%（VAD 前置之后只分离 15%~42%
//     的音频，比原来整段分离快得多），高估 1.5 倍。
//   · 第4步 识别只给 0.35，实测 50.6%，低估 1.4 倍。
//
// 后果：进度到第3步快结束时，按 0.55 的权重算「当前阶段占一半多」，
// 剩余时间被压到接近 0，而实际还有第4步（占 50%）没跑。
#[allow(dead_code)] // v1.15 起 ETA 不再用权重表，但这份实测数据留作参考
    const STAGE_WEIGHTS: [f64; 4] = [0.04, 0.10, 0.36, 0.50];

    /// 按时间权重折算出来的真实完成度（0.0 ~ 1.0）。
    ///
    /// 进度条上的 progress_value 是均分的，这里把它"翻译"成时间意义上的完成度：
    ///   已完成阶段的权重之和 + 当前阶段的权重 × 本阶段内的完成比例
    #[allow(dead_code)]
    fn weighted_progress(&self) -> f64 {
        if !self.running {
            // 没在跑：跑完就是 1.0，没开始就是 0.0
            return if self.done_stages >= pipeline::stages(self.mode).len() {
                1.0
            } else {
                0.0
            };
        }

        let mut acc = 0.0f64;
        for w in Self::STAGE_WEIGHTS.iter().take(self.done_stages.min(4)) {
            acc += w;
        }

        if self.stage >= 1 && self.stage <= 4 {
            // 本阶段内部走到哪了。
            //
            // 原来用的是 (progress_value % 25.0) / 25.0 —— 但
            // progress_value 是【按阶段完成数】算的（每完成一步 +25），
            // 阶段进行中它压根不变，取余永远是 0，等于这段进度没算。
            // 改成用目录文件数推出来的 stage_ratio()。
            let in_stage = self.stage_ratio();
            acc += Self::STAGE_WEIGHTS[self.stage - 1] * in_stage.clamp(0.0, 1.0);
        }
        acc.clamp(0.0, 1.0)
    }

    /// 原始估算（未平滑）。返回 None 表示样本还不够。
    /// 真正显示给用户的那个值由 Msg::Tick 平滑后放进 eta_display。
    /// 数一个目录里的文件个数（目录不存在或读不了就返回 0）。
    #[allow(dead_code)]
    fn count_files(dir: &str) -> usize {
        std::fs::read_dir(dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().is_file())
                    .count()
            })
            .unwrap_or(0)
    }

    /// 当前阶段走到哪儿了（0.0 ~ 1.0）。
    ///
    /// 【为什么需要它】
    ///   原来算 ETA 只用"已完成阶段的耗时"去外推，完全没考虑当前阶段
    ///   跑了多久。实测出的 bug：第1步（ffmpeg 提音频）只用了 16 秒、
    ///   权重 0.03，于是算出"全程 533 秒"；而那时已经跑了 507 秒，
    ///   界面显示「预估剩余约 26 秒」—— 用户一看就知道是错的
    ///   （截图里已用 8 分 27 秒、剩余 26 秒，自相矛盾）。
    ///
    ///   现在补上阶段内进度：脚本是按文件逐个处理的，所以
    ///   【输入目录的文件数】和【输出目录已产出的文件数】之比
    ///   就是一个很准的进度指示，不需要脚本额外上报。
    #[allow(dead_code)]
    fn stage_ratio(&self) -> f64 {
        // ★ 这里是【两层进度】：
        //     文件级：已经处理完几个文件 / 总共几个
        //     文件内：当前文件的 tqdm 进度
        //   总进度 = (已完成文件数 + 当前文件内进度) / 总文件数
        //
        // 【为什么要两层】
        //   原来只返回 stage_frac（单个文件的 tqdm）。处理 14 个文件时，
        //   第 3 个文件跑到 100%，stage_frac 就变成 1.0 ——
        //   但后面还有 11 个文件没跑。ETA 于是以为"本阶段已经完成"，
        //   剩余时间直接被算到接近 0。实测就是"第10/14 个还在跑，
        //   界面显示预估剩余 约 0 秒"。
        let total_files = self.files.len().max(1) as f64;
        let files_done = self.current_idx.min(self.files.len()) as f64;
        let file_frac = if self.stage_frac > 0.0 {
            self.stage_frac.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let two_level = ((files_done + file_frac) / total_files).clamp(0.0, 1.0);

        // 有文件级信息时优先用它（更接近真实总进度）
        if self.running && !self.files.is_empty() && two_level > 0.0 {
            return two_level;
        }
        if self.stage_frac > 0.0 {
            return self.stage_frac.clamp(0.0, 1.0);
        }
        // 拿不到 tqdm（比如刚起步、或者某阶段不输出进度条）时，
        // 退回"数输出目录文件数"的粗粒度估计。
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

    /// 从【已完成文件】的实测数据反推「每秒语音要多少秒处理」。
    ///
    /// 返回 (分离成本率, 识别成本率)，没有可用样本时返回 None。
    ///
    /// ══════════════════════════════════════════════════════════
    /// 为什么必须在线校准，不能硬编码
    /// ══════════════════════════════════════════════════════════
    ///
    /// 两批真实数据的分离成本率：
    ///     第一批 14 个文件：0.0547 ~ 0.0592（平均 0.0572，批内变异 1.7%）
    ///     第二批  4 个文件：0.0677 ~ 0.0740（平均 0.0707，批内变异 4%）
    ///
    /// 批内很稳，两批之间差 24% —— 视频源的编码参数（码率、采样率、
    /// 声道）不同，送进分离模型的频谱特征就不同，耗时跟着变。
    /// 硬编码任何一个常数，都会在另一批数据上偏 20% 以上。
    ///
    /// 这个方法的本质是「移动平均」而不是「拟合模型」，所以不会过拟合：
    /// 它只回答"这台机器、这批文件、刚才那些的平均速度是多少"。
    fn cost_rates(&self) -> (Option<f64>, Option<f64>) {
        // stage_durations 每 4 条一个文件，顺序固定 [提取, VAD, 分离, 识别]
        let n_files = self.stage_durations.len() / 4;
        if n_files == 0 || self.file_speech.len() < n_files {
            return (None, None);
        }
        let mut sep_sum = 0.0f64;
        let mut asr_sum = 0.0f64;
        let mut sp_sum = 0.0f64;
        for i in 0..n_files {
            let sp = self.file_speech[i];
            if sp <= 0.0 {
                continue;
            }
            sep_sum += self.stage_durations[i * 4 + 2] as f64;
            asr_sum += self.stage_durations[i * 4 + 3] as f64;
            sp_sum += sp;
        }
        if sp_sum <= 0.0 {
            return (None, None);
        }
        (Some(sep_sum / sp_sum), Some(asr_sum / sp_sum))
    }

    /// 重算总进度（队列级，三层加权）。
    ///
    /// ══════════════════════════════════════════════════════════
    /// 为什么不能只在 StageDone 里算
    /// ══════════════════════════════════════════════════════════
    ///
    /// 原来的写法是：
    ///     self.progress_value = (stage / STAGES.len()) * 100
    /// 有两个毛病：
    ///   ① 它算的是【当前文件】的进度，不是队列的。4 个文件跑完 2 个时，
    ///      进度条仍然显示 75%（因为第2个文件跑到第4步了）。
    ///   ② 只在某一步【结束】时更新。第4步要跑 4~5 分钟，这期间
    ///      进度条一直钉在 75% 不动，用户以为卡住了。
    ///
    /// 现在的口径：
    ///     总进度 = (已完成文件数 + 当前文件加权进度) / 总文件数 × 100
    ///
    ///   当前文件加权进度 = Σ已完成的步骤权重 + 当前步骤权重 × tqdm进度
    ///   权重 = [0.04, 0.10, 0.36, 0.50]，来自 14 个文件的实测时间占比。
    ///
    /// 【只增不减】是刻意的：剔除文件、阶段切换时偶尔会算出一个略小的值，
    /// 让进度条往回退会让人怀疑程序出错。
    fn recompute_progress(&mut self) {
        if self.files.is_empty() || self.stage == 0 {
            return;
        }
        const W: [f64; 4] = [0.04, 0.10, 0.36, 0.50];

        let n_all = self.files.len() as f64;
        let n_done = self.current_idx as f64;

        // ── 当前文件的加权进度 ──
        let st = self.stage.min(4);
        let mut file_prog = 0.0f64;
        for i in 0..(st - 1) {
            file_prog += W[i];
        }

        // 当前这一步的内部进度。
        //
        // ★ 第1、2步【没有 tqdm 输出】——
        //   提取音频只有一行「提取中: xxx」，VAD 只有几行心跳，
        //   所以 stage_frac 恒为 0，进度条会死钉在上一档不动。
        //   用户反馈"没有平滑地到 14%"就是这个原因：
        //   第1步全程 0%，跑完跳 4%；第2步全程 4%，跑完跳 14%。
        //
        //   这四步都用「已用时间 / 预计耗时」做兜底伪进度。
        //   上限 0.95：宁可到不了头，也不要在真跑完之前先冲到 100%。
        //
        // ★ 原来这里有个 `stage_frac > 0.05` 的阈值，是第5次踩的坑：
        //   第4步识别的 tqdm 只有 49 次更新（0%、2%、4%、6%…），
        //   前 3 次全被 0.05 挡掉，进度条一动不动；
        //   第3步分离 173 次更新，也要等前 9 次（约 4 秒）才够 0.05。
        //   阈值本意是"过滤噪声"，但这些数值都是真实进度，没有噪声可言。
        //   现在只要 > 0 就直接用。
        let all_done: f64 = self.stage_durations.iter().map(|v| *v as f64).sum();
        let stage_used = (self.elapsed_secs as f64 - all_done).max(0.0);
        let sp = self.cur_speech.max(0.0);

        let frac_eff = if self.stage_frac > 0.0 {
            // 有 tqdm 数据（哪怕只有 0.0007 = 1/1376），直接用
            self.stage_frac.clamp(0.0, 1.0)
        } else {
            // 没有 tqdm：用时间比例冒充。
            // expected 留 20% 余量，避免真实耗时超出预测时伪进度先冲到顶。
            let expected = match st {
                1 => 25.0,
                2 => 45.0,
                // 分离 0.062 秒/秒语音、识别 0.11 秒/秒语音（19 个文件的实测中位）
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

    /// 预估【当前正在处理的这个文件】还剩多少秒。
    ///
    /// ══════════════════════════════════════════════════════════════
    /// 为什么只估当前文件，不估整个队列
    /// ══════════════════════════════════════════════════════════════
    ///
    /// 用一次真实的 14 文件批量运行验证过（28 小时音频、总耗时 109 分 49 秒）：
    ///
    ///     整个队列剩余（按已完成文件的平均值外推）     平均误差 31.0%
    ///     当前文件剩余（本函数）                    平均误差 16.4%
    ///
    /// 队列级误差大是因为【不知道后续文件的语音时长】。实测同一批文件里，
    /// 语音时长从 72 秒到 7879 秒，差 109 倍 —— 拿已完成文件的平均值去估
    /// 剩下的，30% 是必然的误差。要压到 13% 只能把 VAD 整批前置先跑一遍，
    /// 但那会让磁盘峰值从 2.7 GB 涨到 15 GB，不划算。
    ///
    /// 当前文件没有这个问题：
    ///   ① 语音时长已知 —— VAD 跑完时日志里有 `VAD: N 段 / X.Xs`
    ///   ② 分离耗时几乎完全由语音时长决定，实测误差只有 1.1%
    ///   ③ 第4步的 tqdm 进度条自带 ETA，拿它外推接近零误差
    ///
    /// 分阶段成本模型（14 个文件实测拟合）：
    ///     第1步 提取    15.9 秒    基本固定
    ///     第2步 VAD     37.1 秒    基本固定
    ///     第3步 分离    0.0568 × 语音秒数      误差 1.1%
    ///     第4步 识别    0.0810 × 语音秒数      误差 16%（仅作提示用）
    ///     其他固定      85.2 秒    模型加载、拼接、填回、字幕改名
    fn compute_eta(&self) -> Option<u64> {
        if !self.running {
            return None;
        }
        let elapsed = self.elapsed_secs as f64;
        // 刚开始的几秒数据不可信，先不给数
        if elapsed < 5.0 {
            return None;
        }

        // ── 本步真正的已用时间 ──
        //
        // ★ 这里必须用【实测】的步骤耗时，不能用硬编码常量。
        //
        //   实测教训（2026-10-18 那批 4 个文件）：
        //     文件1 前3步实测 42+48+218 = 308 秒
        //     而模型常量只有 15.9+37.1+167 = 220 秒
        //     差 88 秒 —— 直接导致算出来的「本步已用」偏小一半，
        //     剩余时间失真、看起来不随时间变化。
        //
        //   stage_durations 里存的就是每一步的真实耗时（按顺序 push），
        //   减去它们的总和，剩下的就是当前这一步已经跑的时间。
        let all_done: f64 = self.stage_durations.iter().map(|v| *v as f64).sum();
        let stage_used = (elapsed - all_done).max(1.0);

        // ── 成本率：优先用已完成文件的实测均值 ──
        let (sep_rate_opt, asr_rate_opt) = self.cost_rates();
        // 没有历史样本时退回全局经验值（第一批 14 个文件的拟合并值）
        // 默认值取自 19 个真实文件的中位数（第一批 14 + 第二批 4 + DVMM）。
        // 识别成本率的实测范围 0.033~0.246，跨度很大，但默认值取 0.11
        // 比原来的 0.081 更接近真实值 —— 原来偏小导致 asr_est 小于
        // 实际已用时间，left 被压成 0，界面卡在 TAIL=10 秒不动。
        let sep_rate = sep_rate_opt.unwrap_or(0.0620);
        let asr_rate = asr_rate_opt.unwrap_or(0.1100);

        let sp = self.cur_speech.max(0.0);

        // ★ 还没有语音时长（第1、2步），不做估算。
        //
        //   硬着头皮算会得到什么？sep_est = asr_est = 0，
        //   只剩 TAIL 常量 —— 界面显示"约 10 秒"并且一直不变。
        //   实测用户就看到了这个假数字（2026-10-18 DVMM-322）。
        //   宁可显示"—"，也不要显示一个骗人的数。
        if sp <= 0.0 {
            return None;
        }

        let sep_est = sep_rate * sp;
        let asr_est = asr_rate * sp;

        // 第1、2步的耗时也拿实测均值，没有就用经验值
        let fixed_first_two = {
            let n_files = self.stage_durations.len() / 4;
            if n_files > 0 {
                let sum: f64 = (0..n_files)
                    .map(|i| (self.stage_durations[i * 4] + self.stage_durations[i * 4 + 1]) as f64)
                    .sum();
                sum / n_files as f64
            } else {
                53.0 // 15.9 + 37.1
            }
        };
        // 第4步之后的收尾（字幕改名、清理）大概 10 秒
        const TAIL: f64 = 10.0;

        // tqdm 进度是否可信
        let frac = self.stage_frac.clamp(0.0, 1.0);
        // 只要有 tqdm 数据就算可信。原来要求 >0.05，第4步前 3 帧（0%、2%、4%）
        // 全被排除，ETA 只能退回成本模型，误差大。
        let frac_ok = frac > 0.001;

        let remain = match self.stage {
            // 第1步：提取。还没跑到 VAD，语音时长未知
            1 => {
                let left = (fixed_first_two * 0.4 - stage_used).max(0.0);
                left + fixed_first_two * 0.6 + sep_est + asr_est + TAIL
            }
            // 第2步：VAD。语音时长要等这一步跑完才知道
            2 => {
                let left = (fixed_first_two * 0.6 - stage_used).max(0.0);
                left + sep_est + asr_est + TAIL
            }
            // 第3步：分离。
            // 优先级：tqdm 自带的 ETA > 按本阶段速度外推 > 成本模型
            3 => {
                let left = if self.had_eta {
                    // ★ tqdm 自己算的，按真实处理速度实时更新，最可信
                    self.tqdm_remain
                } else if frac_ok {
                    (stage_used * (1.0 - frac) / frac).max(0.0)
                } else if sep_est > stage_used {
                    sep_est - stage_used
                } else {
                    stage_used * 0.5
                };
                left + asr_est + TAIL
            }
            // 第4步：识别。
            // 优先级：tqdm 自带的 ETA > 按本阶段速度外推 > 成本模型
            4 => {
                // ★ 前 5 秒不用 tqdm 的 ETA。
                //
                //   实测 tqdm 的前几帧是：72s → 82s → 87s → 84s → 233s，
                //   速度样本太少 + Whisper 还在 warmup。
                //   原来设的是 30 秒，太长 —— 用户要盯着"—"看半分钟。
                //   5 秒足够跳过最乱的起步阶段（tqdm 约 1~2 秒刷新一次，
                //   5 秒能收到 3~5 个样本）。
                let warmed_up = self
                    .stage_started_at
                    .map(|t| t.elapsed().as_secs() >= 5)
                    .unwrap_or(true);
                let left = if self.had_eta && warmed_up {
                    // ★ tqdm 自己算的剩余（<01:12 那段），最可信。
                    //   第4步的 tqdm 是按音频秒数推进的，它给的 ETA
                    //   直接反映了 Whisper 的真实处理速度。
                    self.tqdm_remain
                } else if frac_ok {
                    (stage_used * (1.0 - frac) / frac).max(0.0)
                } else if asr_est > stage_used {
                    asr_est - stage_used
                } else {
                    // ★ 模型低估了（这一阶段已经跑了比预测更久）。
                    //
                    //   原来这里返回 0，最后只剩 TAIL 常量 → 界面显示
                    //   「约 10 秒」并且一直不动，用户以为卡死了。
                    //   实测：SDAM-171 语音 1376.5 秒、识别跑了 172 秒，
                    //   而模型按 0.081 只预测 111 秒 → left=0 → 卡在 10 秒。
                    //
                    //   现在改成「按当前速度再估一半」，至少是个活的数字。
                    stage_used * 0.5
                };
                left + TAIL
            }
            _ => return None,
        };

        Some(remain.max(0.0) as u64)
    }

    /// 当前屏幕的可用尺寸（逻辑像素，已扣掉任务栏大概占的高度）。
    #[cfg(windows)]
    fn screen_work_area() -> (f64, f64) {
        use windows_sys::Win32::UI::HiDpi::GetDpiForSystem;
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN,
        };
        unsafe {
            // 必须按 DPI 换算：本进程是 DPI-aware 的，GetSystemMetrics 返回的是
            // 物理像素（4K 屏上是 3840x2160），而 client_size 收的是逻辑像素，
            // 直接传会被再乘一次缩放，窗口就比屏幕还大了。
            let dpi = GetDpiForSystem();
            let scale = if dpi == 0 { 1.0 } else { dpi as f64 / 96.0 };
            let w = GetSystemMetrics(SM_CXSCREEN) as f64 / scale;
            let h = GetSystemMetrics(SM_CYSCREEN) as f64 / scale;
            (w, h - 90.0)
        }
    }

    #[cfg(not(windows))]
    fn screen_work_area() -> (f64, f64) {
        (1280.0, 720.0)
    }

    /// 开窗尺寸：**固定 1280x720 逻辑像素**。
    ///
    /// client_size 收逻辑像素，系统按 DPI 缩放换算物理像素，所以同一份
    /// 1280x720 在 1080P@100% 上是 1280x720 物理、在 4K@125% 上是
    /// 1600x900 物理 —— **观感完全一致**，这才是 DPI 感知该有的效果。
    ///
    /// 不写成"屏幕的百分之几"：那样屏幕越大窗口越大，正好反了。
    /// 只在屏幕特别小时（低于 900x600）才跟着收缩。
    fn preferred_window_size() -> (f64, f64) {
        let (sw, sh) = Self::screen_work_area();
        (
            1280.0_f64.min(sw * 0.95).max(900.0),
            // 高度从 720 提到 860。
            //
            // 720 装不下五张卡片：日志区 + 进度区各自需要 200 多的净高，
            // 合计加上标题、按钮、间距就超了，于是【右边冒出纵向滚动条】——
            // 而用户明确要求"默认窗口能把卡片完全显示"。
            // 实测 860 才够。
            // 高度从 860 提到 1000。
            //
            // 860 是当年"五张卡片刚好不冒滚动条"的下限，但改版后文件列表
            // 用了自绘行，一屏只放得下 4 行 —— 而一屏能看见多少文件，是
            // 这个工具用得顺不顺手的关键。1000 让文件卡多出 ~2 行、日志卡
            // 也不再贴边。用户是 4K 屏（逻辑高 1600+），放得下。
            1000.0_f64.min(sh * 0.95).max(600.0),
        )
    }

    /// 起一个阶段。
    ///
    /// 把 `live_line` 克隆一份交给后台线程：脚本每输出一行，那里就更新一次。
    /// 界面靠 Msg::Tick（每秒）去读，于是进度卡片上能实时看到
    /// "正在加载模型…"、"提取中: xxx.flac" 这类内容，
    /// 而不是等整个阶段跑完才一次性冒出来。
    fn spawn_stage(&self, stage: usize, context: &ComponentContext<Self>) {
        let live = std::sync::Arc::clone(&self.live_line);
        let ready = std::sync::Arc::clone(&self.model_ready);
        let frac = std::sync::Arc::clone(&self.live_frac);
        let eta = std::sync::Arc::clone(&self.live_remain);
        let sub_end = std::sync::Arc::clone(&self.live_sub_end);
        let mode = self.mode;
        context.spawn_background(move |token| {
            let r: StageResult = pipeline::run_stage(mode, stage, token, live, ready, frac, eta, sub_end);
            Msg::StageDone(r.stage, r.code, r.lines)
        });
    }
}

// ─────────────────────────────────────────────────────────
//  主题色板 —— Claude 暖色（v1.23.0）
//
//  改版前是深色冰川蓝玻璃：深色主题 + Acrylic 半透明，卡片靠 alpha
//  压住壁纸。问题有三个：① 底色跟着壁纸走，永远调不准；
//  ② 卡片和日志区都是半透明，长文本对比度上不去；③ 换主题要逐处改。
//
//  这一版改成不透明的暖米色纸面，颜色只在这里定义：
//    窗口底 #f0eee6 / 卡片 #faf9f5 / 内嵌框 #f5f2ea / 描边 #e3ddd0
//    赤陶橙 #c96442 做强调，橙系文字用 #7d3f26 保证 4.5:1 对比度
//
//  Color::rgb 是 const fn，所以可以直接定义成常量。
// ─────────────────────────────────────────────────────────
const BG: Color = Color::rgb(0xf0, 0xee, 0xe6); // 窗口底色
const CARD: Color = Color::rgb(0xfa, 0xf9, 0xf5); // 卡片面
const CARD_LINE: Color = Color::rgb(0xe3, 0xdd, 0xd0); // 卡片描边
const SUNK: Color = Color::rgb(0xf5, 0xf2, 0xea); // 内嵌框（日志 / 预览）
const SUNK_LINE: Color = Color::rgb(0xe3, 0xdd, 0xd0); // 内嵌框描边
const INK: Color = Color::rgb(0x1f, 0x1e, 0x1d); // 主文字
const INK2: Color = Color::rgb(0x4a, 0x46, 0x3f); // 次要文字
const INK3: Color = Color::rgb(0x6b, 0x64, 0x58); // 更次要（提示、尾注）
const ACCENT: Color = Color::rgb(0xc9, 0x64, 0x42); // 赤陶橙（强调块、进度）
const ACCENT_DEEP: Color = Color::rgb(0xb4, 0x53, 0x2f); // 主按钮底（白字可读）
const ACCENT_SOFT: Color = Color::rgb(0xf7, 0xeb, 0xe4); // 选中胶囊、浅橙底（原稿 --blue-soft）
const ACCENT_INK: Color = Color::rgb(0x7d, 0x3f, 0x26); // 橙色文字（对比度够）
const OK_GREEN: Color = Color::rgb(0x5f, 0x8a, 0x63); // 已完成标记
const DONE_INK: Color = Color::rgb(0x8a, 0x83, 0x74); // 已完成文件的路径（淡化）
const WARN: Color = Color::rgb(0xa4, 0x44, 0x2c); // 警告文字

// ─────────────────────────────────────────────────────────
//  样式 helper —— 统一界面的视觉语言
//
//  参照 Windows 11 设置页（Fluent / WinUI 3）的做法：
//    · 每一块内容装在一张卡片里（圆角 + 细描边）
//    · 卡片有标题，正文靠内边距和间距拉开
//    · 层级靠【字号 + 字重】表达，不靠颜色堆砌
//
//  颜色统一取自上面的色板，不再在 view 里写死 RGB。
// ─────────────────────────────────────────────────────────

/// 一张卡片。body 是卡片内容。
///
/// 后面四个参数是它在父 Grid 里的位置 —— 必须在 `.children()` 【之前】
/// 调 grid_row 之类：children() 返回的是 View，而 GridChildExt 只实现给
/// 从设备描述里抠出显卡短名。
///
///   "兼容模式（Vulkan: NVIDIA GeForce RTX 5070）" → "RTX 5070"
///   检测中（"检测中…"）或纯 CPU 时返回 None，调用处就不显示这一段。
fn gpu_short_name(status: &str) -> Option<&str> {
    for tag in ["RTX", "GTX", "Arc ", "Radeon"] {
        if let Some(pos) = status.find(tag) {
            let rest = &status[pos..];
            let end = rest
                .find(|c| matches!(c, '）' | ')' | '，' | ','))
                .unwrap_or(rest.len());
            return Some(rest[..end].trim());
        }
    }
    None
}

/// 具体控件（Border/StackPanel/...），View 上没有这些方法（E0599）。
///
/// 位置交给 Grid 是为了让卡片随窗口拉伸：放在 STAR 行/列里，
/// 窗口变大卡片就变大，而不是像 StackPanel 那样永远是内容那么高。
fn card(
    title: &str,
    body: View,
    row: i32,
    col: i32,
    row_span: i32,
    col_span: i32,
) -> View {
    // 没胶囊就塞一个 1px 的空文本 —— StackPanel 的 children 是定长元组，
    // 不接受 Option，这是最省事的「空位」写法（本文件其他地方也这么占位）。
    card_x(
        title,
        View::from(TextBlock::new().text("").font_size(1.0)),
        body,
        row,
        col,
        row_span,
        col_span,
    )
}

/// 带「标题右侧小胶囊」的卡片，比如「文件列表 137」。
fn card_x(
    title: &str,
    badge: View,
    body: View,
    row: i32,
    col: i32,
    row_span: i32,
    col_span: i32,
) -> View {
    // ★ orientation(Horizontal) 不能少。
    //   StackPanel 默认是【纵向】的，不写这个，那个 3x16 的强调色小竖条
    //   会和标题竖着排 —— 实测结果是竖条飘到卡片右上角、标题掉到下面。
    let title_row = StackPanel::new()
        .orientation(Orientation::Horizontal)
        .spacing(8.0)
        .children((
            Border::new()
                .background(ACCENT)
                .corner_radius(2.0)
                .width(3.0)
                .height(16.0)
                .content(TextBlock::new().text("").font_size(1.0)),
            TextBlock::new()
                .text(title)
                .font_size(15.0)
                .font_weight(FontWeight::SEMI_BOLD)
                .foreground(INK),
            badge,
        ));
    Border::new()
        .grid_row(row)
        .grid_column(col)
        .grid_row_span(row_span)
        .grid_column_span(col_span)
        // 深蓝半透明玻璃面（不是 5% 白）。
        //
        // 第一版用的是 Color::argb(13,255,255,255) —— 即"5% 白"，理论上是
        // 最标准的玻璃拟态做法。但实测在【花哨壁纸】上不能用：壁纸的暖色
        // 会直接透上来盖住卡片，深色文字压在花哨图案上读起来很累。
        //
        // InternalBeyond 自己用的也不是 5% 白，而是：
        //     --glass: rgba(28,34,52,0.42)     深色玻璃 42%
        //     --amk2:  rgba(22,48,95,0.68)     深蓝 68%
        // 即【带颜色的高不透明度玻璃】。这样既保留透光感（能看出后面有东西），
        // 又把壁纸的干扰压下去，文字始终清晰。
        //
        // 这里取 #152c58（InternalBeyond 的卡片面色）配 59% 不透明度。
        // 冰川蓝玻璃。
        //
        // 调色过程（用户反馈驱动的）：
        //   argb(13,255,255,255)   5% 白  —— 太透，壁纸暖色透上来，文字看不清
        //   argb(150,21,44,88)     #152c58 59% —— 压得住壁纸，但【太蓝太深】
        //   argb(120,168,200,234)  #a8c8ea 47% —— 现在这个：冰川蓝
        //
        // 要点：冰川蓝不是"深蓝加透明度"，而是【高 RGB + 冷色调】——
        //   R 比 G 低、G 比 B 低，整体亮度高（都在 160~240 区间）。
        //   这样才透出"冰面反光"的冷白感，而不是"深海的蓝"。
        .background(CARD)
        // 描边：暖灰，比底色深一档就够，不要抢内容
        .border_brush(CARD_LINE)
        .border_thickness(1.0)
        .corner_radius(20.0)
        .padding(16.0)
        .content(StackPanel::new().spacing(6.0).children((
            // 【不要在这里加"顶部内高光"】
            //
            //   我试过用一条 Border::new().height(1.0) + 白色背景来模拟玻璃
            //   顶部的高光（因为 reactor 没有 box-shadow / inset）。
            //   结果：Border 在 StackPanel 里会被【横向拉伸到满宽】，
            //   于是变成横贯卡片的横线 —— 而且正好压在标题上方，
            //   看起来像一条莫名其妙的分隔线（用户直接要求去掉）。
            //
            //   而且它还实打实占掉了 1px 高度 + 6px spacing = 每张卡 7px，
            //   五张卡就是 35px，正好把下方卡片挤出可视区。
            //
            //   想要玻璃质感就靠【颜色和描边】，不要再加实体线条。
            title_row,
            body,
        )))
}

/// 「标签 + 值」两行。标签小一号、值大一号，靠字号做层级。
fn field(label: &str, value: &str) -> View {
    View::from(StackPanel::new().spacing(3.0).children((
        TextBlock::new().text(label).font_size(11.0).foreground(INK2),
        TextBlock::new().text(value).font_size(14.0).foreground(INK),
    )))
}

/// 卡片标题右侧的计数小胶囊（「文件列表 137」里的 137）。
fn count_pill(text: &str) -> View {
    View::from(
        Border::new()
            .background(Color::rgb(0xf0, 0xdc, 0xd2))
            .corner_radius(9.0)
            .padding(Thickness::xy(8.0, 1.0))
            .vertical_alignment(VerticalAlignment::Center)
            .content(
                TextBlock::new()
                    .text(text)
                    .font_size(11.0)
                    .foreground(ACCENT_INK),
            ),
    )
}

/// 处理模式的「胶囊按钮」。
///
/// 【为什么不用 RadioButton】
///   WinUI 的单选圆点是系统强调色（蓝紫），而 resource_overrides 只有
///   Button 支持 —— RadioButton 改不了配色。换来的代价是三个模式从
///   「一列单选」变成「一排互斥胶囊」，视觉和原稿一致了。
/// 底部操作按钮（开始处理 / 停止 / 暂停）。
///
/// 【为什么不是给 Button 上色】和 `mode_chip` 同一套理由：`ControlCornerRadius`
/// 这个主题资源键覆盖不掉 WinUI 按钮模板的圆角，`corner_radius` 又只长在
/// Border 上（Button 上没有这个方法，试过，E0599）。所以外形一律 Border 画，
/// Button 透明、只保留点击区域。
///
/// `dot` = 主操作前面那个小圆点；禁用态由调用方把颜色换成弱色，
/// 不依赖 WinUI 的 disabled 视觉（那套会把胶囊底色一起冲掉）。
fn pill_button(
    label: &str,
    dot: bool,
    bg: Color,
    line: Color,
    fg: Color,
    enabled: bool,
    on_click: Callback<()>,
) -> View {
    let inner: View = if dot {
        View::from(
            StackPanel::new()
                .orientation(Orientation::Horizontal)
                .spacing(8.0)
                .children((
                    Border::new()
                        .width(8.0)
                        .height(8.0)
                        .corner_radius(4.0)
                        .background(fg)
                        .vertical_alignment(VerticalAlignment::Center)
                        .content(TextBlock::new().text("").font_size(1.0)),
                    TextBlock::new().text(label).font_size(13.0).foreground(fg),
                )),
        )
    } else {
        View::from(TextBlock::new().text(label).font_size(13.0).foreground(fg))
    };

    let face = Border::new()
        .background(bg)
        .border_brush(line)
        .border_thickness(1.0)
        .corner_radius(18.0)
        .padding(Thickness::xy(18.0, 8.0))
        .content(inner);

    Button::new()
        .resource_overrides(
            ResourceOverrides::new()
                .set("ButtonBackground", Color::argb(0, 0, 0, 0))
                .set("ButtonBackgroundPointerOver", Color::argb(0, 0, 0, 0))
                .set("ButtonBackgroundPressed", Color::argb(0, 0, 0, 0))
                .set("ButtonBorderThemeThickness", Thickness::uniform(0.0))
                .set("ButtonPadding", Thickness::uniform(0.0)),
        )
        .is_enabled(enabled)
        // 外观是自己画的，屏幕阅读器读不到里面的 TextBlock 时，
        // 这一句保证「全量模式 / 精简模式 / 兼容模式」仍然有可读名字。
        .automation_name(label)
        .on_click(on_click)
        .content(face)
}

/// 普通小按钮：添加文件 / 更改… / 导出完整日志 这类。
///
/// 外形规则和底部操作键完全一致（同一个 pill_button），只是配色用中性的
/// 卡片白 + 浅描边。原因很直接：WinUI 默认按钮是 4px 小圆角，混在胶囊
/// 按钮中间一看就是"两套控件"，统一之后整个界面只有一种按钮形状。
fn tap_button(label: &str, on_click: Callback<()>) -> View {
    pill_button(label, false, CARD, CARD_LINE, INK2, true, on_click)
}

fn mode_chip(label: &str, selected: bool, enabled: bool, on_click: Callback<()>) -> View {
    // 【为什么是「Border 自绘 + 透明按钮」而不是「给 Button 上色」】
    //   一开始走的是 resource_overrides 改 ButtonBackground / ButtonBorderBrush，
    //   实测不生效：按钮边框能出来，但选中态的橙底完全不见（截图取色仍是卡片色）。
    //   主按钮同样的写法却正常 —— 差别不明，不再赌框架行为。
    //   现在把「好看的那层」交给 Border（背景/描边/圆角全都自己说了算），
    //   Button 只保留点击区域，并且把它的背景和描边全部清零。
    let bg = if selected { ACCENT_SOFT } else { CARD };
    let line = if selected { ACCENT } else { CARD_LINE };
    let fg = if selected { ACCENT_INK } else { INK2 };
    // 左端那个小圆点：原稿每个胶囊前面都有一个，选中时填实。
    let dot = if selected { ACCENT } else { CARD_LINE };

    let face = Border::new()
        .background(bg)
        .border_brush(line)
        .border_thickness(1.0)
        .corner_radius(15.0)
        .padding(Thickness::xy(13.0, 6.0))
        .content(
            StackPanel::new()
                .orientation(Orientation::Horizontal)
                .spacing(7.0)
                .children((
                    Border::new()
                        .width(9.0)
                        .height(9.0)
                        .corner_radius(4.5)
                        .background(dot)
                        .vertical_alignment(VerticalAlignment::Center)
                        .content(TextBlock::new().text("").font_size(1.0)),
                    TextBlock::new().text(label).font_size(13.0).foreground(fg),
                )),
        );

    Button::new()
        .resource_overrides(
            ResourceOverrides::new()
                .set("ButtonBackground", Color::argb(0, 0, 0, 0))
                .set("ButtonBackgroundPointerOver", Color::argb(0, 0, 0, 0))
                .set("ButtonBackgroundPressed", Color::argb(0, 0, 0, 0))
                .set("ButtonBorderThemeThickness", Thickness::uniform(0.0))
                .set("ButtonPadding", Thickness::uniform(0.0)),
        )
        .is_enabled(enabled)
        // 同 mode_chip：外观是自绘的，补一个可读名字给屏幕阅读器。
        .automation_name(label)
        .on_click(on_click)
        .content(face)
}

/// 卡片内部再嵌一层的浅色框。用来区分"一块可滚动的内容"，
/// 比如日志区和预览区 —— 让它们看起来是独立的一块，
/// 而不是直接糊在卡片上。
fn inner_box(body: View) -> View {
    Border::new()
        // 内嵌框比卡片再深一层、再实一点 —— 日志区压在花壁纸上要能看清字
        // 内嵌框要比卡片【深】，才有凹进去的层次；
        // 但底色也往冷蓝偏，不要死黑 —— 死黑在冰川蓝的卡片上很突兀。
        .background(SUNK)
        .border_brush(SUNK_LINE)
        .border_thickness(1.0)
        .corner_radius(12.0)
        .padding(12.0)
        .content(body)
}

impl Component for AppUi {
    type Input = ();
    type Message = Msg;

    fn create(_input: &(), context: &ComponentContext<Self>) -> Self {
        let logs = vec![
            "就绪。点「开始处理」会依次执行四个步骤：".to_string(),
            "  第1步 提取音频 → 第2步 人声检测 → 第3步 分离人声 → 第4步 语音识别翻译".to_string(),
            "".to_string(),
            "先在「文件列表」里添加要处理的视频/音频，再点「开始处理」。".to_string(),
        ];

        // 测试后门：设了 ASMR_AUTOSTART 就自动点一次「开始处理」。
        //
        // 走的是和按钮完全相同的路径 —— 一样是发 Msg::Start 进组件的消息队列，
        // 一样进 update()。区别只是触发者是人手还是环境变量。
        // 之所以需要这个：键鼠自动化点击在这个环境里不可靠（窗口焦点会被
        // 别的程序抢走），而 create() 是在主线程跑的，sender() 可以直接用。
        if std::env::var("ASMR_AUTOSTART").is_ok() {
            let _ = context.sender().send(Msg::Start);
        }

        // 后台问一次 CUDA。torch 首次导入要 2~3 秒，不能卡在界面线程里，
        // 所以走 spawn_background，回来时发 Msg::DeviceDetected。
        context.spawn_background(|_token| {
            let (text, is_gpu, vram, has_vulkan) = pipeline::detect_device();
            Msg::DeviceDetected(text, is_gpu, vram, has_vulkan)
        });

        // 窗口居中的逻辑【不在这里】。
        //
        // 它在 main.rs 的 center_hook 模块里，用 WH_CBT 钩子的 HCBT_ACTIVATE
        // 在窗口激活（显示）之前就把位置摆好，所以窗口第一次出现就已经在中央。
        //
        // 这里不重复做一遍，也不保留"轮询兜底" —— 实测那套做法不稳定
        // （5 次里 2 次会先闪一下），留着只会和钩子抢着设位置。
        // 具体踩过的坑写在 main.rs 顶部的注释里。

        // 调试后门：ASMR_TEST_FILES 用 `;` 分隔若干路径，直接塞进待处理列表。
        // 用途是自动化验证整条流水线时跳过"人手点按钮选文件"那一步，
        // 但走的仍然是完整路径：一样进 self.files，一样经过 prepare_0pre
        // （硬链接/复制）、一样参与字幕收尾的文件名匹配。
        let mut files: Vec<String> = Vec::new();
        if let Ok(v) = std::env::var("ASMR_TEST_FILES") {
            for p in v.split(';') {
                let p = p.trim();
                if !p.is_empty() {
                    files.push(p.to_string());
                }
            }
        }

        // 调试后门：ASMR_FAKE_PREVIEW=1 时塞几条假字幕进预览区。
        // 用途是单独验证预览区的渲染（不用真跑一遍四个脚本，那个要等模型加载）。
        // 正则解析那部分由 config.rs 里的单元测试覆盖，不靠这个后门。
        let preview = if std::env::var("ASMR_FAKE_PREVIEW").is_ok() {
            vec![
                "[12.93s → 17.87s]  就算是机器也是要先烘焙再磨炼的吧？".to_string(),
                "[17.90s → 21.91s]  但是如果全部都要用手烘焙磨炼的话".to_string(),
                "[22.05s → 26.40s]  那已经不是量产工厂而是手工作坊了".to_string(),
                "[27.11s → 31.02s]  不过这样的话品质倒是能保证".to_string(),
                "[31.50s → 35.76s]  毕竟每一批都要靠人手来判断火候".to_string(),
            ]
        } else {
            Vec::new()
        };

        // 启动时读一次上次留下的队列。
        //
        // ★ 只读【一次】，结果用于三个字段。如果写成
        //     pending_queue: Self::load_queue()...
        //     pending_idx:   Self::load_queue()...
        //     show_restore_dialog: Self::load_queue().is_some()
        //   就是读了三次磁盘；万一文件在中途被删，三个字段会互相矛盾。
        let loaded_queue = Self::load_queue();
        let has_pending_queue = loaded_queue.is_some();
        let (pending_queue, pending_idx) = loaded_queue.unwrap_or_default();

        Self {
            running: false,
            stage: 0,
            done_stages: 0,
            files,
            total_lines: 0,
            stop_requested: false,
            selected_index: None,
            preview,
            current_idx: 0,
            pending_queue,
            pending_idx,
            show_restore_dialog: false,
            pending_restore_prompt: has_pending_queue,
            last_preview_line: String::new(),
            stage_frac: 0.0,
            logs,
            progress_value: 0.0,
            // 调试后门：ASMR_NO_COPY=1 时不勾选「字幕放视频旁边」，
            // 用来验证"取消勾选后输出目录可改"这条路径。
            copy_to_video: std::env::var("ASMR_NO_COPY").is_err(),
            force_redo: false,
            device_status: "检测中…".to_string(),
            has_gpu: true,
            has_vulkan: false,
            mode: pipeline::RunMode::Full,
            mode_full_ok: false,
            mode_lite_ok: false,
            paused: false,
            pause_started_at: None,
            lite_mode: false,
            vram_gb: 0.0,
            cpu_ok: false,
            show_cpu_warning: false,
            output_dir: config::dir_asr(),
            // 必须和 preferred_window_size() 一致：写死 1400 而窗口实际
            // 是 1280 的话，两栏会按 1400 排，右边一栏被窗口裁掉。
            window_width: Self::preferred_window_size().0,
            window_height: Self::preferred_window_size().1,
            show_no_file_warning: false,
            started_at: None,
            elapsed_secs: 0,
            stage_durations: Vec::new(),
            file_durations: Vec::new(),
            file_speech: Vec::new(),
            cur_speech: 0.0,
            stage_started_at: None,
            eta_display: None,
            live_line: std::sync::Arc::new(std::sync::Mutex::new(String::new())),
            live_frac: std::sync::Arc::new(std::sync::Mutex::new(0.0)),
            live_remain: std::sync::Arc::new(std::sync::Mutex::new(-1.0)),
            tqdm_remain: 0.0,
            tqdm_ema: 0.0,
            last_raw_remain: -1.0,
            had_eta: false,
            live_sub_end: std::sync::Arc::new(std::sync::Mutex::new(0.0)),
            audio_total: 0.0,
            model_ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            prepare_task: None,
        }
    }

    fn update(&mut self, message: Msg, context: &ComponentContext<Self>) {
        match message {
            Msg::Start => {
                if self.running {
                    return;
                }

                // ★ 没有 NVIDIA 显卡时先警告，别直接开跑。
                //
                //   CPU 回退本身是能跑的（audio-separator 有干净的回退路径），
                //   但速度差 30~100 倍：GPU 上 1 小时视频约 7 分钟，
                //   CPU 可能几小时到十几小时。状态栏那句"处理会很慢"
                //   完全传达不出这个量级 —— 用户会以为只是慢一点，
                //   结果挂一整天。所以这里必须让他明确知道再决定。
                //
                //   注意放在 running 检查【之后】、清状态【之前】：
                //   返回时不改任何状态，用户取消后界面和没点过一样。
                // 无 N 卡、显存不足 6 GB、且【没有 Vulkan 设备】才弹「只能用 CPU」
                // 的确认框。有 Vulkan（AMD/Intel 核显或独显）时走兼容模式，
                // 不需要吓唬用户。
                let no_cuda = !self.has_gpu
                    || (self.vram_gb > 0.0 && self.vram_gb < 6.0);
                let no_vulkan = !self.has_vulkan;
                if no_cuda && no_vulkan && !self.cpu_ok {
                    self.show_cpu_warning = true;
                    return;
                }

                // ★ 一个文件都没加就直接拦下来，弹警告框，什么都不做。
                //
                //   以前这里的行为是「列表为空就沿用 0pre\ 里已有的文件」——
                //   那是开发时为了方便调试留的后门。对用户来说是错的：
                //   他什么都没加，程序却开始跑（或者只在一行日志里报个看不懂的
                //   错），容易以为程序坏了。
                //
                //   注意：这个检查必须在【任何状态被改动之前】，
                //   否则弹完窗界面会停在半启动状态（进度条复位了、日志被清了）。
                // ★ 勾了「强制重新处理」→ 先把已有字幕删掉，这样下面的跳过逻辑就不会生效
                if self.force_redo {
                    let targets = self.files.clone();
                    let mut deleted = 0usize;
                    for f in &targets {
                        deleted += Self::delete_existing_subtitles(f);
                    }
                    if deleted > 0 {
                        self.push_log(format!(
                            "[强制重做] 已删除 {deleted} 个旧字幕文件（彻底删除，不进回收站）"
                        ));
                    } else {
                        self.push_log("[强制重做] 没有找到旧字幕，直接开始");
                    }
                }

                // ★ 开跑前先剔除【已经生成过字幕】的文件。
                //
                //   这是"断点续跑"的核心：产物在不在比索引可靠。
                //   DSH 更新重启时翻译器会被一起杀掉，索引来不及存；
                //   但只要字幕文件写出来了，这个文件就不该再做一遍。
                //
                //   注意放在 files.is_empty() 检查【之前】做剔除 ——
                //   剔完如果空了，下面那个"没有可处理的文件"会正常弹出，
                //   用户就知道"这批都已经做完了"。
                {
                    let before = self.files.len();
                    let skipped: Vec<String> = self
                        .files
                        .iter()
                        .filter(|f| Self::has_subtitle(f))
                        .cloned()
                        .collect();
                    if !skipped.is_empty() {
                        self.files.retain(|f| !Self::has_subtitle(f));
                        self.push_log(format!(
                            "[跳过] {} 个文件已经有字幕，不再重复处理",
                            skipped.len()
                        ));
                        for s in skipped.iter().take(8) {
                            let name = std::path::Path::new(s)
                                .file_name()
                                .map(|x| x.to_string_lossy().to_string())
                                .unwrap_or_else(|| s.clone());
                            self.push_log(format!("        · {name}"));
                        }
                        if skipped.len() > 8 {
                            self.push_log(format!("        …另外 {} 个", skipped.len() - 8));
                        }
                        self.push_log(format!(
                            "[队列] 剩余 {} 个待处理（原有 {} 个）",
                            self.files.len(),
                            before
                        ));
                        // 索引要夹回有效范围，否则第一个文件就取不出来
                        self.current_idx = 0;
                        self.save_queue();
                    }
                }

                if self.files.is_empty() {
                    // 用【原生 WinUI 对话框】，不用 rfd 的系统 MessageBox。
                    //
                    // 为什么换掉 rfd：Windows 系统消息框的宽度按文字长度算，
                    // 但有上限 —— 实测只要某行到 15 个汉字，右边就被硬裁掉
                    // （对话框只有 267px 宽、文字却更长）。缩文案治不了本。
                    // ContentDialog 是 WinUI 自带的，宽度自适应，风格也和
                    // 程序其余部分一致。
                    //
                    // 注意：这里【只置一个标志位】，真正的弹窗在 view() 里
                    // 渲染 —— dialog 是 view 的一部分，不是命令式调用。
                    self.show_no_file_warning = true;
                    return;
                }

                self.running = true;
                self.stop_requested = false;
                self.started_at = Some(std::time::Instant::now());
                self.elapsed_secs = 0;
                if let Ok(mut l) = self.live_line.lock() {
                    l.clear();
                }
                self.eta_display = None;
                self.stage_durations.clear();
                self.file_durations.clear();
                self.file_speech.clear();
                self.cur_speech = 0.0;
                self.stage_started_at = None;
                self.stage = 0;
                self.done_stages = 0;
                self.progress_value = 0.0;
                self.logs.clear();
                self.preview.clear();
                self.total_lines = 0;
                self.push_log("=== 开始处理 ===");

                // 先确认工作目录在不在，省得四个脚本都报一遍同样的错
                let wd = config::work_dir();
                if !std::path::Path::new(wd).is_dir() {
                    self.push_log(format!("[错误] 工作目录不存在: {wd}"));
                    self.running = false;
                    return;
                }

                // 准备 0pre：三个目录都清（马上会往 0pre 放新的文件），
                // 然后再把列表里的文件硬链接/复制进 0pre。
                //
                // 硬链接可能很慢（跨卷时会退化成复制几 GB），所以放后台做。
                self.purge_process_files("", true);

                // 把取消令牌接进复制循环：token.is_cancelled() 一旦为真，
                // 分块复制就会立刻停下并丢弃写了一半的文件。
                // 同时把 ComponentTask 存起来 —— 「停止」按钮靠它 cancel()。
                // ★ 只处理【第一个】文件 —— 逐个文件跑完四步，再下一个。
                //
                //   原来是把整个 files 交给 prepare_0pre，25 个文件的中转
                //   全部堆在 0pre/1audio 里，磁盘直接爆掉（实测 200 GB 塞满）。
                //   现在每轮只写一个文件进 _input.txt。
                self.current_idx = 0;
                let one = vec![self.files[0].clone()];
                self.prepare_task = Some(context.spawn_background(move |token| {
                    Msg::Prepared(pipeline::prepare_0pre_cancellable(&one, &|| {
                        token.is_cancelled()
                    }))
                }));
                return; // 等 Msg::Prepared 回来再开第一阶段
            }

            Msg::Prepared(result) => {
                // 任务已经结束了，句柄没用了，放掉
                self.prepare_task = None;
                match result {
                    Ok((_n, logs)) => {
                        self.push_logs(logs);
                        if self.stop_requested {
                            // Msg::Stop 那边已经报过「任务已停止」了，
                            // 这里只需把状态归位，不再重复写日志。
                            self.running = false;
                            self.stage = 0;
                            return;
                        }
                        self.stage = 1;
                        // 第1步从这一刻开始计时（估算剩余时间要用实测耗时）
                        self.stage_started_at = Some(std::time::Instant::now());
                        // 逐个文件模式下，每轮都要标清楚进度
                        if let Some(cur) = self.files.get(self.current_idx) {
                            let name = std::path::Path::new(cur)
                                .file_name()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_else(|| cur.clone());
                            self.push_log(format!(
                                "===== 文件 {}/{}: {} =====",
                                self.current_idx + 1,
                                self.files.len(),
                                name
                            ));
                        }
                        self.push_log("--- 第1步 提取音频 ---".to_string());
                        // 第一步是新流水线的起点，模型加载标志归零
                        self.model_ready
                            .store(false, std::sync::atomic::Ordering::Relaxed);
                        self.stage_frac = 0.0;
                        self.spawn_stage(1, context);
                    }
                    Err(e) => {
                        self.push_log(format!("[失败] 准备阶段出错: {e}"));
                        self.running = false;
                        self.stage = 0;
                        // 失败了也要清 —— 留着中间产物只会占用空间，
                        // 而且下次跑会和新文件混在一起
                        self.purge_process_files("", true);
                    }
                }
            }

            Msg::StageDone(stage, code, lines) => {
                // 记下这个阶段实际花了多久 —— 估算剩余时间要用
                let stage_secs = self
                    .stage_started_at
                    .take()
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0);
                self.stage_durations.push(stage_secs);
                self.stage_started_at = Some(std::time::Instant::now());
                // 第4步（语音识别翻译）的输出里带着字幕行，先抽出来进预览区。
                // 注意必须在 push_logs 之前做，因为 lines 会被 move 走。
                //
                // 这一步曾经在 Python 版翻过车：正则里漏了 `s?`，导致
                // 一行都匹配不上、预览区永远空白，而且【不报错】。
                // 所以这里用的是 config::translation_re() 里那个带 s? 的版本。
                if stage == 4 {
                    let re = config::translation_re();
                    let mut hit = 0usize;
                    for l in &lines {
                        if let Some(c) = re.captures(l) {
                            let s = c.get(1).map(|m| m.as_str()).unwrap_or("?");
                            let e = c.get(2).map(|m| m.as_str()).unwrap_or("?");
                            let t = c.get(3).map(|m| m.as_str().trim()).unwrap_or("");
                            self.preview.push(format!("[{s}s → {e}s]  {t}"));
                            hit += 1;
                        }
                    }
                    self.push_log(format!("[预览] 本步解析出 {hit} 条字幕"));
                }

                self.push_logs(lines);

                if code != 0 {
                    self.push_log(format!("[失败] 第{stage}步退出码 = {code}，流水线中止"));
                    self.running = false;
                    self.stage = 0;
                    self.purge_process_files("", true);
                    return;
                }

                self.done_stages = stage;
                // 该步已完成 → 把它按 100% 计入当前文件的加权进度
                self.stage_frac = 1.0;
                self.recompute_progress();
                // 把用时写出来：一是让用户知道每步多慢（心里有数），
                // 二是排查性能问题和校准预估算法时能直接读。
                self.push_log(format!(
                    "[完成] 第{stage}步，退出码 0，本步用时 {}",
                    Self::fmt_secs(stage_secs)
                ));

                if self.stop_requested {
                    // 不写额外日志 —— Msg::Stop 已经报过结果了
                    self.running = false;
                    self.purge_process_files("", true);
                    return;
                }

                // 还有下一阶段就继续
                if stage < pipeline::stages(self.mode).len() {
                    self.stage = stage + 1;
                    // ★ 换阶段 → 清掉 ETA 的显示锚点。
                    //
                    //   第3步的「还剩 2 分钟」和第4步的「还剩 8 分钟」是
                    //   两个完全不同的物理过程。拿前者当锚点去限速，结果
                    //   就是数字卡在旧值上慢慢爬（实测用户看到"6分18秒
                    //   停了二十秒"）。
                    //
                    //   清成 None 之后，下一帧会直接取真实值，
                    //   从第 2 帧起才重新启用限速。
                    self.eta_display = None;
                    // 换阶段 → 下一个模型还没加载，标志归零
                    self.model_ready
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    // 换阶段了，上一阶段的最后一行不该再被当成新字幕
                    self.last_preview_line.clear();
                    // 新阶段还没开始推理，tqdm 进度归零（两条通道都要清）
                    self.stage_frac = 0.0;
                    self.tqdm_remain = 0.0;
                    if let Ok(mut f) = self.live_frac.lock() {
                        *f = 0.0;
                    }
                    self.tqdm_ema = 0.0;
                    self.last_raw_remain = -1.0;
                    self.had_eta = false;
                    if let Ok(mut v) = self.live_sub_end.lock() {
                        *v = 0.0;
                    }
                    if let Ok(mut r) = self.live_remain.lock() {
                        *r = -1.0; // -1 = 还没拿到 ETA
                    }
                    self.push_log(format!("--- 第{}步 {} ---",
                        stage + 1,
                        pipeline::stages(self.mode)[stage].2.trim_start_matches(|c: char| c.is_ascii_digit() || c == '第' || c == '步' || c == ' ')));
                    self.spawn_stage(stage + 1, context);
                } else {
                    // 四个阶段都跑完了。
                    // ★ v1.17：先做「字幕时间轴整理」这个纯文本小任务，
                    //   再收尾改名。整理只是几百毫秒的文本处理，
                    //   刻意【不占阶段编号、不进进度体系】—— 进度条和
                    //   剩余时间的算法一行都不用改。
                    //
                    //   为什么放在这里：finalize 会把 3asr 里的 srt 改名
                    //   搬走，整理必须在搬走之前完成。
                    self.push_log("=== 本文件四个阶段完成，开始整理字幕 ===");
                    context.spawn_background(move |_token| {
                        Msg::Tidied(pipeline::tidy_subtitles())
                    });
                }
            }

            Msg::Tidied(lines) => {
                self.push_logs(lines);

                if self.stop_requested {
                    self.running = false;
                    self.purge_process_files("", true);
                    return;
                }

                // 整理完再做字幕收尾：
                // 改名成 .chs.srt，并按用户的勾选决定放视频旁还是留在 3asr\。
                // 这一步要动文件（跨卷时甚至要复制），所以也放后台。
                self.push_log("=== 开始整理字幕 ===");
                // ★ 只传【当前这一个文件】。
                //   传整个 files 会导致每轮都把前面已经完成的文件再"整理"一遍
                //   （虽然文件名对不上会跳过，但日志会很乱，而且 3asr 里的
                //   旧 srt 会被反复处理）。
                let cur = self
                    .files
                    .get(self.current_idx)
                    .cloned()
                    .map(|f| vec![f])
                    .unwrap_or_default();
                let copy = self.copy_to_video;
                let out_dir = self.output_dir.clone();
                context.spawn_background(move |_token| {
                    Msg::Finalized(pipeline::finalize_subtitles(&cur, copy, &out_dir))
                });
            }

            Msg::Finalized(logs) => {
                self.push_logs(logs);

                // 时机2：这个文件的字幕已经生成并归位，它的中间产物没用了。
                // ★ 必须在【下一个文件开始之前】清掉 —— 这正是"逐个文件处理"
                //   的意义所在：磁盘上永远只有一个文件的中间产物。
                self.purge_process_files("", true);

                if self.stop_requested {
                    self.running = false;
                    self.stage = 0;
                    return;
                }

                // ★ 记录这个文件的实测耗时和语音时长，供 ETA 使用。
                //
                //   文件结束时 elapsed 包含了本文件的全部四步，
                //   减去之前文件的累计耗时，就是本文件自己的耗时。
                let prev_sum: f64 = self.file_durations.iter().sum();
                let this_file = (self.elapsed_secs as f64 - prev_sum).max(0.0);
                if this_file > 0.0 {
                    self.file_durations.push(this_file);
                    // 语音时长：第2步解析到了就用它；否则用 0（后续按均值估）
                    self.file_speech.push(self.cur_speech);
                    self.cur_speech = 0.0;
                    self.audio_total = 0.0;
                    if let Ok(mut v) = self.live_sub_end.lock() {
                        *v = 0.0;
                    }
                }

                // ★ 换文件 → 同样清掉 ETA 锚点。
                //   下一个文件的时长/语音占比可能完全不同，
                //   拿上一个文件的值当锚点没有意义。
                self.eta_display = None;

                self.current_idx += 1;
                // ★ 每做完一个就落盘。这样即使程序被强杀（DSH 更新、
                //   断电、任务管理器结束），下次启动至少知道做到哪儿了。
                //   但真正可靠的判据是 has_subtitle() —— 产物在不在。
                self.save_queue();

                if self.current_idx >= self.files.len() {
                    // 全部做完了
                    self.running = false;
                    self.stage = 0;
                    self.done_stages = pipeline::stages(self.mode).len();
                    self.progress_value = 100.0;
                    self.push_log(format!(
                        "=== 全部 {} 个文件处理完成 ===",
                        self.files.len()
                    ));
                    return;
                }

                // 还有下一个 → 回到第1步，重新 prepare
                let one = vec![self.files[self.current_idx].clone()];
                self.push_log(format!(
                    "--- 准备下一个文件（{}/{}）---",
                    self.current_idx + 1,
                    self.files.len()
                ));
                self.prepare_task = Some(context.spawn_background(move |token| {
                    Msg::Prepared(pipeline::prepare_0pre_cancellable(&one, &|| {
                        token.is_cancelled()
                    }))
                }));
            }

            Msg::SetForceRedo(checked) => {
                self.force_redo = checked;
                self.push_log(if checked {
                    "[设置] 强制重新处理：开跑前会删除已有的字幕文件"
                } else {
                    "[设置] 已有字幕的文件会被跳过"
                });
            }

            Msg::SetCopyToVideo(checked) => {
                self.copy_to_video = checked;
                self.push_log(if checked {
                    "[设置] 字幕将直接放到视频旁边（播放器可自动加载）"
                } else {
                    "[设置] 字幕将留在 3asr\\ 目录"
                });
            }

            Msg::DeviceDetected(text, is_gpu, vram_gb, has_vulkan) => {
                self.device_status = text;
                self.has_gpu = is_gpu;
                self.vram_gb = vram_gb;
                self.has_vulkan = has_vulkan;
                // v1.21.1：精简模式的条件放宽为「N 卡显存 ≥ 6 GB」——
                // 大显存卡也可以主动选精简模式（比如想省显存跑别的程序）。
                // vram_gb == 0.0 表示「有 GPU 但显存读不出来」，按未知处理，
                // 不阻止用户选择（否则会被强制塞进兼容模式）。
                let vram_unknown = vram_gb <= 0.0;
                self.mode_full_ok = is_gpu && (vram_unknown || vram_gb >= 8.0);
                self.mode_lite_ok = is_gpu && (vram_unknown || vram_gb >= 6.0);
                self.mode = if self.mode_full_ok {
                    pipeline::RunMode::Full
                } else if self.mode_lite_ok {
                    pipeline::RunMode::Lite
                } else {
                    pipeline::RunMode::Compat
                };
                self.push_log(format!(
                    "[模式] 当前使用：{}",
                    match self.mode {
                        pipeline::RunMode::Full => "全量模式",
                        pipeline::RunMode::Lite => "精简模式",
                        pipeline::RunMode::Compat => "兼容模式",
                    }
                ));
                // 显存 < 8 GB → 精简模式（Python 侧已经自动切 int8 了）
                // 8.1 GB 是 settings.py 算出来的门槛（含 1 GB 安全余量）
                self.lite_mode = is_gpu && vram_gb > 0.0 && vram_gb < 8.1;
                if self.lite_mode {
                    self.push_log(format!(
                        "[精简模式] 当前显卡只有 {:.1} GB 显存，为保证你的其他程序正常运行，识别环节将工作在精简模式。",
                        vram_gb
                    ));
                    self.push_log(
                        "[精简模式] 具体做法：把语音识别的计算精度从 float16 降到 int8，显存占用减半（约 4.0 GB → 2.0 GB）。"
                    );
                    self.push_log(
                        "[精简模式] 代价：识别速度慢约 20%，个别词可能不够准确。"
                    );
                } else if !is_gpu {
                    if has_vulkan {
                        self.push_log(
                            "[兼容模式] 未检测到 NVIDIA 显卡，但发现了 Vulkan 设备，将使用兼容模式（SenseVoice + Qwen）。"
                        );
                    } else {
                        self.push_log(
                            "[兼容模式] 未检测到可用 GPU，将使用纯 CPU 兼容模式（速度较慢）。"
                        );
                    }
                }
            }

            Msg::SetModeFull => {
                if self.mode_full_ok {
                    self.mode = pipeline::RunMode::Full;
                    self.push_log("[模式] 已切换到全量模式");
                }
            }
            Msg::SetModeLite => {
                if self.mode_lite_ok {
                    self.mode = pipeline::RunMode::Lite;
                    self.push_log("[模式] 已切换到精简模式");
                }
            }
            Msg::SetModeCompat => {
                self.mode = pipeline::RunMode::Compat;
                self.push_log("[模式] 已切换到兼容模式");
            }
            Msg::TogglePause => {
                if !self.running {
                    return;
                }
                if self.paused {
                    backend::resume_current();
                    self.paused = false;
                    if let (Some(start), Some(pa)) = (self.started_at, self.pause_started_at.take()) {
                        self.started_at = Some(start + pa.elapsed());
                    }
                    self.push_log("[暂停] 已继续");
                } else {
                    backend::suspend_current();
                    self.paused = true;
                    self.pause_started_at = Some(std::time::Instant::now());
                    self.push_log("[暂停] 已暂停（子进程已挂起）");
                }
            }
            Msg::AgreeCpu => {                self.show_cpu_warning = false;
                self.cpu_ok = true;
                self.push_log(
                    "[兼容模式] 已确认使用纯 CPU 处理。\n                     速度较慢，建议改用带 NVIDIA 显卡的机器。"
                        .to_string(),
                );
            }

            Msg::RefuseCpu => {
                // 用户不接受 CPU 模式 → 直接退出
                std::process::exit(0);
            }

            Msg::BrowseOutputDir => {
                let picked = rfd::FileDialog::new()
                    .set_title("选择输出目录")
                    .pick_folder();
                if let Some(dir) = picked {
                    self.output_dir = dir.to_string_lossy().to_string();
                    self.push_log(format!("[输出目录] 已改为 {}", self.output_dir));
                }
            }

            Msg::Stop => {
                if self.running {
                    self.stop_requested = true;

                    // 硬停止：强杀掉正在跑的子进程树。
                    // 光设 stop_requested 只是"跑完当前阶段就不再继续"——
                    // 但那些脚本跑模型要几十分钟，用户点了停止就得真的停。
                    // 这和 Python 版一致（它用 proc.terminate() + taskkill）。
                    backend::kill_current();

                    // 准备阶段的后台任务也一并取消。
                    // （现在准备阶段只是写一个清单文件，几乎来不及取消；
                    //   但万一以后又加回耗时操作，这个调用能兜住。）
                    if let Some(task) = self.prepare_task.take() {
                        task.cancel();
                    }

                    self.running = false;
                    self.stage = 0;
                    // started_at 不清 —— 保留最后一次的耗时，用户还能看到

                    // 日志只留一行「任务已停止，过程文件已删除」。
                    // 中间那些"正在停止…""已强制结束脚本""删了几个"全部不写 ——
                    // 用户点停止后要的是确认结果，不是内部流水账。
                    self.purge_on_stop();
                }
            }

            Msg::AddFiles => {
                // rfd 的对话框是阻塞式的，会卡住主线程直到用户选完。
                // 这里可以接受：用户正在选文件时本来也不会去点界面。
                let picked = rfd::FileDialog::new()
                    .add_filter("视频 / 音频", &["mp4", "mkv", "flac", "wav", "mp3"])
                    .set_title("选择要处理的视频或音频文件（可多选）")
                    .pick_files();

                let Some(paths) = picked else {
                    // 用户取消，什么都不做，也不写日志
                    return;
                };

                let flat: Vec<String> = paths
                    .into_iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect();
                let (added, dup) = self.add_files(flat);
                self.push_log(format!("[文件] 新增 {added} 个，跳过 {dup} 个重复"));
            }

            Msg::AddFolder => {
                // 多选文件夹（v1.19）：用户经常一次要加好几个目录 ——
                // 一个作品一个文件夹，或者一次下载的一批。单选要点 N 次。
                //
                // rfd 的 pick_folders 背后是 Windows IFileDialog 的
                // FOS_PICKFOLDERS | FOS_ALLOWMULTISELECT，原生支持多选。
                //
                // 用 rfd 而不是自己写一个资源管理器窗口：rfd 背后是 Windows
                // 原生的 IFileDialog，自带盘符导航、快速访问、搜索、最近位置。
                let picked = rfd::FileDialog::new()
                    .set_title("选择要扫描的文件夹（可多选，会递归找出里面的音视频）")
                    .pick_folders();

                let Some(dirs) = picked else {
                    return; // 用户取消
                };

                let mut total = 0usize;
                let mut added = 0usize;
                let mut dup = 0usize;
                let mut scanned: Vec<(String, usize)> = Vec::new();
                for dir in dirs {
                    let dir_s = dir.to_string_lossy().to_string();
                    let found = config::expand_dropped_path(&dir_s);
                    let n = found.len();
                    let (a, d) = self.add_files(found);
                    total += n;
                    added += a;
                    dup += d;
                    scanned.push((dir_s, n));
                }

                // 只选了一个目录时保持原来的日志格式（用户已经习惯了）；
                // 多个目录时先给一行汇总，再逐个列出，方便核对哪个目录没扫到。
                if scanned.len() == 1 {
                    let (dir_s, n) = &scanned[0];
                    self.push_log(format!(
                        "[文件夹] 扫描 {dir_s}：找到 {n} 个音视频，新增 {added} 个，跳过 {dup} 个重复"
                    ));
                } else {
                    self.push_log(format!(
                        "[文件夹] 扫描 {} 个文件夹：共找到 {total} 个音视频，新增 {added} 个，跳过 {dup} 个重复",
                        scanned.len()
                    ));
                    for (dir_s, n) in &scanned {
                        self.push_log(format!("  · {dir_s}（{n} 个）"));
                    }
                }
            }


            Msg::Dropped(data) => {
                // 从资源管理器拖进来的东西。
                // 可能是若干文件，也可能是文件夹 —— 文件夹要递归展开。
                let raw: Vec<String> = match data {
                    DroppedData::StorageItems(items) => {
                        items.into_iter().map(|it| it.path).collect()
                    }
                    DroppedData::Text(_) => {
                        // 拖进来的是纯文本（比如从记事本拖一段路径）。
                        // 当路径用 —— 用户拖文本到这儿，八成就是这个意思。
                        self.push_log("[拖入] 收到的是文本，不是文件，已忽略");
                        return;
                    }
                    DroppedData::Unsupported => {
                        self.push_log("[拖入] 不支持这种内容");
                        return;
                    }
                };

                if raw.is_empty() {
                    return;
                }

                // 展开目录
                let mut flat = Vec::new();
                let mut dirs = 0usize;
                for p in &raw {
                    let expanded = config::expand_dropped_path(p);
                    if expanded.is_empty() {
                        self.push_log(format!("[拖入] 跳过（不是文件也不是目录，或目录里没有音视频）: {p}"));
                    } else if expanded.len() > 1 || std::path::Path::new(p).is_dir() {
                        dirs += 1;
                    }
                    flat.extend(expanded);
                }

                let (added, dup) = self.add_files(flat);
                self.push_log(format!(
                    "[拖入] 收到 {} 项（其中文件夹 {dirs} 个），新增 {added} 个，跳过 {dup} 个重复",
                    raw.len()
                ));
            }

            Msg::SelectFile(i) => {
                // 再点一次同一行就取消选中 —— 列表是自绘的，没有 ListView
                // 帮忙管选中态，这点小交互得自己实现。
                self.selected_index = if self.selected_index == Some(i) { None } else { Some(i) };
            }

            Msg::RemoveSelected => {
                match self.selected_index {
                    Some(i) if i < self.files.len() => {
                        let removed = self.files.remove(i);
                        self.selected_index = None;
                        self.push_log(format!("[文件] 已移除: {removed}"));
                    }
                    _ => {
                        self.push_log("[文件] 没有选中任何文件");
                    }
                }
            }

            Msg::ClearList => {
                if self.files.is_empty() {
                    self.push_log("[文件] 列表已经是空的");
                } else {
                    let n = self.files.len();
                    self.files.clear();
                    self.selected_index = None;
                    self.push_log(format!("[文件] 已清空 {n} 个文件"));
                }
            }

            Msg::Tick => {
                // 启动后第一次心跳：对话框已经挂载完毕，这时候再打开才不会被
                // reactor 的"首帧 is_open"逻辑吞掉。见 pending_restore_prompt 的注释。
                if self.pending_restore_prompt {
                    self.pending_restore_prompt = false;
                    self.show_restore_dialog = true;
                }

                // 只在跑着的时候算；停下来之后时间就冻住，
                // 这样用户还能看到「这次一共花了多久」。
                if self.running {
                    // 暂停期间计时冻结（恢复时会把 started_at 往后推）。
                    if !self.paused {
                        if let Some(t) = self.started_at {
                            self.elapsed_secs = t.elapsed().as_secs();
                        }
                    }
                    // ★ 解析 tqdm 进度条，拿到【当前阶段内部】的精确进度。
                    //
                    //   separator 和 faster-whisper 都会输出形如
                    //       12%|█▏  | 87/706 [01:23<09:53, 1.04it/s]
                    //   的行。`87/706` 是当前文件的处理进度，拿它做
                    //   stage_ratio 比数文件细得多（一个长视频只算 0 或 1）。
                    // ★ 从独立通道读 tqdm 进度（pipeline 的回调里已经算好了比例）。
                    //
                    //   不再从 live_line 里解析 —— 那条通道现在专门给
                    //   「当前动作」和「实时预览」用，两者互不干扰。
                    // ★ 不做 >0 判断：tqdm 的第一帧本来就是 0/1376.54（f=0.0），
                    //   跳过它并不会省什么，反而让「刚开始」和「一直没数据」
                    //   无法区分。直接写进去，0.0 也是合法进度。
                    if self.stage >= 3 && self.stage <= 4 {
                        let f = self.live_frac.lock().map(|v| *v).unwrap_or(0.0);
                        let mut frac = f.clamp(0.0, 1.0);

                        // ★ 第4步：优先用「字幕覆盖到的音频位置 / 音频总长」。
                        //
                        //   字幕行自带时间戳，而且那个时间戳就是原始音频时间轴
                        //   （分离结果被填回了原长度）。所以"最后一条字幕到哪"
                        //   就等于"处理到音频的第几秒"，比 tqdm 的语音秒数
                        //   更直观 —— 用户能直接对着播放器看进度。
                        //
                        //   实测（VEC-738，音频 6392.5 秒）：
                        //     第1501条 → 1432.3s (22.4%)
                        //     第1651条 → 3529.3s (55.2%)   推进 13.6%/50条
                        //     第1851条 → 6364.3s (99.6%)
                        //   完全非线性，正好反映真实处理速度。
                        //
                        //   ★ 取 max(字幕比例, tqdm 比例)：
                        //     静音段字幕不动但 tqdm 还在推进，用 tqdm 兜底，
                        //     进度条不会冻结；语音密集段字幕推得快，用字幕。
                        if self.stage == 4 {
                            let sub_end = self.live_sub_end.lock().map(|v| *v).unwrap_or(0.0);
                            if self.audio_total > 0.0 && sub_end > 0.0 {
                                let sub_frac = (sub_end / self.audio_total).clamp(0.0, 1.0);
                                frac = frac.max(sub_frac);
                            }
                        }
                        self.stage_frac = frac;

                        // ── tqdm 的 ETA：做指数平滑 ──
                        //
                        //   tqdm 每 3 秒按「当时平均速度」重算一次剩余，
                        //   速度波动会让它来回跳。这里用 60/40 加权：
                        //   新值占 40%，历史占 60% —— 趋势跟得上，
                        //   单帧抖动被抹平。
                        //
                        //   平滑后的值仍然保天然"越跑越准"的特性：
                        //   越往后样本越多，tqdm 自己的 ETA 就越接近真值，
                        //   EMA 跟着一起收敛。
                        let raw = self.live_remain.lock().map(|v| *v).unwrap_or(-1.0);
                        if raw >= 0.0 {
                            self.had_eta = true;
                            // ★ 只在新样本到达时才做 EMA。
                            //
                            //   Tick 是 100ms 一次，而 tqdm 每 3 秒才刷新 ——
                            //   也就是说同一个 raw 值会被读 30 次。
                            //   如果每次都做 EMA，等于对同一个数反复平滑 30 遍，
                            //   EMA 会远远落后于真实变化（3 秒才追一次，追的时候
                            //   又被自己 60% 的历史拽住）。实测表现为"ETA 像冻住"。
                            //
                            //   改成只在 raw 真正变化时才算一次，
                            //   这样 EMA 的节奏和 tqdm 的刷新对齐。
                            if (raw - self.last_raw_remain).abs() > 0.01 {
                                self.tqdm_ema = if self.tqdm_ema <= 0.0 {
                                    raw // 第一个样本，直接用
                                } else {
                                    // ★ 系数从 0.6/0.4 收紧到 0.85/0.15。
                                    //
                                    //   实测 tqdm 的 ETA 相邻帧最大跳变 336 秒
                                    //   （Whisper 遇到长静音或慢段时会卡十几秒），
                                    //   0.4 的新值权重让显示跟着一起乱跳。
                                    //   用真实序列模拟过：改成 0.85/0.15 之后，
                                    //   相邻帧平均跳变从 27.7 秒降到 7.1 秒。
                                    // ★ 系数按「时间常数」定，不是拍脑袋。
                                    //   tau ≈ 1/alpha 个 tick。Tick 100ms，
                                    //   alpha=0.05 → tau ≈ 20 tick = 2 秒，
                                    //   比日志里观测到的 12 秒卡顿短，
                                    //   但足够把单帧的 336 秒跳变压下去。
                                    //   （千问建议 3~5 秒，这里取 2 秒折中：
                                    //     太大会让 ETA 明显滞后真实值。）
                                    self.tqdm_ema * 0.95 + raw * 0.05
                                };
                                self.tqdm_remain = self.tqdm_ema;
                                self.last_raw_remain = raw;
                            }
                        }
                    }

                    // ★ 第4步跑到哪里，预览区就跟到哪里。
                    //
                    //   原来预览是在 Msg::StageDone 里一次性解析的 ——
                    //   那要等【整个第4步跑完】才填充，也就是说整段识别
                    //   期间预览框一直是「(等待开始)」。用户截图反馈
                    //   "实时翻译预览没反应"，就是这个原因。
                    //
                    //   现在改成在心跳（100ms 一次）里读 live_line：
                    //   它存的是脚本最新的有信息量的那一行，第4步的字幕
                    //   行格式是 "[596.17s -> 598.34s]: 文本"，
                    //   拿正则解析后追加进 preview 就是实时的。
                    // ★ 从 VAD 的日志行提取【本文件的语音时长】：
                    //      `VAD: 72 段 / 201.3s（占 67.1%）`
                    //   这是 ETA 的关键输入 —— 分离/识别的耗时都正比于
                    //   语音时长，而不是音频总时长。
                    if self.stage == 2 {
                        let line = self
                            .live_line
                            .lock()
                            .map(|l| l.clone())
                            .unwrap_or_default();
                        if let Some(c) = config::vad_speech_re().captures(&line) {
                            if let Ok(v) = c[1].parse::<f64>() {
                                if v > 0.0 {
                                    self.cur_speech = v;
                                }
                            }
                        }
                    }

                    if self.stage == 4 {
                        let line = self
                            .live_line
                            .lock()
                            .map(|l| l.clone())
                            .unwrap_or_default();
                        if !line.is_empty() && line != self.last_preview_line {
                            if let Some(c) = config::translation_re().captures(&line) {
                                let s = c.get(1).map(|m| m.as_str()).unwrap_or("?");
                                let e = c.get(2).map(|m| m.as_str()).unwrap_or("?");
                                let t = c.get(3).map(|m| m.as_str().trim()).unwrap_or("");
                                self.preview.push(format!("[{s}s → {e}s]  {t}"));
                                // 只留最近若干条，别让内存无限涨
                                let keep = 200usize;
                                if self.preview.len() > keep {
                                    let drop = self.preview.len() - keep;
                                    self.preview.drain(0..drop);
                                }
                                self.last_preview_line = line;
                            }
                        }
                    }

                    // 总进度：每 100ms 重算一次（队列级三层加权）
                    self.recompute_progress();

                    // 预估剩余：算出来之后限制变化速率再存起来。
                    match self.compute_eta() {
                        Some(raw) => {
                            // 限速只在【同一状态内】生效。
                            //
                            //   阶段切换（StageDone）和文件切换时已经把
                            //   eta_display 清成 None 了，所以这里走到
                            //   Some(prev) 分支时，prev 一定来自同一个阶段、
                            //   同一个文件 —— 它才是有意义的锚点。
                            //
                            //   曾经加过「阶段切换后前 10 秒不限速」的特例，
                            //   那是用时间戳硬凑：既没解决"锚点错了"的本质，
                            //   又在 10 秒后突然开始限速，行为不连续。

                            let smoothed = match self.eta_display {
                                Some(prev) => {
                                    // ★ 限速必须【按秒】换算，不能直接按 Tick。
                                    //
                                    // ★ 用【绝对增量】而不是百分比。
                                    //
                                    //   百分比的毛病：ETA 越大，一次允许的跳变越大。
                                    //   233 秒时 ×1.12 一次就涨 28 秒，越到后面越跳。
                                    //
                                    //   绝对增量：每秒最多涨 8 秒、降 25 秒。
                                    //   Tick 是 100ms（一秒 10 次），所以每 tick
                                    //   涨 0.8 秒、降 2.5 秒。
                                    //   这样即使内部算出来 84 → 233，
                                    //   显示也只会以每秒 8 秒的速度爬上去（约 18 秒追平），
                                    //   用户看到的是平滑过渡而不是跳跃。
                                    let up = prev as f64 + 0.8;
                                    let down = (prev as f64 - 2.5).max(0.0);
                                    raw.clamp(down as u64, up as u64)
                                }
                                None => raw,
                            };
                            self.eta_display = Some(smoothed);
                        }
                        None => self.eta_display = None,
                    }
                } else {
                    // ★ 已经跑完/没在跑：清掉 ETA。
                    //
                    //   原来这段 match 在 if self.running 块里，
                    //   一旦 running 变 false 就再也不更新 eta_display，
                    //   界面上会一直挂着结束前最后一帧的「约 34 秒」。
                    //   实测用户看到"跑完了还显示还剩34秒"。
                    self.eta_display = None;
                }
            }

            Msg::DismissNoFileWarning => {
                self.show_no_file_warning = false;
            }

            Msg::RestoreQueue => {
                // 用户选了「恢复」：把文件列表填回去，从上次的位置继续。
                let n = self.pending_queue.len();
                let idx = self.pending_idx.min(n);
                self.files = std::mem::take(&mut self.pending_queue);
                self.current_idx = idx;
                self.pending_idx = 0;
                self.show_restore_dialog = false;
                self.pending_restore_prompt = false;
                // 恢复后重存一次：load_queue 可能过滤掉了已删除的文件，
                // 现在的列表才是准确的。
                self.save_queue();
                self.push_log(format!(
                    "[队列] 已恢复 {} 个文件（从第 {} 个继续）",
                    n,
                    idx + 1
                ));
                if let Some(next) = self.files.get(idx) {
                    let name = std::path::Path::new(next)
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| next.clone());
                    self.push_log(format!("[队列] 下一个: {name}"));
                }
                // 已经全部处理完的队列：提示一下，别让用户以为还要跑
                if idx >= n {
                    self.push_log("[队列] 上次已经把列表跑完了，可以直接重新开始");
                }
            }

            Msg::ClearQueue => {
                // 用户选了「清空」：删掉持久化文件，队列归零
                Self::clear_queue_file();
                self.files.clear();
                self.current_idx = 0;
                self.pending_queue.clear();
                self.pending_idx = 0;
                self.show_restore_dialog = false;
                self.pending_restore_prompt = false;
                self.save_queue();
                self.push_log("[队列] 已清空");
            }

            Msg::ExportLog => {
                self.export_log();
            }

            Msg::WindowSized(w, h) => {
                // 只在真的变了才赋值，避免无意义的重复渲染
                if (self.window_width - w).abs() > 0.5 {
                    self.window_width = w;
                }
                if (self.window_height - h).abs() > 0.5 {
                    self.window_height = h;
                }
            }
        }
    }

    fn view(&self, _input: &(), context: &mut ViewContext<Self>) -> View {
        // 窗口材质用 Acrylic —— 磨砂玻璃，比 Mica 更"雾面玻璃"。
        context.window_visuals(
            WindowVisuals::new()
                // 标题栏是非客户区，由 DWM 画，Border 铺不到它。
                // 上一版用 Acrylic 是为了让标题栏和客户区同源，现在客户区
                // 改成不透明米色，只能反过来：用 Mica + Light 把标题栏
                // 也推到浅色，客户区再用 Border 盖成米色，两边才不割裂。
                .backdrop(WindowBackdrop::Mica)
                .theme(WindowTheme::Light)
                // 尺寸按屏幕算（固定 1280x720 逻辑像素，见 preferred_window_size）
                .client_size(Self::preferred_window_size().0, Self::preferred_window_size().1)
                .constraints(WindowConstraints {
                    min_width: Some(900.0),
                    min_height: Some(600.0),
                    ..Default::default()
                }),
        );
        context.window_title(config::app_title());

        // 监听窗口尺寸变化，把新宽度送进状态 —— 这样左右两栏能跟着窗口伸缩。
        // 注意 sender 是 !Send 的，但 on_window_size 只要求 Fn + 'static、
        // 不要求 Send，所以这里可以直接捕获。
        let size_sender = context.sender();
        context.on_window_size(move |size: WindowSize| {
            let _ = size_sender.send(Msg::WindowSized(size.width, size.height));
        });

        // 把 sender 挂到 main.rs 的心跳定时器上。
        // 定时器每秒调一次 Msg::Tick，界面靠它刷新「已用时间」。
        // （每次渲染都设一遍，因为 sender 可能换。）
        crate::tick::set_sender(context.sender());

        // ── 标题区 ──
        // 原稿是「小 logo + 标题 + 副标题」横排。logo 用 exe 旁边的
        // assets\icon.png；没有的话退回一个空占位（源码目录直接跑时就属于这种情况）。
        let logo: View = match config::icon_path() {
            Some(path) => {
                let uri = format!("file:///{}", path.replace('\\', "/"));
                match Image::new().source(uri) {
                    Ok(img) => View::from(
                        img.width(26.0)
                            .height(26.0)
                            .vertical_alignment(VerticalAlignment::Center),
                    ),
                    Err(_) => View::from(TextBlock::new().text("").font_size(1.0)),
                }
            }
            None => View::from(TextBlock::new().text("").font_size(1.0)),
        };

        // 副标题：「本地转写 · 离线翻译 · RTX 5070 加速 · 第 56 / 137 个文件」
        //   显卡短名从 device_status 里抠（"兼容模式（Vulkan: NVIDIA GeForce
        //   RTX 5070）" → "RTX 5070"），检测中或纯 CPU 时这一段直接不出现。
        let subtitle = {
            let total = self.files.len();
            let mut parts: Vec<String> = vec!["本地转写".to_string(), "离线翻译".to_string()];
            if let Some(gpu) = gpu_short_name(&self.device_status) {
                parts.push(format!("{gpu} 加速"));
            }
            if total > 0 {
                let idx = if self.running {
                    (self.current_idx + 1).min(total)
                } else {
                    total
                };
                parts.push(format!("第 {idx} / {total} 个文件"));
            }
            parts.join(" · ")
        };

        let title_block = StackPanel::new()
            .grid_row(0)
            .grid_column(0)
            .grid_column_span(2)
            .orientation(Orientation::Horizontal)
            .spacing(12.0)
            .children((
                logo,
                StackPanel::new().spacing(2.0).children((
                    TextBlock::new()
                        .text("ASMR 日语语音 → 中文翻译")
                        .font_size(26.0)
                        .font_weight(FontWeight::SEMI_BOLD)
                        .foreground(INK),
                    // 副标题：本地 / 离线 / 显卡 / 队列位置，一行说完。
                    //   版本号不写在这里 —— 窗口标题栏已经有 "v1.23.0"，
                    //   副标题再重复一遍纯属占位。
                    TextBlock::new()
                        .text(subtitle)
                        .font_size(12.0)
                        .foreground(INK2),
                )),
            ));

        // ── 卡片 1：环境 ──
        let copy_check = CheckBox::new()
            .is_checked(self.copy_to_video)
            .on_is_checked_changed(context.callback(Msg::SetCopyToVideo))
            .content("字幕直接放到视频旁边（播放器可自动加载）");

        let force_check = CheckBox::new()
            .is_checked(self.force_redo)
            .on_is_checked_changed(context.callback(Msg::SetForceRedo))
            .content("强制重新处理（删除已有字幕）");

        let out_dir = if self.copy_to_video {
            "跟随视频所在目录（自动）".to_string()
        } else {
            self.output_dir.clone()
        };

        // 输出目录那一行的右半部分：
        //   勾选「放视频旁边」时 —— 位置是自动决定的，不给改，放个空占位；
        //   取消勾选时         —— 显示「更改…」，让用户挑目录。
        //
        // 两个分支都转成 View 是为了类型一致（if/else 两支必须同类型）。
        let out_dir_action: View = if self.copy_to_video {
            View::from(TextBlock::new().text("").font_size(1.0))
        } else {
            View::from(
                tap_button("更改…", context.message(Msg::BrowseOutputDir)),
            )
        };

        // 环境卡片用【两列】排。
        //
        // 原来是 4 行竖排（工作目录 / 输出目录 / 设备状态 / 勾选框），
        // 光这一张卡就占约 250px —— 直接把底部的「进度」和「开始处理」
        // 挤出 1400x900 的窗口。改成两列后高度减半。
        let env_card = card(
            "环境",
            StackPanel::new()
                .orientation(Orientation::Horizontal)
                .spacing(72.0)
                .children((
                    // 左列：路径信息 + 两个勾选项（原稿勾选项在左列最下方）
                    //   width 是必须的：不给宽度时这一列只占内容宽度，
                    //   右列会被顶到左边，两列就挤成一条（原稿是左右近乎对半）。
                    StackPanel::new()
                        .width(520.0)
                        .spacing(12.0)
                        .children((
                        field("工作目录", config::work_dir()),
                        // 输出目录：标签在上，值 + 「更改…」在下，两行左对齐
                        StackPanel::new().spacing(2.0).children((
                            TextBlock::new().text("输出目录").font_size(11.0).foreground(INK2),
                            StackPanel::new()
                                .orientation(Orientation::Horizontal)
                                .spacing(8.0)
                                .children((
                                    TextBlock::new().text(&out_dir).font_size(14.0).foreground(INK),
                                    out_dir_action,
                                )),
                        )),
                        StackPanel::new().spacing(6.0).children((copy_check, force_check)),
                    )),
                    // 右列：设备状态 + 处理模式（横排胶囊）
                    StackPanel::new().spacing(12.0).children((
                        // 设备状态：标签在上，(绿点 + 文本) 在下 —— 原稿就是这样
                        StackPanel::new().spacing(3.0).children((
                            TextBlock::new().text("设备状态").font_size(11.0).foreground(INK2),
                            StackPanel::new()
                                .orientation(Orientation::Horizontal)
                                .spacing(6.0)
                                .children((
                                    Border::new()
                                        .width(8.0)
                                        .height(8.0)
                                        .corner_radius(4.0)
                                        .background(OK_GREEN)
                                        .vertical_alignment(VerticalAlignment::Center)
                                        .content(TextBlock::new().text("").font_size(1.0)),
                                    TextBlock::new()
                                        .text(&self.device_status)
                                        .font_size(14.0)
                                        .foreground(INK),
                                )),
                        )),
                        StackPanel::new().spacing(8.0).children((
                            TextBlock::new().text("处理模式").font_size(11.0).foreground(INK2),
                            StackPanel::new()
                                .orientation(Orientation::Horizontal)
                                .spacing(8.0)
                                .children((
                                    mode_chip(
                                        "全量模式",
                                        self.mode == pipeline::RunMode::Full,
                                        self.mode_full_ok,
                                        context.message(Msg::SetModeFull),
                                    ),
                                    mode_chip(
                                        "精简模式",
                                        self.mode == pipeline::RunMode::Lite,
                                        self.mode_lite_ok,
                                        context.message(Msg::SetModeLite),
                                    ),
                                    mode_chip(
                                        "兼容模式",
                                        self.mode == pipeline::RunMode::Compat,
                                        true,
                                        context.message(Msg::SetModeCompat),
                                    ),
                                )),
                            TextBlock::new()
                                .text("全量模式需 N 卡显存 ≥ 8 GB；精简模式 ≥ 6 GB；兼容模式用于低于 6 GB / 非 N 独立显卡 / 核显")
                                .font_size(11.0)
                                .foreground(INK3),
                        )),
                    )),
                )),
        1, 0, 1, 2,
        );

        // ── 卡片 3：待处理文件 ──
        // 文件列表：已处理完（有字幕产物）的文件名用浅绿色标出来，
        // 方便一眼看出还剩哪些没跑。
        let file_items: Vec<(String, View)> = self
            .files
            .iter()
            .enumerate()
            .map(|(i, f)| {
                // 三态，和原稿一致：
                //   · 已处理完（已有字幕产物）→ 路径淡化 + 后面打勾
                //   · 正在处理（current_idx）→ 整行浅橙底 + 序号变橙 + 墨色字
                //   · 排队中 → 正常次要色
                let done = Self::has_subtitle(f);
                let is_current = self.running && i == self.current_idx;

                let name_color = if is_current {
                    INK
                } else if done {
                    DONE_INK
                } else {
                    INK2
                };
                let no_color = if is_current { ACCENT } else { INK3 };
                let tick = if done { "✓" } else { "" };

                // 行内用 Grid 三列，而不是横向 StackPanel：
                //   序号(Auto) | 路径(STAR) | 勾(Auto)
                // 中间列必须是 STAR 才有确定宽度，`text_trimming` 才会真的生效；
                // StackPanel 里 TextBlock 的宽度是"内容宽度"，超长时会把后面的勾
                // 直接顶出可视区（实测：勾完全看不见）。
                let inner = Grid::new()
                    .columns([GridLength::Auto, GridLength::STAR, GridLength::Auto])
                    .column_spacing(10.0)
                    .children((
                        TextBlock::new()
                            .text(format!("{}", i + 1))
                            .font_size(12.0)
                            .foreground(no_color)
                            .width(28.0)
                            .grid_column(0),
                        TextBlock::new()
                            .text(f.clone())
                            .font_size(13.0)
                            .foreground(name_color)
                            .text_trimming(TextTrimming::CharacterEllipsis)
                            .grid_column(1),
                        TextBlock::new()
                            .text(tick)
                            .font_size(13.0)
                            .foreground(OK_GREEN)
                            .grid_column(2),
                    ));

                // 每行都套一层极薄的 Border：高亮时它是浅橙块，
                // 不高亮时背景透明 —— 两种情况下内边距一致，
                // 文字的左边缘才不会在高亮行和普通行之间错位。
                // 两种"被标记"的状态，优先级：正在处理 > 选中。
                //   正在处理用浅橙（和原稿一致），选中的行用更中性的浅灰，
                //   这样同一屏里两种状态不会看混。
                let is_selected = self.selected_index == Some(i);
                let bg = if is_current {
                    ACCENT_SOFT
                } else if is_selected {
                    SUNK
                } else {
                    Color::argb(0, 0, 0, 0)
                };
                let face = Border::new()
                    .background(bg)
                    .corner_radius(7.0)
                    .padding(Thickness::xy(8.0, 2.0))
                    .content(inner);

                // 每行是一个透明按钮：点击区域交给它，外观全在 face 上。
                // 这样行高只由我们自己定的字号 + 内边距决定（26px 上下），
                // 不再受 ListViewItem 那套 40px 起跳的容器样式摆布。
                let row = Button::new()
                    .resource_overrides(
                        ResourceOverrides::new()
                            .set("ButtonBackground", Color::argb(0, 0, 0, 0))
                            .set("ButtonBackgroundPointerOver", Color::argb(14, 0, 0, 0))
                            .set("ButtonBackgroundPressed", Color::argb(26, 0, 0, 0))
                            .set("ButtonBorderThemeThickness", Thickness::uniform(0.0))
                            .set("ButtonPadding", Thickness::uniform(0.0)),
                    )
                    .horizontal_content_alignment(HorizontalAlignment::Stretch)
                    // Button 默认 MinHeight 32，会把每行撑到 48px —— 显式压到 22，
                    // 行高才由里面的字号 + 内边距说了算（约 26px）。
                    .min_height(22.0)
                    .on_click(context.callback(move |_: ()| Msg::SelectFile(i)))
                    .content(face);

                (f.clone(), row.into())
            })
            .collect();

        // 文件列表高度：随窗口高度变化，列表在卡片内部滚动。
        // 不限制高度的话 ListView 会无限长，把 Grid 的 STAR 行撑爆，
        // 结果就是整个界面出现外侧滚动条（用户截图反馈的问题）。
        // 列表最多占多高：上排卡片大约是可用高度的 1/2，扣掉卡片内边距、
        // 标题和下面那两行按钮，剩下的才是列表能用的高度。
        //   系数给小了列表太短，给大了会把按钮顶出卡片（实测 0.32 就会）。
        let list_h = ((self.window_height - 36.0) * 0.195).clamp(110.0, 720.0);

        let list_area: View = if file_items.is_empty() {
            View::from(TextBlock::new().text("（列表为空，点下面的按钮或把文件拖进来）").font_size(13.0))
        } else {
            // 【为什么不用 ListView】
            //   ListViewItem 自带 40px 起跳的最小高度，一屏只能放 4 行；
            //   而且给它设 height / max_height 都压不住 —— 实测它会按内容
            //   撑开，把下面那行按钮整个顶出卡片。ListView 上也没有
            //   resource_overrides 去改 ListViewItemMinHeight（E0599）。
            //   所以列表改成自绘：ScrollViewer + 每行一个透明按钮，
            //   行高 26px 左右，一屏能放十几行，和原稿的密度对得上。
            View::from(
                ScrollViewer::new()
                    // 必须给【确定的】高度。
                    //   试过 max_height：ScrollViewer 也会按内容撑开，
                    //   六个文件就把下面那行按钮顶出卡片了。
                    //   这个系数是按"卡片高度 - 标题 - 按钮行"倒推的经验值，
                    //   大约能完整放下 4 行；再多的部分靠内部滚动。
                    .height(list_h)
                    .vertical_scroll_bar_visibility(ScrollBarVisibility::Auto)
                    .horizontal_scroll_bar_visibility(ScrollBarVisibility::Disabled)
                    .content(
                        StackPanel::new()
                            .spacing(1.0)
                            // 文件数是动态的，children() 只吃定长元组/数组，
                            // 所以这里用 keyed_children —— 它接受任意迭代器，
                            // 顺便给了每行一个稳定的 key（file_0、file_1…），
                            // 列表重建时不会整段重刷。
                            .keyed_children(file_items.iter().enumerate().map(|(i, (_, v))| {
                                KeyedView::new(format!("file_{i}"), v.clone())
                            })),
                    ),
            )
        };

        // 四个按钮排成 2×2。
        // 一栏只有 660 宽，四个按钮挤一行放不下（实测会被压成不均的两行），
        // 所以明确排成两行两列。
        // 四个按钮排成【一行】：四个胶囊加起来约 440px，卡片可用宽度 700 出头，
        // 排两行要多吃掉 48px 高度，全给列表更划算。
        let file_buttons = StackPanel::new()
            .orientation(Orientation::Horizontal)
            .spacing(8.0)
            .children((
                tap_button("添加文件", context.message(Msg::AddFiles)),
                tap_button("添加文件夹", context.message(Msg::AddFolder)),
                tap_button("移除选中", context.message(Msg::RemoveSelected)),
                tap_button("清空列表", context.message(Msg::ClearList)),
            ));

        let files_card = card_x(
            "文件列表",
            count_pill(&self.files.len().to_string()),
            StackPanel::new().spacing(10.0).children((
                list_area,
                file_buttons,
            )),
        2, 0, 1, 1,
        );

        // ── 卡片 4：进度 ──
        let stage_txt = if self.running {
            self.stage_name()
        } else {
            String::new()
        };

        // 已用时间：调用了哪个模型、要等多久，用户看不到（脚本不输出），
        // 但一个【每秒都在变】的数字足以说明"程序还在干活"。
        // 加载 RoFormer / whisper 动辄几十秒到几分钟，没有这个的话
        // 界面是一片死寂，很容易被当成卡死。
        let elapsed_txt = if self.started_at.is_some() {
            let m = self.elapsed_secs / 60;
            let s = self.elapsed_secs % 60;
            if m > 0 {
                format!("{m} 分 {s:02} 秒")
            } else {
                format!("{s} 秒")
            }
        } else {
            String::new()
        };

        // 一句提示：跑起来之后告诉用户"慢是因为在加载模型"，
        // 免得他盯着进度条怀疑是不是出问题了。
        let ready_now = self.model_ready.load(std::sync::atomic::Ordering::Relaxed);
        let hint_txt = if self.running {
            match self.stage {
                2 => {
                    if ready_now {
                        "人声分离模型已加载，正在分离人声"
                    } else {
                        "正在加载人声分离模型 / 准备推理（首次较慢，请稍候）"
                    }
                }
                3 => {
                    if ready_now {
                        "人声检测模型已加载，正在检测人声"
                    } else {
                        "正在加载人声检测模型（首次较慢，请稍候）"
                    }
                }
                4 => {
                    if ready_now {
                        "语音识别模型已加载，正在识别并翻译"
                    } else {
                        "正在加载语音识别模型 / 准备推理（首次较慢，请稍候）"
                    }
                }
                _ => "处理中…",
            }
        } else {
            ""
        };

        // 脚本最后输出的那一行 —— 就是"它现在在干什么"。
        // 加载模型、提取音频、whisper 吐片段，都会在这里实时反映出来。
        let live_txt = self
            .live_line
            .lock()
            .map(|l| l.trim().to_string())
            .unwrap_or_default();
        // 太长就截断，卡片宽度放不下
        let live_txt = if live_txt.chars().count() > 46 {
            let cut: String = live_txt.chars().take(45).collect();
            format!("{cut}…")
        } else {
            live_txt
        };

        // 预估剩余时间。估不出来时显示"计算中…"，
        // 不要显示一个会乱跳的数字 —— 那比没有更让人不安。
        let eta_txt = match self.eta_display {
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
                    // 还在跑但估不出来（第1、2步还没拿到语音时长）
                    "—".to_string()
                } else if self.progress_value >= 99.5 {
                    // 已经跑完：明确归零，别让用户以为还有活没干
                    "0 秒".to_string()
                } else {
                    // 没在跑但也没跑完（手动停止 / 出错中止）
                    "—".to_string()
                }
            }
        };

        // ── 当前文件与队列位置 ──
        //
        //   批量跑的时候用户最想知道"现在处理的是哪个、还剩几个"。
        //   files 是完整路径，这里只取文件名，太长就中间省略。
        let queue_txt = {
            let total = self.files.len();
            // 三种状态要分开说：
            //   · 跑完了（done_stages 满 4 且 not running）→ 已完成，不是"未开始"
            //   · 正在跑 → 当前文件 + 队列位置
            //   · 没跑过 → 队列共 N 个
            // 原来只判断 running，于是全部处理完之后仍然显示"（未开始）"，
            // 用户看到"待处理文件 3 / 队列共 3 个文件（未开始）"会以为没跑成。
            let finished = !self.running
                && self.done_stages >= pipeline::STAGES.len()
                && self.progress_value >= 99.5;
            if total == 0 {
                String::new()
            } else if finished {
                format!("已完成 {total} 个文件的处理")
            } else if self.running {
                let idx = self.current_idx.min(total.saturating_sub(1));
                let name = self
                    .files
                    .get(idx)
                    .and_then(|f| std::path::Path::new(f).file_name().map(|x| x.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "?".to_string());
                let shown = if name.chars().count() > 42 {
                    let head: String = name.chars().take(28).collect();
                    let tail: String = name.chars().skip(name.chars().count() - 11).collect();
                    format!("{head}…{tail}")
                } else {
                    name
                };
                let left = total.saturating_sub(idx + 1);
                format!("正在处理：{shown}    （第 {}/{} 个，还剩 {} 个）",
                        idx + 1, total, left)
            } else {
                format!("队列共 {} 个文件（未开始）", total)
            }
        };

        // 两列排：5 个字段竖着放不进 720 高的窗口（底部会被裁）
        let progress_card = card(
            "进度",
            StackPanel::new().spacing(6.0).children((
                StackPanel::new()
                    .orientation(Orientation::Horizontal)
                    .spacing(24.0)
                    .children((
                        StackPanel::new().spacing(4.0).children((
                            field("当前阶段", &stage_txt),
                            field("已用时间", &elapsed_txt),
                        )),
                        StackPanel::new().spacing(4.0).children((
                            field("当前文件预估剩余", &eta_txt),
                            field("总进度", &format!("{:.0}%", self.progress_value)),
                        )),
                    )),
                // ── 进度条：手搓，不用 ProgressBar ──
                //
                // 为什么不用 ProgressBar：reactor 的 ProgressBar 只有
                // minimum/maximum/value，颜色走系统 accent（WinUI 的蓝），
                // 而 resource_overrides 只挂在 Button 上，改不了它的前景。
                // 这条进度条是整页唯一的「进行中」信号，必须是赤陶橙，
                // 所以用两层 Border 自己画：轨道 + 按百分比算宽度的填充。
                {
                    let bar_w = 460.0_f64;
                    let fill_w = (bar_w * (self.progress_value / 100.0)).clamp(0.5, bar_w);
                    View::from(
                        Border::new()
                            .width(bar_w)
                            .height(8.0)
                            .corner_radius(4.0)
                            .background(Color::rgb(0xe8, 0xe2, 0xd5))
                            .content(
                                Border::new()
                                    .width(fill_w)
                                    .height(8.0)
                                    .corner_radius(4.0)
                                    .background(ACCENT)
                                    .horizontal_alignment(HorizontalAlignment::Left)
                                    .content(TextBlock::new().text("").font_size(1.0)),
                            ),
                    )
                },
                // live_txt 在跑批时会给出三行（完成 / 正在处理 / 正在加载），
                // 三行会把卡片底边顶破，这里限两行 + 省略号。
                StackPanel::new().spacing(3.0).children((
                    TextBlock::new().text("当前动作").font_size(12.0).foreground(INK2),
                    TextBlock::new()
                        .text(live_txt.clone())
                        .font_size(14.0)
                        .foreground(INK)
                        .max_lines(2)
                        .text_trimming(TextTrimming::CharacterEllipsis),
                )),
                // 队列位置和"正在加载模型"这类提示合成一行：
                //   进度卡这已经是最后一行的高度了，两行会把卡片底边顶破
                //   （实测窗口 860 逻辑高下，多一行就裁掉最后一个 TextBlock）。
                TextBlock::new()
                    .text(if hint_txt.is_empty() {
                        queue_txt.clone()
                    } else {
                        format!("{queue_txt}  ·  {hint_txt}")
                    })
                    .font_size(12.0)
                    .foreground(INK3)
                    .text_wrapping(TextWrapping::Wrap),
            )),
        3, 1, 1, 1,
        );

        // ── 卡片 5：实时翻译预览 ──
        let pv_total = self.preview.len();
        let pv_shown: Vec<String> = self
            .preview
            .iter()
            .skip(pv_total.saturating_sub(PREVIEW_SLOTS))
            .cloned()
            .collect();
        let mut pv = pv_shown;
        while pv.len() < PREVIEW_SLOTS {
            pv.push(String::new());
        }
        let pv_hidden = pv_total.saturating_sub(PREVIEW_SLOTS);
        let pv_tail = if pv_total == 0 {
            "（等待开始）".to_string()
        } else if pv_hidden > 0 {
            format!("…（只显示最近 {PREVIEW_SLOTS} 条，另有 {pv_hidden} 条未显示）")
        } else {
            format!("（共 {pv_total} 条）")
        };

        let preview_card = card(
            "实时翻译预览",
            inner_box(StackPanel::new().spacing(2.0).children((
                TextBlock::new().text(pv[0].clone()).font_size(13.0).foreground(INK),
                TextBlock::new().text(pv[1].clone()).font_size(13.0).foreground(INK),
                TextBlock::new().text(pv[2].clone()).font_size(13.0).foreground(INK),
                TextBlock::new().text(pv[3].clone()).font_size(13.0).foreground(INK),
                TextBlock::new().text(pv[4].clone()).font_size(13.0).foreground(INK),
                TextBlock::new().text(pv[5].clone()).font_size(13.0).foreground(INK),
                TextBlock::new().text(pv_tail).font_size(11.0).foreground(INK3),
            ))),
        2, 1, 1, 1,
        );

        // ── 卡片 6：运行日志 ──
        let total = self.logs.len();
        let shown: Vec<String> = self
            .logs
            .iter()
            .skip(total.saturating_sub(LOG_SLOTS))
            .cloned()
            .collect();
        let mut slots = shown;
        while slots.len() < LOG_SLOTS {
            slots.push(String::new());
        }
        let hidden = total.saturating_sub(LOG_SLOTS);
        // 原来这里还有一行 tail（"…只显示最近 N 行，另有 M 行未显示"），
        // 但它是个独立的 TextBlock，即使内容为空也照样占一行高度 ——
        // 日志卡本来就差一行放不下，宁可把这句话挪进卡片标题。
        let log_title = if hidden > 0 {
            format!("运行日志　·　另有 {hidden} 行，可导出查看")
        } else {
            "运行日志".to_string()
        };

        let logs_card = card(
            &log_title,
            StackPanel::new().spacing(6.0).children((
                // 日志区只有固定几个槽位，翻不上去。加个导出按钮，
                // 把内存里的完整日志（最多 LOG_KEEP 行）写成文件，
                // 之后用记事本随便翻 —— 排查"某一步花了多久"就靠它。
                tap_button("导出完整日志", context.message(Msg::ExportLog)),
                inner_box(StackPanel::new().spacing(2.0).children((
                    TextBlock::new().text(slots[0].clone()).font_size(12.0).foreground(INK2),
                    TextBlock::new().text(slots[1].clone()).font_size(12.0).foreground(INK2),
                    TextBlock::new().text(slots[2].clone()).font_size(12.0).foreground(INK2),
                    TextBlock::new().text(slots[3].clone()).font_size(12.0).foreground(INK2),
                    TextBlock::new().text(slots[4].clone()).font_size(12.0).foreground(INK2),
                    // 注意：这里的槽位下标是【写死的】，数量和 LOG_SLOTS 必须一致。
                    //   刚从 6 改成 5 时忘了删掉第 6 个，程序一启动就
                    //   `index out of bounds: the len is 5 but the index is 5`
                    //   直接 panic 退出 —— 改槽位数时先回来核对这里。
                ))),
            )),
        3, 0, 1, 1,
        );

        // ── 底部按钮 ──
        //
        // ★ grid_row / grid_column 必须在 .children() 【之前】调：
        //   children() 返回的是 View，View 上没有这些方法（编译报 E0599）。
        //   先定好位置，挂载时 reactor 会从元素状态里读出来。
        let buttons = StackPanel::new()
            .grid_row(0)
            .grid_column(0)
            .orientation(Orientation::Horizontal)
            .spacing(8.0)
            .children((
                // ★ 按钮要透明，但 Button 上没有 .background() ——
                //   LayoutControl 只提供尺寸/对齐/透明度/边距，没有背景。
                //   Button 自己的 API 里也没有 background。
                //
                //   可行的是 resource_overrides：WinUI 的按钮模板通过
                //   ButtonBackground / ButtonBackgroundPointerOver /
                //   ButtonBackgroundPressed 这几个主题资源取背景刷，
                //   覆盖掉它们就等于改了按钮底色。
                //
                //   ⚠ 必须在 .content() 【之前】调：content() 返回 View，
                //     而 resource_overrides 在具体控件上（同 grid_row 的道理）。
                // 主操作：赤陶橙实心 + 白字 + 前置小圆点；处理中变弱色并禁用。
                //   颜色在调用处决定，不走 WinUI 的 disabled 视觉（见 pill_button 注释）。
                pill_button(
                    if self.running { "处理中…" } else { "开始处理  ➜" },
                    true,
                    if self.running { SUNK } else { ACCENT_DEEP },
                    if self.running { SUNK } else { ACCENT_DEEP },
                    if self.running { INK3 } else { Color::rgb(0xff, 0xff, 0xff) },
                    !self.running,
                    context.message(Msg::Start),
                ),
                // 停止：浅橙底 + 深橙字，和主操作同一色系但明显更弱
                pill_button(
                    "停止",
                    false,
                    SUNK,
                    CARD_LINE,
                    ACCENT_INK,
                    true,
                    context.message(Msg::Stop),
                ),
                // 暂停 / 继续（v1.20）。暂停用 SuspendThread 挂起子进程，
                // 进度和计时都冻结；点「继续」恢复。
                pill_button(
                    if self.paused { "继续  ▶" } else { "暂停  ❚❚" },
                    false,
                    CARD,
                    CARD_LINE,
                    INK2,
                    true,
                    context.message(Msg::TogglePause),
                ),
            ));

        // 布局历史：这里原来用横向 StackPanel 做两栏。
        //
        // ★ 为什么以前不用 Grid：
        //   reactor 的 grid_row / grid_column 只实现给 LayoutControl 的
        //   【具体控件】，而 card() 返回的是擦掉类型的 View —— 直接对
        //   card() 的返回值调 grid_row 会报 E0599。
        //
        // ★ 现在为什么能用了：
        //   把定位参数塞进 card() 内部，在 .children() 之前调 grid_row。
        //   位置被记进元素状态，挂载时读出来 —— 等价于"对具体控件调"。
        //   同理 title_block 也是在自己的构造链上设的 grid_row。

        // ── 卡片区：一个 4行 x 2列 的 Grid ──
        //
        // 为什么从"两栏 StackPanel"改成 Grid：
        //   StackPanel 里的卡片高度永远等于内容高度 —— 窗口拉高，卡片不变，
        //   下面留一大片空白；用户要的是【卡片随窗口一起长】。
        //   放进 Grid 的 STAR 行/列就不一样了：卡片会平分可用空间，
        //   窗口变大卡片跟着变大（日志/预览区能显示更多行）。
        //
        //   行：0=标题(Auto) 1=环境(Auto) 2=上排卡片(STAR) 3=下排卡片(STAR)
        //   列：两列各占一半(STAR)
        //
        // 注意每个元素的位置是在它自己的构造链上设的（card() 的四个参数、
        // title_block 的 grid_row 等），因为 View 上没有 grid_row。
        let content = Grid::new()
            .rows([
                GridLength::Auto,
                GridLength::Auto,
                // 上排 1.15 : 下排 1。
                //   窗口高度已经提到 1000 逻辑，上下两排共有 620 左右；
                //   日志卡（标题 + 导出按钮 + 五行日志）吃 220，再多给它
                //   也不会用上，所以把富余的高度拨给文件列表。
                GridLength::Star(1.15),
                GridLength::STAR,
            ])
            .columns([GridLength::STAR, GridLength::STAR])
            .row_spacing(12.0)
            .column_spacing(12.0)
            .children((title_block, env_card, files_card, preview_card, logs_card, progress_card));

        // ── 拖放 ──
        // 拖放能力在 Border 上（PropertyId::BorderAllowDrop），不在窗口上。
        // 所以整个内容要包进一个 Border 才能接住从资源管理器拖过来的文件。
        let drop_policy = DragDropPolicy::new().storage_items(
            DragDropAction::new(DragDropOperation::Copy).caption("添加到待处理列表"),
        );

        // ── 内容区：占满窗口的剩余高度，装不下就出滚动条 ──
        //
        // ★ 为什么在 .content() 【之前】调 grid_row：
        //   grid_row / grid_column 只实现在具体控件上（ScrollViewer ✓），
        //   而 .content() 会返回 View，之后就没这些方法了。
        //   先设好位置，reactor 会把它们记进元素状态，挂载时再读出来。
        // 外层不再需要滚动条：内容高度由 Grid 的 STAR 行分配，
        // 文件列表在卡片内部自己滚。这里禁用外层滚动条，
        // 避免出现「卡片内一条、界面外侧又一条」的双滚动条。
        let scrolled = ScrollViewer::new()
            .grid_row(0)
            .grid_column(0)
            .vertical_scroll_bar_visibility(ScrollBarVisibility::Disabled)
            .horizontal_scroll_bar_visibility(ScrollBarVisibility::Disabled)
            .content(content);

        // ── 按钮行：已经在上面的 buttons 定义里设好 grid_row(1) 了 ──

        // 两行：内容行吃掉所有剩余高度（STAR），按钮行只占自己需要的高度（Auto）。
        // 这样无论窗口被拉多矮，按钮永远钉在底部。
        //
        // ★ 宽高都要显式给。
        //
        //   高度：减去最外层 Border 的上下 padding(20*2)，剩下的才是 Grid
        //         能用的高度。给了确定高度，STAR 行才有确定的剩余量，
        //         按钮那一行才真正钉住。
        //
        //   宽度：同样要减左右 padding(20*2)。【只给高度不给宽度】的时候，
        //         Grid 会按"父容器可用宽度"铺 —— 而父容器是带 padding 的
        //         Border，于是内容比可视区宽出 40px，右侧被裁、还多出
        //         一条横向滚动条。用户截图反馈"又显示不全了"，就是这里。
        //
        //   注意 ScrollViewer 的横向滚动条是 Disabled 的 —— 所以溢出
        //   不是变成可滚动，而是【直接被裁掉】，更难发现。
        // 按钮行右侧的状态文案（原稿右下角那行）：
        //   「已用 112 分 51 秒 · 剩余约 81 个文件」
        //   running 时给已用时间和剩余文件数，空闲时给队列总数量。
        let footer_txt = {
            let total = self.files.len();
            if self.running && total > 0 {
                let idx = self.current_idx.min(total.saturating_sub(1));
                format!(
                    "已用 {} · 剩余约 {} 个文件",
                    elapsed_txt,
                    total.saturating_sub(idx + 1)
                )
            } else if total > 0 {
                format!("共 {total} 个文件 · 尚未开始")
            } else {
                String::new()
            }
        };

        let footer_status = TextBlock::new()
            .grid_row(0)
            .grid_column(1)
            .text(footer_txt)
            .font_size(12.0)
            .foreground(INK3)
            .horizontal_alignment(HorizontalAlignment::Right)
            .vertical_alignment(VerticalAlignment::Center);

        let body_w = (self.window_width - 36.0).max(300.0);
        let body_h = (self.window_height - 36.0).max(300.0);
        // 按钮行：按钮靠左，状态文字靠右。
        //
        // ★ 这一段原来直接挂在 main_area 的第 2 行上，把 main_area 改成
        //   两列 —— 结果 footer 的 Auto 列吃掉了 ~150px，上方的卡片区跟着
        //   变窄（文件路径肉眼可见被截得更早）。现在按钮和状态先包进一个
        //   自己的 Grid，main_area 仍然只有一列。
        //   button_row 自己要显式定位（grid_row(1)）：不设的话它会被当成
        //   自动放置的子元素，落到 main_area 的第一行 —— 实测按钮会跑到
        //   标题区上面，把标题盖住。
        let button_row = Grid::new()
            .grid_row(1)
            .grid_column(0)
            .columns([GridLength::STAR, GridLength::Auto])
            .children((buttons, footer_status));

        let main_area = Grid::new()
            .width(body_w)
            .height(body_h)
            .rows([GridLength::STAR, GridLength::Auto])
            .columns([GridLength::STAR])
            .row_spacing(6.0)
            .children((scrolled, button_row));

        // ── 「没有可处理的文件」警告框 ──
        //
        // 用户没添加任何文件就点「开始处理」时弹出，程序不会开始跑。
        // （拦截发生在 Msg::Start 里，这里只负责显示。）
        //
        // 为什么用 ContentDialog 而不是 rfd 的系统消息框：
        //   Windows 的系统消息框宽度按文字长度算但有上限，实测一行到 15 个
        //   汉字右边就被硬裁掉。ContentDialog 是 WinUI 原生的，宽度自适应，
        //   风格也和其余界面一致。
        //
        // 为什么内容分行写而不是一段话：这样每行都短，宽度不会撑得过宽。
        let no_file_dialog = ContentDialog::new()
            .title("没有可处理的文件")
            .primary_button_text("好")
            .is_open(self.show_no_file_warning)
            // on_closed 的回调参数是 ContentDialogResult（代表点了哪个按钮），
            // 我们只关心"关掉了"，所以忽略它。
            .on_closed(context.callback(|_: ContentDialogResult| Msg::DismissNoFileWarning))
            .content(
                StackPanel::new().spacing(8.0).children((
                    TextBlock::new()
                        .text("请先添加要处理的视频或音频。")
                        .font_size(14.0),
                    TextBlock::new()
                        .text("· 点「添加文件」或「添加文件夹」")
                        .font_size(14.0),
                    TextBlock::new()
                        .text("· 或者把它们直接拖进窗口")
                        .font_size(14.0),
                    TextBlock::new()
                        .text("添加之后再点「开始处理」。")
                        .font_size(14.0),
                )),
            );

        // 「恢复上次队列」的弹窗。
        //
        //   ContentDialogResult 有三个值：Primary / Secondary / None。
        //   Primary = 「恢复」按钮，Secondary = 「清空」按钮，
        //   None = 按 ESC 或关掉窗口 —— 按清空处理（安全，不会误恢复）。
        let restore_dialog = ContentDialog::new()
            .title("恢复上次的队列？")
            .primary_button_text("恢复")
            .secondary_button_text("清空")
            .is_open(self.show_restore_dialog)
            .on_closed(context.callback(|r: ContentDialogResult| {
                if matches!(r, ContentDialogResult::Primary) {
                    Msg::RestoreQueue
                } else {
                    Msg::ClearQueue
                }
            }))
            .content(
                StackPanel::new().spacing(8.0).children((
                    TextBlock::new()
                        .text(format!(
                            "检测到上次留下的队列（{} 个文件，处理到第 {} 个）。",
                            self.pending_queue.len(),
                            self.pending_idx + 1
                        ))
                        .font_size(14.0),
                    TextBlock::new()
                        .text("· 恢复：填回列表，从上次的位置继续")
                        .font_size(14.0),
                    TextBlock::new()
                        .text("· 清空：丢弃队列，从空列表重新开始")
                        .font_size(14.0),
                    TextBlock::new().text("").font_size(6.0),
                    TextBlock::new()
                        .text("（已删除的源文件会自动跳过）")
                        .font_size(13.0)
                        .foreground(INK3),
                )),
            );

        // 「没有 NVIDIA 显卡」的警告框。
        // 用和上面那个警告框相同的模式（ContentDialog 覆盖层）。
        // ★ 为什么【不提供】"下载 ROCm / DirectML" 这类选项
        //
        //   调研结论（2026-10）：
        //     · ROCm —— Windows 上没有可用产物。PyTorch 的 ROCm 轮子只有
        //       Linux，onnxruntime-rocm 不存在，CTranslate2 也没有 ROCm 后端。
        //       "下载 ROCm 就能跑"是做不到的，不是我们不实现。
        //     · DirectML —— 曾经是那条路，但 torch-directml 的元数据里
        //       硬绑 Requires-Dist: torch==2.4.1，而我们是 torch 2.11.0+cu128。
        //       降级到 2.4.1 会让 sm_120（RTX 50 系）支持全部失效 ——
        //       用 A 卡加速换掉所有 N 卡用户的正常使用，不划算。
        //
        //   所以这里【只告知，不承诺】，并且把"有独显却没被识别"这个
        //   更常见的情况单独指出来 —— 笔记本双显卡被分到核显，
        //   比真的没有 N 卡更常见。
        // 弹窗内容按实际原因动态生成：没卡 / 显存太小
        let cpu_reason = if !self.has_gpu && !self.has_vulkan {
            "没有检测到可用的 GPU（CUDA 和 Vulkan 都不可用）".to_string()
        } else if self.vram_gb > 0.0 {
            format!(
                "检测到显卡显存只有 {:.1} GB，低于本程序要求的 6 GB",
                self.vram_gb
            )
        } else {
            "显卡信息检测失败".to_string()
        };

        let cpu_dialog = ContentDialog::new()
            .title("没有可用的 GPU 加速")
            .primary_button_text("同意，用纯 CPU 继续")
            .secondary_button_text("不同意，退出")
            .is_open(self.show_cpu_warning)
            .on_closed(context.callback(|r: ContentDialogResult| {
                if matches!(r, ContentDialogResult::Primary) {
                    Msg::AgreeCpu
                } else {
                    Msg::RefuseCpu
                }
            }))
            .content(
                StackPanel::new().spacing(8.0).children((
                    TextBlock::new()
                        .text(cpu_reason)
                        .font_size(14.0)
                        .foreground(WARN),
                    TextBlock::new().text("").font_size(4.0),
                    TextBlock::new()
                        .text("· 人声分离需要约 2.5 GB 显存，装不下就跑不了 GPU 版")
                        .font_size(14.0),
                    TextBlock::new()
                        .text("· 如果同意，程序会改用 CPU 处理")
                        .font_size(14.0),
                    TextBlock::new()
                        .text("  速度约为 GPU 的 1/50：1 小时视频可能要几小时")
                        .font_size(14.0),
                    TextBlock::new().text("").font_size(4.0),
                    TextBlock::new()
                        .text("如果你这台电脑【有】NVIDIA 显卡但没被识别：")
                        .font_size(13.0),
                    TextBlock::new()
                        .text("打开 NVIDIA 控制面板 → 管理 3D 设置 → 程序设置")
                        .font_size(13.0),
                    TextBlock::new()
                        .text("→ 添加本程序 → 首选图形处理器选「高性能 NVIDIA 处理器」")
                        .font_size(13.0)
                        .foreground(INK3),
                )),
            );

        // dialog 是覆盖层，要和主内容并列放在同一个 StackPanel 里。
        // 用 keyed_children（而不是 children）：给每个孩子一个稳定身份，
        // 这样重新渲染时 dialog 的开关状态不会被当成"换了一个新控件"。
        // 官方在 reactor 自己的用例里就是这么放的。
        let root = StackPanel::new().keyed_children([
            KeyedView::new("main", main_area),
            KeyedView::new("no_file_dialog", no_file_dialog),
            KeyedView::new("cpu_dialog", cpu_dialog),
            KeyedView::new("restore_dialog", restore_dialog),
        ]);

        // ── 窗口底板：一层深蓝，用来隔开壁纸 ──
        //
        // 【为什么需要这一层】
        //   卡片是 47% 不透明的冰川蓝（argb(120,168,200,234)）。
        // ── 【没有窗口底板了】──
        //
        // 【曾经的方案】在这里套一层深蓝 Border 当"底板"，目的：
        //   卡片是半透明的，直接叠 Acrylic 会透上暖色壁纸、混成脏灰，
        //   加一层深蓝就等于给卡片一个干净的中性底，颜色确实稳了。
        //
        // 【为什么撤掉】底板只铺在【客户区】，而标题栏是窗口的非客户区，
        //   走的是 DWM 的 Acrylic，没有这一层。于是：
        //       标题栏  = Acrylic（透壁纸）
        //       客户区  = Acrylic + 深蓝底板   ← 多一层，偏深偏蓝
        //   两者对不上。平时卡片盖住了看不出来，但标题文字和第一张卡片
        //   之间有一段空隙，底板直接露出来 —— 就是一条突兀的深色带
        //   （用户截图圈出来问"这两处效果不能一致吗"）。
        //
        // 【现在】整窗统一由 Acrylic 提供背景，客户区不再叠任何底板，
        //   标题栏和主体是同一种材质、同一个色源，接缝自然消失。
        //
        //   代价：卡片直接坐在 Acrylic 上，会透到一点壁纸色调。
        //   但卡片本身 alpha 75（约 29% 不透明）已能压住大部分干扰；
        //   如果还是嫌脏，就调卡片的 alpha，不要再加底板 —— 加回来
        //   接缝又会出现。
        // ── 【没有窗口底板】整窗统一由 Acrylic 提供背景 ──
        //
        // 【为什么最终撤掉了底板】
        //   底板能铺的只有【客户区】。标题栏属于窗口的非客户区，由 DWM
        //   单独绘制，Border 铺不到它 —— 试过 DWMWA_USE_HOSTBACKDROPBRUSH
        //   让它借用宿主背景刷，但 DWM 那边只认 Windows 自己的材质模型，
        //   给不出"和客户区完全相同"的效果。
        //
        //   结论：想让标题栏和客户区真正一致，只能让两边【共享同一个背景
        //   源】—— 也就是客户区也只用 Acrylic，不叠任何东西。于是接缝
        //   从根上消失：标题栏、标题区、卡片之间的空隙全是同一种磨砂玻璃。
        //
        //   代价由卡片自己扛：底板没了之后卡片直接坐在壁纸上，所以把卡片
        //   alpha 从 75 提到 110（约 43% 不透明），靠卡片自身压住壁纸干扰。
        Border::new()
            .background(BG)
            .padding(18.0)
            .drop_policy(drop_policy)
            .on_drop(context.callback(Msg::Dropped))
            .content(root)
    }
}
