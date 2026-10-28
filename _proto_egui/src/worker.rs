//! worker.rs —— 在后台线程跑真实流水线，用 channel 把事件送回界面。
//!
//! 为什么这样写：WinUI 版（app/src/ui.rs）用的是 reactor 的
//! `spawn_background`，每个阶段结束回一条 Msg、由 update 驱动下一阶段。
//! egui 是即时模式，天然适合「后台推事件 + 每帧收事件」，所以这里用一个
//! 普通线程 + `std::sync::mpsc`，逻辑顺序和 ui.rs 保持一致：
//!
//!   prepare（写 _input.txt）→ 第1..4步 → tidy（字幕整理）→ finalize（改名归位）
//!   → purge（清中间产物）→ 下一个文件
//!
//! 停止 = `CancellationToken::cancel()` + `backend::kill_current()`（强杀子进程树）。
//! 暂停 = `backend::suspend_current()` / `resume_current()`（挂起/恢复进程）。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::backend;
use crate::cancel::CancellationToken;
use crate::config;
use crate::pipeline::{self, RunMode, StageResult};

/// 后台线程 → 界面的事件
pub enum Ev {
    Log(String),
    StageStart(usize),
    /// frac = tqdm 进度 0.0~1.0；eta = 脚本自己报的剩余秒数（0 = 没有）
    Tick {
        frac: f64,
        eta: f64,
        latest: String,
    },
    StageDone {
        stage: usize,
        code: i32,
        secs: f64,
    },
    Preview(Vec<String>),
    FileStart {
        idx: usize,
        total: usize,
        name: String,
    },
    FileDone {
        idx: usize,
    },
    /// 这个文件没跑完就让整个批次中止了（某一步退出码非 0，或用户点了停止）。
    ///
    /// 为什么单独一个事件、而不是复用 FileDone：界面上「完成 ✓（绿）」和
    /// 「失败（红）」是两回事。以前失败时只 break、什么也不发，那一行就
    /// 永久停在「处理中」—— 用户明明已经终止了任务，界面还写着处理中。
    FileAborted {
        idx: usize,
        /// true = 用户主动停止；false = 脚本报错中止
        cancelled: bool,
    },
    AllDone,
}

pub struct Worker {
    pub cancel: CancellationToken,
    pub rx: Receiver<Ev>,
}

/// `jobs` 里是 `(在界面文件列表里的下标, 完整路径)` —— 下标要带过来，
/// 因为「已有字幕、被跳过」的文件不在 jobs 里，不能按顺序号反推。
pub fn spawn(
    jobs: Vec<(usize, String)>,
    mode: RunMode,
    copy_to_video: bool,
    out_dir: String,
) -> Worker {
    let (tx, rx) = channel::<Ev>();
    let cancel = CancellationToken::new();
    let token = cancel.clone();
    thread::spawn(move || {
        run_all(tx, token, jobs, mode, copy_to_video, out_dir);
    });
    Worker { cancel, rx }
}

fn log(tx: &Sender<Ev>, s: impl Into<String>) {
    let _ = tx.send(Ev::Log(s.into()));
}

fn logs(tx: &Sender<Ev>, v: Vec<String>) {
    for l in v {
        let _ = tx.send(Ev::Log(l));
    }
}

fn run_all(
    tx: Sender<Ev>,
    token: CancellationToken,
    jobs: Vec<(usize, String)>,
    mode: RunMode,
    copy_to_video: bool,
    out_dir: String,
) {
    let total = jobs.len();
    let n_stages = pipeline::stages(mode).len();

    for (ord, (idx, file)) in jobs.iter().enumerate() {
        let idx = *idx;
        if token.is_cancelled() {
            break;
        }
        let name = std::path::Path::new(file)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| file.clone());

        let _ = tx.send(Ev::FileStart {
            idx,
            total,
            name: name.clone(),
        });
        log(
            &tx,
            format!("--- 准备文件（{}/{}）: {name} ---", ord + 1, total),
        );

        match pipeline::prepare_0pre_cancellable(&[file.clone()], &|| token.is_cancelled()) {
            Ok((_n, lines)) => logs(&tx, lines),
            Err(e) => {
                log(&tx, format!("[失败] 准备阶段出错：{e}"));
                break;
            }
        }

        // 本文件这次处理的起点：产物兜底时用它区分「本次新产出的字幕」
        // 和「上一轮残留的旧字幕」（否则真失败会被误判成成功）。
        let file_start = std::time::SystemTime::now();

        // 阶段间共享的实时通道（和 ui.rs 里那套一模一样）
        let live: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let ready = Arc::new(AtomicBool::new(false));
        let frac: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
        let eta: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
        let sub_end: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));

        let mut failed = false;
        for stage in 1..=n_stages {
            if token.is_cancelled() {
                break;
            }
            let title = pipeline::stages(mode)[stage - 1].2;
            let _ = tx.send(Ev::StageStart(stage));
            log(&tx, format!("--- {title} ---"));

            let t0 = Instant::now();
            let res = run_one_stage(
                mode,
                stage,
                token.clone(),
                Arc::clone(&live),
                Arc::clone(&ready),
                Arc::clone(&frac),
                Arc::clone(&eta),
                Arc::clone(&sub_end),
                &tx,
            );
            let secs = t0.elapsed().as_secs_f64();

            let r: StageResult = match res {
                Some(r) => r,
                None => {
                    log(&tx, "[失败] 阶段线程异常退出");
                    failed = true;
                    break;
                }
            };

            // 阶段输出里带字幕行 → 抽出来进预览区（正则来自 config）。
            // 不限定阶段号：全量模式字幕在第 4 步，兼容模式在第 3 步。
            {
                let mut hit = Vec::new();
                for l in &r.lines {
                    if let Some(s) = fmt_sub(l) {
                        hit.push(s);
                    }
                }
                if !hit.is_empty() {
                    log(&tx, format!("[预览] 本步解析出 {} 条字幕", hit.len()));
                    let _ = tx.send(Ev::Preview(hit));
                }
            }

            logs(&tx, r.lines);
            let _ = tx.send(Ev::StageDone {
                stage,
                code: r.code,
                secs,
            });

            if r.code != 0 {
                // 【产物兜底】最后一个阶段正常写完字幕、却在进程退出时崩掉
                // （实测 0xC0000409 = -1073740791，torch/ctranslate2 的析构里）。
                // 字幕已经是好的，按失败处理会让用户白等十几分钟还拿不到字幕，
                // 所以这里先看 3asr 有没有产物：有就继续走 tidy + finalize。
                // 不限制阶段号：兼容模式出字幕的是第 3 步（翻译），
                // 全量/精简是第 4 步（识别），只靠时间戳判定更通用。
                if pipeline::has_any_subtitle_for(&file, file_start) {
                    log(
                        &tx,
                        format!(
                            "[警告] 第{stage}步退出码 = {}（进程收尾时崩溃），但字幕已生成，继续整理归位",
                            r.code
                        ),
                    );
                } else {
                    log(
                        &tx,
                        format!("[失败] 第{stage}步退出码 = {}，流水线中止", r.code),
                    );
                    failed = true;
                    break;
                }
            }
            // 换阶段：清掉上一阶段残留
            if let Ok(mut f) = frac.lock() {
                *f = 0.0;
            }
            if let Ok(mut l) = live.lock() {
                l.clear();
            }
            ready.store(false, Ordering::Relaxed);
        }

        if failed || token.is_cancelled() {
            let _ = pipeline::purge_process_files(true);
            // 告诉界面这一行结束了（失败 / 被停止）—— 否则它会一直停在
            // 「处理中」，用户看到的就是「明明终止了还写着处理中」。
            let _ = tx.send(Ev::FileAborted {
                idx,
                cancelled: token.is_cancelled(),
            });
            break;
        }

        // 四个阶段跑完 → 字幕整理（纯 Rust）→ 收尾改名归位
        log(&tx, "=== 本文件四个阶段完成，开始整理字幕 ===");
        logs(&tx, pipeline::tidy_subtitles());
        log(&tx, "=== 开始整理字幕 ===");
        logs(
            &tx,
            pipeline::finalize_subtitles(&[file.clone()], copy_to_video, &out_dir),
        );
        // 中间产物在下一个文件开始前清掉：磁盘上永远只有一个文件的中间产物
        let _ = pipeline::purge_process_files(true);
        let _ = tx.send(Ev::FileDone { idx });
    }

    let _ = tx.send(Ev::AllDone);
}

/// 一行输出如果是字幕行（`[起] -> [止]: 文本`），格式化成预览区要显示的样子。
fn fmt_sub(l: &str) -> Option<String> {
    let re = config::translation_re();
    let c = re.captures(l)?;
    let s = c.get(1).map(|m| m.as_str()).unwrap_or("?");
    let e = c.get(2).map(|m| m.as_str()).unwrap_or("?");
    let t = c.get(3).map(|m| m.as_str().trim()).unwrap_or("");
    Some(format!("[{s}s → {e}s]  {t}"))
}

/// 跑一个阶段，同时每 250ms 把实时进度/最新行推给界面。
///
/// 为什么开子线程：`pipeline::run_stage` 是阻塞的（要等子进程结束），
/// 而实时进度（tqdm 的 frac、最新一行输出）放在共享变量里 ——
/// 只有从外面轮询才拿得到。
#[allow(clippy::too_many_arguments)]
fn run_one_stage(
    mode: RunMode,
    stage: usize,
    token: CancellationToken,
    live: Arc<Mutex<String>>,
    ready: Arc<AtomicBool>,
    frac: Arc<Mutex<f64>>,
    eta: Arc<Mutex<f64>>,
    sub_end: Arc<Mutex<f64>>,
    tx: &Sender<Ev>,
) -> Option<StageResult> {
    let slot: Arc<Mutex<Option<StageResult>>> = Arc::new(Mutex::new(None));
    let slot2 = Arc::clone(&slot);

    // 这些 Arc 主循环还要继续读（轮询进度），所以进线程前先 clone 一份
    let (live_t, ready_t, frac_t, eta_t, sub_end_t) = (
        Arc::clone(&live),
        Arc::clone(&ready),
        Arc::clone(&frac),
        Arc::clone(&eta),
        Arc::clone(&sub_end),
    );

    let h = thread::spawn(move || {
        let out = pipeline::run_stage(
            mode, stage, token, live_t, ready_t, frac_t, eta_t, sub_end_t,
        );
        if let Ok(mut s) = slot2.lock() {
            *s = Some(out);
        }
    });

    let mut last_seen = String::new();
    loop {
        let done = slot.lock().map(|s| s.is_some()).unwrap_or(true);
        if done {
            break;
        }
        let fv = frac.lock().map(|x| *x).unwrap_or(0.0);
        let ev = eta.lock().map(|x| *x).unwrap_or(0.0);
        let lv = live.lock().map(|x| x.clone()).unwrap_or_default();
        // 实时字幕：字幕行一出现就推给预览区（不等阶段跑完）
        if lv != last_seen {
            if let Some(s) = fmt_sub(&lv) {
                let _ = tx.send(Ev::Preview(vec![s]));
            }
            last_seen = lv.clone();
        }
        let _ = tx.send(Ev::Tick {
            frac: fv,
            eta: ev,
            latest: lv,
        });
        thread::sleep(Duration::from_millis(250));
    }

    let _ = h.join();
    slot.lock().ok().and_then(|mut s| s.take())
}

/// 强杀当前子进程（界面「停止」按钮用）。
pub fn kill() -> bool {
    backend::kill_current()
}

/// 暂停 / 恢复（挂起整个进程树）。
pub fn suspend() -> bool {
    backend::suspend_current()
}

pub fn resume() -> bool {
    backend::resume_current()
}
