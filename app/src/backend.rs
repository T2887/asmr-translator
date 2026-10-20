// backend.rs —— 调用 Python 脚本并【实时】读取它的输出
//
// 为什么要写这么细，而不是直接用 BufReader::read_line：
//
//   1. 那些脚本输出的是 **GBK**（Windows 中文 cmd 的默认编码）。
//      BufReader::read_line 要求输入是合法 UTF-8，碰到中文会直接返回
//      InvalidData 错误 —— 结果是「一行中文都读不到」，但不报错、不崩，
//      只是日志区一片空白。所以必须【读原始字节】再自己按 \n 切。
//
//   2. stdout 和 stderr 必须【各开一个线程】读。只读一路的话，另一路的
//      管道缓冲区写满时，子进程会阻塞在 write 上永远不退出 —— 经典死锁。
//
//   3. on_line 是 Fn，两个线程都要用，所以包一层 Arc。

use std::io::{BufRead, BufReader, Read};
use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;

/// 一次性跑一小段 Python 代码，返回它 stdout（去掉首尾空白）。
///
/// 和 run_script 的区别：这个是【同步、短命】的，只为拿一个返回值，
/// 不需要逐行回调、也不需要能强杀。用来做设备检测这类一次性查询。
///
/// 编码处理和 run_script 一致：强制子进程 UTF-8 输出，这边按 UTF-8 解，
/// 不赌系统 locale。
pub fn run_python_inline(python_exe: &str, code: &str) -> Result<String, String> {
    let mut cmd = Command::new(python_exe);
    cmd.arg("-c")
        .arg(code)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .creation_flags(CREATE_NO_WINDOW);

    let out = cmd
        .output()
        .map_err(|e| format!("启动 {python_exe} 失败: {e}"))?;

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(if err.is_empty() {
            format!("退出码 {:?}", out.status.code())
        } else {
            err
        });
    }

    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// CREATE_NO_WINDOW：不给子进程建控制台窗口。
/// 不加这个，每次跑脚本都会弹一个黑框出来。
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// 当前正在跑的子进程 PID。用于「停止」时强杀。
///
/// 为什么搞个全局变量：run_script 是阻塞执行的（在 spawn_background 的闭包里），
/// 外界拿不到那个 Child 的所有权。但要杀进程只需要 PID —— 有 PID 就能
/// 调 taskkill。这比把 Child 塞进 Arc<Mutex<>> 再和读管道线程抢所有权简单得多。
static CURRENT_PID: std::sync::Mutex<Option<u32>> = std::sync::Mutex::new(None);

/// 强杀当前正在跑的子进程（连同它的子进程树）。
///
/// 对应 Python 版的 proc.terminate() + taskkill /F /T /PID。
/// 用 /T 是因为 python.exe 可能又起了 ffmpeg 之类的孙进程，
/// 只杀父进程会留下孤儿占着显存。
///
/// 这个是【异步】的：返回时进程可能还没死透，文件句柄也没释放。
/// 所以调用方紧接着清理文件时要能容忍失败并重试（purge_process_files 有 4 轮重试）。
pub fn kill_current() -> bool {
    let pid = match CURRENT_PID.lock() {
        Ok(g) => *g,
        Err(_) => return false,
    };
    let Some(pid) = pid else {
        return false;
    };

    let mut cmd = std::process::Command::new("taskkill");
    cmd.args(["/F", "/T", "/PID", &pid.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW);
    let _ = cmd.spawn().and_then(|mut c| c.wait());

    if let Ok(mut slot) = CURRENT_PID.lock() {
        *slot = None;
    }
    true
}

type LineCallback = Arc<dyn Fn(&str) + Send + Sync + 'static>;

/// 挂起/恢复当前子进程（暂停按钮用）。
///
/// 用 ntdll 的 NtSuspendProcess / NtResumeProcess：一次挂起整个进程的
/// 所有线程，比逐个 SuspendThread 简单，也不会漏掉新起的线程。
/// 这两个函数未文档化，但自 XP 起就存在，行为稳定。
#[cfg(windows)]
fn nt_process_call(pid: u32, proc_name: &[u8]) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_SUSPEND_RESUME};

    type NtFn = unsafe extern "system" fn(isize) -> i32;

    unsafe {
        let ntdll = GetModuleHandleA(b"ntdll.dll\0".as_ptr());
        if ntdll.is_null() {
            return false;
        }
        let f = GetProcAddress(ntdll, proc_name.as_ptr());
        let Some(f) = f else {
            return false;
        };
        let f: NtFn = std::mem::transmute(f);
        let handle = OpenProcess(PROCESS_SUSPEND_RESUME, 0, pid);
        if handle.is_null() {
            return false;
        }
        let rc = f(handle as isize);
        CloseHandle(handle);
        rc == 0
    }
}

/// 暂停当前阶段（挂起 python 子进程）。
pub fn suspend_current() -> bool {
    let pid = CURRENT_PID.lock().ok().and_then(|g| *g);
    match pid {
        Some(p) => nt_process_call(p, b"NtSuspendProcess\0"),
        None => false,
    }
}

/// 恢复当前阶段（继续被挂起的 python 子进程）。
pub fn resume_current() -> bool {
    let pid = CURRENT_PID.lock().ok().and_then(|g| *g);
    match pid {
        Some(p) => nt_process_call(p, b"NtResumeProcess\0"),
        None => false,
    }
}

/// 把一路管道里的内容按行抽出来，逐行交给回调。
///
/// 关键点：用 read_until 读【字节】，不用 read_line。
/// read_line 要求 UTF-8，而脚本输出是 GBK，中文会直接报错。
fn pump<R>(reader: R, on_line: LineCallback, tag: &'static str) -> thread::JoinHandle<()>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut raw: Vec<u8> = Vec::with_capacity(4096);

        loop {
            raw.clear();
            match reader.read_until(b'\n', &mut raw) {
                // 读到 0 字节 = 管道关闭 = 子进程结束了这一路输出
                Ok(0) => break,
                Ok(_) => {
                    // 去掉行尾的 \r\n
                    while matches!(raw.last(), Some(b'\n') | Some(b'\r')) {
                        raw.pop();
                    }
                    // 按 UTF-8 解。run_script 里已经用 PYTHONIOENCODING 钉死了
                    // 子进程的输出编码，所以这里不用再猜是 GBK 还是 UTF-8。
                    // 用 lossy 是兜底：万一某行混进了非 UTF-8 字节（比如脚本里
                    // 直接读了 GBK 文件），也只是那一行出现替换字符，不会中断整个流程。
                    let text = String::from_utf8_lossy(&raw);

                    // ★ 再按 \r 切一次。
                    //
                    //   上游 read_until(b'\n') 只认换行符，但 tqdm 进度条
                    //   是用 \r 回车的（原地刷新同一行）。所以一个"行"里
                    //   可能塞了好几次进度更新：
                    //       0%|...\r  5%|...\r 10%|...\r 15%|...
                    //   整个字符串当一行交给界面的话，正则 search 只能
                    //   匹配到【最早】那个进度（0%），后面的全被忽略 ——
                    //   进度条于是永远停在起点，直到这一步结束才跳变。
                    //
                    //   实测症状：第3步分离的 tqdm 明明是整数格式 8/173，
                    //   进度条却从 14% 直接跳到 50%。
                    for seg in text.split(['\r', '\n']) {
                        let s = seg.trim();
                        if !s.is_empty() {
                            on_line(s);
                        }
                    }
                }
                Err(e) => {
                    // 读管道出错通常意味着子进程异常退出，记一笔就结束本路
                    eprintln!("[{tag}] pipe read error: {e}");
                    break;
                }
            }
        }
    })
}

/// 跑一个 Python 脚本，把它的每一行输出实时交给 `on_line`。
///
/// 参数：
///   python_exe —— 解释器完整路径，例如 `D:\ASMRASR\env\python.exe`
///   work_dir   —— 工作目录，脚本里的相对路径（0pre/、1audio/ 之类）都基于它
///   script     —— 脚本文件名，例如 `a_提取音频.py`
///   on_line    —— 每读到一行就调用一次。注意：**它是在后台线程里被调用的**，
///                 回调里绝不能碰 UI 对象，只能往消息队列里塞。
///
/// 返回：子进程的退出码。
///
/// 这个函数会阻塞到子进程结束为止，所以要放在后台线程里调用。
pub fn run_script(
    python_exe: &str,
    work_dir: &str,
    script: &str,
    on_line: impl Fn(&str) + Send + Sync + 'static,
) -> Result<i32, String> {
    let mut cmd = Command::new(python_exe);
    cmd.arg(script)
        .current_dir(work_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // ★ 强制子进程用 UTF-8 输出，不要依赖系统 locale。
        //
        // 为什么必须显式设：Python 在 Windows 下默认按系统 locale 决定 stdout
        // 编码（中文系统 = GBK）。但 GBK 还是 UTF-8 取决于【环境变量】是否被设过
        // —— 用户环境里可能有 PYTHONIOENCODING，也可能没有。
        // 赌环境 = 一半机器会乱码。所以这里直接钉死 UTF-8，Rust 端也按 UTF-8 解，
        // 两边约定明确，跟系统是什么语言、外面设过什么变量都无关。
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .env("PYTHONUNBUFFERED", "1")   // 关掉块缓冲，保证输出实时送出
        // 数据目录：脚本里的 0pre/1audio/2vad/3asr 是相对路径，而 cwd 是
        // 程序目录（脚本要能找到 model/、ffmpeg/）。装进 Program Files 时
        // 程序目录不可写，所以由这个变量把「该往哪写」告诉 Python。
        .env("ASMR_DATA_DIR", crate::config::data_dir())
        // 程序目录不可写时（装进了 Program Files），Python 会试着把
        // __pycache__ 写到脚本旁边，每次启动都失败一次、还刷一行警告。
        // 这种情况直接关掉字节码缓存；可写时保留，启动能快点。
        .env(
            "PYTHONDONTWRITEBYTECODE",
            if crate::config::data_dir() == crate::config::work_dir() {
                "0"
            } else {
                "1"
            },
        )
        .creation_flags(CREATE_NO_WINDOW);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("启动失败 {python_exe} {script}: {e}"))?;

    // 记下 PID，供 kill_current() 强杀用。
    // 为什么需要：用户点「停止」时，Python 版的作法是 taskkill /F /T /PID
    // 把整棵进程树杀掉，而不是等它自己结束 —— 那几个脚本跑模型时
    // 一跑就是几十分钟，软停止等于停不下来。
    {
        let id = child.id();
        if let Ok(mut slot) = CURRENT_PID.lock() {
            *slot = Some(id);
        }
    }

    // 用 take() 把管道的所有权拿走，这样 child 之后还能自由 wait()
    // （如果只 as_ref() 借用，后面 wait() 会撞借用检查）
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "拿不到 stdout 管道".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "拿不到 stderr 管道".to_string())?;

    let cb: LineCallback = Arc::new(on_line);

    let h_out = pump(stdout, Arc::clone(&cb), "stdout");
    let h_err = pump(stderr, Arc::clone(&cb), "stderr");

    // 等两路都读完（等于等管道关闭），再等进程退出
    if h_out.join().is_err() {
        return Err("stdout 读取线程 panic".to_string());
    }
    if h_err.join().is_err() {
        return Err("stderr 读取线程 panic".to_string());
    }

    let status = child
        .wait()
        .map_err(|e| format!("等待子进程退出失败: {e}"))?;

    // 进程已经退了，把 PID 记空 —— 否则下次点「停止」会去 taskkill 一个
    // 已经不存在（甚至可能被系统复用了）的 PID。
    if let Ok(mut slot) = CURRENT_PID.lock() {
        *slot = None;
    }

    Ok(status.code().unwrap_or(-1))
}

#[cfg(test)]
mod tests {
    use super::*;

    // 「停止」按钮要能真的停下正在跑的脚本。
    //
    // 这个测试的存在理由：脚本跑模型一次几十分钟，软停止（等它自己结束）
    // 等于没停。用户点停止时必须用 taskkill 把整棵进程树干掉。
    // 这条路径涉及真实的进程创建和强杀，光看代码看不出来对不对。
    #[test]
    fn kill_current_really_kills_the_child() {
        let dir = std::env::temp_dir().join("asmr_kill_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // 一个会睡很久的脚本，模拟跑模型的阶段
        std::fs::write(
            dir.join("_sleep.py"),
            "import time\nprint('started', flush=True)\ntime.sleep(120)\n",
        )
        .unwrap();

        let py = crate::config::python_exe().to_string();
        let wd = dir.to_string_lossy().to_string();

        let started = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s2 = started.clone();

        let handle = std::thread::spawn(move || {
            run_script(&py, &wd, "_sleep.py", move |_line| {
                s2.store(true, std::sync::atomic::Ordering::SeqCst);
            })
        });

        // 等它真的跑起来（最长 15 秒）
        let t0 = std::time::Instant::now();
        while !started.load(std::sync::atomic::Ordering::SeqCst)
            && t0.elapsed().as_secs() < 15
        {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(
            started.load(std::sync::atomic::Ordering::SeqCst),
            "子进程没能启动，测试前提不成立"
        );

        // 强杀
        assert!(kill_current(), "kill_current 应该报告杀掉了东西");

        // run_script 应该很快返回，而不是等到 120 秒
        let t1 = std::time::Instant::now();
        let result = handle.join().expect("后台线程 panic 了");
        let took = t1.elapsed().as_secs();

        assert!(
            took < 20,
            "杀掉之后应该马上返回，实际等了 {took} 秒 —— 说明没杀成功"
        );
        match result {
            Ok(code) => assert_ne!(code, 0, "被强杀的进程退出码不该是 0"),
            Err(_) => {} // 也算失败，可以接受
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // kill_current 在没有子进程时不能 panic，应该老老实实返回 false
    #[test]
    fn kill_current_without_child_is_false() {
        // 先确保没有残留
        if let Ok(mut slot) = CURRENT_PID.lock() {
            *slot = None;
        }
        assert!(!kill_current(), "没有子进程时应该返回 false");
    }
}
