// main.rs —— 入口
//
// 模块划分：
//   config.rs   —— 路径常量、版本号（集中一处，避免同一路径写三遍）
//   backend.rs  —— 调 Python 子进程并实时读输出（已单独验证过）
//   pipeline.rs —— 四阶段流水线（对应四个 Python 脚本）
//   ui.rs       —— 界面与状态

// ★ 发布版用 Windows GUI 子系统，双击才不弹黑框。
//   调试版【不加】这个 —— 否则 println! 的输出没地方去，
//   排查问题时看不到任何东西。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod config;
mod pipeline;
mod srt;
mod ui;

use windows_reactor::*;

// ─────────────────────────────────────────────────────────────
// 窗口居中：为什么要在 main 里装 CBT 钩子
// ─────────────────────────────────────────────────────────────
//
// 【问题】窗口打开时会先在系统给的级联位置（比如 51,51）露一下，
//         再跳到屏幕中央，视觉上闪一下。
//
// 【为什么会闪】先把尝试过的错误做法记下来，免得以后再走一遍：
//
//   ① 后台 sleep 700ms 再居中
//      → 窗口早就显示完了，跳变最明显。
//
//   ② 后台高频轮询（第一轮不 sleep、之后每 2ms 一次）
//      → 好一些但【仍然不稳定】。实测 5 次里 3 次干净、2 次闪一下。
//        原因：窗口的「创建」和「显示」几乎同时发生，轮询本质上是在
//        追着窗口跑 —— 只要慢一拍，用户就看见了。
//        实测数据：窗口从启动到可见只要约 40ms，级联位置每次 +25px。
//
//   ③ 在 Component::create 里同步轮询
//      → 【会死锁】。create 跑在 UI 线程上，而窗口的创建本身要靠
//        UI 线程完成；阻塞它等于让窗口永远建不出来。
//
// 【正确做法】拦在窗口「激活/显示之前」。
//
//   WH_CBT 钩子的 HCBT_ACTIVATE 通知，会在系统激活窗口【之前】回调 ——
//   这时窗口已经创建（HWND 有效）但还没画到屏幕上。
//   在这个回调里 SetWindowPos，窗口第一次出现就已经在中央，全程不闪。
//
//   钩子必须在窗口创建【之前】装好，所以放在 main 里、App::run_component 之前。
//   钩子只作用于本线程（dwThreadId = GetCurrentThreadId()），不影响别的程序。
//
//   注：hwnd 用 wparam 直接拿，不枚举。HCBT_ACTIVATE 的 wparam 就是那个窗口。
//      但要排掉一些本进程的辅助窗口（它们也会触发 CBTA），判据是
//      「有标题」——主窗口的标题是 view 里设的 window_title，
//      辅助窗口没有标题。

#[cfg(windows)]
mod center_hook {
    use std::sync::atomic::{AtomicBool, Ordering};
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::Graphics::Dwm::{
        DwmSetWindowAttribute, DWMWA_BORDER_COLOR, DWMWA_CAPTION_COLOR, DWMWA_COLOR_NONE,
        DWMWA_SYSTEMBACKDROP_TYPE, DWMWA_TEXT_COLOR, DWMWA_WINDOW_CORNER_PREFERENCE,
        DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWCP_ROUND,
        DWMSBT_MAINWINDOW,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, GetSystemMetrics, GetWindowRect, GetWindowTextLengthW, SetWindowPos,
        SetWindowsHookExW, HCBT_ACTIVATE, SM_CXSCREEN, SM_CYSCREEN, SWP_NOACTIVATE, SWP_NOSIZE,
        SWP_NOZORDER, WH_CBT,
    };

    /// 只在第一次激活时摆位置。之后用户自己拖动窗口，不该再被拽回去。
    static DONE: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" fn cbt_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        unsafe {
            if code == HCBT_ACTIVATE as i32 && !DONE.load(Ordering::Relaxed) {
                let hwnd = wparam as HWND;
                // 有标题的才算主窗口 —— 本进程的 IME、消息窗之类也会触发 CBTA。
                if GetWindowTextLengthW(hwnd) > 0 {
                    DONE.store(true, Ordering::Relaxed);
                    paint_title_bar(hwnd);
                    place(hwnd);
                }
            }
            CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam)
        }
    }

    /// 把系统标题栏染成和界面主体一致的深蓝。
    ///
    /// 【为什么需要这个】
    ///   reactor 的 WindowVisuals 只有 backdrop / client_size / icon /
    ///   constraints / theme —— 没有标题栏配色；ViewContext 也没有。
    ///   所以标题栏一直是 Windows 的强调色（蓝紫色），和里面的冰川蓝
    ///   玻璃完全不搭（用户直接问"顶上这块能和主体一致吗"）。
    ///
    ///   绕过去的办法是 DWM 的 Win32 接口：DwmSetWindowAttribute 配
    ///   DWMWA_CAPTION_COLOR / DWMWA_TEXT_COLOR，Windows 11 起支持。
    ///   center_hook 这里本来就拿到了主窗口的 hwnd，顺手设掉。
    ///
    /// 【颜色格式】DWM 收的是 COLORREF，即 0x00BBGGRR —— 注意是
    ///   【BGR 倒序】，和平时写的 #RRGGBB 相反，很容易搞混。
    ///   0x007E4C2C 展开是 B=2C G=4C R=7E，也就是 #2C4C7E。
    unsafe fn paint_title_bar(hwnd: HWND) {
        // 这里不用再套一层 unsafe {}：
        // 本函数已经是 unsafe fn，而唯一需要 unsafe 的
        // DwmSetWindowAttribute 调用被关在下面 set() 自带的 unsafe 块里，
        // 外层再套一层编译器会报 unnecessary unsafe block。
        fn set(hwnd: HWND, attr: i32, value: u32) {
            let _ = unsafe {
                DwmSetWindowAttribute(
                    hwnd,
                    attr as u32,
                    &value as *const u32 as *const core::ffi::c_void,
                    std::mem::size_of::<u32>() as u32,
                )
            };
        }

            // ── 让标题栏也用磨砂玻璃，而不是纯色 ──
            //
            // 【走过一次弯路】
            //   第一版是给标题栏设纯色（DWMWA_CAPTION_COLOR = 深蓝 #2C4C7E），
            //   想让"顶上和主体一致"。颜色确实变了，但它是【纯色】——
            //   DWM 一旦拿到明确的 caption 颜色，标题栏就退出材质渲染路径。
            //   结果主体是磨砂玻璃、顶上是一块死板色块，反而更不协调
            //   （用户接着就问"标题栏不能一样磨砂玻璃效果吗"）。
            //
            // 【正确做法】两步：
            //   1. DWMWA_CAPTION_COLOR = DWMWA_COLOR_NONE
            //      把纯色清掉，交还给系统材质
            //   2. DWMWA_SYSTEMBACKDROP_TYPE = DWMSBT_TRANSIENTWINDOW
            //      指定标题栏用 Acrylic（瞬态窗口材质），和客户区
            //      WindowVisuals::backdrop(Acrylic) 对上
            //   这样整块窗口（标题栏 + 客户区）才是同一种磨砂玻璃。
            //
            //   备选：DWMSBT_MAINWINDOW(2) 是 Mica，更"实"、颗粒感弱，
            //   如果 Acrylic 显得太糊可以换过去。
            //
            // 【2026-10-20 第三次改：深色玻璃 → 浅色纸面】
            //   UI 换成 Claude 暖色纸面后，标题栏就不能再是深蓝磨砂了：
            //   一深一浅，两块拼在一起像没渲染完。
            //   现在整窗走浅色 —— 材质用 Mica（和 ui.rs 的 backdrop 对上），
            //   并强制关掉沉浸式深色，标题文字改成墨色。
            set(hwnd, DWMWA_CAPTION_COLOR, DWMWA_COLOR_NONE);
            set(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_MAINWINDOW as u32);

            // 注：试过 DWMWA_USE_HOSTBACKDROPBRUSH 让标题栏借用宿主背景刷，
            // 但客户区现在已经没有底板了（整窗统一 Acrylic），
            // 两边是同一个背景源，不再需要这个开关。

            // 【关键】0 = 浅色标题栏。
            //   这里原来是 1（深色），是为了让标题文字在深蓝玻璃上发白；
            //   现在底板是浅米色，再开深色就会在浅色客户区顶上盖一块深灰，
            //   而且系统还会把标题栏文字刷成白色 —— 白字压浅底根本看不清。
            set(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 0);

            // 标题文字改成墨色 #2B2621（COLORREF 是 0x00BBGGRR，倒着写）
            set(hwnd, DWMWA_TEXT_COLOR, 0x0021262B);

            // 窗口边框留一条淡灰 #E5DFD2，让窗口边缘不糊在浅色壁纸上
            set(hwnd, DWMWA_BORDER_COLOR, 0x00D2DFE5);

            // ── 窗口四角圆角 ──
            //
            // 用户反馈「圆角不一致」：卡片是 20px 圆角，窗口四角却是直角，
            // 米色客户区一直顶到方形边框上，四角看着像被切了一刀。
            //
            // Win11 的窗口圆角由 DWM 统一裁（DWMWA_WINDOW_CORNER_PREFERENCE）。
            // 不显式设的话，某些创建路径（自定义标题栏 / 关掉系统边框）会落到
            // DONOTROUND，客户区就保持直角。这里明确要 ROUND。
            //
            // 注意必须在窗口显示【之前】设（这里在 HCBT_ACTIVATE 钩子里，
            // 正好赶在第一次绘制前），显示之后再改也能生效，但会闪一下。
            set(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND as u32);

            // ── 窗口 / 任务栏图标 ──
            //
            // 【为什么要手动设】
            //   reactor 的 WindowVisuals 虽然有 icon 字段，但 winui 后端根本
            //   没用它（后端源码里 icon 只出现在 Button 的 SymbolIcon 分支），
            //   所以窗口图标一直是 exe 的默认图标 —— 任务栏上那个白方块。
            //   exe 自身也没有嵌入 RT_GROUP_ICON（项目没有 build.rs），
            //   于是只能运行时用 WM_SETICON 把 .ico 塞给窗口。
            //
            //   任务栏图标有缓存：改完要把程序彻底关掉重开才看得到。
            unsafe {
                use windows_sys::Win32::UI::WindowsAndMessaging::{
                    ICON_BIG, ICON_SMALL, IMAGE_ICON, LR_DEFAULTSIZE, LR_LOADFROMFILE, LoadImageW,
                    SendMessageW, WM_SETICON,
                };
                if let Some(path) = icon_ico_path() {
                    let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
                    let hicon = LoadImageW(
                        std::ptr::null_mut(),
                        wide.as_ptr(),
                        IMAGE_ICON,
                        0,
                        0,
                        LR_LOADFROMFILE | LR_DEFAULTSIZE,
                    );
                    if !hicon.is_null() {
                        let h = hicon as usize;
                        SendMessageW(hwnd, WM_SETICON, ICON_BIG as usize, h as isize);
                        SendMessageW(hwnd, WM_SETICON, ICON_SMALL as usize, h as isize);
                    }
                }
            }
    }

    /// exe 同目录下的 `assets\icon.ico`。
    ///
    /// 和 ui.rs 里 `icon_path()` 是同一套查找规则：都按 exe 所在目录找，
    /// 这样把程序整个目录拷走也能用；只拷 exe 不拷 assets 就退回默认图标。
    fn icon_ico_path() -> Option<String> {
        let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
        let p = dir.join("assets").join("icon.ico");
        p.is_file().then(|| p.to_string_lossy().to_string())
    }

    /// 把窗口摆到主显示器正中。
    ///
    /// ★ 只改位置，【不改尺寸】。
    ///
    ///   这里原来写死了 1400x900 一起设进 SetWindowPos，结果是：
    ///   ui.rs 里按屏幕算出来的 client_size（比如 2900x1500）刚设好，
    ///   立刻被这个钩子覆盖回 1400x900 —— 自适应尺寸完全失效
    ///   （实测：窗口一直是 1750x1125，正好是 1400x900 在 125% 下的物理值）。
    ///
    ///   现在带上 SWP_NOSIZE，尺寸交给 reactor 的 client_size 决定，
    ///   这里只负责挪到中间。
    unsafe fn place(hwnd: HWND) {
        unsafe {
            // 先问窗口当前有多大，再据此算居中位置 ——
            // 尺寸不是我们定的，不能自己假设。
            let mut rect = windows_sys::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            };
            if GetWindowRect(hwnd, &mut rect) == 0 {
                return;
            }
            let w = rect.right - rect.left;
            let h = rect.bottom - rect.top;

            let sw = GetSystemMetrics(SM_CXSCREEN);
            let sh = GetSystemMetrics(SM_CYSCREEN);
            let x = ((sw - w) / 2).max(0);
            let y = ((sh - h) / 2).max(0);

            SetWindowPos(
                hwnd,
                std::ptr::null_mut(),
                x,
                y,
                0,
                0,
                SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE,
            );
        }
    }

    /// 装上钩子。返回 false 表示装失败（不影响程序运行，只是不居中）。
    pub fn install() -> bool {
        unsafe {
            let h = SetWindowsHookExW(
                WH_CBT,
                Some(cbt_proc),
                std::ptr::null_mut(),
                windows_sys::Win32::System::Threading::GetCurrentThreadId(),
            );
            !h.is_null()
        }
    }
}

// ─────────────────────────────────────────────────────────────
// 每秒心跳：让界面上的"已用时间 / 当前状态"动起来
// ─────────────────────────────────────────────────────────────
//
// 【为什么需要这个】
//   第2步加载 RoFormer、第4步加载 whisper 都要几十秒到几分钟。这段时间
//   脚本一行输出都没有，进度条也停在原地 —— 用户盯着一个完全静止的窗口，
//   会以为程序卡死了。
//
// 【为什么不直接用 reactor 的机制】
//   · reactor 没有定时器 API（`interval` 那个是 RepeatButton 的参数）
//   · `LocalSender` 内部是 `Rc<RefCell<..>>`，是 !Send 的，
//     没法存进后台线程，所以后台线程没法"主动推"消息给界面
//   · `spawn_background` 的闭包只能返回【一个】消息，做不了循环
//
// 【所以绕道 Win32】
//   `SetTimer(hWnd=NULL, 0, 1000, Some(proc))` 会由系统按周期调用 `proc`，
//   而且【在创建定时器的那个线程】上调用 —— 也就是 UI 线程。
//   于是可以在 proc 里安全地取出存在 thread_local 里的 sender 并 send。
//   thread_local 正是为了绕开 LocalSender 的 !Send：它不跨线程，只在
//   UI 线程内部传递。
#[cfg(windows)]
mod tick {
    use crate::ui::Msg;
    use std::cell::RefCell;
    use windows_reactor::LocalSender;
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::UI::WindowsAndMessaging::{SetTimer, WM_TIMER};

    thread_local! {
        /// 界面组件的消息发送端。`view()` 每次渲染都会刷新一次。
        /// 用 thread_local 而不是全局静态：LocalSender 是 !Send 的。
        static SENDER: RefCell<Option<LocalSender<Msg>>> = const { RefCell::new(None) };
    }

    /// 由 ui.rs 的 view() 调用，把当前 sender 挂上来。
    pub fn set_sender(s: LocalSender<Msg>) {
        SENDER.with(|c| *c.borrow_mut() = Some(s));
    }

    /// 定时器回调。跑在 UI 线程上，所以能安全访问 thread_local。
    unsafe extern "system" fn on_timer(_hwnd: HWND, msg: u32, _id: usize, _time: u32) {
        if msg != WM_TIMER {
            return;
        }
        SENDER.with(|c| {
            if let Some(s) = c.borrow().as_ref() {
                // send 失败（队列满/组件已卸载）就忽略，下一拍还会再来
                let _ = s.send(Msg::Tick);
            }
        });
    }

    /// 启动周期心跳。返回 false 表示没装上（界面不会动，但不影响功能）。
    ///
    /// 周期 100ms：界面上「已用时间」本来 1 秒刷一次就够，但长按判定
    /// （卡片按住约 450ms 才开始拖动）需要更细的粒度 —— 用 1 秒的话
    /// 用户按住要等整整一秒才有反应，手感很差。
    pub fn start() -> bool {
        unsafe {
            let id = SetTimer(std::ptr::null_mut(), 0, 100, Some(on_timer));
            id != 0
        }
    }
}

/// 非 Windows 平台的空实现（这个程序本来就只跑 Windows，
/// 留着是为了让代码在别的平台也能通过语法检查）。
#[cfg(not(windows))]
mod tick {
    pub fn set_sender(_s: windows_reactor::LocalSender<super::ui::Msg>) {}
    pub fn start() -> bool {
        false
    }
}

fn main() {
    // 必须在 reactor 创建窗口【之前】装 —— 装晚了就错过第一次 HCBT_ACTIVATE 了。
    #[cfg(windows)]
    {
        // 失败也不致命：窗口会按系统默认位置出现，功能不受影响。
        let _ = center_hook::install();
    }

    // 每秒一次的心跳定时器。装不上不影响功能，只是界面上的
    // "已用时间 / 当前状态"不会自动刷新。
    let tick_ok = tick::start();

    // 有个隐藏的调试后门：ASMR_TEST_FILES 之类的环境变量在 ui.rs 里处理。
    // 这里只记一下定时器状态，方便排查"为什么秒数不动"。
    if !tick_ok && std::env::var("ASMR_DEBUG_TICK").is_ok() {
        eprintln!("[tick] SetTimer 失败，界面不会每秒刷新");
    }

    App::run_component::<ui::AppUi>(()).unwrap();
}
