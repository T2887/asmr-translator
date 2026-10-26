use crate::native::winui::bindings::*;
use windows_core::*;

// Win10 没有 api-ms-win-appmodel-runtime-l1-1-5.dll（AddPackageDependency 仅 Win11 22H2+）。
// 动态探测不到时，回退到 Windows App Runtime 官方 Bootstrap DLL（Win10 也支持）。
windows_core::link!("kernel32.dll" "system" fn GetModuleHandleW(name: PCWSTR) -> *mut std::ffi::c_void);
windows_core::link!("kernel32.dll" "system" fn LoadLibraryW(lpLibFileName: PCWSTR) -> *mut std::ffi::c_void);
windows_core::link!("kernel32.dll" "system" fn GetProcAddress(hModule: *mut std::ffi::c_void, lpProcName: PCSTR) -> *mut std::ffi::c_void);

const BOOTSTRAP_DLL: &str = "Microsoft.WindowsAppRuntime.Bootstrap.dll";
const WIN10_MAJOR_MINOR: u32 = 0x0002_0000;

const FRAMEWORK_FAMILY: PCWSTR = w!("Microsoft.WindowsAppRuntime.2_8wekyb3d8bbwe");
const PACKAGE_DEPENDENCY_LIFETIME_KIND_PROCESS: i32 = 0;

static BOOTSTRAPPED: std::sync::Mutex<bool> = std::sync::Mutex::new(false);

/// 临时诊断：把 bootstrap 各步骤写进 %TEMP%\asmr_boot.log，便于在无控制台的 GUI 里定位失败点。
fn boot_log(msg: &str) {
    use std::io::Write;
    let path = std::env::temp_dir().join("asmr_boot.log");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{msg}");
    }
}

/// Initializes the Windows App Runtime for framework-dependent apps by resolving the installed
/// framework package and adding it to the process package graph directly.
///
/// Returns an error (and shows an install dialog) if the runtime framework package is not
/// installed on the machine.
pub fn bootstrap() -> Result<()> {
    let mut bootstrapped = BOOTSTRAPPED.lock().unwrap();
    if *bootstrapped {
        return Ok(());
    }
    unsafe { bootstrap_inner()? };
    *bootstrapped = true;
    Ok(())
}

unsafe fn bootstrap_inner() -> Result<()> {
    // Win10 没有 AddPackageDependency；另外留一个环境变量开关便于在 Win11 上强制测回退路径。
    let force = std::env::var_os("ASMR_FORCE_BOOTSTRAP").is_some();
    let native = unsafe { native_appmodel_api_available() };
    boot_log(&format!("enter force={force} native={native}"));
    if force || !native {
        let r = unsafe { bootstrap_via_dll() };
        boot_log(&format!(
            "bootstrap_via_dll -> {}",
            match &r {
                Ok(()) => "OK".to_string(),
                Err(e) => format!("ERR 0x{:08X}", e.code().0 as u32),
            }
        ));
        return r;
    }
    let mut dependency_id: PWSTR = PWSTR::null();
    let hr = unsafe {
        TryCreatePackageDependency(
            std::ptr::null_mut(),
            FRAMEWORK_FAMILY,
            PACKAGE_VERSION {
                Anonymous: PACKAGE_VERSION_0 {
                    Version: WINDOWSAPPSDK_RUNTIME_VERSION_UINT64,
                },
            },
            process_architecture().flags | PackageDependencyProcessorArchitectures_Neutral,
            PACKAGE_DEPENDENCY_LIFETIME_KIND_PROCESS,
            PCWSTR::null(),
            0,
            &mut dependency_id,
        )
    };

    if hr == STATEREPOSITORY_E_DEPENDENCY_NOT_RESOLVED {
        show_install_dialog();
        return Err(Error::new(
            hr,
            "Microsoft.WindowsAppRuntime.2 framework package is not installed.",
        ));
    }
    hr.ok()?;

    // This handle is unused as the package dependency lifetime is configured for the lifetime
    // of the process.
    let mut handle = std::ptr::null_mut();
    let mut package_full_name = PWSTR::null();
    unsafe {
        let add_result = AddPackageDependency(
            PCWSTR(dependency_id.0),
            0,
            0,
            &mut handle,
            &mut package_full_name,
        );

        let _ = HeapFree(
            GetProcessHeap(),
            0,
            dependency_id.0 as *mut std::ffi::c_void,
        );

        let _ = HeapFree(
            GetProcessHeap(),
            0,
            package_full_name.0 as *mut std::ffi::c_void,
        );

        if add_result == STATEREPOSITORY_E_DEPENDENCY_NOT_RESOLVED {
            show_install_dialog();
            return Err(Error::new(
                add_result,
                "Microsoft.WindowsAppRuntime.2 framework package is not installed.",
            ));
        }

        add_result.ok()?;
    };

    Ok(())
}

#[derive(Clone, Copy)]
struct ProcessArchitecture {
    display: &'static str,
    flags: PackageDependencyProcessorArchitectures,
}

fn process_architecture() -> ProcessArchitecture {
    match std::env::consts::ARCH {
        "x86_64" => ProcessArchitecture {
            display: "x64",
            flags: PackageDependencyProcessorArchitectures_X64,
        },
        "x86" => ProcessArchitecture {
            display: "x86",
            flags: PackageDependencyProcessorArchitectures_X86,
        },
        "aarch64" => ProcessArchitecture {
            display: "arm64",
            flags: PackageDependencyProcessorArchitectures_Arm64,
        },
        "arm" => ProcessArchitecture {
            display: "arm",
            flags: PackageDependencyProcessorArchitectures_Arm,
        },
        _ => ProcessArchitecture {
            display: std::env::consts::ARCH,
            flags: PackageDependencyProcessorArchitectures_None,
        },
    }
}

unsafe fn native_appmodel_api_available() -> bool {
    let module = unsafe { GetModuleHandleW(w!("api-ms-win-appmodel-runtime-l1-1-5.dll")) };
    if module.is_null() {
        return false;
    }
    !unsafe { GetProcAddress(module, s!("AddPackageDependency")) }.is_null()
}

/// Win10 路径：走 Windows App Runtime 官方 Bootstrap API（MddBootstrapInitialize）。
unsafe fn bootstrap_via_dll() -> Result<()> {
    let module = unsafe { LoadLibraryW(w!("Microsoft.WindowsAppRuntime.Bootstrap.dll")) };
    boot_log(&format!(
        "LoadLibraryW -> {:?} err={:?}",
        module,
        std::io::Error::last_os_error()
    ));
    if module.is_null() {
        show_install_dialog();
        return Err(Error::new(
            HRESULT(0x8007_007E_u32 as i32),
            "找不到 Microsoft.WindowsAppRuntime.Bootstrap.dll（请放在 exe 同目录）。",
        ));
    }
    let proc = unsafe { GetProcAddress(module, s!("MddBootstrapInitialize")) };
    if proc.is_null() {
        show_install_dialog();
        return Err(Error::new(
            HRESULT(0x8007_007F_u32 as i32),
            "Bootstrap DLL 中找不到 MddBootstrapInitialize。",
        ));
    }
    type InitFn = unsafe extern "system" fn(u32, PCWSTR, PACKAGE_VERSION) -> HRESULT;
    let init: InitFn = unsafe { std::mem::transmute(proc) };
    let hr = unsafe {
        init(
            WIN10_MAJOR_MINOR,
            w!(""),
            PACKAGE_VERSION {
                Anonymous: PACKAGE_VERSION_0 {
                    Version: WINDOWSAPPSDK_RUNTIME_VERSION_UINT64,
                },
            },
        )
    };
    if hr.is_err() {
        boot_log(&format!("MddBootstrapInitialize -> HRESULT 0x{:08X}", hr.0 as u32));
        show_install_dialog();
    } else {
        boot_log("MddBootstrapInitialize -> S_OK");
    }
    hr.ok()
}

fn process_caption() -> HSTRING {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.file_name().map(|f| f.to_string_lossy().into_owned()))
        .map_or_else(
            || HSTRING::from("This application could not be started"),
            HSTRING::from,
        )
}

fn show_install_dialog() {
    let caption = process_caption();
    let text = HSTRING::from(format!(
        "You must install Windows App Runtime ({WINDOWSAPPSDK_RUNTIME_VERSION_MAJOR}.{WINDOWSAPPSDK_RUNTIME_VERSION_MINOR}.{WINDOWSAPPSDK_RUNTIME_VERSION_BUILD}.{WINDOWSAPPSDK_RUNTIME_VERSION_REVISION}, {}) to run this application.\n\
         \n\
         Do you want to download it now?",
        process_architecture().display
    ));

    let result = unsafe {
        MessageBoxW(
            HWND::default(),
            PCWSTR::from_raw(text.as_ptr()),
            PCWSTR::from_raw(caption.as_ptr()),
            (MB_YESNO | MB_ICONERROR) as u32,
        )
    };

    if result == IDYES {
        unsafe {
            ShellExecuteW(
                HWND::default(),
                w!("open"),
                w!("https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads"),
                PCWSTR::null(),
                PCWSTR::null(),
                SW_SHOWNORMAL,
            );
        }
    }
}
