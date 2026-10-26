use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 找到 Windows SDK 自带的 rc.exe（取 bin 下版本号最大的那个）。
fn find_rc() -> Option<PathBuf> {
    if let Ok(p) = env::var("RC_EXE") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let kits = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut best: Option<(u64, PathBuf)> = None;
    if let Ok(entries) = std::fs::read_dir(kits) {
        for e in entries.flatten() {
            let rc = e.path().join("x64").join("rc.exe");
            if !rc.is_file() {
                continue;
            }
            let name = e.file_name().to_string_lossy().to_string();
            let ver: u64 = name
                .split('.')
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if best.as_ref().is_none_or(|(v, _)| ver >= *v) {
                best = Some((ver, rc));
            }
        }
    }
    best.map(|(_, p)| p)
}

fn main() {
    println!("cargo:rerun-if-changed=selfcontained.manifest");
    println!("cargo:rerun-if-changed=reactor.rc");
    println!("cargo:rerun-if-env-changed=RC_EXE");

    let Some(rc) = find_rc() else {
        println!("cargo:warning=rc.exe not found: self-contained manifest NOT embedded (Win10 会报找不到入口)");
        return;
    };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // 输出到无中文的临时目录，避免 rc.exe 的 ANSI 路径问题
    let res = env::temp_dir().join("asmr_reactor_manifest.res");
    let _ = std::fs::remove_file(&res);

    let status = Command::new(&rc)
        .current_dir(&manifest_dir)
        .arg("/nologo")
        .arg(&format!("/fo{}", res.display()))
        .arg("/I")
        .arg(&manifest_dir)
        .arg("reactor.rc")
        .status();

    match status {
        Ok(s) if s.success() && res.is_file() => {
            println!("cargo:rustc-link-arg-bins={}", res.display());
        }
        Ok(s) => println!("cargo:warning=rc.exe failed (status {s}), manifest not embedded"),
        Err(e) => println!("cargo:warning=rc.exe could not run: {e}"),
    }
}
