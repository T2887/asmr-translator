//! 版本检查与自动更新
//!
//! 设计目标（用户 2026-10-31 定：方案 C）：
//!   1. 启动后台静默检查，有新版本就在界面顶部显示一条横幅
//!   2. 点「立即更新」尝试程序内下载（进度条），下完直接运行安装包
//!   3. 下载失败（常见：国内连不上 GitHub）就退回「打开下载页」，
//!      页面上给出网盘镜像链接，用户自己挑一个能下载的
//!
//! 网络请求不引第三方 HTTP 库：优先用系统自带的 `curl.exe`
//! （Windows 10 1803+ 一定有），没有就退回 PowerShell 的
//! `Invoke-WebRequest`。两者都失败则直接报错，不阻塞主线程 ——
//! 检查发生在后台线程，结果通过 channel 回传。

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

/// ★ 用户实测反馈「每次打开都闪过一个黑窗口」—— 根因就在这里。
///
///   本文件启动 2 秒后会自动检查更新，走的是 `curl.exe`（老系统退
///   PowerShell）。这些子进程如果【不带 CREATE_NO_WINDOW】，Windows 就会
///   给它们开一个黑色控制台窗口，在界面上闪一下。
///
///   v1.28 加入自动更新时漏了这个标志；同一时期 softsub.rs 的 ffmpeg
///   探测也漏了。凡是起子进程的地方都必须带，一个都不能漏。
///
/// 下面给 Command 加一个 `.no_window()` 扩展方法，各调用点串上即可。
#[cfg(windows)]
trait NoWindow {
    fn no_window(&mut self) -> &mut Self;
}

#[cfg(windows)]
impl NoWindow for Command {
    fn no_window(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        self.creation_flags(CREATE_NO_WINDOW)
    }
}

#[cfg(not(windows))]
trait NoWindow {
    fn no_window(&mut self) -> &mut Self;
}

#[cfg(not(windows))]
impl NoWindow for Command {
    fn no_window(&mut self) -> &mut Self {
        self
    }
}

/// 版本清单地址。
///
/// 放 Gitee 而不是 GitHub 的原因：`raw.githubusercontent.com` 在国内
/// 绝大多数网络下直接连不上，用户拿到的将永远是「检查失败」。
pub const VERSION_URL: &str = "https://gitee.com/aaa15/asmr-translator/raw/main/version.json";

/// 项目主页（下载页兜底用）
pub const RELEASES_PAGE: &str = "https://gitee.com/aaa15/asmr-translator";

/// 一次检查结果
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    /// 远端最新版本号，如 "1.28"
    pub version: String,
    /// 更新说明（可空）
    pub notes: String,
    /// 主下载地址（安装包直链）
    pub download_url: String,
    /// 备用镜像：(名字, 地址)，一般是百度网盘 / 夸克网盘
    pub mirrors: Vec<(String, String)>,
    /// ★ 环境包补丁（v1.30 起）。
    ///
    /// 背景：老用户装的是 1.29 那版运行环境包，里面**没有 Haas-ja 模型**
    /// （459 MB）。只更新主程序的话，Haas-ja 加载失败会静默降级到 SenseVoice ——
    /// 功能不坏，但拿不到这次识别准确率的提升。
    ///
    /// 重下 12 GB 全量环境包对老用户太贵，所以单独做一个 490 MB 的补丁包。
    /// 清单里给出版本和地址，程序自己判断「我缺不缺」再决定提示。
    pub runtime_patch: Option<RuntimePatch>,
}

/// 环境包补丁
#[derive(Debug, Clone)]
pub struct RuntimePatch {
    /// 这个补丁对应的环境包版本，如 "1.30.0"
    pub version: String,
    /// 补丁包下载地址
    pub url: String,
    /// 补丁包大小（MB），用来在界面上给用户预期
    pub size_mb: u32,
    /// 补丁说明
    pub notes: String,
}

impl UpdateInfo {
    /// 有没有比 `current` 新
    pub fn is_newer_than(&self, current: &str) -> bool {
        version_cmp(&self.version, current) == std::cmp::Ordering::Greater
    }
}

/// 语义化版本比较（只处理 1 ~ 4 段数字，忽略前面的 v）
pub fn version_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    fn parse(v: &str) -> Vec<u64> {
        v.trim()
            .trim_start_matches('v')
            .trim_start_matches('V')
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u64>().unwrap_or(0))
            .collect()
    }
    let (va, vb) = (parse(a), parse(b));
    let n = va.len().max(vb.len());
    for i in 0..n {
        let x = va.get(i).copied().unwrap_or(0);
        let y = vb.get(i).copied().unwrap_or(0);
        if x != y {
            return x.cmp(&y);
        }
    }
    std::cmp::Ordering::Equal
}

/// 从 JSON 文本里抠出一个字符串字段（不引 serde，用最朴素的定位）。
///
/// 只支持 `"key": "value"` 这种扁平结构 —— version.json 是我们自己写的，
/// 不需要完整 JSON 解析器。找不到返回 None。
fn json_str(raw: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let pos = raw.find(&pat)?;
    let rest = &raw[pos + pat.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let mut chars = after.char_indices();
    // 找起始引号
    let (start, _) = chars.find(|(_, c)| *c == '"')?;
    let body = &after[start + 1..];
    let mut out = String::new();
    let mut esc = false;
    for c in body.chars() {
        if esc {
            out.push(match c {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                other => other,
            });
            esc = false;
        } else if c == '\\' {
            esc = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

/// 抠出 `"mirrors": [ {"name":"", "url":""}, ... ]` 里的每一项
fn json_mirrors(raw: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(pos) = raw.find("\"mirrors\"") else {
        return out;
    };
    let rest = &raw[pos..];
    // 截到匹配的 ] 为止（够用：镜像数组里不会有嵌套方括号）
    let Some(end) = rest.find(']') else {
        return out;
    };
    let arr = &rest[..end];
    for chunk in arr.split('{').skip(1) {
        let name = json_str(chunk, "name");
        let url = json_str(chunk, "url");
        if let (Some(n), Some(u)) = (name, url) {
            if !u.trim().is_empty() {
                out.push((n, u));
            }
        }
    }
    out
}

/// 解析 version.json
pub fn parse_manifest(raw: &str) -> Option<UpdateInfo> {
    let version = json_str(raw, "version")?;
    let download_url = json_str(raw, "url").unwrap_or_default();
    Some(UpdateInfo {
        version,
        notes: json_str(raw, "notes").unwrap_or_default(),
        download_url,
        mirrors: json_mirrors(raw),
        runtime_patch: json_runtime_patch(raw),
    })
}

/// 解析可选的 `runtime_patch` 段。
///
/// 形状：
/// ```json
/// "runtime_patch": {
///   "version": "1.30.0",
///   "url": "https://.../ASMR-运行环境补丁_v1.30.0.exe",
///   "size_mb": 490,
///   "notes": "补上 Haas-ja 语音识别模型"
/// }
/// ```
/// 没有这一段（老清单）就返回 None，不报错。
fn json_runtime_patch(raw: &str) -> Option<RuntimePatch> {
    let i = raw.find("\"runtime_patch\"")?;
    let rest = &raw[i..];
    let l = rest.find('{')?;
    let r = rest.find('}')?;
    if r <= l {
        return None;
    }
    let body = &rest[l..=r];

    let url = json_str(body, "url").unwrap_or_default();
    if url.trim().is_empty() {
        return None; // 没有地址等于没有补丁
    }
    Some(RuntimePatch {
        version: json_str(body, "version").unwrap_or_default(),
        url,
        size_mb: json_u32(body, "size_mb").unwrap_or(0),
        notes: json_str(body, "notes").unwrap_or_default(),
    })
}

/// 取一个整数型字段（`"size_mb": 490` 这种，不带引号）
fn json_u32(raw: &str, key: &str) -> Option<u32> {
    let pat = format!("\"{key}\"");
    let i = raw.find(&pat)?;
    let rest = &raw[i + pat.len()..];
    let colon = rest.find(':')?;
    let tail = rest[colon + 1..].trim_start();
    let num: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

/// 用系统自带工具把 URL 内容读成字符串。
///
/// `curl.exe` 是 Windows 10 1803+ 的标准组件；老系统上退回 PowerShell。
fn http_get_text(url: &str, timeout_secs: u64) -> Result<String, String> {
    // ① curl.exe
    if let Ok(out) = Command::new("curl.exe")
        .args([
            "-sS",
            "-L",
            "--max-time",
            &timeout_secs.to_string(),
            "-H",
            "Cache-Control: no-cache",
            url,
        ])
        .stdin(Stdio::null())
        .no_window()
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).to_string();
            if !s.trim().is_empty() {
                return Ok(s);
            }
            return Err("服务器返回空内容".into());
        }
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        if !err.is_empty() {
            return Err(format!("curl 失败：{}", err));
        }
    }

    // ② PowerShell 兜底
    let ps = format!(
        "$ProgressPreference='SilentlyContinue';\
         try{{ $r=Invoke-WebRequest -Uri '{}' -UseBasicParsing -TimeoutSec {};\
         [Console]::OutputEncoding=[Text.Encoding]::UTF8; $r.Content }}catch{{ exit 1 }}",
        url.replace('\'', "''"),
        timeout_secs
    );
    match Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &ps])
        .stdin(Stdio::null())
        .no_window()
        .output()
    {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).to_string();
            if s.trim().is_empty() {
                Err("下载内容为空".into())
            } else {
                Ok(s)
            }
        }
        Ok(_) => Err("网络请求失败（curl 与 PowerShell 都没拿到内容）".into()),
        Err(e) => Err(format!("无法启动下载工具：{}", e)),
    }
}

/// 同步检查更新。**必须放在后台线程里调用**。
pub fn check_blocking(current: &str, timeout_secs: u64) -> Result<Option<UpdateInfo>, String> {
    // 允许用环境变量临时指向别处：本地自测（file:// 也行）、
    // 或者用户自己搭了镜像想换源，都不用重新编译。
    let url = std::env::var("ASMR_UPDATE_URL").unwrap_or_else(|_| VERSION_URL.to_string());
    let raw = http_get_text(&url, timeout_secs)?;
    let info = parse_manifest(&raw).ok_or_else(|| "版本清单格式不对（缺 version 字段）".to_string())?;
    if info.is_newer_than(current) {
        Ok(Some(info))
    } else {
        Ok(None)
    }
}

/// 启动后台检查，返回接收端。UI 每帧 try_recv 一次即可。
pub fn check_async(current: String) -> Receiver<Result<Option<UpdateInfo>, String>> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        // 稍微等一下：启动瞬间正在读配置/枚举设备，别抢带宽
        std::thread::sleep(Duration::from_secs(2));
        let _ = tx.send(check_blocking(&current, 12));
    });
    rx
}

/// 下载进度回调：已下载字节、总字节（总长未知时 None）
pub type ProgressFn<'a> = &'a mut dyn FnMut(u64, Option<u64>);

/// 下载安装包到 `dest`。**后台线程调用**。
///
/// 直接调 `curl.exe -o`，它能显示总长度，失败信息也清楚；
/// 没有 curl 时退回 PowerShell。
pub fn download_blocking(
    url: &str,
    dest: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
    cancel: &dyn Fn() -> bool,
) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_file(dest);

    // 用 curl 的分块输出模式：-s 静默，但保留进度靠我们自己读 stdout 不行，
    // 所以先拿 Content-Length，再分块写文件。简单起见：整体下载 + 轮询文件大小。
    let total: Option<u64> = http_head_length(url).ok();

    let mut child = Command::new("curl.exe")
        .args([
            "-sS",
            "-L",
            "--fail",
            "--max-time",
            "600",
            "-o",
            &dest.to_string_lossy(),
            url,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .no_window()
        .spawn()
        .map_err(|e| format!("启动 curl 失败：{}", e))?;

    // 轮询文件体积喂进度条
    loop {
        if cancel() {
            let _ = child.kill();
            let _ = std::fs::remove_file(dest);
            return Err("已取消".into());
        }
        match child.try_wait() {
            Ok(Some(st)) => {
                if !st.success() {
                    let mut buf = String::new();
                    if let Some(mut e) = child.stderr.take() {
                        let _ = e.read_to_string(&mut buf);
                    }
                    let _ = std::fs::remove_file(dest);
                    let msg = buf.trim();
                    return Err(if msg.is_empty() {
                        "下载失败".into()
                    } else {
                        format!("下载失败：{}", msg)
                    });
                }
                break;
            }
            Ok(None) => {
                let cur = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
                on_progress(cur, total);
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(format!("等待下载进程失败：{}", e)),
        }
    }

    let size = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    if size < 1024 {
        let _ = std::fs::remove_file(dest);
        return Err("下载下来的文件太小，可能不是安装包".into());
    }
    on_progress(size, Some(size));
    Ok(())
}

/// 只取 Content-Length，拿不到就算了（进度条退化成「已下载 xx MB」）
fn http_head_length(url: &str) -> Result<u64, String> {
    let out = Command::new("curl.exe")
        .args(["-sSI", "-L", "--max-time", "10", url])
        .stdin(Stdio::null())
        .no_window()
        .output()
        .map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let l = line.to_ascii_lowercase();
        if let Some(rest) = l.strip_prefix("content-length:") {
            if let Ok(n) = rest.trim().parse::<u64>() {
                return Ok(n);
            }
        }
    }
    Err("没有 Content-Length".into())
}

/// 用默认浏览器打开网址（下载页 / 网盘镜像都走这个）
pub fn open_url(url: &str) {
    // cmd /c start 同样会闪一个控制台窗口，这里也压掉。
    // 注意：压的是【cmd 自己】的窗口，浏览器照常正常弹出。
    let _ = Command::new("cmd")
        .args(["/c", "start", "", url])
        .stdin(Stdio::null())
        .no_window()
        .spawn();
}

/// 运行下好的安装包（Inno Setup，`/SILENT` 静默安装、装完自动启动新版本）
pub fn run_installer(path: &Path) -> Result<(), String> {
    Command::new(path)
        .args(["/SILENT", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS"])
        .stdin(Stdio::null())
        .no_window()
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("无法启动安装程序：{}", e))
}

/// 下载目录：`<工作目录>/_更新`
pub fn download_dir(data_dir: &str) -> PathBuf {
    Path::new(data_dir).join("_更新")
}

// ─────────────────────── 环境包补丁：本地缺不缺 ───────────────────────

/// 判断本地是否【缺少】Haas-ja 语音识别模型。
///
/// 用途：决定要不要在「发现新版本」弹窗里额外提示装环境包补丁。
///
/// 为什么要检测而不是单纯按版本号判断：
///   · 全新安装的用户拿到的环境包【已经带】这个模型，不该再被提示装补丁
///   · 老用户升级主程序后环境里没有它，才需要提示
///   只看版本号会把前者也骚扰一遍。
///
/// 判定标准：`model\Haas-ja-v1.0\` 下五个文件齐全才算有。
/// 少任何一个都算缺（残缺的模型加载会失败，等于没有）。
pub fn needs_runtime_patch(work_dir: &str) -> bool {
    const FILES: [&str; 5] = [
        "encoder.quantized.onnx",
        "encoder.data",
        "decoder.onnx",
        "joiner.onnx",
        "tokens.txt",
    ];
    let dir = Path::new(work_dir).join("model").join("Haas-ja-v1.0");
    if !dir.is_dir() {
        return true;
    }
    !FILES.iter().all(|f| dir.join(f).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_ordering() {
        assert_eq!(version_cmp("1.27.4", "1.27.3"), std::cmp::Ordering::Greater);
        assert_eq!(version_cmp("1.28", "1.27.9"), std::cmp::Ordering::Greater);
        assert_eq!(version_cmp("1.27", "1.27.0"), std::cmp::Ordering::Equal);
        assert_eq!(version_cmp("v1.27.4", "1.27.4"), std::cmp::Ordering::Equal);
        assert_eq!(version_cmp("1.26", "1.27"), std::cmp::Ordering::Less);
    }

    #[test]
    fn manifest_parse() {
        let raw = r##"{
          "version": "1.28",
          "url": "https://example.com/a.exe",
          "notes": "修复了若干问题\n第二行",
          "mirrors": [
            {"name": "百度网盘", "url": "https://pan.baidu.com/s/abc"},
            {"name": "夸克网盘", "url": "https://pan.quark.cn/s/def"}
          ]
        }"##;
        let info = parse_manifest(raw).expect("应能解析");
        assert_eq!(info.version, "1.28");
        assert_eq!(info.download_url, "https://example.com/a.exe");
        assert!(info.notes.contains("第二行"));
        assert_eq!(info.mirrors.len(), 2);
        assert_eq!(info.mirrors[0].0, "百度网盘");
        assert!(info.is_newer_than("1.27.4"));
        assert!(!info.is_newer_than("1.28"));
        // 老清单没有 runtime_patch 段，不该报错，也不该有补丁
        assert!(info.runtime_patch.is_none());
    }

    /// 带 runtime_patch 的新清单要能解析出补丁信息。
    #[test]
    fn manifest_parse_runtime_patch() {
        let raw = r##"{
          "version": "1.30.0",
          "url": "https://example.com/main.exe",
          "notes": "换了识别引擎",
          "mirrors": [],
          "runtime_patch": {
            "version": "1.30.0",
            "url": "https://example.com/patch.exe",
            "size_mb": 490,
            "notes": "补上 Haas-ja 模型"
          }
        }"##;
        let info = parse_manifest(raw).expect("应能解析");
        let p = info.runtime_patch.expect("应该有补丁段");
        assert_eq!(p.version, "1.30.0");
        assert_eq!(p.url, "https://example.com/patch.exe");
        assert_eq!(p.size_mb, 490);
        assert!(p.notes.contains("Haas-ja"));
    }

    /// 补丁段里没有 url 等于没有补丁 —— 不能因为字段残缺就弹个装不成的窗。
    #[test]
    fn runtime_patch_without_url_is_ignored() {
        let raw = r##"{
          "version": "1.30.0",
          "url": "https://example.com/main.exe",
          "runtime_patch": { "version": "1.30.0", "size_mb": 490 }
        }"##;
        let info = parse_manifest(raw).expect("应能解析");
        assert!(info.runtime_patch.is_none(), "没有 url 就不该算补丁");
    }

    /// needs_runtime_patch：模型文件齐全才算有，缺一个都算缺。
    #[test]
    fn runtime_patch_detection() {
        let root = std::env::temp_dir().join("asmr_patch_detect_test");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let root_s = root.to_string_lossy().to_string();

        // ① 目录都不存在 → 缺
        assert!(needs_runtime_patch(&root_s), "没有 model 目录应判缺");

        // ② 建目录但文件不全 → 仍缺
        let d = root.join("model").join("Haas-ja-v1.0");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("encoder.quantized.onnx"), b"x").unwrap();
        assert!(needs_runtime_patch(&root_s), "只有部分文件应判缺");

        // ③ 五个文件齐全 → 不缺
        for f in ["encoder.data", "decoder.onnx", "joiner.onnx", "tokens.txt"] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        assert!(!needs_runtime_patch(&root_s), "文件齐全应判不缺");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// 用仓库里真实的 version.json 验证解析 —— 保证补丁段能被读到。
    #[test]
    fn real_version_json_has_patch() {
        // 从 _proto_egui/target/release/ 往上找到仓库根
        let mut p = std::env::current_dir().unwrap();
        for _ in 0..6 {
            let cand = p.join("version.json");
            if cand.is_file() {
                let raw = std::fs::read_to_string(&cand).unwrap();
                let info = parse_manifest(&raw).expect("真实清单应能解析");
                println!("版本 = {}", info.version);
                let patch = info.runtime_patch.expect("真实清单里应该有补丁段");
                println!("补丁 = v{} ({} MB)", patch.version, patch.size_mb);
                assert!(patch.size_mb > 0);
                assert!(!patch.url.is_empty());
                return;
            }
            if !p.pop() {
                break;
            }
        }
        panic!("找不到 version.json");
    }

}
