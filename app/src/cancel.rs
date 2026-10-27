//! 取消令牌的间接层。
//!
//! 为什么要多这一层：`pipeline.rs` 需要 `CancellationToken`，Windows 版用的是
//! windows-reactor 提供的那个类型；而 egui 原型（`_proto_egui`）用 `#[path]`
//! 直接引 pipeline.rs 源码复用，却不想把整个 windows-reactor 拖进来编译。
//!
//! 于是两边各自提供 `crate::cancel`，pipeline.rs 只认这个名字：
//!   · 主程序（这里）  → 转发 reactor 的类型，与 ui.rs 里 spawn_background
//!                        拿到的 token 是同一个类型，无需转换
//!   · egui 原型        → 自己的 `Arc<AtomicBool>` 实现
//!
//! 这样这处「唯一的外部依赖」就不会把两边的编译绑死。

pub use windows_reactor::CancellationToken;
