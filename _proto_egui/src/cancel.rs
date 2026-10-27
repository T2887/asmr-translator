//! `CancellationToken` 的 egui 原型版实现。
//!
//! 为什么有这个文件：`pipeline.rs` 里有一处 `use crate::cancel::CancellationToken;`
//! （原本是 `use windows_reactor::CancellationToken;`）。Windows 版主程序在
//! `app/src/cancel.rs` 里直接转发 windows-reactor 的类型；这个 egui 原型用
//! `#[path]` 引同一份 pipeline.rs 源码，但**不想**把整个 windows-reactor
//! （连带它的 WinUI 绑定）拖进依赖树，于是这里给一个语义等价的最小实现。
//!
//! 两个实现的公开接口一致：`new()` / `cancel()` / `is_cancelled()`（Clone 可用）。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    pub fn reset(&self) {
        self.flag.store(false, Ordering::SeqCst);
    }
}
