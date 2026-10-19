# Git 工作流规范

本项目是单人维护的桌面工具，用轻量的 trunk-based 流程，不引入多余分支。

## 分支

| 分支 | 用途 |
|---|---|
| `main` | 稳定线，任何时候都能编译、测试通过 |
| `work/<主题>` | 较大改动（跨多轮的重构）临时开，完成后合回 `main` |

小改动（一个 bug、一个小功能）直接在 `main` 上做，不为了流程而流程。

## 提交信息

格式：

```
<type>: <一句话说明>

<可选：为什么改 / 影响范围>
```

| type | 含义 |
|---|---|
| `feat` | 用户可见的新功能 |
| `fix` | 修 bug（不改变预期行为之外的东西） |
| `refactor` | 重构，行为不变 |
| `perf` | 性能优化 |
| `docs` | 文档 |
| `chore` | 版本号、打包、配置、依赖 |
| `test` | 只加/改测试 |

示例：

```
feat: 兼容模式支持独显优先（显存安全空间不足时退核显）
fix: 字幕整理在空目录时误报失败
refactor: 字幕整理从 Python 迁到 Rust（srt.rs）
chore: v1.21.0
```

**一次提交只做一件事。** 不要把「升级版本号」和「修 bug」混在一个提交里。

## 版本与标签

- 版本号规则见 `AGENTS.md` 第 9 节：修 bug 第三位 +1，加功能第二位 +1，架构大改第一位 +1
- 三处必须同步：`app/src/config.rs` 的 `APP_VERSION`、`_candidate/b1_main.iss`、`_candidate/b2_runtime.iss`
- 每次发版：测试 → 编译 → 启动看标题 → 提交 → 打标签

```bash
git tag -a v1.21.0 -m "v1.21.0 纯逻辑 Rust 化"
```

## 发版检查清单

```powershell
cargo test                                   # 必须全过
cargo build --release
# 复制 exe 到发布目录，启动一次确认窗口标题
git add -A
git commit -m "chore: v1.21.0"
git tag -a v1.21.0 -m "..."
```

## 回滚

```bash
git checkout v1.20.0 -- .        # 取某个文件的旧版本
git revert <commit>              # 撤销某次提交（保留历史）
git checkout v1.20.0             # 临时切到旧版本查看
```

## 不纳入版本库的内容

见 `.gitignore`，主要是：

- `env\`（Python 运行环境，数 GB）
- `model\`、`llama\`（模型权重和 llama.cpp 二进制，体积大、许可独立）
- `0pre\` / `1audio\` / `2vad\` / `3asr\`（运行时中间文件）
- 日志、队列文件、安装包、备份

源码（Rust + Python）和配置模板（词表、热词、使用说明）才进仓库。
