# 第三方组件与许可证声明

本项目以 **MIT** 许可证发布（见 [LICENSE](LICENSE)）。

下面是运行时用到的第三方组件及其许可证。**本项目的 MIT 许可不改变这些组件各自的许可条款** —— 使用、再分发它们时请遵守各自的要求。

---

## 一、Rust 依赖

全部通过 Cargo 引入。用 `cargo metadata` 统计，共 77 个 crate，许可证分布：

| 许可证 | 数量 |
|---|---|
| MIT OR Apache-2.0 | 44 |
| MIT | 14 |
| MIT/Apache-2.0 | 3 |
| Zlib OR Apache-2.0 OR MIT | 3 |
| Apache-2.0 OR MIT | 3 |
| Unlicense OR MIT | 2 |
| Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | 2 |
| (Apache-2.0 OR MIT) AND BSD-3-Clause | 1 |
| ISC | 1 |
| UNKNOWN | 1 |

直接依赖：

| crate | 版本 | 许可证 |
|---|---|---|
| `encoding_rs` | 0.8.42 | (Apache-2.0 OR MIT) AND BSD-3-Clause |
| `regex` | 1.13.1 | MIT OR Apache-2.0 |
| `rfd` | 0.17.2 | MIT |
| `windows-reactor` | 0.100 | MIT OR Apache-2.0 |
| `windows-sys` | 0.61.2 | MIT OR Apache-2.0 |

**这些许可证都允许在 MIT 项目中使用，且不要求本项目改为 copyleft。**

---

## 二、Python 依赖

| 包 | 许可证 | 用途 |
|---|---|---|
| **torch** | BSD-3-Clause | 深度学习框架 |
| **torchaudio** | BSD-2-Clause | 音频处理 |
| **numpy** | BSD-3-Clause | 数组运算 |
| **librosa** | ISC | 音频分析 |
| **soundfile** | BSD-3-Clause | 读写音频文件 |
| **pydub** | MIT | 音频拼接 |
| **pyyaml** | MIT | 配置文件 |
| **audio-separator** | MIT | 人声分离封装 |
| **faster-whisper** | MIT | Whisper 推理（CTranslate2 后端） |
| **CTranslate2** | MIT | 推理引擎 |
| **ONNX Runtime** | MIT | ONNX 推理 |
| **pyannote.audio** | MIT | 语音活动检测 |

> **注**：项目曾使用 `pysrt`（GPLv3）生成字幕文件。GPLv3 具有传染性，会使整个项目被迫以 GPLv3 发布。为了能以 MIT 发布，已改为使用项目自带的 `app/srt_util.py`（无第三方依赖，约 60 行）。

---

## 三、模型权重（**不随本项目分发**）

本项目**不包含**以下模型文件。它们由使用者自行下载，各自遵守原作者的许可：

| 模型 | 来源 | 说明 |
|---|---|---|
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) 社区 | 人声分离，610 MB |
| `whisper-large-v2-translate-zh-v0.2-st-ct2` | [4evergr8](https://huggingface.co/4evergr8)（HuggingFace） | 日语→中文翻译转写，2.9 GB |
| `pyannote/segmentation-3.0` | [pyannote](https://huggingface.co/pyannote/segmentation-3.0) | VAD。**gated model**，需要在 HuggingFace 同意使用条款后下载 |

**特别是 pyannote 的模型**：它是 gated（需申请访问权限）的，**请勿把它的权重文件打包进任何再分发产物**。项目代码通过 `Model.from_pretrained()` 在运行时加载，不内嵌权重。

---

## 四、其他

| 组件 | 许可证 | 说明 |
|---|---|---|
| **ffmpeg** | LGPL-2.1+ / GPL（取决于构建选项） | 音频提取。本项目通过子进程调用 `ffmpeg.exe`，**不链接其库**。GPL 构建的 ffmpeg 可以这样调用而不会传染本项目（进程边界隔离）。 |

---

## 五、如果你要再分发本项目的成品

**只分发代码**（不含 `env/`、`model/`、`ffmpeg`）：

```
✓ 直接遵守本项目的 MIT 许可证
✓ 在合适位置保留 LICENSE 文件和版权声明
```

**分发带运行环境的成品**（比如你自己打的安装包）：

```
需要额外注意：
① ffmpeg —— 如果用的是 GPL 构建（如 gyan.dev 的 full 版），
   分发时要一并提供其源码或源码获取方式
② 各个 Python 包 —— 保留它们的 LICENSE 文件
③ 模型权重 —— 不要分发 pyannote 的 gated 模型
```

**本项目附带的 `使用说明.txt` 里已经提醒用户自行确认模型的使用条款。**
