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

> **注**：字幕读写使用项目自带的 `app/srt_util.py`，无任何第三方依赖。

---

## 三、模型权重

**本仓库（git）不包含任何模型权重** —— `.gitignore` 里排除了 `model/`、`*.ckpt`、`*.onnx`、`*.gguf`、`*.bin`、`*.safetensors`。

**但发布的「运行环境安装包」里打包了这些权重**，这样使用者装完就能用，不需要 HuggingFace 账号、不需要联网下载。它们各自的许可如下：

| 模型 | 文件 | 大小 | 来源 | 许可证 |
|---|---|---|---|---|
| **BS-Roformer**（人声分离） | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 610 MB | Ultimate Vocal Remover 社区 | MIT |
| **whisper-large-v2-translate-zh-v0.2-st-ct2**（识别 + 日→中翻译） | `models--4evergr8--…`（HF cache） | 2.9 GB | [4evergr8](https://huggingface.co/4evergr8) | MIT（基于 OpenAI Whisper 的 MIT 许可派生） |
| **SenseVoiceSmall**（兼容模式识别） | `SenseVoiceSmall/model.pt` | 893 MB | 阿里 FunAudioLLM | MIT |
| **Qwen2.5-1.5B-Instruct GGUF**（兼容模式翻译） | `qwen2.5-1.5b-q4km.gguf` | 940 MB | 阿里通义千问 | Apache-2.0 |
| **pyannote/segmentation-3.0**（VAD） | `models--4evergr8--pyannote-segmentation-3.0`（HF cache） | 5.6 MB | [4evergr8](https://huggingface.co/4evergr8) 的转存版 | MIT |

> **关于 pyannote**：官方 `pyannote/segmentation-3.0` 是 gated model（需要登录 HF 并同意条款才能下载），但本项目用的是 `4evergr8` 转存到公开仓库的同一份权重（MIT 许可），随包分发的也是这份缓存。**使用者不需要 HF 账号**。
>
> 如果你要自己重新分发，请保留上表的来源与许可说明；如果替换成官方 gated 版本，就需要你自己持有 HF token，并且**不得**把权重再分发出去。

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
① ffmpeg —— 如果用的是 GPL 构建（如 gyan.dev 的 full 版），
   分发时要一并提供其源码或源码获取方式
② 各个 Python 包 —— 保留它们的 LICENSE 文件
③ 模型权重 —— 按上面第三节的表格保留来源与许可说明；
   pyannote 若改用官方 gated 版本，则不得随包分发
④ 建议在下载页或安装向导里放一句「本软件包含第三方模型，
   各自遵循其原始许可，详见 THIRD_PARTY_LICENSES.md」
```

**本项目附带的 `使用说明.txt` 与 `docs/RELEASE.md` 里都已写明模型来源与许可。**
