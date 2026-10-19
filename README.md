# ASMR 自动翻译工具

把日语 ASMR 音声 / 视频批量翻译成中文字幕（`.chs.srt`），生成后直接放在源文件旁边，播放器可以自动加载。

**全程离线**，不上传任何文件。

Windows 桌面程序：**Rust + WinUI 3**（界面）+ **Python**（AI 推理）。

---

## 它做什么

四步流水线，**顺序是 VAD 在前**：

```
① 提取音频     ffmpeg 把视频/音频统一转成 flac
② 人声检测     pyannote VAD 圈出「有语音的片段」
③ 分离人声     只分离上一步圈出的片段（这一步决定了速度）
④ 语音识别     whisper 直接翻译成中文，输出 .srt
```

**为什么 VAD 要放在分离之前**：实测语音只占音频的 **15%~42%**（ASMR 里大量静音、喘息、环境音），而人声分离的耗时与音频长度严格成正比。先圈出语音段再只分离那些段，整条链路能省 **60%** 的时间。

> 实测：12 分钟视频 145 秒 → 58.7 秒；30 分钟音声 分离 122 秒 → 52 秒。

---

## 硬件要求

**必须有 NVIDIA 显卡。**

| 显存 | 状态 | 说明 |
|---|---|---|
| **≥ 8.1 GB** | 全速（float16） | 峰值约 7.3 GB，给其他程序留 5 GB |
| **6 ~ 8.1 GB** | 精简模式（int8） | 峰值约 5 GB，识别慢约 20%，个别词可能不准 |
| **< 6 GB** | **只能用 CPU** | 程序会弹窗确认，同意则跑（慢 50 倍），不同意则退出 |
| AMD / Intel | 不支持 | PyTorch / ONNX Runtime / CTranslate2 在 Windows 上都没有可用的 A 卡后端 |

**显存实测（RTX 5070 12GB，5 分钟音频）**：

```
第2步 VAD       +869 MiB
第3步 分离     +2585 MiB
第4步 识别     +4265 MiB
全程峰值        6860 MiB
```

**笔记本双显卡**：如果程序检测不到 N 卡，在 NVIDIA 控制面板 → 管理 3D 设置 → 程序设置里，把这个程序指定为「高性能 NVIDIA 处理器」。

---

## 速度

以 1 小时视频为例：**约 3 分钟**（RTX 5070）。

明细（12 分钟视频实测）：

```
VAD        1.1 秒
分离       9.5 秒     ← 只处理 15% 的音频
识别       7.2 秒
其余      41.0 秒     ← 模型加载、拼接、填回、文件读写
────────────────────
总        58.7 秒
```

---

## 目录结构

```
ASMR翻译器-rust/
├── app/                    Rust 主程序
│   ├── src/
│   │   ├── main.rs         入口、窗口居中钩子
│   │   ├── ui.rs           全部界面代码
│   │   ├── pipeline.rs     四个脚本的调度、字幕整理与收尾
│   │   ├── backend.rs      Python 子进程管理
│   │   └── config.rs       路径、版本号、路径解析
│   ├── a_提取音频.py       ← 下面这些是 Python 侧
│   ├── b_分离人声.py
│   ├── c_人声检测.py
│   ├── d_语音识别.py
│   ├── e_字幕整理.py       字幕时间轴整理（也能 --batch 批量修旧字幕）
│   ├── srt_util.py         自写的 SRT 读写（替代 GPL 的 pysrt）
│   ├── events.py           结构化事件（@@EV@@ 行，给 Rust 侧读进度/字幕）
│   ├── 词表.txt            固定错译替换表（每行 错误=>正确，可自己加）
│   ├── asmr词汇.txt        ASMR 常用日语词表（参考用，非默认热词）
│   ├── settings.py         路径与模型配置、显存自适应
│   └── Cargo.toml
├── 开发笔记.md              踩过的坑都在这里，改代码前建议先翻
├── LICENSE                 MIT
└── THIRD_PARTY_LICENSES.md 第三方组件声明
```

---

## 如何搭起来

**本项目不包含模型权重和 Python 环境**，需要自己准备。

### 1. 准备 Python 环境

```bash
python -m venv env
env\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install audio-separator[cpu] faster-whisper pyannote.audio librosa soundfile pydub pyyaml
```

**注意**：`pysrt` **不要装** —— 项目已经用自带的 `srt_util.py` 替代了它（见下方「许可证」一节）。

### 2. 准备模型

三个模型都**不在本仓库内**，需要自己下载放进 `model/` 目录。

---

#### ① 人声分离：BS-Roformer

```
文件名   model_bs_roformer_ep_317_sdr_12.9755.ckpt
大小     610 MB
```

**最省事的方式**：安装 `audio-separator` 之后**第一次运行会自动下载**到 `model/` 目录，不用手动找。

如果要手动下载，从这里找（`model_bs_roformer_ep_317_sdr_12.9755.ckpt` + 同名的 `.yaml` 配置文件都要）：

- **UVR 模型仓库**：https://github.com/TRvlvr/model_repo/releases
- **UVR 主项目**：https://github.com/Anjok07/ultimatevocalremovergui
- **audio-separator 项目**（含模型列表与文档）：https://github.com/nomadkaraoke/python-audio-separator

> 这个权重来自 UVR（Ultimate Vocal Remover）社区。许可证情况请以原仓库声明为准。

---

#### ② 语音识别：Whisper large-v2 日→中翻译版

```
文件名   whisper-large-v2-translate-zh-v0.2-st-ct2
大小     2.9 GB
仓库     https://huggingface.co/4evergr8/whisper-large-v2-translate-zh-v0.2-st-ct2
```

**注意**：这是**社区微调的翻译专用模型**，不是官方原版 Whisper。

原版 Whisper 的 `task='translate'` 只能日→英；这个模型被改成了**日→中**，所以能一步出中文。

**基座是 large-v2 而不是 v3** —— 这是有意的：v3 把 mel 特征从 80 改到 128，在长音频和带噪声的素材上更容易产生幻觉，而 ASMR 正好是这类素材。

**下载方式**（会自动进 HuggingFace 缓存）：

```python
from faster_whisper import WhisperModel
m = WhisperModel("4evergr8/whisper-large-v2-translate-zh-v0.2-st-ct2",
                 device="cuda", compute_type="float16")
```

或者直接把仓库 clone 到 `model/` 下任意子目录，程序会自动找（靠 `model.bin` + `config.json` 识别）。

---

#### ③ 语音活动检测：pyannote

```
仓库     https://huggingface.co/pyannote/segmentation-3.0
大小     5.6 MB
```

**⚠ 这是 gated model，必须先申请访问权限**：

```
1. 登录 HuggingFace
2. 打开 https://huggingface.co/pyannote/segmentation-3.0
3. 点「Agree and access repository」同意使用条款
4. 在 https://huggingface.co/settings/tokens 建一个 read token
5. 程序里加载时传 token，或者设置环境变量 HF_TOKEN
```

**⚠ 请勿把它的权重文件打包进任何再分发产物** —— gated model 的条款不允许。程序是运行时用 `Model.from_pretrained()` 加载的，不内嵌权重。

---

#### 模型目录结构

```
model/
├── model_bs_roformer_ep_317_sdr_12.9755.ckpt     ← 分离权重
├── model_bs_roformer_ep_317_sdr_12.9755.yaml     ← 分离配置
├── models--4evergr8--whisper-large-v2-translate-zh-v0.2-st-ct2/   ← HF 缓存目录
└── models--4evergr8--pyannote-segmentation-3.0/                   ← HF 缓存目录
```

**后两个是 HuggingFace 的缓存结构**（`blobs/` 存真文件，`snapshots/` 放符号链接），用 `huggingface_hub` 下载会自动长这样。

### 3. 准备 ffmpeg

放到 `env/ffmpeg/ffmpeg.exe`。**必须内置**，不能靠系统 PATH —— 用户机器上很可能没有。

### 4. 编译 Rust

```bash
cd app
cargo build --release
```

产物 `target/release/asmr_translator.exe` 放到项目根目录，和 `app/`、`env/`、`model/` 同级。

---

## 许可证

**MIT**（见 [LICENSE](LICENSE)）。

第三方组件的许可见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。

> **关于 `pysrt`**：项目早期用 `pysrt` 生成字幕，但它是 **GPLv3**，具有传染性 —— 只要 import 它，整个项目就必须以 GPLv3 发布。为了能以 MIT 发布，已经改成项目自带的 `app/srt_util.py`（约 60 行，无第三方依赖）。
>
> **贡献代码时请不要引入 GPL / AGPL 系的依赖**，否则会破坏整个项目的许可。

**模型权重不在本仓库内**，各自遵守原作者的许可。特别是 `pyannote/segmentation-3.0` 是 gated model，**不要把它打包进任何再分发产物**。

---

## 已知限制

- **只支持 NVIDIA 显卡**（AMD 在 Windows 上没有可用的加速后端）
- **4 GB 及以下显存**只能跑 CPU 模式，速度约为 GPU 的 1/50
- **Whisper large-v2 比 v3 更适合日语**（v3 改了 mel 特征，在长音频和噪声上更容易产生幻觉）
- 模型加载每步各一次（4 个独立进程），约占总时间的 70%。合并成单进程能再省 5-6 秒/文件，但改动较大，暂未做
