# ASMR 自动翻译工具

把日语 ASMR 音声 / 视频批量翻译成中文字幕（`.chs.srt`），生成后直接放在源文件旁边，播放器可以自动加载。

**全程离线**，不上传任何文件。

Windows 桌面程序：**Rust + egui**（界面，自绘）+ **Python**（AI 推理）。
界面自绘，不依赖 Windows App Runtime / VC++ 运行库：**Win10、Win11 双击即跑**。

---

## 下载安装（普通使用者看这里）

程序分**两个包**，两个都要下、都要装 —— 这样升级主程序时只需要重下 131 MB，而不是 6.7 GB。

| 包 | 内容 | 大小 | 下载 |
|---|---|---|---|
| **主程序安装包** | 程序本体 + 界面 + llama.cpp | **131 MB** | 👉 [Releases](../../releases) 的 Assets，或百度 / 夸克网盘（链接见 [Release 说明](docs/RELEASE.md)） |
| **运行环境**（安装包分卷或 zip） | Python 环境 + 全部模型权重 | **6.7 GB** | 百度网盘 / 夸克网盘（链接见 [Release 说明](docs/RELEASE.md)） |

两个包都装到 `C:\Program Files\ASMR自动翻译工具`（安装时会弹一次管理员授权）。
日志和中间文件在 `%LOCALAPPDATA%\ASMR自动翻译工具\`，字幕仍旧直接生成在视频旁边。

装完双击桌面「ASMR 自动翻译工具」即可，**全程离线**。

详细步骤、系统要求、常见问题见 **[docs/RELEASE.md](docs/RELEASE.md)**。

> 下面「如何搭起来」一节是**从源码自己编译**的流程，普通使用者不需要看。

---

## 三种处理模式

程序按显卡情况选模式，也可以在界面上手动切。

| 模式 | 适用 | 怎么跑的 |
|---|---|---|
| **全量模式** | GTX16系 / RTX20系以上，显存 ≥8 GB | whisper large-v2 **日→中翻译特化模型**一步出中文，不经过日文文本层 |
| **精简模式** | 同上，显存 ≥6 GB | 同上，但用 int8 量化，约省 2 GB 显存，识别慢约 20% |
| **兼容模式** | GTX10系及更早、非 N 卡、核显（Vulkan） | 两段式：sherpa-onnx 的日语 ASR 出日文 → llama.cpp 跑微调过的 1.8B 翻译模型出中文 |

**全量 / 精简**是端到端：音频进去，中文字幕出来。
**兼容模式**是两段式，有一个**日文文本层**——所以只有它能额外导出 `.jpn.srt` 日文字幕。

### 四步流水线

```
① 提取音频     ffmpeg 把视频/音频统一转成 flac
② 人声检测     圈出「有语音的片段」；兼容模式同时做语音转写
③ 分离人声     只分离上一步圈出的片段（这一步决定了速度）
④ 语音识别     whisper 直接翻译成中文，输出 .srt
```

**为什么 VAD 要放在分离之前**：实测语音只占音频的 **15%~42%**（ASMR 里大量静音、喘息、环境音），而人声分离的耗时与音频长度严格成正比。先圈出语音段再只分离那些段，整条链路能省 **60%** 的时间。

> **全量模式默认关掉 whisper 自己的 VAD（v1.30.6 起）**：whisper 内置的 VAD 会把喘息、耳语、气声当成静音丢掉，实测覆盖率从 71% 掉到 87%（关 VAD 后反而上去），漏译从 0.85 分钟降到 7 秒。
> 第 ② 步的 VAD 仍然要做 —— 它是给第 ③ 步省时间的，两者是不同的东西。

---

## 字幕校对

翻译完成后，**「实时翻译预览」卡片右上角**会出现「✎ 校对字幕」按钮。

- 逐句列出时间戳和文本，**单击时间戳可以试听那一段**
- **双击文本直接改**
- 每行右侧 `×` 删除整句，点 `↺` 撤回
- 兼容模式多一列「识别原文」（日文），全量模式没有
- 点「保存并写回字幕」直接改掉已经生成的 `.chs.srt`（原文件备份在数据目录的 `backup\` 里）

日文原文明细保存在 `2vad\<名字>.json`，只在**本次程序运行期间**可用——关掉程序就等于任务结束，不需要留这个文件。需要日文字幕请勾「生成日文字幕」。

---

## 幻觉治理

全量模式下 whisper 在纯音效/静音段会编出不存在的台词（倒计时、拟声词、结尾的「感谢您的视听」）。按**「宁可漏翻，不可错翻」**的原则，v1.40.0 加了三道过滤：

| 过滤 | 拦掉的 |
|---|---|
| **纯数字** | whisper 训练数据里的倒计时 `10 9 8 7 … 0` |
| **纯拟声 / 元描述** | `叭叭叭叭`、`滋滋滋`、`约20秒的舌吻` 这类 |
| **人声闸** | 可疑的长段（>6 秒且字密度低）用日语 ASR 复核，它只听到拟声词或什么都没听到，就整段丢弃 |

人声闸的阈值是用 **69,808 条中日对照语料**校准的，误杀率约 **1%**。
另外 `initial_prompt` 也缩短了——之前那串术语会被 whisper 原样吐成一条字幕。

`ASMR_NO_JUNK=1` / `ASMR_NO_ONO=1` / `ASMR_SPEECH_GATE=0` 可以单独关掉对应的过滤。

---

## 硬件要求

**有 NVIDIA 显卡最好**（能用 CUDA，最快）。没有也能跑 —— AMD / Intel 核显走「兼容模式」（Vulkan 或 CPU），只是慢。

| 显存 | 状态 | 说明 |
|---|---|---|
| **≥ 8.1 GB** | 全量模式（float16） | 峰值约 7.3 GB，给其他程序留 5 GB |
| **6 ~ 8.1 GB** | 精简模式（int8） | 峰值约 5 GB，识别慢约 20%，个别词可能不准 |
| **< 6 GB** | **兼容模式** | 自动切到 sherpa-onnx + llama.cpp（Vulkan 后端），比 CUDA 慢，但能出结果 |
| AMD / Intel 核显 | **兼容模式** | 同上，走 Vulkan 或纯 CPU |

**显存实测（RTX 5070 12GB，5 分钟音频，全量模式）**：

```
第2步 VAD       +869 MiB
第3步 分离     +2585 MiB
第4步 识别     +4265 MiB
全程峰值        6860 MiB
```

**笔记本双显卡**：如果程序检测不到 N 卡，在 NVIDIA 控制面板 → 管理 3D 设置 → 程序设置里，把这个程序指定为「高性能 NVIDIA 处理器」。

---

## 速度

以 1 小时视频为例：**约 3 分钟**（RTX 5070，全量模式）。

明细（12 分钟视频实测）：

```
VAD        1.1 秒
分离       9.5 秒     ← 只处理 15% 的音频
识别       7.2 秒
其余      41.0 秒     ← 模型加载、拼接、填回、文件读写
────────────────────
总        58.7 秒
```

兼容模式明显更慢：7 分钟的音频约 4~5 分钟（两段式模型 + CPU 推理）。

---

## 目录结构

```
asmr-translator/
├── _proto_egui/            当前界面（Rust + egui，打包用的就是这个）
│   └── src/
│       ├── main.rs         全部界面代码
│       ├── worker.rs       子进程调度、控制台输出解析
│       ├── proofread.rs    字幕校对窗口
│       ├── updater.rs      版本检查（从 Gitee 拉 version.json）
│       └── subscan.rs      沿目录扫描已有字幕
├── app/src/                共用逻辑（被 _proto_egui 以 #[path] 引用）
│   ├── pipeline.rs         四个脚本的调度、字幕整理与收尾
│   ├── backend.rs          Python 子进程管理
│   ├── config.rs           路径、版本号
│   ├── srt.rs              SRT 读写、词表规则
│   └── ui.rs               旧 WinUI3 界面（已废弃，仅存档）
├── package/ASMR自动翻译工具/
│   ├── app/                实际随包发布的 Python 脚本
│   │   ├── a_提取音频.py
│   │   ├── b_分离人声.py
│   │   ├── c_人声检测.py
│   │   ├── d_语音识别.py       全量 / 精简模式
│   │   ├── light_asr.py        兼容模式 第2步
│   │   ├── light_translate.py  兼容模式 第3步
│   │   ├── whisper_split.py    分段、合并、刷屏折叠、幻觉过滤
│   │   ├── status_probe.py     GPU / CPU 状态采集
│   │   ├── terms_fix.py        术语字典
│   │   ├── srt_util.py         SRT 读写（自带实现，无第三方依赖）
│   │   └── events.py           结构化事件（@@EV@@ 行）
│   ├── fixes.txt           固定错译替换表（可自己加）
│   ├── hotwords.txt        ASMR 常用词表
│   └── model/              模型权重（不进 git）
├── docs/                   发布说明、踩坑记录
├── LICENSE                 MIT
└── THIRD_PARTY_LICENSES.md 第三方组件声明
```

---

## 如何搭起来

**本仓库不包含模型权重和 Python 环境** —— 普通使用者请直接下安装包（见上面的「下载安装」），不用看这一节。

下面是从源码自己搭的流程，需要自己准备环境和模型。

### 1. 准备 Python 环境

```bash
python -m venv env
env\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install audio-separator[cpu] faster-whisper pyannote.audio librosa soundfile pydub pyyaml
pip install sherpa-onnx            # 兼容模式的日语 ASR
```

### 2. 准备模型

模型都**不在本仓库内**，需要自己下载放进 `model/` 目录。

---

#### ① 人声分离：BS-Roformer

```
文件名   model_bs_roformer_ep_317_sdr_12.9755.ckpt
大小     610 MB
```

**最省事的方式**：安装 `audio-separator` 之后**第一次运行会自动下载**到 `model/` 目录。

如果要手动下载（`.ckpt` 和同名 `.yaml` 都要）：

- **UVR 模型仓库**：https://github.com/TRvlvr/model_repo/releases
- **audio-separator 项目**：https://github.com/nomadkaraoke/python-audio-separator

> 权重来自 UVR 社区，许可证以原仓库声明为准。

---

#### ② 全量模式：Whisper large-v2 日→中翻译版

```
文件名   whisper-large-v2-translate-zh-v0.2-st-ct2
大小     2.9 GB
仓库     https://huggingface.co/4evergr8/whisper-large-v2-translate-zh-v0.2-st-ct2
```

**注意**：这是**社区微调的翻译专用模型**，不是官方原版 Whisper。原版 `task='translate'` 只能日→英；这个被改成了**日→中**，一步出中文。

**基座是 large-v2 而不是 v3** —— 这是实测结论：v3 把 mel 特征从 80 改到 128，长音频/带噪声素材上更容易幻觉，而且我们实测过 v3 的中文特化版（`chickenrice0721/whisper-large-v3-translate-zh-v0.1-lt-ct2`）覆盖率反而掉 6.6 个点，还有 25 秒只出「就这样」三个字的段长失控问题。

**下载方式**：

```python
from faster_whisper import WhisperModel
m = WhisperModel("4evergr8/whisper-large-v2-translate-zh-v0.2-st-ct2",
                 device="cuda", compute_type="float16")
```

或者把仓库 clone 到 `model/` 下任意子目录，程序靠 `model.bin` + `config.json` 自动找。

---

#### ③ 兼容模式：日语 ASR + 翻译模型

```
日语 ASR    Haas-ja（sherpa-onnx 格式）
翻译模型    Hy-MT2-1.8B + LoRA（gguf，约 74 MB）
```

这两个是兼容模式用的。`model/` 里放 `hy.gguf`（主模型）和 `hy-mt2-asmr-lora.gguf`（LoRA 适配器），llama.cpp 启动时用 `--lora` 挂载。

---

#### ④ 语音活动检测：pyannote

```
仓库     https://huggingface.co/pyannote/segmentation-3.0
大小     5.6 MB
```

**⚠ 官方仓库是 gated model**，直接下载要先申请访问权限。**但本项目用的是 `4evergr8/pyannote-segmentation-3.0`——同一份权重转存到公开仓库的版本（MIT）**，不需要 token、不需要申请。发布的「运行环境」里缓存的就是这一份。

如果你坚持用**官方 gated 版本**，那就需要自己申请 token，并且**不要**把它的权重打包进任何再分发产物。

---

#### 模型目录结构

```
model/
├── model_bs_roformer_ep_317_sdr_12.9755.ckpt     ← 分离权重
├── model_bs_roformer_ep_317_sdr_12.9755.yaml     ← 分离配置
├── models--4evergr8--whisper-large-v2-translate-zh-v0.2-st-ct2/   ← HF 缓存目录
├── models--4evergr8--pyannote-segmentation-3.0/                   ← HF 缓存目录
├── hy.gguf                                       ← 兼容模式翻译主模型
├── hy-mt2-asmr-lora.gguf                         ← LoRA 适配器
└── vad_asmr_encdec.onnx                          ← ASMR 专用 VAD（可选）
```

**HF 缓存结构**是 `blobs/` 存真文件、`snapshots/` 放符号链接，用 `huggingface_hub` 下载会自动长这样。

### 3. 准备 ffmpeg

放到 `env/ffmpeg/ffmpeg.exe`。**必须内置**，不能靠系统 PATH —— 用户机器上很可能没有。

### 4. 编译 Rust

```bash
cd _proto_egui
cargo build --release
```

产物 `target/release/asmr_proto_egui.exe`，改名为 `ASMR自动翻译工具.exe` 放到 `package/ASMR自动翻译工具/` 下，和 `app/`、`model/` 同级。

> 注意：`_proto_egui` 用 `#[path = "../../app/src/xxx.rs"]` 直接引用 `app/src/` 里的模块，所以这两个目录的相对位置不能变。

### 5. 打包

用 Inno Setup 编译 `package/_candidate/b1_main.iss`，产物约 131 MB。

---

## 许可证

**MIT**（见 [LICENSE](LICENSE)）。

第三方组件的许可见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。

> **贡献代码时请不要引入 GPL / AGPL 系的依赖** —— 这类许可证有传染性，会让整个项目被迫换许可证。字幕读写用的是项目自带的 `srt_util.py`，没有任何第三方依赖。

**模型权重不在本仓库（git）内**，各自遵守原作者的许可。它们被打包在「运行环境」里随程序分发（详见 THIRD_PARTY_LICENSES.md 第三节），其中 pyannote 用的是公开转存版而非官方 gated 版。

---

## 图标

图标**是纯代码画出来的**，没有任何第三方素材：赤陶橙圆角方 + 纸白「三根声波竖条 + 一条字幕线」。

- 生成脚本：`tools/make_icon.py`（Python + Pillow，约 130 行）
- 产物：`app/assets/icon.png`（1024，运行时窗口图标）、`app/assets/icon.ico`（16~256 共 9 个尺寸）

**许可证**：图标与本项目同为 **MIT**，可以随项目一起再分发，不需要额外署名。

> **关于图标相似性**：本项目是**开源、非商业**项目，图标里用的是「声波 / 字幕」这类通用视觉符号 —— 同类软件大量使用相似构型，这是语义决定的，不是复刻某个具体产品。本项目未就该图标申请商标，也不主张独占。如果你打算 fork 后**商业发行**，建议换一个更有辨识度的图标，并自行做一次商标检索。

---

## 已知限制

- **CUDA 加速只支持 NVIDIA 显卡**；AMD / Intel 核显走兼容模式（Vulkan / CPU），明显更慢
- **显存低于 6 GB 的 N 卡**会落到兼容模式：转写用 sherpa-onnx、翻译用 llama.cpp 的 **Vulkan** 后端，比 CUDA 慢但远没到「不可用」；只有机器上连 Vulkan 设备都没有时才退回纯 CPU
- **Whisper large-v2 比 v3 更适合 ASMR**（v3 改了 mel 特征，在长音频和噪声上更容易幻觉；实测覆盖率反而低 6.6 个点）
- **全量模式没有日文文本层** —— 想要日文字幕只能走兼容模式，或者在全量模式下额外勾选「生成日文字幕」（那一份是另跑一次 ASR 得来的）
- 模型加载每步各一次（4 个独立进程），约占总时间的 70%。合并成单进程能再省 5-6 秒/文件，但改动较大，暂未做
- **全量模式在纯音效段仍可能漏进极短的音效字幕**（3 字以内的 `んっ…`、`あ`），三道过滤的 6 秒门槛拦不到
