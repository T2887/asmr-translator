"""兼容模式的语音识别引擎：Haas-ja（首选）与 SenseVoice（回退）。

为什么要有这一层
────────────────
v1.30 起兼容模式默认用 Haas-ja-v1.0（日语专用，615M 参数），实测比
SenseVoiceSmall 准得多 —— 120 秒街头对话样本里 SenseVoice 有 3 条整句
崩坏，Haas-ja 全对（详见交接文档的对比表）。

但 Haas-ja 的 ONNX 版有个隐患：它是社区转换的，靠 sherpa-onnx 的
**Parakeet-TDT 路径**加载，依赖两个内部实现细节（Kimi Delta Attention
状态被展平成 [2,B,20480] 当 LSTM 状态传；靠 metadata url 里的 `tdt`
识别模型类型）。sherpa-onnx 将来一改就可能加载不了。

所以这里做成双引擎：
  1. 优先 Haas-ja
  2. 任何环节失败（模型缺失 / sherpa-onnx 版本不对 / 加载异常）
     → 自动回退 SenseVoice，并在日志里说清楚原因
  3. 用户也可以用环境变量强制指定，方便排查

绝对不能让用户因为换个模型就整个跑不了 —— 兼容模式面向的正是
GTX10 系、核显、老机器这些最脆弱的场景。
"""
import math
import os
import sys

# ── 引擎选择 ──
#
# ASMR_ASR_ENGINE 环境变量：
#   auto（默认）→ 按下面 AUTO_ORDER 的顺序逐个尝试
#   haas        → 强制 Haas-ja
#   zipformer   → 强制 zipformer-ja
#   sensevoice  → 强制 SenseVoice（应急开关）
#
# ★ 默认顺序：haas → sensevoice
#
#   Haas-ja 排第一 —— 实测最准，两个样本 13 条字幕全对。
#
#   zipformer-ja **故意不在自动链里**：它 RTF 只有 0.009（比 SenseVoice
#   快 3.5 倍），但在 ASMR 素材上会吞掉开头音节、把整句变成「はい」。
#   同一个 120 秒样本里它丢了 2 整句，30 秒样本里「ご主人様」变成「主人様」。
#   这种「更快的错」对用户没价值，所以只留手动开关：
#       set ASMR_ASR_ENGINE=zipformer
#   给那些机器实在慢到跑不动 Haas-ja 的人一个退路。
#
#   实测数据（i7 级 CPU / 4 线程，详见 _haas_test/bench_haas.py）：
#     SenseVoice   RTF 0.032~0.047   内存 +50MB    多处崩坏
#     Haas-ja      RTF 0.078~0.096   内存 +200MB   全对
#     zipformer-ja RTF 0.009~0.013   内存 +85MB    漏字/整句丢失
ENGINE_ENV = os.environ.get("ASMR_ASR_ENGINE", "auto").strip().lower()

AUTO_ORDER = ("haas", "sensevoice")

# 可以手动指定、但不在自动链里的引擎
MANUAL_ENGINES = ("haas", "zipformer", "sensevoice")

# ── 模型目录 ──
#
# ★ 用【相对脚本位置】的绝对路径，不依赖 cwd。
#
#   light_asr.py 里原来的 MODEL = os.path.join("model", "SenseVoiceSmall")
#   是相对路径，靠 Rust 侧 .current_dir(work_dir) 命中。那种写法在
#   「Rust 正常调用」时没问题，但两种情况会失效：
#     · 命令行直接跑脚本（cwd 不在程序目录）
#     · 将来有别的调用方忘了设 current_dir
#   加载失败的表现是「模型缺失」，排查起来要绕一圈。
#
#   这里统一算成基于 __file__ 的绝对路径，放哪儿都能跑。
#   （脚本在 <程序目录>\app\ 下，往上跳一层就是程序根）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Haas-ja 量化版
HAAS_DIR = os.path.join(_ROOT, "model", "Haas-ja-v1.0")

# SenseVoice（回退用）
SENSEVOICE_DIR = os.path.join(_ROOT, "model", "SenseVoiceSmall")

# ★ sherpa-onnx 版本白名单。
#
#   为什么要在运行时再查一遍（requirements 里已经用 == 锁了）：
#     user 的 env 是打包分发的，我们没法保证对方是真的按 requirements 装的；
#     也可能是老用户升级主程序、env 还是旧版。
#     Haas-ja 走的是 sherpa-onnx 的 Parakeet-TDT 内部路径，
#     版本不对时**不一定报错，可能静默输出乱码或空串** —— 那比直接崩溃更糟，
#     用户会以为「这软件翻译质量变差了」而不知道是版本问题。
#
#   实测可用的版本：2.0.1
#   升级流程：改 requirements-asr.txt → 重装 → 跑 _haas_test/bench_haas.py
#             → 确认结果没劣化 → 再把新版本加进这个集合
SHERPA_OK_VERSIONS = {"2.0.1"}

# Haas-ja 需要的四个文件；缺任何一个都算没装好
HAAS_FILES = (
    "encoder.quantized.onnx",
    "encoder.data",
    "decoder.onnx",
    "joiner.onnx",
    "tokens.txt",
)


def haas_ready():
    """Haas-ja 的模型文件是否齐全。"""
    return all(os.path.isfile(os.path.join(HAAS_DIR, f)) for f in HAAS_FILES)


# ─────────────────────────────────────────────────────────────
# Zipformer-ja（k2-fsa 官方发布，ReazonSpeech 35k 小时训练）
#
# ★ 为什么把它也纳入候选：
#   Haas-ja 虽然参数多（615M）、官方 benchmark 更好，但它的 ONNX 版是
#   社区转换的，要走 Parakeet-TDT 内部路径，稳定性有隐患。
#   zipformer-ja 是 sherpa-onnx **官方发布**的标准 transducer 模型：
#     · 走最常规的 from_transducer 路径，没有 hack
#     · int8 版只有 153 MB（Haas-ja 要 459 MB）
#     · 159.34M 参数，35k 小时 ReazonSpeech 训练，纯日语
#   代价是模型小、理论上准确率不如 Haas-ja —— 但**实测说了算**，
#   所以两个都留着，用哪种交给配置文件决定。
# ─────────────────────────────────────────────────────────────
# transducer 共用基类
# ─────────────────────────────────────────────────────────────

class _TransducerBase:
    """transducer 引擎（Haas-ja / zipformer）共用逻辑。

    sherpa-onnx 的离线 transducer 识别结果带三类有用信息，旧代码只取 text：
      · text        —— 识别文本
      · ys_log_probs —— 逐 token 的对数概率，exp(均值) 就是「置信度」
      · timestamps   —— 逐 token 的起始时间（秒，相对本次输入音频）
      · tokens       —— 逐 token 文本，配合 timestamps 可在长段内部切句

    置信度是**过滤幻觉**的关键手段：实测纯环境音产生的幻觉
    （「れる。」）置信度 0.40，而真实台词 0.88、真实拟声 0.77，
    用 0.5 作门槛能滤掉幻觉又保住拟声。
    """

    def _decode(self, wav_path):
        import soundfile as sf

        samples, sr = sf.read(wav_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        stream = self._rec.create_stream()
        stream.accept_waveform(sr, samples)
        self._rec.decode_stream(stream)
        return stream.result

    def transcribe(self, wav_path):
        """返回识别出的文本（日文）。失败返回空串。"""
        try:
            return str(self._decode(wav_path).text).strip()
        except Exception:
            return ""

    def transcribe_ex(self, wav_path):
        """返回 {text, conf, tokens, timestamps}。

        conf 是 exp(平均 log 概率)，0~1；空输出时给 0.0。
        tokens/timestamps 用于在长段内部按停顿切句，长度不一定相等
        （个别实现有 padding），调用方要按 min(len) 截断。
        """
        r = self._decode(wav_path)
        text = str(r.text).strip()
        lps = list(getattr(r, "ys_log_probs", None) or [])
        if lps:
            conf = math.exp(sum(lps) / len(lps))
        else:
            conf = 0.0
        toks = list(getattr(r, "tokens", None) or [])
        ts = list(getattr(r, "timestamps", None) or [])
        n = min(len(toks), len(ts))
        return {
            "text": text,
            "conf": conf,
            "tokens": toks[:n],
            "timestamps": [float(x) for x in ts[:n]],
        }


# ─────────────────────────────────────────────────────────────

ZIPFORMER_DIR = os.path.join(_ROOT, "model", "sherpa-onnx-zipformer-ja")

# zipformer 的文件名带 epoch 后缀，所以按前缀匹配而不是写死全名
ZIPFORMER_PREFIXES = ("encoder", "decoder", "joiner")


def _find_prefixed(d, prefix, prefer_int8=True):
    """在目录里找以 prefix 开头的 .onnx 文件。

    zipformer 的官方包里文件名形如：
        encoder-epoch-99-avg-1.int8.onnx
        decoder-epoch-99-avg-1.onnx      ← decoder 通常只有 fp32
        joiner-epoch-99-avg-1.int8.onnx
    优先挑 .int8.onnx（体积小 4 倍，准确率损失极小）。
    """
    if not os.path.isdir(d):
        return None
    cands = [f for f in os.listdir(d)
             if f.startswith(prefix) and f.endswith(".onnx")]
    if not cands:
        return None
    if prefer_int8:
        int8 = [f for f in cands if ".int8." in f]
        if int8:
            return os.path.join(d, sorted(int8)[0])
    return os.path.join(d, sorted(cands)[0])


def zipformer_ready():
    """zipformer-ja 的模型文件是否齐全。"""
    if not os.path.isdir(ZIPFORMER_DIR):
        return False
    ok = all(_find_prefixed(ZIPFORMER_DIR, p) for p in ZIPFORMER_PREFIXES)
    return ok and os.path.isfile(os.path.join(ZIPFORMER_DIR, "tokens.txt"))


class ZipformerRecognizer(_TransducerBase):
    """sherpa-onnx 官方 zipformer-ja（标准 transducer，无 hack）。"""

    name = "zipformer-ja"

    def __init__(self, num_threads=4):
        import sherpa_onnx

        self.engine_version = getattr(sherpa_onnx, "__version__", "")
        enc = _find_prefixed(ZIPFORMER_DIR, "encoder")
        dec = _find_prefixed(ZIPFORMER_DIR, "decoder", prefer_int8=False)
        joi = _find_prefixed(ZIPFORMER_DIR, "joiner")
        if not (enc and dec and joi):
            raise RuntimeError(f"zipformer 模型文件不全：{ZIPFORMER_DIR}")

        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=enc,
            decoder=dec,
            joiner=joi,
            tokens=os.path.join(ZIPFORMER_DIR, "tokens.txt"),
            num_threads=num_threads,
            # zipformer 用标准 fbank（80 维），不要像 Qwen3-ASR 那样传 128
            model_type="zipformer2",
        )


class HaasRecognizer(_TransducerBase):
    """Haas-ja-v1.0 包装。接口对齐下面 SenseVoiceRecognizer 的 transcribe()。"""

    name = "Haas-ja-v1.0"

    def __init__(self, num_threads=4):
        import sherpa_onnx

        # ★ 版本白名单校验 —— 见文件顶部 SHERPA_OK_VERSIONS 的说明。
        #   版本不符时直接抛异常（由 load_engine 接住并回退 SenseVoice），
        #   因为「能加载但输出乱码」比「加载失败」更难排查。
        ver = getattr(sherpa_onnx, "__version__", "")
        self.engine_version = ver
        if ver not in SHERPA_OK_VERSIONS:
            raise RuntimeError(
                f"sherpa-onnx 版本 {ver or '未知'} 不在白名单 "
                f"{sorted(SHERPA_OK_VERSIONS)} 内；"
                f"Haas-ja 依赖其 TDT 内部实现，版本不符可能静默出错"
            )

        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(HAAS_DIR, "encoder.quantized.onnx"),
            decoder=os.path.join(HAAS_DIR, "decoder.onnx"),
            joiner=os.path.join(HAAS_DIR, "joiner.onnx"),
            tokens=os.path.join(HAAS_DIR, "tokens.txt"),
            num_threads=num_threads,
            model_type="nemo_transducer",
            # FastConformer 用 128 维 log-mel，不是 whisper 的 80
            feature_dim=128,
        )


# ─────────────────────────────────────────────────────────────
# SenseVoice（原来的实现，保留作回退）
# ─────────────────────────────────────────────────────────────

class SenseVoiceRecognizer:
    """SenseVoiceSmall 包装（v1.29 及以前的默认引擎）。"""

    name = "SenseVoiceSmall"

    def __init__(self, model_dir=SENSEVOICE_DIR):
        from funasr import AutoModel

        # ★ funasr 的 AutoModel 不认绝对路径 —— 它把 model= 当成相对路径
        #   去拼 <model>\xxx.bpe.model。传绝对路径会拼出畸形路径然后
        #   报 NOT_FOUND。所以这里临时切到程序根目录，用完切回来。
        old = os.getcwd()
        try:
            os.chdir(_work_dir())
            self._model = AutoModel(
                model=model_dir, device="cpu", disable_update=True
            )
        finally:
            os.chdir(old)

    def transcribe(self, wav_path):
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        r = self._model.generate(input=wav_path, language="ja", use_itn=True)
        raw = r[0].get("text", "") if r else ""
        return clean_marks(rich_transcription_postprocess(raw))


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────

# SenseVoice 的 rich transcription 会把事件/情绪标签转成 emoji
# （<|BGM|> → 🎼、<|ANGRY|> → 😡）。它们不是台词，混进 SRT 毫无意义。
_MARKS = str.maketrans("", "", "🎼👏😀😭🤧😷😊😔😡😰🤢😮❓")


def clean_marks(text):
    if isinstance(text, (list, tuple)):
        text = " ".join(map(str, text))
    return " ".join(str(text).translate(_MARKS).split())


def _work_dir():
    """程序根目录（脚本在 app\ 下，往上跳一层）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────
# 对外入口
# ─────────────────────────────────────────────────────────────

def _try_engine(kind, num_threads):
    """尝试加载一个引擎。成功返回 (engine, None)，失败返回 (None, 原因)。"""
    if kind == "zipformer":
        if not zipformer_ready():
            return None, f"模型文件缺失（目录 {ZIPFORMER_DIR}）"
        try:
            return ZipformerRecognizer(num_threads=num_threads), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    if kind == "haas":
        if not haas_ready():
            missing = [f for f in HAAS_FILES
                       if not os.path.isfile(os.path.join(HAAS_DIR, f))]
            return None, f"模型文件缺失 {', '.join(missing)}（目录 {HAAS_DIR}）"
        try:
            return HaasRecognizer(num_threads=num_threads), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    if kind == "sensevoice":
        try:
            return SenseVoiceRecognizer(), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    return None, f"未知引擎 {kind!r}"


def load_engine(num_threads=4):
    """加载识别引擎，带多级降级。

    返回 (engine, note)：
        engine — 可用的识别器实例；**绝不会返回 None**
        note   — 给日志看的说明

    降级链（auto 模式）：zipformer → haas → sensevoice
    指定单一引擎时：只试那一个，失败就抛异常（方便排查）
    """
    # 指定了具体引擎 → 只试它，不降级
    if ENGINE_ENV in MANUAL_ENGINES:
        eng, err = _try_engine(ENGINE_ENV, num_threads)
        if eng is not None:
            return eng, f"{eng.name} 就绪（sherpa-onnx {getattr(eng, 'engine_version', '—')}）"
        raise RuntimeError(
            f"指定的识别引擎 {ENGINE_ENV} 加载失败：{err}\n"
            f"  去掉 ASMR_ASR_ENGINE 环境变量即可自动降级。"
        )

    # auto：按顺序试
    tried = []
    for kind in AUTO_ORDER:
        eng, err = _try_engine(kind, num_threads)
        if eng is not None:
            if tried:
                print(f"[识别] 降级到 {eng.name}（前面失败的：{'；'.join(tried)}）",
                      flush=True)
            return eng, f"{eng.name} 就绪（sherpa-onnx {getattr(eng, 'engine_version', '—')}）"
        tried.append(f"{kind}({err})")
        if kind != AUTO_ORDER[-1]:
            print(f"[识别] {kind} 不可用，继续下一个 —— {err}", flush=True)

    # 全挂
    raise RuntimeError(
        "语音识别引擎全部不可用：\n  · "
        + "\n  · ".join(tried)
        + "\n  请检查 model\\ 目录是否完整，或重新安装运行环境包。"
    )
