"""兼容模式 第2步：Silero VAD + 语音识别（全 CPU）。

输入：0pre\\*.flac / *.wav / *.mp3 / *.m4a
输出：2vad\\<名字>.json   [{start, end, text}, ...]

识别引擎见 asr_engine.py：默认 Haas-ja-v1.0（日语专用），
加载失败自动回退 SenseVoice，可用 ASMR_ASR_ENGINE 强制指定。

为什么不用 pyannote：兼容模式面向没有可用 CUDA 的机器，
pyannote 在 CPU 上很慢；Silero VAD 是 ONNX 小模型，毫秒级。

── 为什么还要「补扫」──
实测真实 ASMR 素材（300 秒），Silero VAD 默认阈值只把 11.9% 的音频
判为语音，把阈值降到 0.02（几乎失效）也只有 22.3%。S 音过重的 ASMR
讲解、耳语、拟声全部被当成噪音丢掉——而 Haas-ja 拿到的那 12% 几乎
全对。也就是说瓶颈在「喂给模型的音频太少」，不在模型本身。

所以这里改成两段式：
  1. VAD 主切段 —— 高置信语音，边界清楚，识别最准（主力）
  2. 漏段补扫   —— VAD 没覆盖的长空白里按 10 秒窗口补跑，捞回讲解词
补扫必然引入幻觉（纯环境音也会「听」出字），所以用模型自己返回的
ys_log_probs 算置信度过滤：实测幻觉「れる。」0.40、真台词 0.88、
真实拟声 0.77，门槛设 0.5 刚好滤掉幻觉保住拟声。
"""
import glob
import json
import os
import sys
import tempfile
import time

# ★ 识别/VAD 的 CPU 线程数（v1.30.4 起可调，默认 4 = 旧行为）。
#   实测（H:\_probe_asr\bench_cuda.py，Haas-ja encoder，8.6s 一段）：
#     1/2/4/8/16 线程 = 4122/1994/1260/872/2082 ms
#   → 4→8 线程白拿 1.45x，比走 CUDA EP（892 ms，1.29x）还快。
#   原因：这个 int8 模型 70% 的时间是 ConvInteger 卷积，而 CUDA EP 没有
#   int8 卷积 kernel（255 个 ConvInteger 全部回退 CPU），GPU 只能加速
#   22% 的 MatMulNBits，还要付 PCIe 拷贝成本。
#   默认按核数给：4 核及以下给 4（再高会互相抢核，16 线程实测反而更慢），
#   8 核机器给 4，16 核给 8。上层也可以直接用环境变量覆盖，
#   例如 ASMR_ASR_THREADS=8。
_ASR_THREADS = min(8, max(4, (os.cpu_count() or 4) // 2))
try:
    _env_thr = int(os.environ.get("ASMR_ASR_THREADS", "") or 0)
    if _env_thr > 0:
        _ASR_THREADS = max(1, min(16, _env_thr))
except Exception:
    pass

# ★ 绕过 torch 的一个重复注册 bug：
#   某些环境里 torchcodec 解码音频时会重新执行 torch._dynamo.package，
#   而 package.py 在模块级调 CacheArtifactFactory.register，
#   第二次注册同一个类型就抛 AssertionError（precompile already registered）。
#   这里把 register 包一层：同类型已注册时直接返回，不报错。
try:
    import torch  # noqa: F401
    from torch.compiler._cache import CacheArtifactFactory as _CAF

    _orig_register = _CAF.__dict__["register"].__func__

    def _safe_register(cls, artifact_cls):
        try:
            key = artifact_cls.type()
        except Exception:
            return _orig_register(cls, artifact_cls)
        if key in cls._artifact_types:
            return artifact_cls
        return _orig_register(cls, artifact_cls)

    _CAF.register = classmethod(_safe_register)
except Exception:
    pass

import soundfile as sf
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

# 识别引擎（Haas-ja 优先，失败回退 SenseVoice）。详见 asr_engine.py。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_engine import load_engine
from events import emit

# ── 数据目录前缀 ──
#
# 0pre / 2vad / 3asr 都是【写】的。装进 Program Files 时程序目录不可写，
# Rust 侧会通过 ASMR_DATA_DIR 告诉我们该写到哪；不传时（命令行直跑）行为不变。
_DATA_DIR = os.environ.get("ASMR_DATA_DIR", "").strip()


def _p(name):
    return os.path.join(_DATA_DIR, name) if _DATA_DIR else name


PRE = _p("0pre")
VAD_DIR = _p("2vad")
MODEL = os.path.join("model", "SenseVoiceSmall")
EXTS = (".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg")

# ── 补扫参数（全部经真实素材实测调整，改动前请重跑 _haas_test） ──
VAD_PAD_SEC = 0.45      # VAD 段前后外扩，补回被边界切掉的音节（不重叠）
# ★ v1.30.3：单条字幕的最大时长。
#   ASMR-VAD 在长对话上会判出「连续语音」的超长段（实测最长 64 秒），
#   一句字幕持续 64 秒在播放器上等于一直不换行。12 秒是「一屏能读完、
#   又不至于切得太碎」的经验值（与 whisper 的 30s 窗口、字幕阅读速度都兼容）。
MAX_SEG_SEC = 12.0
MIN_HOLE_SEC = 6.0      # 空白区间至少这么长才值得补扫
SCAN_WIN = 6.0          # 补扫窗口长度；10 秒太粗，时间戳误差会大到无法定位
SCAN_EDGE = 0.35        # 补扫窗口与 VAD 段边界保持的距离，避免重复识别
SCAN_MIN_SEC = 3.0      # 太短的补扫窗口没有意义
MIN_CONF = 0.50         # 补扫结果最低置信度
MIN_CONF_SHORT = 0.70   # 短文本（<=3 字）需要更高置信度，滤掉碎片
REPEAT_MIN = 3          # 同一句话在补扫里出现这么多次，判为固定幻觉

files = sorted(
    f for f in glob.glob(os.path.join(PRE, "*")) if f.lower().endswith(EXTS)
)
if not files:
    print(f"在 {PRE} 中没有找到音频文件")
    sys.exit(1)

print(f"兼容模式：{len(files)} 个音频")
print("加载 VAD + 语音识别模型 ...", flush=True)
t0 = time.time()

# ★ v1.30.3：ASMR 专用 VAD（约 500h 日文 ASMR 训练，标签含 whispered-speech）
#   替代原 Silero。实测数据见下面 segs 处的注释。
#   回退策略是【自动】的：模型缺失或加载异常 → 静默用回 Silero，
#   不让用户因为一个可选模型跑不起来。不提供环境变量开关。
vad = load_silero_vad(onnx=True)          # 兜底 + 回退路径用
asmr_vad = None
_vp = ""
for _name in ("vad_asmr_encdec.onnx", "asmr_vad.onnx", "model.onnx"):
    _cand = os.path.join("model", _name)
    if os.path.isfile(_cand):
        _vp = _cand
        break
if _vp:
    try:
        from asmr_vad import AsmrVad
        asmr_vad = AsmrVad(_vp, threads=_ASR_THREADS)
        print(f"  VAD: ASMR 专用（{os.path.basename(_vp)}，{_ASR_THREADS} 线程）", flush=True)
    except Exception as _e:
        print(f"  VAD: ASMR 加载失败（{type(_e).__name__}: {_e}），"
              f"自动回退 Silero", flush=True)
        asmr_vad = None
else:
    print("  VAD: 未找到 ASMR VAD，自动回退 Silero", flush=True)

# ★ v1.30.4：ASR 热词。
#
#   和翻译层共用同一批词表文件（`hotwords.txt` + `hotwords_<音频名>.txt`，
#   逗号分隔、# 注释），但喂给识别器的形态不同：这里是「一行一个词」。
#   一次任务里多个作品会合并成一份词表（识别器只建一次，按文件重建要 2 秒）。
#
#   为什么要单开这个开关：热词会把解码从 greedy 换成 modified_beam_search，
#   120 秒音频实测 13.4 s → 15.1~16.0 s（+12~19%）。所以只在词表里
#   真有词（用户主动填了人名）时才开；`ASMR_ASR_HOTWORDS=off` 可强制关闭。
def _load_hotwords(stems):
    """从词表文件里收集热词（去重、保序）。"""
    if os.environ.get("ASMR_ASR_HOTWORDS", "").lower() in ("off", "0", "no"):
        return []
    names = ["hotwords.txt"] + [f"hotwords_{s}.txt" for s in stems]
    terms = []
    for name in names:
        p = os.path.join(".", name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8-sig") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    for w in ln.replace("，", ",").split(","):
                        w = w.strip()
                        if w and w not in terms:
                            terms.append(w)
        except OSError:
            pass
    return terms


_stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
_hw = _load_hotwords(_stems)
if _hw:
    print(f"  热词: {len(_hw)} 个（{'、'.join(_hw[:6])}"
          f"{'…' if len(_hw) > 6 else ''}）— 解码改用 beam search", flush=True)

# ★ 识别引擎：默认 Haas-ja（实测最准），失败自动降级。
#   具体策略和实测数据见 asr_engine.py 顶部注释。
asr, _engine_note = load_engine(num_threads=_ASR_THREADS, hotwords=_hw)
_HAS_EX = hasattr(asr, "transcribe_ex")
print(f"模型加载完成（{time.time() - t0:.1f}s）— {_engine_note}", flush=True)
if not _HAS_EX:
    print("  注意：该引擎不返回置信度，补扫结果不做幻觉过滤", flush=True)

os.makedirs(VAD_DIR, exist_ok=True)
ok = 0
fail = 0
total = len(files)
t_start = time.time()


def merge_segments(segs, gap_sec=0.6, max_sec=10.0):
    """合并短停顿，但绝不把连续对话粘成一整屏字幕。

    Silero 已用 ``max_speech_duration_s=10`` 限制单个原始段；旧代码随后又把
    间隔小于 0.6 秒的段无限合并，连续对话可能从 1:20 一路粘到 2:23，最终
    63 秒、十几句话同时显示。合并后的跨度达到 ``max_sec`` 就另起一段。
    """
    merged = []
    for s in segs:
        can_join = False
        if merged:
            gap = (s["start"] - merged[-1]["end"]) / 16000.0
            joined_span = (s["end"] - merged[-1]["start"]) / 16000.0
            can_join = gap < gap_sec and joined_span <= max_sec
        if can_join:
            merged[-1]["end"] = s["end"]
        else:
            merged.append({"start": s["start"], "end": s["end"]})
    return merged


def split_long_segments(segs, max_sec=12.0):
    """把过长的语音段强制切开（v1.30.3）。

    ★ 为什么必须切：实测 ASMR 素材上 ASMR-VAD 会判出「连续语音」的
      超长段 —— 最长 64.2 秒（真实对话里有换气和停顿，但没到 VAD 的
      静音阈值），561 条里有 23 条 >12 秒。

    后果有两层：
      1. 识别引擎按段送，超长段一次性出几十个字，字幕一屏放不下
      2. 一句字幕持续 64 秒，播放器上等于「一直不换行」

    切法：段内若已有更细的静音边界（不做二次 VAD，省算力），
    没有就按 ceil(时长/max_sec) 均分 —— 均分点落在词中间的概率不高，
    而且比「一句 64 秒」好得多。
    """
    out = []
    for a, b in segs:
        dur = b - a
        if dur <= max_sec:
            out.append((a, b))
            continue
        n = int(dur // max_sec) + 1
        step = dur / n
        for k in range(n):
            out.append((a + k * step, min(a + (k + 1) * step, b)))
    return out


def pad_segments(segs, pad, dur):
    """把每段前后各外扩 pad 秒，但不许越过相邻段的中点。

    Silero 的 speech_pad_ms 只加在原始段上，合并后句首句尾仍可能被切掉
    半个音节（实测「こっちから」被切成「こっち」）。这里统一再松一点。
    """
    out = []
    for i, (a, b) in enumerate(segs):
        lo = segs[i - 1][1] if i > 0 else 0.0
        hi = segs[i + 1][0] if i + 1 < len(segs) else dur
        na = max(lo, a - pad)
        nb = min(hi, b + pad)
        out.append((na, nb))
    return out


def scan_holes(segs, dur, min_hole):
    """返回 VAD 完全没覆盖、且长度 >= min_hole 的空白区间。"""
    holes = []
    prev = 0.0
    for a, b in segs:
        if a - prev >= min_hole:
            holes.append((prev, a))
        prev = max(prev, b)
    if dur - prev >= min_hole:
        holes.append((prev, dur))
    return holes


def _write_tmp(wav16, a, b):
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp, wav16[int(a * 16000):int(b * 16000)], 16000)
    return tmp


def recognize(wav16, a, b):
    """识别 [a,b) 区间，返回 dict(text, conf, tokens, timestamps)。"""
    tmp = _write_tmp(wav16, a, b)
    try:
        if _HAS_EX:
            return asr.transcribe_ex(tmp)
        return {"text": asr.transcribe(tmp), "conf": 1.0,
                "tokens": [], "timestamps": []}
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def keep_scan(res):
    """补扫结果是否可信。空文本直接丢；置信度低则视为幻觉。"""
    text = (res.get("text") or "").strip()
    if not text:
        return False
    core = text.strip("。、，,.!?！？…ー~～ 　")
    if not core:
        return False                      # 只剩标点
    conf = float(res.get("conf") or 0.0)
    if conf < MIN_CONF:
        return False
    if len(core) <= 3 and conf < MIN_CONF_SHORT:
        return False                      # 「れろれろ」「くん。」这类碎片
    return True


def refine_span(res, win_start, win_len):
    """用词级时间戳把补扫窗口收窄到真实语音范围。

    ★ Haas-ja 的 ``timestamps[0]`` 恒为 0（占位），不可信；从第二个开始才是
      真实起点。实测 t=135s 的窗口「これ」标 0.00s、「いい」标 8.32s，
      说明语音其实在窗口尾部——不修的话字幕会从窗口开头挂 6 秒。
    相邻间隔的中位数用来估一个 token 的时长，这样中间夹着长静音也不会
    把语音范围拉满整个窗口。
    """
    ts = [float(x) for x in (res.get("timestamps") or [])]
    if len(ts) < 2:
        return round(win_start, 2), round(win_start + win_len, 2)

    body = ts[1:]
    gaps = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1)
                  if ts[i + 1] >= ts[i])
    avg = gaps[len(gaps) // 2] if gaps else 0.25
    avg = min(max(avg, 0.08), 0.60)       # 夹住，防极端值

    start = win_start + max(0.0, body[0] - avg)
    end = win_start + min(win_len, ts[-1] + avg * 2.0)
    if end - start < 0.6:                 # 收窄过头，退回窗口中段
        mid = win_start + win_len / 2.0
        start, end = mid - 0.3, mid + 0.3
    return round(start, 2), round(end, 2)


for idx, fp in enumerate(files, 1):
    stem = os.path.splitext(os.path.basename(fp))[0]
    out = os.path.join(VAD_DIR, stem + ".json")
    print(f"\n处理: {os.path.basename(fp)}", flush=True)
    try:
        wav16 = read_audio(fp, sampling_rate=16000)
        dur = len(wav16) / 16000.0

        if asmr_vad is not None:
            # ★ v1.30.3：ASMR 专用 VAD（约 500h 日文 ASMR 训练，标签含
            #   whispered-speech）。PC 实测（3 素材各 10 分钟，同 ASR，
            #   唯一变量是 VAD）：
            #     CER  31.4% → 25.8%（-5.6 点）
            #     命中率 65.8% → 70.1%（+4.3 点）
            #   三个素材方向一致，速度基本持平（185s vs 190s）。
            #   段数更少更整（78~84 vs 127~128），碎片更少。
            #   模型缺失/加载失败时 asmr_vad 为 None，自动走下面 Silero 分支。
            _sec = asmr_vad.segments(
                wav16, threshold=0.5, min_speech=0.3,
                min_silence=0.4, pad=0.0, normalize=True)
            seg_sec = [(a, b) for a, b in _sec]
            _vad_name = "ASMR-VAD"
        else:
            segs = get_speech_timestamps(
                wav16, vad, sampling_rate=16000,
                min_speech_duration_ms=300,
                # ★ v1.30.2 VAD 放宽（79 条音频 × 5 组参数网格实测）：
                #   阈值 0.5→0.35、静音切分 600→400ms，能把官方字幕里
                #   「听得到但没圈进语音段」的漏检从 28.2% 提到 31.5%，
                #   识别时长只 +4%。再放宽（0.3/300ms）只多救 3 个点，
                #   却开始圈喘息/水声 —— 幻觉由置信度闸（MIN_CONF）兜底，
                #   不值得再冒。
                min_silence_duration_ms=400,
                speech_pad_ms=120,
                threshold=0.35,
                max_speech_duration_s=10,
            )
            segs = merge_segments(segs)
            seg_sec = [(s["start"] / 16000.0, s["end"] / 16000.0) for s in segs]
            _vad_name = "Silero"

        # ★ v1.30.3：先切超长段（>12s），再外扩。
        #   顺序很重要：先外扩会把本来 11 秒的段推到 12+ 秒，
        #   而且外扩后相邻段边界更近，再切容易切错。
        _n_long = sum(1 for a, b in seg_sec if b - a > 12.0)
        seg_sec = split_long_segments(seg_sec, MAX_SEG_SEC)
        if _n_long:
            print(f"  [切分] {_n_long} 条超长段已切开（上限 {MAX_SEG_SEC:.0f}s）")
        seg_sec = pad_segments(seg_sec, VAD_PAD_SEC, dur)

        covered = sum(b - a for a, b in seg_sec)
        print(f"  VAD[{_vad_name}] {len(seg_sec)} 段，覆盖 {covered:.1f}s "
              f"({covered / max(0.1, dur) * 100:.1f}%)", flush=True)

        rows = []
        # ★ v1.30.3：段进度（借鉴安卓版设计）。
        #   为什么需要：ASR 阶段一跑十几分钟，界面只知道"在第2步"，
        #   用户看到进度条不动会以为卡死。这里按段打点，输出
        #   已处理段数/总段数 + 每分钟处理几段 + 预计剩余，
        #   同时用 @@EV@@ 让界面能读（协议见 events.py）。
        _n_total = len([1 for a, b in seg_sec if b - a >= 0.15])
        _t_seg0 = time.time()
        _done = 0
        for a, b in seg_sec:
            if b - a < 0.15:
                continue
            res = recognize(wav16, a, b)
            if res["text"]:
                rows.append({
                    "start": round(a, 2),
                    "end": round(b, 2),
                    "text": res["text"],
                })
            # ★ 段进度：每 20 段或最后一段打点（每段都打会把日志刷爆）
            _done += 1
            if _done % 20 == 0 or _done == _n_total:
                _el = time.time() - _t_seg0
                _spm = _done / max(0.01, _el) * 60.0      # 每分钟处理几段
                _left = (_n_total - _done) / max(0.001, _spm) if _spm > 0 else 0
                print(f"    识别 {_done}/{_n_total} 段"
                      f"（{_done / max(1, _n_total) * 100:.0f}%）"
                      f"  {_spm:.1f} 段/分  剩约 {_left:.0f} 分", flush=True)
                emit("segs", _done, _n_total, round(_spm, 2), int(_left))

        # ── 漏段补扫 ──
        holes = scan_holes(seg_sec, dur, MIN_HOLE_SEC)
        n_scan = 0
        scan_rows = []
        for ha, hb in holes:
            cur = ha + SCAN_EDGE
            stop = hb - SCAN_EDGE
            while cur < stop:
                nxt = min(cur + SCAN_WIN, stop)
                if nxt - cur >= SCAN_MIN_SEC:
                    n_scan += 1
                    res = recognize(wav16, cur, nxt)
                    if keep_scan(res):
                        s2, e2 = refine_span(res, cur, nxt - cur)
                        scan_rows.append((s2, e2, res["text"]))
                    # 大文件补扫很慢（实测约为纯 VAD 的 8 倍），
                    # 中途报进度，否则界面看起来像卡死。
                    if n_scan % 10 == 0:
                        print(f"    补扫 {n_scan} 窗"
                              f"（{cur / max(0.1, dur) * 100:.0f}%）",
                              flush=True)

                cur = nxt

        # ── 跨窗口重复幻觉过滤 ──
        #
        # 实测 Haas-ja 对某些环境音会**固定输出同一句话**（真实素材里
        # 「音楽」在 63/134/253/263 秒四处不连续位置反复出现，置信度
        # 都过 0.5）。真台词极少在短时间内一字不差重复这么多次，
        # 所以对**短文本**做计数过滤；长句不动，避免误杀正常复述。
        def _norm(t):
            return t.strip("。、，,.!?！？…ー~～ 　")

        from collections import Counter
        cnt = Counter(_norm(t) for _, _, t in scan_rows)
        n_rep = 0
        for s2, e2, txt in scan_rows:
            key = _norm(txt)
            if len(key) <= 8 and cnt[key] >= REPEAT_MIN:
                n_rep += 1
                continue
            rows.append({"start": s2, "end": e2, "text": txt})

        rows.sort(key=lambda r: r["start"])
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        extra = ""
        if n_scan:
            extra = f"，补扫 {n_scan} 窗留 {len(scan_rows)}"
            if n_rep:
                extra += f"（去重复幻觉 {n_rep}）"
        print(f"  转写完成：{len(rows)} 段{extra}", flush=True)
        ok += 1
    except Exception as e:
        print(f"  [跳过] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        fail += 1
        continue

    pct = idx * 100.0 / total
    bar = "#" * int(pct / 5)
    el = int(time.time() - t_start)
    print(f"  {pct:3.0f}%|{bar:<20}| {idx}/{total} "
          f"[{el // 60:02d}:{el % 60:02d}<00:00]", flush=True)

print(f"\n兼容模式转写完成：成功 {ok} 个，失败 {fail} 个")
if ok == 0:
    print("[失败] 没有任何音频转写成功")
    sys.exit(1)
