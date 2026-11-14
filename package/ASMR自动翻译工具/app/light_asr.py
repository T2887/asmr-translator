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
print("加载 Silero VAD + 语音识别模型 ...", flush=True)
t0 = time.time()
vad = load_silero_vad(onnx=True)

# ★ 识别引擎：默认 Haas-ja（实测最准），失败自动降级。
#   具体策略和实测数据见 asr_engine.py 顶部注释。
asr, _engine_note = load_engine(num_threads=4)
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
        seg_sec = pad_segments(seg_sec, VAD_PAD_SEC, dur)

        covered = sum(b - a for a, b in seg_sec)
        print(f"  VAD {len(seg_sec)} 段，覆盖 {covered:.1f}s "
              f"({covered / max(0.1, dur) * 100:.1f}%)", flush=True)

        rows = []
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
