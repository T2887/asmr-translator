# -*- coding: utf-8 -*-
"""whisper 分段策略（移植自 TransWithAI ChickenRice 引擎，MIT）。

原仓库：TransWithAI/Faster-Whisper-TransWithAI-ChickenRice
       src/faster_whisper_transwithai_chickenrice/infer.py
只移植两个纯算法（不引入依赖、不改许可证）：算法本身也属 MIT。

为什么我们需要它
────────────────
whisper 硬切 30 秒窗口会切在词中间 → 首尾音节丢失、字幕时间轴偏移，
这是全量模式长期存在的坑。ChickenRice 的解法是「在决策窗里找最长
语音间隙再切」，比固定 30s 硬切明显合理。

merge_segments 则专治 whisper 的**跨段重复输出**
（同一句在相邻段重复出现，是 whisper 的经典毛病）。

⚠️ 这两个函数是纯逻辑，与模型无关，不改变音频内容。
"""
from dataclasses import dataclass


@dataclass
class SpeechSpan:
    start: float
    end: float


@dataclass
class AudioChunk:
    index: int
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    text: str


# ── 1. 智能切块：在决策窗内找最长语音间隙 ──────────────────────
def create_contiguous_chunks(spans, max_duration, total_duration,
                            split_window_factor=0.4):
    """把 [0, total_duration] 切成 ≤max_duration 的连续块，切点落在语音间隙。

    spans: [(start, end)] 语音段（VAD 结果）。
    split_window_factor: 决策窗起点 = current_start + max_duration*(1-factor)。
      0.4 → 最后 12 秒（30s 窗口时）里找间隙。
    找不到间隙（连续说话）→ 退化为硬切，与原实现一致。
    """
    if max_duration <= 0:
        raise ValueError("max_duration must be greater than 0")
    if total_duration <= 0:
        return []
    if total_duration <= max_duration:
        return [AudioChunk(0, 0.0, total_duration)]

    ordered = sorted((SpeechSpan(float(a), float(b)) for a, b in spans
                      if b > a), key=lambda s: s.start)
    chunks = []
    cur = 0.0
    while cur < total_duration:
        potential_end = cur + max_duration
        if potential_end >= total_duration:
            chunks.append(AudioChunk(len(chunks), cur, total_duration))
            break
        zone_start = cur + max_duration * (1 - split_window_factor)
        best_split, best_gap = None, 0.0
        for prev, nxt in zip(ordered, ordered[1:]):
            gs, ge = prev.end, nxt.start
            if zone_start <= gs and ge <= potential_end:
                gap = ge - gs
                if gap > 0.1 and gap > best_gap:
                    best_gap = gap
                    best_split = gs + gap / 2
        split = best_split if best_split is not None else potential_end
        split = max(cur, min(split, potential_end, total_duration))
        chunks.append(AudioChunk(len(chunks), cur, split))
        cur = split
    return chunks


# ── 2. 段合并：按文本前后缀包含去重 ───────────────────────────
def _norm_merge_text(t):
    """合并判重用的归一化：去空白与常见标点，只留文字。"""
    import re
    return re.sub(r"[\s。、，,.!?！？…〜~ー：:；;「」『』\"'()（）\[\]【】]", "",
                  t or "")


def merge_segments(segments, max_gap=2.0, max_duration=20.0, enabled=True):
    """合并相邻重复段。segments: [Segment]。

    判定（与原实现一致）：
      · 间隔 gap > max_gap → 不合并
      · 合并后时长 > max_duration → 不合并
      · 文本【前缀/后缀互相包含】才合并 —— 这是去重复的关键，
        单纯"相邻"不合并，否则会把正常的两句话粘成一句。
    合并时保留较长/较全的那个文本，时间取并集。
    """
    segs = [s for s in segments if (s.text or "").strip()]
    segs.sort(key=lambda s: (s.start, s.end))
    if not enabled:
        return segs
    out = []
    for seg in segs:
        if not out:
            out.append(seg)
            continue
        last = out[-1]
        if seg.start - last.end > max_gap:
            out.append(seg)
            continue
        if seg.end - last.start > max_duration:
            out.append(seg)
            continue
        ln = _norm_merge_text(last.text)
        sn = _norm_merge_text(seg.text)
        if not ln or not sn:
            out.append(seg)
            continue
        # 后段包含前段 → 用后段（更全）
        if sn.startswith(ln):
            out[-1] = Segment(last.start, max(last.end, seg.end), seg.text)
            continue
        # 前段包含后段（头或尾）→ 保留前段
        if ln.startswith(sn) or ln.endswith(sn):
            out[-1] = Segment(last.start, max(last.end, seg.end), last.text)
            continue
        # 后段以前段结尾 → 用后段
        if sn.endswith(ln):
            out[-1] = Segment(last.start, max(last.end, seg.end), seg.text)
            continue
        out.append(seg)
    return out


def enforce_timeline(segments):
    """强制时间轴单调不重叠（原实现的 enforce_segment_timeline）。

    whisper 偶尔给出重叠/倒退的时间戳，播放器会闪跳。
    规则：每段起点不早于上一段终点；终点不小于起点。
    """
    out = []
    prev_end = 0.0
    for s in sorted(segments, key=lambda x: (x.start, x.end)):
        a = max(s.start, prev_end)
        b = max(s.end, a)
        out.append(Segment(a, b, s.text))
        prev_end = b
    return out


if __name__ == "__main__":
    # 自检：构造三种典型情况
    spans = [(0, 5), (7, 20), (22, 34), (36, 60), (61, 90)]
    ch = create_contiguous_chunks(spans, 30, 90)
    print("切块（应在 20~22 附近的间隙切，而不是硬切 30）:")
    for c in ch:
        print(f"  #{c.index} {c.start:6.2f} ~ {c.end:6.2f}")

    print("\n去重复合并:")
    segs = [Segment(0, 3, "你好啊"), Segment(3.5, 6, "你好啊，今天"),
            Segment(10, 12, "完全无关的一句")]
    for s in merge_segments(segs):
        print(f"  {s.start:5.1f}~{s.end:5.1f}  {s.text}")

    print("\n时间轴修正:")
    bad = [Segment(0, 5, "A"), Segment(3, 8, "B"), Segment(9, 7, "C")]
    for s in enforce_timeline(bad):
        print(f"  {s.start:5.1f}~{s.end:5.1f}  {s.text}")
