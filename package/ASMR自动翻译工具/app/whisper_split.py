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
import os
import re


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


# ★ v1.30.8：单条字幕内部的重复刷屏折叠。
#
#   实测（巧可 RJ277717 第 6 条，2026-11-25）whisper 会吐：
#     00:01:51,000 --> 00:02:21,000
#     小穴 高潮 高潮 高潮 高潮 ...（54 次「高潮」，30 秒）
#   这是 novad 模式独有的病灶：关掉 VAD 后没有「这窗有人声」的约束，
#   长静音里模型用同一个词填满整窗。
#
#   为什么不能简单折叠所有重复：ASMR 里「喜欢 喜欢 喜欢」「老师 老师」
#   是**真实台词**（人工字幕也这么写），折叠会毁掉表演。实测 5 素材
#   207 条字幕里只有 2.9% 含重复，其中大半是真实语气。
#
#   所以判据要严：
#     ① 重复单元 2~12 字符（覆盖「高潮」「不要停」「好舒服」这类）
#     ② 重复 **≥5 次**才动手（「喜欢×5」保留）
#     ③ 且重复占该条**正文 50% 以上**（排比句里夹一次不算）
#   满足时折成 `单元…（×N）`，保留前两次让人看出是重复。
_REP_UNIT = None


def collapse_repeats(text, min_repeat=5, min_share=0.5):
    """折叠单条字幕里的极端重复。返回 (新文本, 是否改动)。"""
    import re as _re
    # v1.40.2：门槛从 20 字降到 12 字、单元下限从 2 字降到 1 字。
    #   实测漏网样本「亲 亲 亲 亲 亲 亲」（11 字）「呃 呃 呃 …」（23 字但
    #   单元只有 1 个字）都是被这两个下限挡在外面，而它们恰恰是纯幻觉。
    #   折叠只保留前两遍 + 计数标注，所以即便误判也不会丢信息。
    if not text or len(text) < 10:
        return text, False
    core = _re.sub(r"[\s、。，．！？!?…♡♥♪~〜「」『』（）()]", "", text)
    if len(core) < 5:
        return text, False

    best = None                              # (单元, 次数, 覆盖长度)
    for unit_len in range(1, 13):
        i = 0
        n = len(core)
        while i + unit_len * 2 <= n:
            u = core[i:i + unit_len]
            k = 1
            while core[i + k * unit_len:(i + (k + 1) * unit_len)] == u:
                k += 1
            if k >= min_repeat and k * unit_len >= len(core) * min_share:
                if best is None or k * unit_len > best[2]:
                    best = (u, k, k * unit_len)
                i += k * unit_len
            else:
                i += 1
    if not best:
        return text, False

    u, k, cov = best
    # 保留前两次 + 计数标注，比整段删掉更诚实（用户能看出这段是重复）
    kept = text
    # 用文本里的原始形态替换（core 去过标点，不能直接拿它做替换）
    pat = _re.compile(r"(?:%s[\s、。，．！？!?…♡♥♪~〜]*){%d,}"
                      % (_re.escape(u), k))
    m = pat.search(kept)
    if not m:
        # 标点把它切碎了，退化成「只保留前两遍」
        idx = kept.find(u)
        if idx < 0:
            return text, False
        tail = kept[idx + len(u) * 2:]
        if len(tail.strip(" 、。，．！？!?…♡♥♪~〜")) > 0:
            return text, False
        return kept[:idx + len(u) * 2] + f"…（×{k}）", True
    new = kept[:m.start()] + f"{u}…（×{k}）" + kept[m.end():]
    return new, True


def collapse_rows(rows, min_repeat=5, drop_over=6):
    """对 [(start, end, text)] 整批折叠，返回 (rows, 改动条数)。

    ★ v1.40.2：`drop_over` 次以上的重复**整段丢弃**（8 → 6，用户 m23735 拍板），不再折成 `（×N）` 保留。

    起因（用户实测）：全量模式产出
        `请把你的肉棒插到小穴深处 尽情地摩擦…（×11）`
    时间窗 6 秒 —— 6 秒说 11 遍 = 每遍 0.55 秒，人说不出来。
    这是 whisper 在静音/音效段的循环幻觉（novad 下没有「这窗有人声」的
    约束），折叠成带计数的标注只是把一个错误包装得更像真的，用户点时间
    播放就会发现内容对不上。

    阈值 8：真实台词「喜欢 喜欢 喜欢」「老师 老师 老师」最多 5~6 次。
    """
    out, n = [], 0
    dropped = 0
    for a, b, t in rows:
        nt, hit = collapse_repeats(t, min_repeat=min_repeat)
        if hit:
            # 从折叠结果里把次数抠出来（形如 `…（×11）`）
            _k = 0
            _p = nt.rfind("（×")
            if _p >= 0:
                _q = nt.find("）", _p)
                if _q > _p:
                    try:
                        _k = int(nt[_p + 2:_q])
                    except ValueError:
                        _k = 0
            # v1.40.2：> 改成 >= —— 用户截图那条「小穴高潮了…（×8）」正好 8 次，
            #    判据放过了它（m23540「这不是还有吗」）。8 次重复在 ASMR
            #   里也绝不可能是正常台词（每遍 0.6 秒）。
            if _k >= drop_over:
                dropped += 1
                print(f"  [丢弃幻觉] {a:.1f}-{b:.1f}  ×{_k} 次重复，"
                      f"整段丢弃：{t[:30]}")
                continue
            n += 1
            print(f"  [折叠重复] {a:.1f}-{b:.1f}  {t[:36]} → {nt[:36]}")
        out.append((a, b, nt))
    if dropped:
        print(f"  [丢弃幻觉] 共 {dropped} 条（重复 >={drop_over} 次）")
    return out, n



# ════════════════════════════════════════════════════════════════
# 人声闸（v1.30.8）：用独立 ASR 复核「长时段少字」的段
# ════════════════════════════════════════════════════════════════
#
# 为什么需要它
# ────────────
# novad（v1.30.6 起的默认）不再用 VAD 裁解码窗口，好处是不丢喘息、气声、
# 耳语（覆盖率 71% → 87%），代价是 whisper 会拿到纯音效的窗口，在这种
# 窗口里它经常顺着 initial_prompt 编出整句台词。
#
# 实测（04 号素材，144 段）—— 常规置信度手段全部失效：
#   · no_speech_prob 70/70 段恒为 0.000（译中特化模型把这个头训坏了）；
#   · avg_logprob / compression_ratio 是**窗口级**的值，相邻 5~7 段完全
#     同值（-0.562/1.25 连 7 段），分不出同窗内的真假段；
#   · 能量反向 —— 接吻声峰值 -5~-10 dB，耳语真语音帧 RMS 才 -40 dB；
#   · ASMR VAD 有漏检：它对 272.6-274.3s 打 0.000，而 Haas-ja 在那里识出
#     「私のおへそのあたりに」(conf 0.948)，是真语音。拿它当硬闸会删掉
#     35 段真台词。
#
# 所以唯一可靠的信号是「换一个模型再听一遍」：
#   · 独立 ASR 返回空串           → 铁证幻觉（1133.9-1145.9s 那句「约20秒
#                                   的舌吻 结束」确实是编的）；
#   · 返回的字全是极少数几种     → 那是拟声词/音效，不是台词。
#
# 成本：只有「字/秒 < min_cps」的段才会被送复核（同素材 28 段 46.4 秒），
# 而且 Haas-ja 在 CPU 上跑，可以和 whisper 的 GPU 并行。
#
# 为什么还要 min_dur 这一层：`え私?`(conf 0.966) / `じゃあ` / `ほら` /
# `いいわよ` / `出して?` 这些真语音的字/秒同样很低，字符种类也只有 2~4，
# 但它们都在 4 秒以内 —— 只用「长 + 少」两个条件同时成立才丢，短句一个
# 都不碰。
# ════════════════════════════════════════════════════════════════
_CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def cjk_count(t):
    """数一行字幕里有多少个假名/汉字（标点、空格、数字不算）。"""
    return len(_CJK_RE.findall(t or ""))


# 纯拟声字表：一整行如果只由这些字组成，那它描述的是音效，不是台词。
# 没有放「哈」「呼」「吸」「舔」「揉」—— 它们是真的台词用字
# （「哈哈」「呼呼大睡」「吸出来」「舔舐」「揉揉」人工字幕里都有）。
_ONO = set("叭啪啾滋咕咚噗咻吮啧嘟啵咂嘎咔嗒吭嘶咝嚓啪叽咕嘟亲呲")
# whisper 的元描述幻觉：「约20秒的舌吻 结束」「约10秒的舌吻时间」——
# 它不是在翻译台词，是在描述这段音频。
_META_RE = re.compile(r"[约約]?\s*\d+\s*秒")
_META_HINT = ("舌吻", "接吻", "吻", "时间", "口交", "舔", "结束", "开始")


_ASCII_JUNK_HEAD = re.compile(r"^[A-Za-z][A-Za-z'\-]{1,7}\s+")
_ASCII_JUNK_TAIL = re.compile(r"\s+[A-Za-z][A-Za-z'\-]{0,7}$")


def strip_ascii_junk(txt):
    """去掉句首/句尾的拉丁字母垃圾。

    whisper 在日语音频上偶尔会在句首吐一段拉丁字母，实测
    RJ01426911 第一句是「hisp 仅限一周 和最喜欢的声优一起度过…」——
    那个 'hisp' 是「ウィスプ」被听岔了，留下来只会拉低和人工字幕的相似度。

    只删【句首】且【纯字母 2~8 个】且【后面还有中文】的片段；
    句尾同理。'OK 好的'、'BGM 响起来了' 这种前面是常见词的保留 ——
    所以这里额外要求：被删的片段不在白名单里。
    """
    if not txt:
        return txt
    _keep = {"ok", "bm", "bgm", "se", "cv", "as", "no", "yes", "hi", "hey", "oh"}
    m = _ASCII_JUNK_HEAD.match(txt)
    if m:
        w = m.group(0).strip().lower()
        rest = txt[m.end():]
        if w not in _keep and cjk_count(rest) >= 2:
            txt = rest
    m = _ASCII_JUNK_TAIL.search(txt)
    if m:
        w = m.group(0).strip().lower()
        rest = txt[:m.start()]
        if w not in _keep and cjk_count(rest) >= 2:
            txt = rest
    return txt.strip()


def drop_ono(rows, log=None):
    """丢掉「整行只由拟声字组成」的段，以及「约N秒的X」这类元描述。

    这是 novad 下第二大类假字幕（第一类是纯数字倒计时，见 drop_junk）。
    whisper 拿到接吻/舔耳的音效窗口时，会老老实实把音效音写成中文
    拟声词：`叭叭叭叭`、`滋滋滋`、`啪 啾啪`、`咕咚咕咚`、`亲 亲`。
    这些不是台词，却会占掉一整条字幕的位置。

    判据只看「整行去掉标点后是否全部由拟声字构成」，所以
    `摸摸头 摸摸头`（真台词）、`揉揉揉`（「揉」不在表里）、`哈哈`
    （「哈」不在表里）都不会被误删。

    返回 (新 rows, 丢弃条数)。
    """
    if log is None:
        log = print
    out, n = [], 0
    for (s, e, t) in rows:
        txt = (t or "").strip()
        core = re.sub(r"[\s\W_]+", "", txt)
        if core and all(ch in _ONO for ch in core):
            n += 1
            log("  [拟声] 丢弃 %.1f-%.1f 秒纯音效行：%r" % (s, e, txt))
            continue
        if len(txt) <= 20 and _META_RE.search(txt) and any(
                h in txt for h in _META_HINT):
            n += 1
            log("  [拟声] 丢弃 %.1f-%.1f 秒元描述幻觉：%r" % (s, e, txt))
            continue
        # 判据 C（排比复读）：长句 + 字符种类极少。
        #
        #   whisper 在静音段不只复读单个词，还会套模板排比：
        #   「我会用手指摩擦你的屁股 用小穴摩擦你的肉棒 用小穴摩擦你的乳头
        #     用小穴摩擦你的肉棒 用小穴摩擦你的屁股」
        #   实测 47 个字只有 17 种字符，uniq 比 **0.36**；而正常台词实测
        #   0.87~0.95（20 条样本：0.87/0.92/0.94/0.94/0.95）。
        #
        #   长度门槛 20 字是给真实台词留的：ASMR 里「喜欢 喜欢 喜欢 喜欢 喜欢」
        #   （15 字，uniq 比 0.13）「老师 老师 老师 老师」（8 字）都是表演，
        #   不能因为重复就删。折叠归 collapse_repeats 管，这里只管长句。
        if len(core) >= 20 and len(set(core)) / len(core) < 0.55:
            n += 1
            log("  [拟声] 丢弃 %.1f-%.1f 秒排比复读：%r（%d 字仅 %d 种）"
                % (s, e, txt, len(core), len(set(core))))
            continue
        out.append((s, e, t))
    return out, n


_TITLE_RE = re.compile(r"第\s*\d+\s*[话話集回章]|^\s*(?:EP|Ep|ep|#)\s*\d+$")


def drop_junk(rows, max_len=4, log=None):
    """丢掉「整行不含假名/汉字且很短」的段。

    这是 whisper 最出名的一类幻觉：训练数据里有大量带倒计时的视频，
    于是它在纯音效窗口里会老老实实输出 `10 9 8 7 6 5 4 3 2 1 0`。
    实测 04 号素材（`H:/_probe_gate/3asr/04_16k.srt`，109 条）里有 11 条
    是这样的纯数字，占 10%。它们不含任何假名/汉字，也没有任何信息量。

    只删 max_len(4) 个字以内、且一个假名/汉字都没有的行，所以
    `…` / `♡` / `！` 也会被清掉（这些本来就该由前一段带着走），
    而 `啊` / `亲` / `嗅嗅` 这类带字的短句一个都不碰。

    返回 (新 rows, 丢弃条数)。
    """
    if log is None:
        log = print
    out, n = [], 0
    for (s, e, t) in rows:
        txt = (t or "").strip()
        if cjk_count(txt) == 0 and len(txt) <= max_len:
            n += 1
            log("  [杂讯] 丢弃 %.1f-%.1f 秒纯数字/符号：%r" % (s, e, txt))
            continue
        # v1.40.2：不限长度的纯数字序列。
        #   whisper 的倒计时幻觉不一定短：实测「10 9 8 7 6 5 4 3 2 1 0」
        #   有 21 个字符，被上面那条 max_len=4 的门槛挡在外面（m23540 用户质疑）。
        #   整条一个字都没有、只剩数字，绝不可能是台词。
        _bare = re.sub(r"[\s\W_]+", "", txt)
        # ★ v1.40.4：标题类幻觉。
        #   whisper 的训练数据里有大量番剧字幕，开场安静段会吐「第10话」
        #   「第3集」「EP12」这种东西，而且能拉满十几秒（实测用户截图
        #   m23893：0.0-16.0 秒整条就三个字「第10话」）。它不是台词。
        if _TITLE_RE.search(txt) and len(_bare) <= 8:
            n += 1
            log("  [杂讯] 丢弃 %.1f-%.1f 秒标题幻觉：%r" % (s, e, txt))
            continue
        # ★ v1.40.4：极低密度长段。
        #   正常人说话不会「十几秒只吐两三个字」——实测真台词的密度最低也有
        #   0.87 字/秒（calib_haas.py 抽 200 条量出来的）。而 whisper 在开场
        #   安静段能给出「0.0-16.0 秒 / 第10话」这种 0.19 字/秒的鬼东西。
        #   时长门槛 10 秒是给 ASMR 留的：喘息段里「啊……」确实可以拖十秒。
        _dur = float(e) - float(s)
        if _dur >= 10.0 and len(_bare) <= 6:
            n += 1
            log("  [杂讯] 丢弃 %.1f-%.1f 秒极低密度长段（%.0f 秒只有 %d 字）：%r" % (s, e, _dur, len(_bare), txt))
            continue
        if _bare and _bare.isdigit():
            n += 1
            log("  [杂讯] 丢弃 %.1f-%.1f 秒数字倒计时：%r" % (s, e, txt))
            continue
        out.append((s, e, t))
    return out, n


# 日文拟声/气声常用假名。用来判断 Haas-ja「听到的东西」是不是只有音效。
#
#   ★ 为什么需要这条：max_uniq(字符种类<=4) 抓得到 `れろれろ`，
#     但抓不到 `ちゅっ、ん、れろれろ`（6 种字符）—— 后者同样是纯舔耳音效。
#     实测 05 号素材 357.0-361.6 秒就是这样漏过去的。
#   故意不放「あいうえお」这类能被真词用到的元音单字，避免误杀。
_ONO_JA = set(
    # ★ v1.40.6：只留**真正的拟声字**。旧表把基本五十音（かきくけこさしすせそ…）
    #   全塞进来了，于是任何纯假名的日语短语都被判成拟声：
    #      → rest='' → 误杀（真台词「呵呵 开玩笑的」）
    #      → rest='耳'（1 字）→ 误杀（真台词）
    #   现在只有「舔/吸/水声/口内音」这些字才算拟声；判据也收紧到 rest == 0。
    "れろちゅっんぁぃぅぇぉゃゅょゎー～〜"
    "ぺぷぴぱぽぴゃぴゅぴょ"
    #   不要再往里加普通浊音假名（だ で ど ば び ぶ べ ぼ が ぎ ぐ げ ご
    #   ざ じ ず ぜ ぞ ちゃ ちょ しゃ しゅ しょ）。它们是日语最常用的字符，
    #   放进来就会把真台词判成拟声 —— 实测两次误杀都是这个原因：
    #     `90.0-91.2 那样的话`  ← Haas `だろ`（句尾「だろう」，不是拟声）
    #   现在表里只剩舔耳/口内音专用的那些（れ ろ ち ゅ ん っ ぺ ぴ ぱ ぽ）。
    #   代价是 `ちゅくちゅう` 这种含 `く`/`う` 的拟声判不出来，
    #   由 uniq 判据和字比失调兜底。
)


def _ono_only(heard):
    """Haas 听到的内容去掉拟声假名后，还剩几个「实词」字符。

    日文里的实词（名词/动词/形容词）一定有汉字或「非拟声」假名，
    纯音效（れろれろ、ちゅっ、ん、ぺろぺろ）去掉拟声后基本不剩。
    """
    import re as _re
    core = _re.sub(r"[\s\W_]+", "", heard or "")
    if not core:
        return True            # 一个字都没听到：交给「空串」那条判据
    if len(core) < 5:
        # ★ v1.40.6：短词也要看是不是纯拟声。旧版一律 return False，
        #   于是 `れろ`(2 字) 被放过，whisper 在它上面编了「脸上也是」（实测）。
        #   判据：短词里**一个非拟声字都没有**才算拟声 —— 这样
        #   `すぐ`/`はい`/`じゃあ`/`もう`/`え私?` 都不受影响（含非拟声字）。
        # ★ v1.40.6：要同时查中文表和日文表。Haas 返回的是日文拟声（れろ），
        #   但 profile_rows 拿到的是 whisper 的中文输出（啪嗒 啾啪嗒）——
        #   只查日文表会让「拟声行占比」永远是 0，档位判据失效。
        return all(ch in _ONO_JA or ch in _ONO for ch in core)
    rest = "".join(ch for ch in core if ch not in _ONO_JA and ch not in _ONO)
    return len(rest) == 0


def profile_rows(rows):
    """给一份转写结果打画像，用来决定过滤强度走哪一档。

    ★ v1.40.6：参数不再写死。04 号那种「舔耳/接吻音效占大半」的素材需要
      严过滤（min_dur=1.2 / min_cps=4.0），但换成对话为主的素材，同样的参数
      会把「え?私?」这类真短句一起删掉。所以先用 whisper 自己的输出判断
      这是一篇什么样的音频，再决定闸门开多大。

    三个指标（都是无量纲比例，和素材时长、语速无关）：
      ono_frac     : 疑似拟声/音效行占比 —— 音效重的素材能到 0.2 以上
      verylow_frac : 字/秒 < 1.2 的行占比 —— 长时段只说两三个字，幻觉高发
      short_frac   : 短于 1.5 秒的行占比 —— 碎片多说明边界不稳

    返回 dict，其中 tier 是 'strict' / 'normal' / 'loose'。
    """
    n = len(rows)
    if not n:
        return {"n": 0, "ono_frac": 0.0, "verylow_frac": 0.0,
                "short_frac": 0.0, "cps_med": 0.0, "tier": "normal"}
    cps = []
    ono = 0
    short = 0
    for s, e, t in rows:
        dur = float(e) - float(s)
        c = (cjk_count(t) / dur) if dur > 0 else 9.9
        cps.append(c)
        if dur < 1.5:
            short += 1
        try:
            if _ono_only(t):
                ono += 1
        except Exception:
            pass
    cps_sorted = sorted(cps)
    return {
        "n": n,
        "ono_frac": ono / n,
        "verylow_frac": sum(1 for c in cps if c < 1.2) / n,
        "short_frac": short / n,
        "cps_med": cps_sorted[n // 2],
        "tier": ("strict" if (ono / n >= 0.22 or sum(1 for c in cps if c < 1.2) / n >= 0.14)
                 else "loose" if (ono / n <= 0.06 and sum(1 for c in cps if c < 1.2) / n <= 0.04)
                 else "normal"),
    }


# 三档过滤参数。键与 gate_suspect 的关键字参数同名，直接 **展开传。
#   strict : 音效重（舔耳/接吻/水声占大半），宁可误杀也不能留假字幕
#   normal : 常规 ASMR（有台词也有音效）
#   loose  : 对话为主（几乎没有音效段），别乱删
TIER_PARAMS = {
    "strict": {"min_dur": 1.2, "min_cps": 4.0, "max_uniq": 4},
    "normal": {"min_dur": 1.8, "min_cps": 3.5, "max_uniq": 5},
    "loose":  {"min_dur": 2.5, "min_cps": 2.5, "max_uniq": 6},
}


def gate_suspect(rows, wav_path, asr_fn, min_cps=1.0, min_dur=1.2,
                 max_uniq=4, tmpdir=None, log=None, lowconf=None):
    """复核「长时段少字」的段，把纯幻觉 / 纯拟声的整段丢掉。

    rows     : [(start, end, text), ...]
    wav_path : 16k 单声道 wav（生产第 4 步读的就是这个）
    asr_fn   : callable(path) -> str，注入 Haas-ja；
               本模块不 import asr_engine，保持「纯逻辑、无模型依赖」
    lowconf  : 可选 list；逃过闸门但复核文本仍然可疑的段会 append 进来。
               供 UI 标「低置信」用 —— 不删，只提示。
    返回     : (新 rows, 丢弃条数)

    ★ min_dur 从 6.0 → 3.5（v1.40.2）→ 2.0（v1.40.6，用户要求
    「宁可误杀，不允许乱翻」）：
      6.0 是拍脑袋定的，实测把大量 4~6 秒的幻觉挡在门外。巧可 05 号里
      `小穴高潮了…（×8）`(5.4s) 和 `肉棒又在一抖一抖的了`(4.6s) 两条
      纯音效假字幕就是这样漏出去的，用户反馈「播放的这段没有说话的」。
      3.5 秒以下 Haas-ja 的识别置信度会明显下降，所以停在 3.5。

    ★ v1.40.6 再降到 1.2：2.0 仍然挡住了 1~2 秒的段，而 whisper 在纯音效上
      编出来的短句正是这个长度。实测 `725.3-726.8 脸上也是`(1.5 秒) 对应的
      Haas 输出是 `れろ`（纯舔耳拟声），因为 1.5 < 2.0 根本没进闸门。
      1.2 是「Haas 还能稳定出字」的下限；再短就只剩 `あ`、`ん` 这类单音，
      删与不删都不影响观感，还会增加耗时。
    """
    if not rows:
        return rows, 0
    if log is None:
        log = print

    try:
        import wave
        import tempfile
        import numpy as np
    except Exception as e:          # 缺库就整体跳过：宁可漏删，不能崩
        log(f"  [人声闸] 跳过（缺依赖：{e}）")
        return rows, 0

    try:
        with wave.open(wav_path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except Exception as e:
        log(f"  [人声闸] 跳过（读音频失败：{e}）")
        return rows, 0
    if sw != 2:
        log("  [人声闸] 跳过（只支持 16bit PCM）")
        return rows, 0

    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)

    own = tmpdir is None
    if own:
        tmpdir = tempfile.mkdtemp(prefix="speechgate_")

    out, drop = [], 0
    try:
        for (s, e, t) in rows:
            dur = float(e) - float(s)
            n = cjk_count(t)
            # 极短碎片：split_long_rows 按字数比例切时间，末尾一小截会分到
            # 0.06 秒（实测 06 号 277.78-277.84「下次见」）。播放器里这种字幕
            # 根本来不及显示，留着只会在评测里算成一条假幻觉。0.12 秒是
            # 「人眼还能看到一闪」的下限。
            if dur <= 0.12:
                drop += 1
                log("  [人声闸] 丢弃 %.1f-%.1f 秒（极短碎片）：%s"
                    % (s, e, (t or "")[:24]))
                continue
            cps = (n / dur) if dur > 0 else 9.9
            # 不够长、或者字够密 —— 都不动。
            #
            # ★ v1.40.6 例外：**短段却塞了很多字**也要查。
            #   旧写法 `cps >= min_cps` 直接跳过，于是「字比失调」判据永远不生效
            #   —— 中文写得越长 cps 越高、越是被跳过，逻辑正好反过来。
            #   实测 `414.8-416.8 用我的小穴 让你的肉棒变得硬邦邦`（2.0 秒 15 字，
            #   cps=7.5）就在这一步被漏掉，Haas 实际只听到 `で`（1 个假名）。
            #   真台词 2 秒说 15 个字是可能的（译为中文后会变长），所以这里
            #   只是「纳入复核」，仍由 Haas 的 heard 决定去留，不会误杀。
            # ★ v1.40.6：物理上不可能的字速，直接判幻觉，不必再问 Haas。
            #   拆分把长句切开后会出现 0.3 秒的碎片（原来的长句时间轴没变），
            #   于是「把我的小穴 变成你的专属飞机杯吧」（15 字）被标成 0.3 秒
            #   = 50 字/秒。正常人说话 5~8 字/秒，12 已经是极快语速的两倍。
            #   这类段连 Haas 都不用跑（0.3 秒的切片它也听到不东西）。
            if n >= 6 and dur > 0 and (n / dur) >= 12.0:
                drop += 1
                log("  [人声闸] 丢弃 %.1f-%.1f 秒（%.1f 秒塞了 %d 字 = %.0f 字/秒，"
                    "物理不可能）：%s" % (s, e, dur, n, n / dur, (t or "")[:24]))
                continue
            # n == 0（没有汉字，比如 `biubiubiu`、`oh yeah`）以前一律跳过。
            # 实测 `658.6-661.7 biubiubiu`(3.1 秒) 对应 Haas `んれろれろ`，
            # 是纯舔耳音效上编出来的，应当删。所以只有**很短**的无汉字行才跳过
            # （`a`、`oh`、`ok` 这类真正的语气词）。
            _t_core = "".join(ch for ch in (t or "") if ch.isalnum())
            if dur < min_dur or (n == 0 and len(_t_core) < 6):
                out.append((s, e, t))
                continue
            # 门槛 10 → 6：`1146.0-1147.4 请大家多多关照哦 quzimingyue`（1.4 秒
            # 8 个汉字，cps=5.7）原本被跳过 —— 那是 whisper 在片尾音乐上编的
            # 署名幻觉（`quzimingyue` 这种拼音就是它的版权幻觉模板），
            # Haas 一个字也没听到，本该删掉。6 字以上、3.5 秒以内的碎片一律复核。
            # 上限 3.5 → 4.0：`468.5-472.2 国王大人 您的耳朵 碰到我的脖子了`
            # （3.7 秒 16 字，cps=4.3）刚好超过 3.5 被漏掉，Haas 听到的是
            # `れろれろれろ`（纯舔耳）。4 秒以内、6 字以上的碎片一律复核。
            _dense_short = (dur <= 4.0 and n >= 6)
            if cps >= min_cps and not _dense_short:
                out.append((s, e, t))
                continue

            i0 = max(0, int(round(s * sr)))
            i1 = min(len(a), int(round(e * sr)))
            if i1 - i0 < sr // 4:
                out.append((s, e, t))
                continue

            p = os.path.join(tmpdir, "g%04d.wav" % len(out))
            try:
                seg = np.clip(a[i0:i1] * 32767.0, -32768, 32767).astype(np.int16)
                with wave.open(p, "wb") as o:
                    o.setnchannels(1)
                    o.setsampwidth(2)
                    o.setframerate(sr)
                    o.writeframes(seg.tobytes())
                heard = (asr_fn(p) or "").strip()
            except Exception:
                out.append((s, e, t))      # 复核失败就原样保留
                continue

            u = len(set(_CJK_RE.findall(heard)))
            ono = _ono_only(heard)
            # ★ v1.40.6：u <= max_uniq 这条判据只在 dur >= 3.5 时才用。
            #   min_dur 从 3.5 放到 2.0 之后，一批 2~3.5 秒的真短句被这条误杀：
            #        （u=3）→ 真台词「诶？我吗？」
            #     （u=6）→ 真台词「呵呵 开玩笑的」
            #       （u=2）→ 真台词「地板上也是」
            #   短句本来字符种类就少，uniq 判据在这个时候没有分辨力；
            #   短段只信  和空串两条（实测无误杀）。
            # ★ v1.40.6：门槛从 3.5 提到 6.0。实测 3.5 仍然误杀：
            #   `118.8-123.9 真是的 诶？我吗？`（5 秒 6 字）对应 Haas `え?私?`，
            #   字符种类只有 4 种 → 被 uniq 判据删掉，但内容其实完全对得上。
            #   5 秒以内的句子，日文字符种类天然就少（`え?私?` 全是短音），
            #   uniq 在这一档没有分辨力；6 秒以上还只有 4 种字符才真的可疑。
            # ★ 6.0 → 8.0（六素材标定）：6~8 秒的真台词仍会被 uniq 判据误杀，
            #   实测 `655.1-669.9 耳朵里面 轻轻地 轻轻地`（15 秒，Haas `すぅ。ああ`）、
            #   `469.1-475.2 刚刚才吹进去的耳垢…`（6 秒，Haas `任せおかせ`）、
            #   `758.0-765.1 呵呵 好可爱 啊`（7 秒，Haas `可愛いあー`）三条
            #   真台词全在 6~15 秒这一档。8 秒以内只信「空串 / 拟声 / 字比失调」。
            _short_ok = dur < 8.0
            # ★ 06 号素材（`おまけ～朝も夜も子づくりH`）暴露的**主要误杀源**：
            #   Haas 在这些段只听到 `ああ。` / `はあはあ。` / `あ`，于是
            #   `u = 1 <= max_uniq` 命中，整段被删。但那几段中文是
            #   `啊 舒服 啊 里面 里面 好舒服` —— 呻吟声确实存在，whisper 只是
            #   把呻吟转成了字。**「Haas 只听到感叹词」不等于「这段是空的」**，
            #   它恰恰证明有人在场。所以纯感叹一律不走 uniq 判据。
            # ★ `す` 是呼吸声（实测 `655.1-669.9 耳朵里面 轻轻地 轻轻地` 的 Haas 输出
            #   是 `すぅ。ああ`，`す` 不在表里就判不出「只有感叹词」，真台词被 uniq 误杀）。
            _INTERJ = set("あっああんっんうんはあふうええおおさあまあやあわあいやはっふっはーあーむぅうーす")
            _hset0 = set("".join(_CJK_RE.findall(heard or "")))
            _interj_only = bool(_hset0) and _hset0 <= _INTERJ
            # ★ v1.40.6：字比失调 —— 中文十几字，Haas 只听到一两个假名。
            #   实测 `414.8-416.8 用我的小穴 让你的肉棒变得硬邦邦` 对应 Haas 只听到 `で`。
            #   真台词不会出现这种比例（Haas 对真实语音的召回很全，见 calib_gate.py）。
            _hcore = "".join(_CJK_RE.findall(heard or ""))
            # 短段里的极端失调：`495.5-496.4 再一次射精吧`（0.9 秒 6 字）对应 Haas `む`，
            # 0.9 秒低于 min_dur 根本没进闸门。中文 5 字以上而日语只剩 1 个假名，
            # 说明 whisper 不是在翻译而是在编。`诶？`(2 字) / `射吧`(2 字) 这类不受影响。
            # ★ 06 号素材（`おまけ～朝も夜も子づくりH`）暴露的误判：Haas 在这几段
            #   只听到 `ああ。` / `あっ!` —— 那是**真语音的感叹**，不是"没人说话"。
            #   06 号 4 条被删全是这类（13.8% 误杀），所以纯感叹词一律不算失调。
            _lopsided = (not _interj_only) and (
                # ★ 13 → 16（六素材标定）：13~15 字这一档有三条真台词被误杀
                #   （`407.8-410.6 唉嘿 你怎么了？刚才 颤抖了一下对吧？` 15 字 ← `パチ`）。
                #   代价是 `414.8-416.8 用我的小穴 让你的肉棒变得硬邦邦`（15 字 ← `で`）
                #   会漏网，幻觉 6→7 = 3.6% 仍远低于 5% 红线。
                (n >= 16 and len(_hcore) <= 3)
                or (n >= 5 and len(_hcore) <= 1))
            # `ono` 不要求中文也是拟声。试过收紧成「Haas 判拟声 且 中文也拟声」，
            # 确实救回 `667.8-669.4 射出来`，但同时放过了
            # `986.1-997.8 给你充分地清理干净`（11.7 秒，Haas 只有 `ちゅっ、れろぺろ`）
            # 和 `1107.5-1108.7 诶？`（Haas 只有 `ん`）。
            # 用户的标准是「宁可误杀不可乱翻」，所以这里按 Haas 判。
            if (not heard) or (u <= max_uniq and not _short_ok and not _interj_only) or ono or _lopsided:
                drop += 1
                if not heard:
                    why = "一个字也没听到"
                elif _lopsided:
                    why = "字比失调（中文 %d 字 / 日语 %d 字）「%s」" % (n, len(_hcore), heard[:12])
                elif ono:
                    why = "只听出拟声「%s」" % heard[:12]
                else:
                    why = "只听出「%s」" % heard[:12]
                log("  [人声闸] 丢弃 %.1f-%.1f 秒（%.0f 秒只有 %d 字，%s）：%s"
                    % (s, e, dur, n, why, (t or "")[:24]))
                continue

            # 逃过了闸门，但 Haas-ja 听到的内容偏少（≤ 2×max_uniq = 8 种字符）。
            #
            #   阈值校准：`calib_gate.py` 抽 200 条 >6 秒真台词喂 Haas-ja，
            #   uniq 分布 min=5 / p10=13 / 中位=22。取 8 会标到约 4% 的真台词
            #   —— 但那几条恰好都是「6 秒只说 7 个字」这类信息量极低的段，
            #   标橙让用户听一下是合理的（用户定调：宁可漏翻不可错翻）。
            #   取 6 只标 1%，太保守；取 10 以上会误标正常长句。
            if lowconf is not None and u <= max_uniq * 2:
                lowconf.append((s, e, t, heard))

            out.append((s, e, t))
    finally:
        if own:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    return out, drop

# ─────────────────────────────────────────────────────────────
# 稳定化总入口（v1.40.1）
#
# 四个过滤原本散在 d_语音识别.py 的四处调用里，各自读环境变量、各自打印
# 日志。功能是对的，但「想加一条新规则」时要翻整个文件找调用点，而且
# 顺序一旦排错（比如先跑人声闸再折叠重复）结果会不一样。
#
# 这里把它们收成一个函数，顺序就是下面这个顺序，理由写在每一步上。
# 每一步都可以用对应的 ASMR_NO_* 关掉，关掉后行为和以前完全一致。
#
# 参考：Finesub（github.com/caca2331/finesub，GPL-3.0，只读了它的文档）
# 把「去噪、合并碎片、丢弃幻觉」单独叫「稳定化」一个阶段。我们这边做的是
# 同样的事，只是更保守 —— 它用云端大模型判断，我们用 Haas-ja 的独立复核。
# ─────────────────────────────────────────────────────────────
_SPLIT_PUNCT = "，。！？、；：…"


def _split_once(r, max_len):
    """拆一次；拆不开返回 None。"""
    txt = (r[2] or "").strip()
    if len(txt) < max_len:
        return None
    mid = len(txt) // 2
    cut = -1
    for i, ch in enumerate(txt):
        if ch in _SPLIT_PUNCT and 6 <= i + 1 <= len(txt) - 6:
            if cut < 0 or abs(i + 1 - mid) < abs(cut - mid):
                cut = i + 1
    if cut < 0:
        best = -1
        for i, ch in enumerate(txt):
            if ch == " " and 6 <= i <= len(txt) - 6:
                if best < 0 or abs(i - mid) < abs(best - mid):
                    best = i
        cut = best
    if cut < 0:
        return None
    a, b = txt[:cut].strip(), txt[cut:].strip()
    if len(a) < 6 or len(b) < 6:
        return None
    s0, e0 = float(r[0]), float(r[1])
    span = max(e0 - s0, 0.1)
    t = s0 + span * len(a) / max(len(a) + len(b), 1)
    return (s0, round(t, 2), a), (round(t, 2), e0, b)


def split_long_rows(rows, max_len=24, log=None):
    """把过长的字幕按标点拆成两条。

    实测（lib02 RJ01426911，广播剧风格）：我们 83 条 / 人工 110 条，
    最长 57 字。内容本身对（Haas-ja 复核 0% 幻觉），但一行 40~50 个
    汉字在播放器里要占三行，读不完就切走了。

    拆法：找【最靠近中点】的标点；没有标点就找最靠近中点的空格；
    两半都至少 6 个字才拆（防止「啊，」被单独拆出来）；时间按
    【两半的字数比例】分而不是对半。

    最多拆 3 轮：38 字按中点拆开两半可能还是 20+ 字，但再拆就成
    一句一屏，播放器跳得比听得还快。
    """
    out = list(rows)
    n = 0
    for _ in range(3):
        changed = False
        nxt = []
        for r in out:
            pair = _split_once(r, max_len)
            if pair is None:
                nxt.append(r)
            else:
                nxt.extend(pair)
                changed = True
                n += 1
        out = nxt
        if not changed:
            break
    if n and log:
        log("  [拆分] 共拆开 %d 条过长字幕" % n)
    return out


_TAIL_WORDS = ("结束", "完毕", "谢谢观看", "谢谢收看", "全剧终", "完", "终")


def drop_tail_words(rows, min_dur=2.0, log=None):
    """丢掉「整条恰好是片尾词」的段。

    whisper 的训练数据里有大量视频结尾，音频末尾它会吐「结束」这种字。
    实测用户截图 m23939：RJ01556415 Track-01 最后一条是 00:10:07 的
    「结束」，Haas-ja 在那段一个字都没听到。

    只对【整条完全等于】那些词、且【时长 >= min_dur】的段生效 ——
    「结束了」「结束了吗」都是正常台词，一律不碰。
    """
    if log is None:
        log = print
    out, n = [], 0
    for (s0, e0, t) in rows:
        txt = (t or "").strip().strip("。.!！?？")
        if txt in _TAIL_WORDS and (float(e0) - float(s0)) >= min_dur:
            n += 1
            log("  [杂讯] 丢弃 %.1f-%.1f 秒片尾词幻觉：%r" % (s0, e0, t))
            continue
        out.append((s0, e0, t))
    return out, n


def stabilize(rows, wav_path=None, asr_fn=None, lowconf=None, log=None,
              head_model=None, head_path=None, head_th=0.95):
    """把一条 whisper 原始输出过一遍稳定化。返回 (rows, stats)。

    rows     : [(start, end, text), ...]
    wav_path : 数组化后的 16k wav；给 None 就跳过需要音频的步骤（人声闸）
    asr_fn   : callable(path) -> str，人声闸用的独立 ASR（Haas-ja）。
               给 None 人也声闸整步跳过。
    lowconf  : 可选 list，收集「留但可疑」的段，给校对界面标橙
    log      : 日志输出，默认 print
    stats    : {"rep": 折了几条, "junk": 丢了几条, "ono": 丢了几条,
                "gate": 丢了几段, "lowconf": 标了几段}
    """
    if log is None:
        log = print
    stats = {"rep": 0, "junk": 0, "ono": 0, "gate": 0, "lowconf": 0, "split": 0, "tail": 0}
    if not rows:
        return rows, stats

    # ⓪ 句首/句尾的拉丁字母垃圾（「hisp 仅限一周 …」里的 hisp）。
    #    放在最前面：后面的折叠/拟声判据都是按「正文」算的，
    #    留个 hisp 在前面会让 uniq 比虚高、该折的折不动。
    _cleaned = []
    for (a, b, t) in rows:
        t2 = strip_ascii_junk(t)
        if t2:
            _cleaned.append((a, b, t2))
    rows = _cleaned

    # ① 单条内部的极限重复。
    #    最早做，因为后面的 drop_ono 判「全由拟声字组成」时，刷屏行
    #    （「高潮高潮高潮…×54」）会被误判成拟声而删掉 —— 删了也对，
    #    但日志会说成「拟声」，排查时误导人。
    # ⓿ 素材画像 —— 用 whisper 的原始输出判断这篇音频是什么类型，
    #    再决定后面几道过滤开多大。参数不再写死（见 profile_rows 的注释）。
    _prof = profile_rows(rows)
    _tier = os.environ.get("ASMR_GATE_TIER", "").strip().lower() or _prof["tier"]
    if _tier not in TIER_PARAMS:
        _tier = "normal"
    _tp = dict(TIER_PARAMS[_tier])
    stats["tier"] = _tier
    stats["profile"] = _prof
    log("  [画像] %d 条 / 拟声行 %.0f%% / 超低密度 %.0f%% / 短碎片 %.0f%% "
        "/ 中位密度 %.1f 字秒 → 档位 %s"
        % (_prof["n"], _prof["ono_frac"] * 100, _prof["verylow_frac"] * 100,
           _prof["short_frac"] * 100, _prof["cps_med"], _tier))

    if os.environ.get("ASMR_NO_COLLAPSE", "").strip() != "1":
        try:
            rows, stats["rep"] = collapse_rows(rows)
            if stats["rep"]:
                log(f"  [折叠重复] 处理 {stats['rep']} 条内部刷屏")
        except Exception as e:
            log(f"  [折叠重复] 跳过（{type(e).__name__}: {e}）")

    # ② 纯数字/符号（whisper 的倒计时幻觉：10 9 8 7 … 0）。
    if os.environ.get("ASMR_NO_JUNK", "").strip() != "1":
        try:
            rows, stats["junk"] = drop_junk(rows)
            if stats["junk"]:
                log(f"  [杂讯] 丢弃 {stats['junk']} 条纯数字/符号")
        except Exception as e:
            log(f"  [杂讯] 跳过（{type(e).__name__}: {e}）")

    # ③ 纯拟声行 + 「约N秒的X」元描述。
    if os.environ.get("ASMR_NO_ONO", "").strip() != "1":
        try:
            rows, stats["ono"] = drop_ono(rows)
            if stats["ono"]:
                log(f"  [拟声] 丢弃 {stats['ono']} 条纯音效/元描述行")
        except Exception as e:
            log(f"  [拟声] 跳过（{type(e).__name__}: {e}）")

    # ④ 人声闸：换个模型（Haas-ja）再听一遍。
    #    放最后，因为它最慢（整篇 1147 秒音频约 18 秒 CPU），
    #    而且前三步已经把大部分假字幕清掉了，可疑段数量大幅减少。
    # ④ 过长的字幕先按标点拆成两条，**再**进人声闸。
    #
    #    ★ v1.40.6 顺序调整（原来是拆分的放在闸门之后）。实测踩到的坑：
    #    `414.840-416.760 用我的小穴 让你的肉棒变得硬邦邦`（1.92 秒 15 字）
    #    原本是某个 10 秒长段拆出来的碎片。闸门看的是**拆之前**的长段
    #    （9.8 秒 30 字，cps=3.1），Haas 对整段能听到若干字，于是判「不是幻觉」
    #    整段放行；随后拆分把那一句切出来，就成了一条没人复查的假字幕。
    #    先拆后判，碎片会各自过闸（实测该段 Haas 只听到 `で`，被字比失调判据删掉）。
    if os.environ.get("ASMR_NO_SPLIT", "").strip() != "1":
        try:
            _before = len(rows)
            rows = split_long_rows(rows, max_len=24, log=None)
            if len(rows) != _before:
                stats["split"] = len(rows) - _before
                log(f"  [拆分] {_before} 条 → {len(rows)} 条（超 24 字的切开）")
        except Exception as e:
            log(f"  [拆分] 跳过（{type(e).__name__}: {e}）")

    _gate_mode = os.environ.get("ASMR_SPEECH_GATE", "").strip()
    if _gate_mode != "0" and wav_path:
        # ★ v1.40.6：**默认走 Haas，不再用分类头**。理由（实测）：
        #   分类头判的是「这段有没有人声」，而我们要判的是「whisper 翻得对不对」，
        #   这是两个任务。17 条真幻觉里 16 条的 encoder p 高达 0.96~1.00 —— 因为
        #   那些段**确实有人在说话**（只是 whisper 听岔了），所以「有人声」判据
        #   对它们永远是高分，任何阈值都救不了。ASMR_SPEECH_GATE=3 才启用分类头，
        #   仅供实验对比，不要在生产打开。
        _by_head = False
        if _gate_mode == "3" and head_model is not None and head_path:
            try:
                n0 = len(rows)
                _r, _n = gate_by_head(rows, wav_path, head_model, head_path,
                                      th=head_th, lowconf=lowconf, log=log)
                if _r is not None:
                    rows = _r
                    stats["gate"] = _n
                    _by_head = True
                    if _n:
                        log(f"  [人声闸] 丢弃 {_n} 段纯音效/幻觉："
                            f"{n0} → {len(rows)} 条")
            except Exception as e:
                log(f"  [人声闸] 分类头不可用（{type(e).__name__}: {e}），退回 Haas")
        if not _by_head and asr_fn is not None:
            try:
                n0 = len(rows)
                # log 必须传进去：不传的话 gate_suspect 内部的
                # 「丢弃 a-b 秒（原因）」全部被静默，生产日志里只剩一行汇总，
                # 排查误杀时根本不知道是哪条判据干的（实测踩过）。
                rows, stats["gate"] = gate_suspect(
                    rows, wav_path, asr_fn, lowconf=lowconf, log=log, **_tp)
                if stats["gate"]:
                    log(f"  [人声闸] 丢弃 {stats['gate']} 段纯音效/幻觉："
                        f"{n0} → {len(rows)} 条")
            except Exception as e:
                log(f"  [人声闸] 跳过（{type(e).__name__}: {e}）")

    if lowconf:
        stats["lowconf"] = len(lowconf)
    # ⑥ 片尾词幻觉（「结束」这类整条只有一个词的）。
    #    放在最后：拆分可能把一条长句拆出「结束」，所以要再查一次。
    try:
        rows, _n_tail = drop_tail_words(rows)
        stats["tail"] = _n_tail
    except Exception as e:
        log(f"  [杂讯] 片尾词检查跳过（{type(e).__name__}: {e}）")

    return rows, stats


# ─────────────────────────────────────────────────────────────────────────
# ★ v1.40.5：人声分类头（替代 Haas-ja 逐段复核）
#
#   训练：1500 条真台词（704 部作品）+ 327 条真音效难负样本
#        （同段音频相邻台词之间 >=4.5 秒的间隙，Haas 复核只听到拟声）
#   特征：CT2 encoder 输出 (301,1280) 沿时间轴取 std / max / min
#   指标：GroupKFold(5, 按作品分组) AUC 0.9976
#        th=0.95 → 召回 0.957 / 精确 0.997（327 条真音效只误报 13 条）
#   权重：model/voice_head.npz，42 KB
#
#   为什么换掉 Haas：
#   ① 快几百倍 —— encoder 在转写时本来就跑过，这里只是矩阵乘；
#      Haas 每段要切 wav + 单独转写，22 分钟素材 +2 分钟。
#   ② Haas 在 CPU 上，会和 GPU 上的 whisper 抢核。
#   ③ 精度不输 —— Haas 的判据（空串 / 字符种类 <=4）本来也是间接信号。
#
#   ★ 特征必须和训练时完全一致：6 秒 16k 单声道 → fe() → encode() →
#     取中心 6 秒，不能整段塞进去（训练样本全是 6 秒）。
# ─────────────────────────────────────────────────────────────────────────

_HEAD_CACHE = {}


def load_voice_head(path):
    """载入 voice_head.npz。失败返回 None。"""
    if path in _HEAD_CACHE:
        return _HEAD_CACHE[path]
    try:
        import numpy as np
        d = np.load(path)
        head = {
            "coef": d["coef"].astype(np.float32),
            "intercept": np.float32(d["intercept"]),
            "mean": d["mean"].astype(np.float32),
            "scale": d["scale"].astype(np.float32),
        }
    except Exception:
        head = None
    _HEAD_CACHE[path] = head
    return head


def _prob_one(fw_model, seg6, head):
    """一个 6 秒 float32 片段的「有人声」概率。

    fw_model 必须是 **faster_whisper.WhisperModel**（不是它内部的
    ctranslate2.models.Whisper）—— feature_extractor 在 WhperModel 上，
    encode() 在 fw_model.model 上，前者没有 encode、后者没有 fe。
    """
    import ctranslate2
    import numpy as np
    fe = fw_model.feature_extractor
    ct2 = getattr(fw_model, "model", fw_model)
    x = np.expand_dims(fe(seg6), 0).astype(np.float32)
    hs = ct2.encode(ctranslate2.StorageView.from_array(x))
    hs = hs.to_device(ctranslate2.Device.cpu)
    arr = np.array(hs, dtype=np.float32)[0]
    f = np.hstack([arr.std(0), arr.max(0), arr.min(0)]).astype(np.float32)
    z = ((f - head["mean"]) / head["scale"]) @ head["coef"] + head["intercept"]
    return float(1.0 / (1.0 + np.exp(-z)))


def gate_by_head(rows, wav_path, fw_model, head_path, th=0.95,
                 lowconf=None, log=None):
    """用 encoder 特征分类头判断「这段到底有没有人声」。

    与 gate_suspect 的差别：这个对**每一段**都算（一次矩阵乘而已），
    不再用「字/秒 < min_cps」先挑可疑段。

    时间窗：段长 >= 6 秒时取**中心 6 秒**（训练样本都是 6 秒）；
    短于 6 秒 pad 到 6 秒。取中心而非开头，因为 ASMR 里台词经常
    拖在长段末尾（实测 150.0-170.0 的真语音在 162.8-167.2）。
    """
    import wave
    import numpy as np

    head = load_voice_head(head_path)
    if head is None:
        return None, 0          # None = 头不可用，调用方退回 Haas

    try:
        with wave.open(wav_path, "rb") as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
    except Exception as e:
        if log:
            log("  [人声闸] 读不了 wav（%s），跳过" % e)
        return None, 0

    if sr != 16000:
        return None, 0
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    total = audio.shape[0]
    WIN = 16000 * 6

    out, dropped = [], 0
    for s, e, txt in rows:
        a = max(0, min(int(round(float(s) * 16000)), total))
        b = max(a, min(int(round(float(e) * 16000)), total))
        if b - a <= 0:
            out.append((s, e, txt))
            continue
        # ★ 和训练窗口对齐（关键）。
        #   训练样本是从「句对起点前 1 秒」切 6 秒（make_samples3.py 的
        #   ss = start - 1.0），所以真实语音落在窗口的第 1 秒之后。
        #   推理时若直接拿段本身当窗口，语音在开头，特征分布对不上 ——
        #   实测阈值 0.95 会误杀大量真台词（p 只有 0.03~0.85）。
        _pre = 16000 * 1
        try:
            if b - a >= WIN:
                # 长段滑窗：每 2 秒起一个 6 秒窗，取最高分（语音可能在任一处）
                _ps = []
                _lo = a
                while _lo + WIN <= b:
                    _ps.append(_prob_one(fw_model, audio[_lo:_lo + WIN], head))
                    _lo += 16000 * 2
                if not _ps:
                    _ps = [_prob_one(fw_model, np.pad(audio[a:b], (0, WIN - (b - a))), head)]
                p = max(_ps)
            else:
                # 训练样本是从「句对起点前 1 秒」开始切 6 秒，那 1 秒是
                # **真实音频**（静音或音效），不是零。所以这里要往前取
                # 1 秒真实样本，不能补零 —— 补零会让 p 掉到 0.9 左右
                # （实测：同一段纯切 6 秒 p=0.9999，前面补 1 秒零 → 0.92）。
                _lo = a - _pre
                _pad_head = 0
                if _lo < 0:
                    _pad_head = -_lo
                    _lo = 0
                _seg = audio[_lo:_lo + WIN]
                if _pad_head:
                    _seg = np.concatenate([np.zeros(_pad_head, dtype=np.float32), _seg])
                if _seg.shape[0] < WIN:
                    _seg = np.pad(_seg, (0, WIN - _seg.shape[0]))
                else:
                    _seg = _seg[:WIN]
                p = _prob_one(fw_model, _seg, head)
        except Exception as e:
            if log:
                log("  [人声闸] 特征失败（%s），本段保留" % e)
            out.append((s, e, txt))
            continue
        if p >= th:
            out.append((s, e, txt))
            if lowconf is not None and p < th + 0.03:
                lowconf.append((s, e, txt, "p=%.3f" % p))
        else:
            dropped += 1
            if log:
                log("  [人声闸] 丢弃 %.1f-%.1f 秒（有人声概率 %.3f < %.2f）：%r"
                    % (s, e, p, th, txt))
    if dropped and log:
        log("  [人声闸] 共丢弃 %d 段（分类头 th=%.2f）" % (dropped, th))
    return out, dropped


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
