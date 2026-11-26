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
_ONO = set("叭啪啾滋咕咚噗咻吮啧嘟啵咂嘎咔嗒吭嘶咝嚓啪叽咕嘟")
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
    "れろちゅっんぁぃぅぇぉゃゅょゎー～〜"
    "ぺぷぴぱぽぴゃぴゅぴょ"
    "ぐじずぜぞだでどばびぶべぼ"
    "ちゃちゅちょしゃしゅしょ"
    "くかきこさすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわを"
    "ぱぴぷぺぽげござ"
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
        # 太短的句子不做拟声判断 —— すぐ / はい / じゃあ / 出して 这些
        # 真台词本身就两三个字，去掉假名后几乎不剩东西，会全被误杀。
        # 短句另有 uniq 判据兜着，不需要这条。
        return False
    rest = "".join(ch for ch in core if ch not in _ONO_JA)
    return len(rest) <= 1


def gate_suspect(rows, wav_path, asr_fn, min_cps=1.0, min_dur=3.5,
                 max_uniq=4, tmpdir=None, log=None, lowconf=None):
    """复核「长时段少字」的段，把纯幻觉 / 纯拟声的整段丢掉。

    rows     : [(start, end, text), ...]
    wav_path : 16k 单声道 wav（生产第 4 步读的就是这个）
    asr_fn   : callable(path) -> str，注入 Haas-ja；
               本模块不 import asr_engine，保持「纯逻辑、无模型依赖」
    lowconf  : 可选 list；逃过闸门但复核文本仍然可疑的段会 append 进来。
               供 UI 标「低置信」用 —— 不删，只提示。
    返回     : (新 rows, 丢弃条数)

    ★ min_dur 从 6.0 降到 3.5（v1.40.2）：
      6.0 是拍脑袋定的，实测把大量 4~6 秒的幻觉挡在门外。巧可 05 号里
      `小穴高潮了…（×8）`(5.4s) 和 `肉棒又在一抖一抖的了`(4.6s) 两条
      纯音效假字幕就是这样漏出去的，用户反馈「播放的这段没有说话的」。
      3.5 秒以下 Haas-ja 的识别置信度会明显下降，所以停在 3.5。
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
            cps = (n / dur) if dur > 0 else 9.9
            # 不够长、或者字够密 —— 都不动
            if dur < min_dur or n == 0 or cps >= min_cps:
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
            if (not heard) or u <= max_uniq or ono:
                drop += 1
                if not heard:
                    why = "一个字也没听到"
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
def stabilize(rows, wav_path=None, asr_fn=None, lowconf=None, log=None):
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
    stats = {"rep": 0, "junk": 0, "ono": 0, "gate": 0, "lowconf": 0}
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
    if (os.environ.get("ASMR_SPEECH_GATE", "").strip() != "0"
            and wav_path and asr_fn):
        try:
            n0 = len(rows)
            rows, stats["gate"] = gate_suspect(
                rows, wav_path, asr_fn, min_cps=4.0, lowconf=lowconf)
            if stats["gate"]:
                log(f"  [人声闸] 丢弃 {stats['gate']} 段纯音效/幻觉："
                    f"{n0} → {len(rows)} 条")
        except Exception as e:
            log(f"  [人声闸] 跳过（{type(e).__name__}: {e}）")

    if lowconf:
        stats["lowconf"] = len(lowconf)
    return rows, stats


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
