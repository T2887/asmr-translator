"""e_字幕整理.py —— 第5步：统一整理字幕时间轴（v1.17）

【为什么单独做这一步】

  识别阶段拿到的 segment.end 来自 faster-whisper 的「压缩时间轴」映射：
  VAD 把静音去掉后音频被拼短，Whisper 在压缩坐标里解码，最后再用
  total_silence_before + t 映射回原始时间轴 —— 于是 segment 跨越的
  段间静音全被算进了字幕显示时长。Whisper 又常把段尾推到 30 秒解码
  窗口的末尾，语音占比低时 30 秒压缩音频 = 原音频 75~220 秒。

  实测真实产物：一条「好的 拜托了」挂了 220 秒，一条 13 秒的字幕只有
  一句话，96% 的字幕 end 恰好等于下一条 start。用户看到的就是
  「这句话说完了，字幕还一直挂着」。

  这不是识别质量问题，而是「时间轴后处理」这一层缺位。所以拆出本脚本：
    · 识别脚本只负责「文本 + 粗略时间」；
    · 这里统一做：时长上限、VAD 段对齐、最小显示时长、相邻不重叠。

  整理只依赖 SRT 文本，所以也能对【已有旧字幕】批量重跑（见 --batch），
  不用重新跑一遍 ASR。

【用法】

  python e_字幕整理.py                    # 流水线模式：处理 3asr 下所有 srt
  python e_字幕整理.py --dry-run          # 流水线模式，只统计不写盘
  python e_字幕整理.py --batch D:\\字幕目录 # 批量修旧字幕（保守模式，默认只报告）
  python e_字幕整理.py --batch D:\\字幕目录 --write   # 真正覆盖写回

【参数（按实测语速校准）】

  · high = 0.90 秒/字 + 0.8 秒 —— 慢语速上限。实测正常字幕语速
    p90 = 0.89 秒/字，取这个值是为了不误杀正常长句。
  · low  = 0.30 秒/字 + 0.6 秒 —— 正常语速保底，避免 VAD 段太短时
    字幕一闪就没。
  · 该条 start 所在 VAD 段的结尾 —— 真实语音边界，跨静音就截这里。
"""

import glob
import os
import re
import sys
from pathlib import Path

from srt_util import read_srt, write_srt

try:
    from events import emit
except Exception:  # 旧包里可能没有 events.py，不影响整理本身
    def emit(*_a, **_k):
        pass


# ── 固定错译词表（v1.17.1）────────────────────────────────
# 词表放在脚本同目录：
#   词表.txt              全局规则
#   词表_<音频名>.txt     只对某个音频生效（按作品的人名/术语）
# 只放高置信度的固定错译；同义改写/润色不属于这里。
_WORDLIST_DIR = os.path.dirname(os.path.abspath(__file__))
_WORDLIST_FILE = os.path.join(_WORDLIST_DIR, "词表.txt")
_WORD_HITS = [0]  # 用 list 包一层，方便在函数里累加


def _load_wordlist(stem=None):
    """读全局词表 + 可选的按作品词表。stem 是音频名（不带 _(Vocals) 后缀）。"""
    files = [_WORDLIST_FILE]
    if stem:
        files.append(os.path.join(_WORDLIST_DIR, f"词表_{stem}.txt"))
    pairs = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8-sig") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#") or "=>" not in ln:
                        continue
                    a, b = ln.split("=>", 1)
                    a, b = a.strip(), b.strip()
                    if a and b and a != b:
                        pairs.append((a, b))
        except FileNotFoundError:
            pass
    # 长词优先，避免短词先把长词切坏
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def _apply_words(text, pairs):
    for a, b in pairs:
        if a in text:
            _WORD_HITS[0] += 1
            text = text.replace(a, b)
    return text


_REPEAT_WHITELIST = {
    # ASMR 的正常拟声叠词（「ABAB」形状），折了会丢语义（千问评审指出）
    "咕嘟", "咕噜", "扑通", "叮咚", "吧唧", "哗啦", "滴答", "呼噜", "啪嗒", "咔嚓",
    "咕叽", "啾啾", "啵啵", "噗噗", "嗡嗡", "沙沙", "簌簌", "潺潺", "呼呼", "咚咚",
    "砰砰", "啪啪", "嗒嗒", "嘶嘶", "哼哼", "哈哈",
}


def _collapse_repeats(text, max_len=4):
    """折叠相邻重复片段：`欢迎欢迎` → `欢迎`（v1.24.3）。

    为什么需要：第 4 步开着 initial_prompt / hotwords 时，whisper 会把碎段
    合并成更流畅的长句（实测 22 条 → 11 条，字幕质量明显更好），代价是在
    块拼接边界重复吐出同一段词：
        「哎呀 欢迎欢迎 欢迎来到我的保健室」← 日语只说了一次「欢迎」
    分别只开 prompt 或只开 hotwords 都能触发 —— 所以不能靠关掉它们来修，
    关掉字幕会碎成一条一条的。

    规则：
      · 只折叠**重复单元长度 >= 2** 的（「谢谢 / 看看 / 嗯嗯」这种单字重复
        单元天然进不了循环，不动）
      · 单元长度上限 4（原 8；长句重复罕见，折了反而丢内容）
      · 白名单里的拟声叠词跳过
      · 连续重复多次一次折成一份
    与 srt.rs 的 collapse_repeats 等价。
    """
    cs = list(text)
    n = len(cs)
    out = []
    i = 0
    while i < n:
        hit = False
        for l in range(min(max_len, (n - i) // 2), 1, -1):
            if cs[i:i + l] == cs[i + l:i + 2 * l]:
                if "".join(cs[i:i + l]) in _REPEAT_WHITELIST:
                    break          # 正常拟声叠词，整串保留
                k = 2
                while i + (k + 1) * l <= n and cs[i:i + l] == cs[i + k * l:i + (k + 1) * l]:
                    k += 1
                # 重复之后紧跟「的 / 地 / 得 / 着」的是正常叠词
                # （一抖一抖的 / 大口大口地 / biubiu地 / 摩擦摩擦的）；
                # 重复幻觉后面跟的是新内容或句尾（欢迎欢迎 欢迎来到…）。
                if i + k * l < n and cs[i + k * l] in "的地得着":
                    break
                out.extend(cs[i:i + l])
                i += k * l
                hit = True
                break
        if not hit:
            out.append(cs[i])
            i += 1
    return "".join(out)


# ── 时间轴整理 ──────────────────────────────────────────────

def _speech_tail(start, vad_segs, gap=0.5, max_span=15.0):
    """从 start 所在段起向后连段，返回同一口气说到最后的段末。"""
    idx = None
    for i, seg in enumerate(vad_segs):
        if seg[0] - 0.25 <= start <= seg[1] + 0.25:
            idx = i
            break
    if idx is None:
        return None
    tail = vad_segs[idx][1]
    for seg in vad_segs[idx + 1:]:
        va, vb = seg[0], seg[1]
        if va - tail <= gap and vb - start <= max_span:
            tail = vb
        else:
            break
    return tail


def _cap_end(start, end, text, vad_segs, relax=1.0):
    """给一条字幕算最长合理的 end。

    relax 是批量模式的宽容系数：
      · 流水线模式（有 VAD 段，边界准）= 1.0
      · 批量修旧字幕（没有 VAD，只能靠字速）= 2.0
        —— 实测 1.2~2x 的条目里混着正常的慢语速（ASMR 很常见），
        按 1.0 写回会误伤 25% 的正常字幕（千问 review 指出）。
    """
    n = len(text.replace(" ", ""))
    low = 0.30 * n + 0.6
    high = (0.90 * n + 0.8) * relax
    cap = start + high

    # v1.24.2：段末改成「同一口气连起来之后的段末」，并且允许把 whisper
    # 偏早的 end 补到段末（最多 1.2 秒）。与 d_语音识别.py 保持一致。
    seg_end = _speech_tail(start, vad_segs)
    if seg_end is not None:
        cap = min(cap, max(seg_end, start + low))
        if end < seg_end and seg_end - end <= 2.0:
            end = max(end, min(seg_end, end + 1.2))
    return min(end, cap)


def tidy_rows(rows, vad_segs, relax=1.0, wordlist=None):
    """整理一份字幕，返回 (新 rows, 收缩条数, 整理前总长, 整理后总长)。"""
    if wordlist is None:
        wordlist = []
    out = []
    shrunk = 0
    before = 0.0
    for i, (a, b, t) in enumerate(rows):
        nb = _cap_end(a, b, t, vad_segs, relax)
        # 不压到下一句
        nxt = rows[i + 1][0] if i + 1 < len(rows) else None
        if nxt is not None and nxt > a:
            nb = min(nb, nxt)
            # 最少显示 0.5 秒，但不超过下一句
            nb = max(nb, min(a + 0.5, nxt))
        if nb <= a:
            nb = a + 0.3
        if nb < b - 0.01:
            shrunk += 1
        before += b - a
        # 词表替换 + 折叠重复（v1.24.3）：流水线和 --batch 都会经过这一步。
        out.append((a, nb, _collapse_repeats(_apply_words(t, wordlist))))
    after = sum(b - a for a, b, _ in out)
    return out, shrunk, before, after


def _vad_for(srt_path, vad_dir):
    """按字幕文件名找对应的 VAD 段；找不到返回空列表（保守模式）。"""
    stem = os.path.splitext(os.path.basename(srt_path))[0]
    m = re.match(r"^(.+?)_\(Vocals\)", stem)
    base = m.group(1) if m else stem
    vad_path = os.path.join(vad_dir, base + ".srt")
    if not os.path.isfile(vad_path):
        return []
    try:
        return [(a, b) for a, b, _t in read_srt(vad_path)]
    except Exception:
        return []


def _validate(rows):
    """整理结果的自检：时间戳合法、单调、无重叠。

    写回用户已有字幕之前必须过这一关 —— 宁可这次不整理，
    也不能把一份本来能用的字幕写成坏文件。
    """
    prev_end = -1.0
    for a, b, _t in rows:
        if b <= a:
            return False
        if a < prev_end - 0.001:
            return False
        prev_end = b
    return True


def _max_dur(rows):
    return max((b - a for a, b, _ in rows), default=0.0)


def _process_one(path, vad_segs, write, tag="", with_sub_events=True,
                 relax=1.0, out_dir=None):
    rows = read_srt(path)
    if not rows:
        print(f"  [跳过] {os.path.basename(path)}：空字幕")
        return None
    # 词表：全局 + 按作品（文件名里的音频名）
    _stem = os.path.splitext(os.path.basename(path))[0]
    _m = re.match(r"^(.+?)_\(Vocals\)", _stem)
    _base = _m.group(1) if _m else _stem
    wordlist = _load_wordlist(_base)

    out, shrunk, before, after = tidy_rows(rows, vad_segs, relax, wordlist)
    name = os.path.basename(path)
    msg = (f"  [整理] {name}：{len(rows)} 条，收缩 {shrunk} 条，"
           f"总长 {before:.0f}s → {after:.0f}s，"
           f"最长 {_max_dur(rows):.1f}s → {_max_dur(out):.1f}s")
    print(msg)
    emit("tidy", f"before={before:.1f}", f"after={after:.1f}",
         f"shrunk={shrunk}", f"n={len(rows)}")
    # 逐条 sub 事件只给流水线模式的实时预览用；批量修 4 万条字幕时
    # 再发 4 万行事件纯属刷屏（Rust 也不在跑）。
    if with_sub_events:
        for a, b, t in out:
            emit("sub", f"{a:.2f}", f"{b:.2f}", t)
    if write:
        if not _validate(out):
            print(f"  [跳过写回] {name}：整理结果未通过校验（保留原文件）")
            return None
        target = path
        if out_dir:
            # --out 模式：不碰原文件，把整理结果写成副本，供人工试看。
            os.makedirs(out_dir, exist_ok=True)
            target = os.path.join(out_dir, name)
        # 先写临时文件再原子替换：中途失败也不会留下半截字幕。
        tmp = target + ".tmp"
        write_srt(tmp, out)
        os.replace(tmp, target)
    return (len(rows), shrunk, before, after, _max_dur(rows), _max_dur(out))


def pipeline_main(write, relax=1.0):
    """流水线模式：整理 3asr 下的所有 srt，VAD 段从 2vad 取。"""
    # ★ 延迟导入 settings：它顶部会 import torch（2~3 秒 + 几百 MB）。
    #   批量修旧字幕（--batch）根本不需要它，别为了一条命令去加载 torch。
    from settings import config

    asr_dir = config["path"]["asr"]
    vad_dir = config["path"]["vad"]
    files = sorted(glob.glob(os.path.join(asr_dir, "*.srt")))
    if not files:
        print(f"在 {asr_dir} 中没有找到字幕文件")
        # 非 0 退出码：没有字幕 = 这一步没做成，不能让界面以为成功
        sys.exit(1)

    emit("stage", "start", "tidy")
    total = len(files)
    agg = []
    for idx, path in enumerate(files, 1):
        emit("file", idx, total, os.path.basename(path))
        vad = _vad_for(path, vad_dir)
        if not vad:
            print(f"  [注意] {os.path.basename(path)}：没有找到 VAD 段，"
                  f"只按字速上限整理（保守模式）")
        r = _process_one(path, vad, write, relax=relax)
        if r:
            agg.append(r)

    _summary(agg, write)
    emit("stage", "done", "tidy")


def batch_main(root, write, relax=2.0, out_dir=None):
    """批量模式：递归修旧 .chs.srt。没有 VAD 段，只用字速上限 + 不重叠。

    默认 relax=2.0 —— 批量模式没有 VAD 边界，只能靠字速；
    实测 1.2~2x 里混着正常的慢语速，按 1.0 会误伤约 25% 的字幕。
    """
    if os.path.isfile(root):
        paths = [root]
    else:
        paths = sorted(glob.glob(os.path.join(root, "**", "*.chs.srt"),
                                 recursive=True))
    if not paths:
        print(f"在 {root} 下没有找到 *.chs.srt")
        sys.exit(1)

    print(f"批量模式：{len(paths)} 个文件，宽容系数 {relax:.1f}，"
          f"{'写回' if write else '只报告（加 --write 才写盘）'}")
    agg = []
    for idx, path in enumerate(paths, 1):
        emit("file", idx, len(paths), os.path.basename(path))
        r = _process_one(path, [], write, with_sub_events=False,
                         relax=relax, out_dir=out_dir)
        if r:
            agg.append(r)
    _summary(agg, write)


def _summary(agg, write):
    if not agg:
        print("\n没有可整理的字幕")
        return
    n_files = len(agg)
    n_rows = sum(a[0] for a in agg)
    n_shrunk = sum(a[1] for a in agg)
    before = sum(a[2] for a in agg)
    after = sum(a[3] for a in agg)
    max_before = max(a[4] for a in agg)
    max_after = max(a[5] for a in agg)
    print("\n" + "=" * 58)
    print(f"字幕整理完毕：{n_files} 个文件 / {n_rows} 条字幕"
          f"（{'已写回' if write else 'DRY-RUN，未写盘'}）")
    print(f"  收缩了 {n_shrunk} 条的结束时间")
    if _WORD_HITS[0]:
        print(f"  词表替换 {_WORD_HITS[0]} 处（全局 {len(_load_wordlist())} 条规则）")
    print(f"  总时长 {before:.0f}s → {after:.0f}s")
    print(f"  最长一条 {max_before:.1f}s → {max_after:.1f}s")
    print("=" * 58)


def main():
    argv = sys.argv[1:]
    write = "--write" in argv
    dry = "--dry-run" in argv

    def _opt(name):
        if name not in argv:
            return None
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) else None

    relax = 1.0
    try:
        if _opt("--relax") is not None:
            relax = float(_opt("--relax"))
    except Exception:
        relax = 1.0

    out_dir = _opt("--out")

    if "--batch" in argv:
        root = _opt("--batch") or "."
        # 批量模式默认 2.0：没有 VAD 边界，只能靠字速，必须更宽容。
        batch_main(root, (write or bool(out_dir)) and not dry,
                   relax if _opt("--relax") is not None else 2.0, out_dir)
    else:
        pipeline_main(not dry, relax)


if __name__ == "__main__":
    main()
