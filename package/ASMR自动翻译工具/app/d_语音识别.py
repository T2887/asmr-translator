# ─────────────────────────────────────────────────────────────
# 【必须在任何第三方 CUDA 库之前执行】
#
# 让 ONNX Runtime / CTranslate2 能找到 cuDNN 和 cuBLAS。
#
# 背景：onnxruntime-gpu 的 CUDA provider 在加载时会去 LoadLibrary
#   "cudnn64_9.dll" 这类文件。这些 DLL 其实随 PyTorch 一起装在了
#   env\Lib\site-packages\torch\lib 下，但【那个目录不在系统 PATH 里】。
#   结果：provider 加载失败 → ONNX Runtime 只打一行 warning 就
#   【静默回退 CPU】→ 程序照常跑完，只是慢几十倍。
#   实测现象：BS-Roformer 人声分离在 5070 上跑了 7 分钟还没完，
#   而 GPU 利用率只有 8%。
#
#   PyTorch 自己不受影响（它内部知道自己的 DLL 在哪），
#   倒霉的只有那些"借用"CUDA 的第三方运行时。
#
#   add_dll_directory 必须在 import 这些库之前调 —— 它们是在
#   import 那一刻就去加载 provider DLL 的，之后再补就晚了。
#   这里用 sys.executable 推出路径，避免提前 import torch
#   （import torch 本身也会锁住 DLL 搜索路径）。
# ─────────────────────────────────────────────────────────────
import os as _dll_os
import sys as _dll_sys
import pathlib as _dll_pl

try:
    # 注意是 .parent 而不是 .parent.parent：
    #   env\python.exe  ->  .parent = env\      （site-packages 就在 env\Lib 下）
    #   写多一层会得到项目根目录，那个路径下没有 Lib，判断为 False 就静默跳过了。
    _env_root = _dll_pl.Path(_dll_sys.executable).parent
    _torch_lib = _env_root / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.is_dir():
        _dll_os.add_dll_directory(str(_torch_lib))
        _dll_os.environ["PATH"] = str(_torch_lib) + _dll_os.pathsep + _dll_os.environ.get("PATH", "")
except Exception:
    pass

import os
import sys
import glob
from pathlib import Path
from srt_util import read_srt, write_srt
import faster_whisper.vad
import faster_whisper.transcribe
from unittest.mock import patch
from faster_whisper import WhisperModel
from settings import config
from events import emit


# ─────────────────────────────────────────────────────────────
# 领域提示 + 热词（v1.17.1）
#
# 目的：把「专有词、语气词、高频表达」提前告诉解码器，减少
#   把「秘书」译成「卑处」这类固定错译，也让口语更自然。
#
# 为什么 hotwords 用【中文】而不是日文：
#   这个模型是 task="translate"（日→中一步出中文）。hotwords 是
#   作为 prompt 注入的，用中文能直接引导解码器输出正确的中文词；
#   塞日文词反而可能让模型在日文 token 和中文输出之间打架
#   （实测建议来自千问的 review）。
#
# 为什么不开 no_repeat_ngram_size=3：
#   它禁止同一段解码里出现重复 3-gram，会把「去了 去了 去了」这种
#   【真实重复喊叫】也压掉；而本项目 8% 的相邻重复大多发生在
#   【不同 segment 之间】，这个参数根本管不到。要压跨段重复应该动
#   condition_on_previous_text（那是另一个 A/B，不在本次范围）。
#
# 想调就改这两行，改完建议用同一段素材对比，别只看单条。
# ─────────────────────────────────────────────────────────────
_INITIAL_PROMPT = "以下是日语成人向音声的中文翻译，口语化、句子短促，术语用肉棒、小穴、舒服、去了，保留语气词。"
# ★ v1.29：从全局热词里移除「秘书」「社长」这两个【角色词】。
#
#   这两个词原本是为了某个特定作品加的（那个作品里有「社长」角色），
#   放在全局 _HOTWORDS 里会污染所有其它作品 —— 实测 04 号文件
#   （「ゆりか/由梨香」的个人音声）里，「社长」出现了至少 12 次：
#     「对不起 社长」「真是的 社长」「最喜欢社长了」……
#   用户明确反馈这不是该作品的称呼。
#
#   热词是作为 prompt 注入的，模型会顺着它把相近的称呼都往「社长」上靠
#   （原文多半是「お客様＝客人先生」或某个角色名）。
#
#   角色词/人名这类作品级词汇应该放 hotwords_<stem>.txt，只在对应作品生效
#   （见 _load_extra_hotwords）。全局只保留【通用的性行为术语】。
_HOTWORDS = "肉棒,小穴,舒服,去了,高潮,舔,精液,射精,口交,乳交"


def _load_extra_hotwords(stem):
    """按作品补充热词（v1.17.1）。

    读取工作目录（程序根目录）下的两个可选文件：
        hotwords.txt            对所有作品生效的补充词
        hotwords_<stem>.txt     只对某个音频生效（stem = 音频文件名）

    每行一个词，也支持逗号分隔；# 开头是注释。

    为什么必须按作品：人名/地名是作品级的（例如「凪咲」「三野轮凪咲」），
    塞进全局 _HOTWORDS 会污染其他作品 —— 别的作品可能把角色名硬拽过来
    （千问 review 指出过）。
    实测：把「凪咲/神社/巫女/迷途者」按作品加进去后，
    「渚沙」错译从 6 次降到 0，「神社」从 0 次变回 2 次。
    """
    words = []
    cwd = os.getcwd()
    for name in ("hotwords.txt", f"hotwords_{stem}.txt"):
        p = os.path.join(cwd, name)
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
                        if w and w not in words:
                            words.append(w)
        except Exception:
            pass
    return words


# ─────────────────────────────────────────────────────────────
# 心跳：给"要等很久又没有任何输出"的操作配一个后台报时线程。
#
# 为什么需要：加载模型（RoFormer / pyannote / whisper）期间脚本一行都不打印，
# 界面上的进度条也不动。用户盯着一个毫无变化的窗口，会以为程序卡死了。
# 有了这个，每 3 秒会有一行带秒数的输出流到界面，证明它还在干活。
#
# 用 daemon 线程 + Event：主流程一结束就 stop()，线程自己退出，不用 join。
# flush=True 是必须的 —— 否则输出会攒在缓冲区里，界面还是看不到。
# ─────────────────────────────────────────────────────────────
import threading as _hb_threading
import time as _hb_time


class Heartbeat:
    def __init__(self, label):
        self.label = label
        self._stop = _hb_threading.Event()
        self._t0 = _hb_time.time()
        self._thread = _hb_threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(3):
            print(f"  [等待] {self.label} 已用 {_hb_time.time() - self._t0:.0f} 秒 ...", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False

extensions = ("wav", "mp3", "aac")
pattern = os.path.join(config["path"]["audio"], "*.*")
files = [
    f for f in glob.glob(pattern)
    if f.lower().endswith(extensions)
]

if not files:
    # 第3步可能因为「无语音 / 短音频」跳过了这个文件，没有产出 1audio。
    # 第3步真失败会以非 0 退出码中止流水线（见 b_分离人声.py 末尾），
    # 所以能走到这里就说明是「跳过」而不是失败 —— 正常结束，继续下一个。
    print("[跳过] 没有可识别的音频（第3步已跳过该文件），流水线继续")
    sys.exit(0)

import re as _re


# ─────────────────────────────────────────────────────────────
# 字幕显示时长修正（v1.7.1）
#
# 症状：成片字幕里，一句话说完后字幕还一直挂着，直到下一句出现才换。
#   实测真实产物：一条 "好的 拜托了" 挂了 220 秒，一条
#   "好厉害 接吻的声音好厉害 想要更多" 挂了 143 秒；96% 的字幕
#   end 恰好等于下一条的 start。
#
# 根因（不是识别质量问题）：
#   VAD 前置之后，Whisper 拿到的是【去掉静音的压缩音频】。它先在
#   压缩坐标里解码，再用 restore_speech_timestamps 把时间戳映射回
#   原始时间轴 —— 映射公式是 total_silence_before + t，于是 segment
#   跨越的【段间静音】全被算进了这条字幕的显示时长。
#   更糟的是 Whisper 常把 segment.end 推到 30 秒解码窗口的末尾：
#   语音占比只有 15%~40% 时，压缩音频的 30 秒 = 原音频 75~220 秒。
#   两者叠加，就出现了"一句话挂几分钟"。
#
# 修法：写盘前给每条字幕算一个最长合理时长 ——
#   · high = 0.90 秒/字 + 0.8 秒 —— 慢语速上限（实测 p90 = 0.89 秒/字，
#     取这个值是为了【不误杀正常长句】）
#   · low  = 0.30 秒/字 + 0.6 秒 —— 正常语速保底，避免 VAD 段太短时
#     字幕闪一下就没
#   · 该字幕 start 所在 VAD 段的结尾 —— 真实语音边界，跨静音就截这里
# 最后再夹到"下一条字幕 start"之前，保证不重叠。
# ─────────────────────────────────────────────────────────────
def _speech_tail(start, vad_subs, gap=0.9, max_span=20.0):
    """返回「从 start 这一口气说到哪」：start 所在段起，向后把间隔
    <= gap 的段连起来，取最后一个段末。

    ASMR 里一口气说好几句、中间几乎不换气时，VAD 会切成很多小段。
    只看 start 所在的那一段，就会把「话还没说完」的地方当成句尾。
    v1.24.2 实测：191 条字幕里 29 条（15.2%）的 end 早于本条覆盖范围
    内最后一个语音段末，最多早 4.17 秒 —— 用户听到的就是这句抱怨。
    """
    idx = None
    for i, seg in enumerate(vad_subs):
        if seg[0] - 0.25 <= start <= seg[1] + 0.25:
            idx = i
            break
    if idx is None:
        return None
    tail = vad_subs[idx][1]
    for seg in vad_subs[idx + 1:]:
        va, vb = seg[0], seg[1]
        if va - tail <= gap and vb - start <= max_span:
            tail = vb
        else:
            break
    return tail


def _build_clip_timestamps(vad_subs, gap_merge=1.2, max_len=15.0, short_pad=0.0):
    """把 VAD 段合并成「解码窗口」列表，返回 [(start, end), ...]。

    ★ v1.28.8 核心修复：一条字幕横跨几十秒的根因和解法。

    根因（在 faster_whisper\vad.py 的 collect_chunks 里）：
      原来走的是 vad_filter=True + patch 注入 VAD 段。collect_chunks 会用
      max_duration=chunk_length（30 秒）把【多个 VAD 段拼进同一个解码窗口】，
      段间静音被整段去掉 —— 解码器看到的是「连续语流」，于是输出一条
      横跨 30 秒的 segment；restore_speech_timestamps 再把首末词映射回
      原始时间轴，字幕就跟着横跨 30 秒，中间的静音全被算进显示时长。

    实测（02 号文件 980s，注入项目 VAD 后 101 条）：
      最长一条 34.5s，>15s 有 7 条；词级时间戳 25.6% 是零宽度、
      8.5% 明显不合理（1 个字占 1.9 秒），说明词时间是被摊平出来的假数据。

    解法：改用 clip_timestamps 传扁平时间点列表。这条路径
    （transcribe.py:426-451）【不经过 collect_chunks】，每个 clip 独立成
    一个解码窗口，所以一条 segment 不可能横跨多个 VAD 段。

    参数：
      gap_merge — 相邻 VAD 段落间隔小于它才合并（同一口气/一口气说几句，
                  VAD 会切碎，合并回去才不会把一句话拆成两条字幕）
      max_len   — 单个窗口最长秒数。超过就断开，保证一条字幕不会太长。
                  实测 15 秒时最长条 10.8s、无 >15s 条目。
      short_pad — ★ v1.28.9：默认 0.0（不补边）。留作实验开关。

    ★ v1.28.9 关于 short_pad 的结论：**实测证明补边有害，默认关闭。**
      最初的想法是：03 号文件 78.31s 处 VAD 只判 0.39 秒、前后各是长静音，
      单独解码时 whisper 会顺着 initial_prompt 编出 92 个字。于是想「给它
      补点上下文，让它听到前后在说什么」。
      拿 0.0 / 0.3 / 0.5 / 1.0 四档实测（同一文件、同参数）：

        short_pad=0.0  幻觉 0 条   38.56s ->「结束」      78.31s ->「谢谢你 社长」
        short_pad=0.3  幻觉 0 条   38.26s ->「诶」
        short_pad=0.5  幻觉 0 条   38.06s ->「变态」
        short_pad=1.0  幻觉 2 条   37.56s ->「肉棒大人 射出来 射出来…」17字/0.37s
                                   151.55s ->「请把精液射给我吧」8字/0.39s

      结论：补边非但没压住幻觉，反而**把幻觉从 0 条变成 2 条** ——
      补进来的邻居语音和静音让解码器听到更多上下文，更容易顺着成人向
      prompt 续写；同时把原本干净的短句（「结束」「呵呵」）也搅乱了。
      所以默认 short_pad=0.0，真正的防线是下面的幻觉拦截
      （按 VAD 真实语音时长算语速，离谱就丢）。
      这个参数保留着，方便以后有人想再试别的取值。
    """
    if not vad_subs:
        return []

    clips = []
    cur_a, cur_b = float(vad_subs[0][0]), float(vad_subs[0][1])
    for seg in vad_subs[1:]:
        a, b = float(seg[0]), float(seg[1])
        # ★ 注意：先判断「合并后是否超长」，但单个 VAD 段本身就可能 > max_len
        #   （实测最长的一个 VAD 段 26.9 秒）。那种情况下不能因为「已经超了」
        #   就把它和后面的段拼在一起 —— 单独成窗口即可，让它自己超长。
        merged_len = max(b, cur_b) - cur_a
        cur_len = cur_b - cur_a
        if a - cur_b <= gap_merge and merged_len <= max_len:
            cur_b = max(cur_b, b)
        else:
            clips.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    clips.append((cur_a, cur_b))

    # ★ v1.28.9：孤立极短窗口补上下文（原地扩边，不新增窗口）。
    if short_pad > 0:
        out = []
        for i, (a, b) in enumerate(clips):
            d = b - a
            if d >= 0.8:
                out.append((a, b))
                continue
            prev_b = clips[i - 1][1] if i > 0 else 0.0
            next_a = clips[i + 1][0] if i + 1 < len(clips) else float("inf")
            # 只补到不碰上邻居为止，避免把邻居的音频混进本窗口
            pa = max(prev_b + 0.05, a - short_pad)
            pb = min(next_a - 0.05, b + short_pad)
            out.append((pa, pb))
        clips = out

    return clips


def _cap_row_end(start, end, text, vad_subs):
    """给一条字幕算合理的 end；vad_subs 是原始音频上的 VAD 段。

    v1.24.2 改了两处：
      · 段末不再只看 start 所在那一段，而是用 _speech_tail 把同一口气的
        连续段连起来（一句话被 VAD 切碎时不至于提前收尾）
      · 原来「只缩不伸」（end 一律 min(end, cap)），导致 whisper 偏早的
        end 原封不动写盘；现在允许把 end 补到段末，单条最多补 1.2 秒，
        差值超过 2 秒的多半是 ASR 漏词/误检，保持原样不硬补。
    """
    n = len(text.replace(" ", ""))
    low = 0.30 * n + 0.6
    # v1.28.2：high 从 0.90n+0.8 放宽到 1.30n+1.8。
    #   原因：这里数的是【中文字数】，但真实语音时长由日语音节数决定。
    #   中文译文普遍比日文短（「姐姐你回来啦」6 字对应「お姉さんおかえり」10 音节），
    #   按中文字数估时长必然偏短 → 用户听到「话还没说完字幕就消失」。
    high = 1.30 * n + 1.8
    cap = start + high

    seg_end = _speech_tail(start, vad_subs)
    if seg_end is not None:
        # ★ v1.28.3 关键修正：真实语音边界优先。
        #   原来是 cap = min(cap, max(seg_end, start + low)) —— 把「按字数
        #   估出来的 high」当成天花板，即使 VAD 明确说这段语音还在继续，
        #   超过 high 的部分照样被切掉 → 「话没说完字幕就消失」。
        # ★ v1.28.7：不再用 VAD 段尾当唯一天花板。
        #   原来的 cap = min(max(seg_end, ...), start+20) 会出问题：whisper 的
        #   end 来自解码对齐，通常比 VAD 更贴近真实说话结束；而 VAD 对耳语 /
        #   气声 / ASMR 背景音不敏感，段尾常早于实际收尾 → 字幕在 whisper
        #   认为还在说的位置就被切掉，用户听到「话没说完字幕就消失」。
        #   现在：取 VAD 段尾、字数下限、whisper 自己的 end 三者最大，但整体
        #   再压一道「按字数算的宽松上限」= start + max(6, 0.75n + 6)：
        #   3 个字最多 9 秒、23 个字最多 23 秒 —— 既有足够余量，又不会出现
        #   「很不错」这种三个字挂 30 秒的情况。
        _soft = start + max(6.0, 0.75 * n + 6.0)
        cap = min(max(seg_end, start + low, min(end, start + 30.0)), _soft)
        if end < seg_end:
            # ★ v1.28.4：补长上限从 2.5 秒放宽到 5.0 秒 —— ASMR 里 whisper
            #   经常把一句长耳语的 end 判在句中，VAD 才是真实结束点。
            end = max(end, min(seg_end, end + 5.0))

    return min(end, cap)


audio_tasks = []
for file_str in files:
    file_path = Path(file_str)
    raw_stem = file_path.stem

    # ★ v1.7：VAD 现在跑在【原始音频】上，所以它的产物名是原始 stem
    #   （例：test60s.srt），而 1audio 里的分离结果叫
    #   test60s_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav。
    #   要把那个后缀剥掉才能对上 VAD 的文件名。
    #   不剥的话会提示「跳过：找不到 VAD 字幕文件」然后整批什么都不做，
    #   而退出码还是 0 —— 实测踩过这个坑。
    _m = _re.match(r"^(.+?)_\(Vocals\)", raw_stem)
    basename = _m.group(1) if _m else raw_stem

    vad_log_path = os.path.join(config["path"]["vad"], f"{basename}.srt")
    # ASR 输出仍用完整名字（下游 finalize 靠它找视频）
    asr_log_path = os.path.join(config["path"]["asr"], f"{raw_stem}.srt")

    if not os.path.exists(vad_log_path):
        print(f"跳过：找不到 VAD 字幕文件 {vad_log_path}")
        continue
    audio_tasks.append((str(file_path), vad_log_path, asr_log_path))

print('设备:', config["device"], '类型:', config["compute_type"])


# 2. 定义补丁函数（用于注入你之前 VAD 得到的时间轴）
def get_custom_vad_patch(audio, vad_parameters=None):
    if hasattr(faster_whisper.vad, "_custom_segments"):
        segment_count = len(faster_whisper.vad._custom_segments)
        print(f"  [Patch Success] 注入来自 SRT 的 {segment_count} 个片段")
        return faster_whisper.vad._custom_segments
    print("  [Patch Warning] 未发现预设片段数据！")
    return []


# ★ v1.27.4 / v1.29：加载模型前先确认「提交内存」够。
#   实测（26:55 素材，RTX 5070，float16）：ASR 阶段
#     WorkingSet 峰值 3.55 GB，Commit 峰值 5.26 GB
#   用物理内存判会放行，然后在中途 malloc 失败（DefaultCPUAllocator:
#   not enough memory），表现为 0xC0000409 或退出码 1 ——
#   用户看到的是「第4步失败」，但真正原因是页文件/提交额度不够。
def _memory_status():
    """返回 (可用物理, 还可提交) 字节数，读不到返回 (0, 0)。"""
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys), int(st.ullAvailPageFile)
    except Exception:
        pass
    return 0, 0


_phys0, _commit0 = _memory_status()

# ─────────────────────────────────────────────────────────────
# ★ v1.29：内存不足时【自动降级】，不再直接退出。
#
#   背景（真实用户案例，2026-11-01）：
#     一台 RTX 3080 Ti Laptop + 32 GB 内存的机器，跑 27 分钟 / 284 个 VAD 段
#     的素材，第 4 步开始推理后 14 秒崩溃，退出码 -1073740791（0xC0000409）。
#     进度条停在 0%，一条字幕都没出 —— 因为崩在推理刚开始的 malloc 上。
#
#   根因是「提交内存」不足：malloc 记账看的是【物理内存 + 页文件】这个池子，
#   不是任务管理器的「可用物理内存」。页文件设小了，可用物理还剩 15 GB 时
#   依然可能分配失败，而 ctranslate2 在 C++ 层分配失败会走 __fastfail，
#   直接杀进程，不会抛可捕获的异常。
#
#   v1.27.4 的做法是「检测到不足就报错退出」—— 用户被拦下，什么也做不了。
#   v1.29 改成三级降级：只要还没到「连 int8 都跑不动」的地步，就自动降精度
#   继续跑，并在日志里说明原因和代价。
#
#   为什么降 compute_type 能救：
#     float16    → Commit 峰值约 5.3 GB（基线）
#     int8_float16 → 权重减半，Commit 峰值约 2.5~3 GB，直接腰斩
#   代价是速度慢约 20%、个别词准确度略降 —— 但总比崩掉强。
# ─────────────────────────────────────────────────────────────
def _gib(n):
    return n / 1024 ** 3


# 各档位的提交额度需求（GB）。
#   float16       —— 实测 26:55 素材峰值 5.26 GB，留余量取 5.0
#   注解：门槛【略低于】峰值是有意的 —— 峰值是短时尖峰，卡在均值附近
#   既能拦住确定跑不动的机器，又不会把刚好够用的机器误拦。
_NEED_F16 = 5.0
_NEED_I8 = 2.8   # int8 大约减半后再留一点余量
# 连 int8 都拿不到这么多时，才真的没救（交给兼容模式）
_NEED_MIN = 1.8

_degraded = None   # None / "int8_float16"

if _commit0 and _commit0 < int(_NEED_F16 * 1024 ** 3):
    if _commit0 >= int(_NEED_I8 * 1024 ** 3):
        # 能跑 int8 —— 自动降级
        _degraded = "int8_float16"
        print(f"[内存] 提交额度不足：需要约 {_NEED_F16:.1f} GB，"
              f"当前只能再拿到 {_gib(_commit0):.2f} GB")
        print(f"[内存] 已自动降级为 int8_float16（权重减半，提交占用约 {_NEED_I8:.1f} GB）")
        print("[内存] 代价：识别速度慢约 20%，个别词准确度可能略降")
        print("[内存] 想用回原来的速度：此电脑→属性→高级系统设置→性能设置→高级→"
              "虚拟内存→改成「系统管理的大小」或手工 32768 MB，重启后再试")
    elif _commit0 >= int(_NEED_MIN * 1024 ** 3):
        # 连 int8 都很勉强 —— 还是试一把，但把风险说清楚
        _degraded = "int8_float16"
        print(f"[内存] 提交额度很紧张：需要约 {_NEED_F16:.1f} GB，"
              f"当前只有 {_gib(_commit0):.2f} GB")
        print(f"[内存] 仍将尝试 int8_float16（最低需求约 {_NEED_MIN:.1f} GB），但可能中途失败")
        print("[内存] 如果这一步还是崩，请调大虚拟内存，或改用兼容模式")
    else:
        # 真的没救：明确报错退出，别让用户白等
        print(f"[失败] 内存严重不足：语音识别最低需要约 {_NEED_MIN:.1f} GB 提交额度，"
              f"当前只能再拿到 {_gib(_commit0):.2f} GB")
        print("       这是「提交内存」不足 —— 虚拟内存（页文件）设得太小或被关闭了。")
        print("       注意：任务管理器里的「可用物理内存」看着够并不能说明问题，")
        print("             malloc 记账看的是「物理内存 + 页文件」的合计额度。")
        print("       修法：此电脑→属性→高级系统设置→性能设置→高级→虚拟内存→"
              "改成「系统管理的大小」或手工 32768 MB，重启后再试。")
        print("       也可以改用兼容模式（内存需求约 1~2 GB）。")
        sys.exit(1)
else:
    if _commit0:
        print(f"[内存] 提交额度 {_gib(_commit0):.1f} GB，满足语音识别需求")


# 3. 初始化 Whisper 模型
_compute = config["compute_type"]
if _degraded:
    _compute = _degraded
    print(f"[内存] 本次使用 compute_type = {_compute}")

with Heartbeat("加载语音识别模型"):
    model = WhisperModel(
        model_size_or_path=config["model"]["asr"],
        device=config["device"],
        compute_type=_compute,
        download_root=config["path"]["model"]
    )

# 4. 开始推理循环
emit("stage", "start", 4)
_total_tasks = len(audio_tasks)
for _task_idx, (audio_path, vad_log_path, asr_log_path) in enumerate(audio_tasks, 1):
    emit("file", _task_idx, _total_tasks, Path(audio_path).name)
    print(f"\n开始推理: {Path(audio_path).name}")

    # 按作品补充热词（hotwords.txt / hotwords_<stem>.txt）
    _stem = _re.sub(r"_\(Vocals\).*$", "", Path(audio_path).stem)
    _extra = _load_extra_hotwords(_stem)
    _hw = ",".join([w.strip() for w in _HOTWORDS.split(",") if w.strip()] + _extra)
    if _extra:
        print(f"  [热词] 按作品补充 {len(_extra)} 个：{','.join(_extra)}")

    # 加载 VAD 得到的片段
    vad_subs = read_srt(vad_log_path)
    custom_chunks = []
    sr = config["vad"]["sample_rate"]  # 16000

    for _a, _b, _t in vad_subs:
        custom_chunks.append({
            'start': int(_a * sr),
            'end': int(_b * sr)
        })

    # 临时挂载到 faster_whisper 模块上
    faster_whisper.vad._custom_segments = custom_chunks

    # ★ v1.28.8：改用 clip_timestamps，让每个 VAD 段（按 gap_merge/max_len
    #   合并后）各自成为一个解码窗口。
    #
    #   原来走 vad_filter=True + patch 注入 VAD 段，会经过 vad.py 的
    #   collect_chunks：它用 max_duration=chunk_length（30s）把【多个 VAD 段
    #   拼进同一个窗口】，段间静音被去掉 → 解码器看到连续语流 → 输出一条
    #   横跨 30 秒的 segment，中间的静音全被算进字幕时长。
    #   实测 02 号文件最长一条 34.5s、>15s 有 7 条，就是这个机制造成的。
    #
    #   clip_timestamps 走的是另一条路径（transcribe.py:426-451），
    #   【不经过 collect_chunks】，每个 clip 独立解码，所以一条 segment
    #   不可能横跨多个 VAD 段 —— 这才是「字幕提前消失」的根治。
    _clips = _build_clip_timestamps(vad_subs)
    _clip_flat = []
    for _ca, _cb in _clips:
        _clip_flat.extend([round(_ca, 3), round(_cb, 3)])
    _use_clip = bool(_clip_flat)
    if _clips:
        print(f"  [分段] {len(vad_subs)} 个 VAD 段 → {len(_clips)} 个解码窗口"
              f"（最长 {max(b - a for a, b in _clips):.1f}s）")

    # clip_timestamps 与 vad_filter 互斥：给了 clip_timestamps 后 vad_filter 被忽略。
    # 保留 patch 作为兜底 —— 老版本 faster-whisper 若不走 clip 分支仍能工作。
    with patch(
            "faster_whisper.transcribe.get_speech_timestamps",
            side_effect=get_custom_vad_patch
    ):
        segments, _ = model.transcribe(
            audio=audio_path,
            language="ja",
            task='translate',
            # 解码束宽。faster-whisper 默认 5，不传就是 5。
            # 用环境变量 ASMR_BEAM 临时试更小的值（2 / 1），默认仍是 5。
            beam_size=int(os.environ.get("ASMR_BEAM", "5")),
            # ★ v1.28.8：clip_timestamps 优先；为空时退回 vad_filter。
            clip_timestamps=_clip_flat if _clip_flat else None,
            vad_filter=True,  # clip_timestamps 生效时会被忽略，保留作兜底
            # ★ v1.17.1：关闭跨段上下文（原来是 True）。
            #
            #   拿 R4222 官方中文字幕（8 轨，529 句）逐轨对比实测：
            #     True ：字数覆盖 72%，短句大量被并进上一句，
            #            还出现 3 次「渚沙」（应「凪咲」）
            #     False：字数覆盖 85%，短句漏译明显减少，错译 0 次
            #   代价：一句话被拆成多条时没有前文可参考；实测句子仍通顺。
            condition_on_previous_text=False,
            initial_prompt=_INITIAL_PROMPT,
            hotwords=_hw,
            log_progress=True
        )

        raw_rows = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            _a, _b = float(segment.start), float(segment.end)

            # ★ v1.28.9：拦截「短窗口里的长幻觉」。
            #
            #   实测（03 号文件，78.31s）：VAD 只判出 0.39 秒语音（78.31~78.70），
            #   下一个语音段要到 91.88s 才开始，中间 13 秒是纯静音。
            #   whisper 却输出 78.31 -> 83.57、92 个字：
            #     「随时都可以射在里面哦 随时都可以 随时允许你射在里面
            #       随时都可以 随时都可以 把精液射在我的手手上 把蛋蛋汁涂满 …」
            #   这种重复排比是幻觉的典型特征 —— 真实语速根本容纳不下。
            #
            #   两道拦截：
            #     ① 时间夹到所属解码窗口内（向上留 0.6 秒缓冲，解码器确实会
            #        略微越过 clip 边界，实测正常条目超出 0.5~1.2 秒）。
            #     ② 字数超过「真实语音时长 × 语速上限」就丢弃。
            #
            #   ★ 注意②用的是【VAD 真实语音时长】，不是窗口时长。
            #   窗口在短段两侧补过上下文（short_pad），按窗口算会放宽上限，
            #   等于让幻觉又溜过去。这里取本条实际覆盖到的 VAD 语音总量。
            #   中文口语实测中位 3.5 字/秒，取 9.0 作为硬上限，
            #   正常条目碰不到，只拦「5 秒塞 92 字」这种。
            if _use_clip and _clip_flat:
                _win = None
                for _i in range(0, len(_clip_flat) - 1, 2):
                    if abs(_clip_flat[_i] - _a) < 0.4:
                        _win = (_clip_flat[_i], _clip_flat[_i + 1])
                        break
                if _win is not None:
                    # 本条覆盖的真实语音时长
                    _speech_len = 0.0
                    for _vs in vad_subs:
                        _va, _vb = float(_vs[0]), float(_vs[1])
                        _ov = min(_b, _vb) - max(_a, _va)
                        if _ov > 0:
                            _speech_len += _ov
                    if _speech_len <= 0:
                        _speech_len = _win[1] - _win[0]
                    _n = len(_re.sub(r"[^\u4e00-\u9fff]", "", text))
                    # ★ 阈值要同时满足「绝对字数多」和「语速离谱」，缺一不可：
                    #   · 只卡语速会误杀真短句 —— 实测 38.56s 处 VAD 判 0.37 秒，
                    #     whisper 出「呵呵」2 个字（5.4 字/秒，完全正常），
                    #     但按 9.0 字/秒的硬线算，「呵呵」是 5.4<9 不该拦；
                    #     17 个字的幻觉才是 45.8 字/秒，由「≥8 字」这一条一起卡住。
                    #   · 只卡字数会误杀长句 —— 15 秒说 30 个字很正常。
                    #   所以：字数 ≥ 8 且 语速 > 12 字/秒，才判定为幻觉。
                    #   真实的短促叫喊（「再来一次 再来一次」之类）字数也够，
                    #   但它的语速不会超过 12 —— 因为 VAD 判定的语音时长会跟着变长。
                    if _n >= 8 and _n / max(0.25, _speech_len) > 12.0:
                        print(f"  [幻觉拦截] {_a:.2f}->{_b:.2f} 真实语音 {_speech_len:.2f}s "
                              f"却识别出 {_n} 字（{_n/_speech_len:.1f} 字/秒），"
                              f"丢弃：{text[:40]}…")
                        continue

                    # ★ v1.29：短条过滤 ——「完」「结束」「再见」这类假字幕。
                    #
                    #   实测（04 号文件）：一条「完」显示 1.00 秒，但 pyannote VAD
                    #   判定那里真实语音只有 0.43 秒（43% 覆盖）。同文件里 22 条
                    #   短于 1.2 秒的字幕，真实语音普遍只有 0.3~0.5 秒：
                    #     2.39s  0.42s →「结束」
                    #    81.08s  0.34s →「结束」
                    #   165.64s  0.32s →「结束」
                    #   197.86s  0.35s →「再见」
                    #   201.76s  0.43s →「完」
                    #   856.22s  0.32s →「结束」
                    #
                    #   0.3~0.4 秒的音频装不下一个完整词，whisper 在这么短的窗口里
                    #   把 ASMR 的唇齿音、气声、轻舔声这些【音效】解读成了音节。
                    #   这类字幕纯属噪声，会打断观看节奏。
                    #
                    #   ★ 判据必须两头卡，否则会误杀真实的短促应答：
                    #     · 真实语音 < 0.45 秒（物理上装不下一个词）
                    #     · 且输出 ≥ 2 个中文字（排除「诶」「啊」这类单字语气词 ——
                    #       它们本来就短，是真实内容）
                    #   实测对照组：452.29s 那条「我这边会更加卖力...」真实语音
                    #   0.95 秒，不受影响；「诶？」「好」这类单字也都能留下。
                    #
                    # ★ v1.29 补充：单字的「完」也要拦，但不能一刀切按单字拦。
                    #   实测 04 号文件里最后一轮仍有几条漏网，它们的真实语音时长
                    #   和真实语气词几乎一样，光靠时长区分不了：
                    #     165.64s  VAD 0.32s →「完」     ← 该拦
                    #     201.76s  VAD 0.44s →「完」     ← 该拦
                    #     103.37s  VAD 1.35s →「完」     ← 真实短句，保留
                    #       5.84s  VAD 0.40s →「哇」     ← 真实语气词，保留
                    #      42.76s  VAD 0.51s →「呵呵」   ← 真实语气词，保留
                    #   区别在【词义】：完成/结束类词在 ASMR 里多半是「終わり」
                    #   之类的音效被误读；「哇/呵呵/唉/好/诶」是真的短促反应。
                    #   所以补一张完成类词表，只在时长也极短时才按它拦。
                    END_WORDS = (
                        "完", "完毕", "结束", "完了", "完蛋", "再见", "拜拜",
                        "完成", "到此为止", "时间到",
                    )
                    _is_end_word = text.strip().rstrip("。！？.!?").strip() in END_WORDS
                    if _speech_len < 0.45 and (_n >= 2 or _is_end_word):
                        print(f"  [短条过滤] {_a:.2f}->{_b:.2f} 真实语音仅 {_speech_len:.2f}s，"
                              f"疑似音效误识，丢弃：{text}")
                        continue
                    _b = min(_b, _win[1] + 0.6)

            raw_rows.append((_a, _b, text))
            print(f"[{_a:.2f}s -> {_b:.2f}s]: {text}")
            # 结构化事件：和上面那行【同一份数据】，只是给 Rust 侧一个
            # 不依赖文本格式的入口（正则继续兜底，两者等价）。
            emit("sub", f"{_a:.2f}", f"{_b:.2f}", text)

        # ★ v1.28.8：clip_timestamps 模式下，segment 的 start/end 就是该解码
        #   窗口的真实语音区间，【不需要也不该】再做事后收缩/补长。
        #
        #   为什么必须跳过：_cap_row_end 里有一条「end 偏早就补到 VAD 段尾，
        #   最多补 5 秒」的规则。那是为【旧方案】准备的 —— 旧方案把多个 VAD 段
        #   拼进一个 30 秒窗口，segment.end 被映射得很不可靠，才需要拿 VAD 段尾
        #   来兜。换成 clip 方案后 segment.end 已经准了，这条规则反而把每条
        #   的 end 往后撑 5 秒。
        #
        #   实测（02 号文件）：跳过前 121 条里有 26 处重叠、其中 23 处重叠 >2 秒，
        #   而重叠量恰好集中在 5.00 秒 —— 正是这里的 5.0 秒补长造成的。
        #   边界情况（起点重复、极短窗口）只做最小修正，不做时长猜测。
        asr_rows = []
        for i, (a, b, t) in enumerate(raw_rows):
            if _use_clip:
                nb = b
                # 解码窗口之间天然不重叠；这里只保证时长下限，避免一闪而过。
                if i + 1 < len(raw_rows):
                    nxt = raw_rows[i + 1][0]
                    if nxt > a:
                        # 允许最多重叠 0.6 秒（后一条覆盖前一条），与 srt.rs 一致
                        nb = min(nb, nxt + 0.6)
                        nb = max(nb, min(a + 1.2, nxt))
                if nb <= a:
                    nb = a + 0.8
                asr_rows.append((a, nb, t))
                continue
            nb = _cap_row_end(a, b, t, vad_subs)
            if i + 1 < len(raw_rows):
                nxt = raw_rows[i + 1][0]
                if nxt > a:
                    # ★ v1.28.3：原来这里无条件 nb = min(nb, nxt) —— 只要
                    #   whisper 把下一句的 start 判早了，上一句就被硬切到那里，
                    #   即使 VAD 明确说这句话还在说 → 用户听到「话没说完字幕
                    #   就消失」。现在改成：只有当 VAD 也认为语音在 nxt 之前
                    #   就结束时才截断；否则允许本条多挂一会儿（最多盖到真实
                    #   语音尾），重叠部分由播放器用后一条覆盖，不影响阅读。
                    _tail = _speech_tail(a, vad_subs)
                    _limit = nxt if (_tail is None or _tail <= nxt + 0.15) else min(_tail, a + 20.0)
                    nb = min(nb, _limit)
                    nb = max(nb, min(a + 1.2, nxt))   # ★ v1.28.4：最少显示 1.2 秒（原 0.5）
            if nb <= a:
                nb = a + 0.8
            asr_rows.append((a, nb, t))

        _shrunk = sum(1 for r0, r1 in zip(raw_rows, asr_rows) if r1[1] < r0[1] - 0.01)
        _before = sum(b - a for a, b, _t in raw_rows)
        _after = sum(b - a for a, b, _t in asr_rows)
        if _use_clip:
            print(f"  [时间轴] 解码窗口模式：{len(asr_rows)} 条，总时长 {_after:.0f}s"
                  f"（未做事后收缩）")
        else:
            print(f"  [时间轴] 收缩 {_shrunk}/{len(asr_rows)} 条字幕的结束时间"
                  f"（总时长 {_before:.0f}s → {_after:.0f}s）")

        # 一次性写盘（不要放在循环里逐条写）
        write_srt(asr_log_path, asr_rows)

    print(f"处理完成，保存至: {asr_log_path}")

emit("stage", "done", 4)
print("\n所有 ASR 任务处理完毕！")

# ★ v1.25.1：主动跳过 CUDA 析构，别让进程以 0xC0000409 退出。
#
#   实测（RTX 3060 Laptop 6GB，1 小时音频 / 410 段字幕）：字幕已经全部写完、
#   "处理完成，保存至 ..." 和 "所有 ASR 任务处理完毕！" 都打印了，然后
#   ctranslate2 在进程退出时销毁 CUDA context / cuBLAS handle 崩掉，退出码
#   -1073740791（0xC0000409 = STATUS_STACK_BUFFER_OVERRUN）。
#   业务早就做完了，崩的只是清理阶段 —— 但上游只按退出码判成败，会把这轮
#   判成失败，后面的字幕整理和"搬到视频旁边"整段跳过。
#
#   os._exit 不跑 atexit / finally、不析构 C++ 对象。对马上要结束的进程没有
#   代价（显存和句柄由内核回收）。必须先手动 flush：os._exit 不会替你刷缓冲，
#   否则最后几行日志会丢。
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
