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
_HOTWORDS = "肉棒,小穴,舒服,去了,高潮,舔,精液,射精,口交,乳交,秘书,社长"


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
def _speech_tail(start, vad_subs, gap=0.5, max_span=15.0):
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
    high = 0.90 * n + 0.8
    cap = start + high

    seg_end = _speech_tail(start, vad_subs)
    if seg_end is not None:
        # 段尾是真实语音结束；但不早于"正常语速该有的时长"
        cap = min(cap, max(seg_end, start + low))
        if end < seg_end and seg_end - end <= 2.0:
            # 补到段末，单条最多补 1.2 秒（千问评审建议 0.8；取 1.2 是
            # 为了覆盖诊断里 0.87~1.59 秒那批的主要分布）
            end = max(end, min(seg_end, end + 1.2))

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


# 3. 初始化 Whisper 模型
with Heartbeat("加载语音识别模型"):
    model = WhisperModel(
        model_size_or_path=config["model"]["asr"],
        device=config["device"],
        compute_type=config["compute_type"],
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

    # 使用 patch 强制 Whisper 使用我们的时间轴，而不是让它重新检测 VAD
    with patch(
            "faster_whisper.transcribe.get_speech_timestamps",
            side_effect=get_custom_vad_patch
    ):
        segments, _ = model.transcribe(
            audio=audio_path,
            language="ja",
            task='translate',
            vad_filter=True,  # 必须开启，Patch 才会生效
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
            raw_rows.append((float(segment.start), float(segment.end), text))
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {text}")
            # 结构化事件：和上面那行【同一份数据】，只是给 Rust 侧一个
            # 不依赖文本格式的入口（正则继续兜底，两者等价）。
            emit("sub", f"{segment.start:.2f}", f"{segment.end:.2f}", text)

        # ★ v1.7.1：写盘前收缩每条的 end。
        #   直接写 segment.end 会出现"一句话挂两百秒"，原因见 _cap_row_end 上方。
        asr_rows = []
        for i, (a, b, t) in enumerate(raw_rows):
            nb = _cap_row_end(a, b, t, vad_subs)
            if i + 1 < len(raw_rows):
                nxt = raw_rows[i + 1][0]
                if nxt > a:
                    nb = min(nb, nxt)                 # 不压到下一句
                    nb = max(nb, min(a + 0.5, nxt))   # 最少显示 0.5 秒
            if nb <= a:
                nb = a + 0.3
            asr_rows.append((a, nb, t))

        _shrunk = sum(1 for r0, r1 in zip(raw_rows, asr_rows) if r1[1] < r0[1] - 0.01)
        _before = sum(b - a for a, b, _t in raw_rows)
        _after = sum(b - a for a, b, _t in asr_rows)
        print(f"  [时间轴] 收缩 {_shrunk}/{len(asr_rows)} 条字幕的结束时间"
              f"（总时长 {_before:.0f}s → {_after:.0f}s）")

        # 一次性写盘（不要放在循环里逐条写）
        write_srt(asr_log_path, asr_rows)

    print(f"处理完成，保存至: {asr_log_path}")

emit("stage", "done", 4)
print("\n所有 ASR 任务处理完毕！")