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

    # ★ 随包分发的 ffmpeg（env\ffmpeg\ffmpeg.exe）优先于系统 PATH 上的。
    #
    #   发布包现在自带 ffmpeg（静态链接的单个 exe，211 MB），
    #   这样目标机器上【没装过 ffmpeg 也能跑】，不需要联网下载。
    #   放在 torch\lib 之后加：两者互不干扰，先加谁都可以。
    _ff_dir = _env_root / "ffmpeg"
    if (_ff_dir / "ffmpeg.exe").is_file():
        _dll_os.environ["PATH"] = str(_ff_dir) + _dll_os.pathsep + _dll_os.environ.get("PATH", "")
except Exception:
    pass

import os
import glob
import subprocess
import sys
from pathlib import Path
import ffmpeg_downloader
import torch
from srt_util import write_srt
# ★ 优化：ffmpeg 装过就别再装。
#   原来每次都跑 `python -m ffmpeg_downloader install`，实测 0.56 秒/次，
#   三个脚本加起来 1.7 秒纯属白跑（而且它会联网检查更新）。
#   改成先看 ffmpeg 在不在 PATH 上，不在才装。
ffmpeg_downloader.add_path()
import shutil as _shutil
if _shutil.which("ffmpeg") is None:
    subprocess.run(
        [sys.executable, "-m", "ffmpeg_downloader", "install", "8.0@full-shared"],
        input="y\n",
        text=True
    )
    ffmpeg_downloader.add_path()
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from settings import config
from events import emit


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

# ─────────────────────────────────────────────────────────────
# 【注意：这一版的输入是 0pre 的原始音频，不是 1audio 的人声】
#
# 顺序调整（v1.7）：
#   原来是  提取 → 分离 → VAD → 识别
#   现在改成 提取 → VAD → 分离 → 识别
#
# 为什么：
#   实测 L 盘一个 30 分钟音声，VAD 判定语音只占 41.8%；
#   另一个 12 分钟视频只占 15%。也就是说【大量时间是静音/喘息/环境音】。
#   而人声分离要占全程 62% 的时间，且耗时和音频长度严格成正比。
#
#   先做 VAD（在原始音频上就能做，不需要人声），
#   再只分离 VAD 圈出来的那些段：
#     30 分钟音频：分离 121.78s → 51.84s（省 57%）
#     12 分钟视频：分离  65s  →  9.45s（省 85%）
#   实测整条链路 145s → 58.7s（省 60%），字幕质量不变（零漏检）。
# ─────────────────────────────────────────────────────────────
extensions = ("wav", "mp3", "flac")
pattern = os.path.join(config["path"]["pre"], "*.*")
audio_files = [
    f for f in glob.glob(pattern)
    if f.lower().endswith(extensions)
]

if not audio_files:
    print(f"在 {config['path']['pre']} 中没有找到可处理的音频文件")
    # ★ 非 0 退出码：没有输入文件就等于这一步没做成，不能让界面以为成功了
    sys.exit(1)

print('设备:', config["device"])

# 2. 初始化 VAD 模型
vad = Model.from_pretrained(
    checkpoint=config["model"]["vad"],
    cache_dir=config["path"]["model"]
)
vad.to(torch.device(config["device"]))

with Heartbeat("加载 VAD 模型"):
    vad_pipeline = VoiceActivityDetection(segmentation=vad)
# 注意：修正了之前代码中的变量名错误 config.vad_min.duration_off -> config["vad"]["min_duration_off"]
vad_pipeline.instantiate({
    "min_duration_on": config["vad"]["min_duration_on"],
    "min_duration_off": config["vad"]["min_duration_off"],
})

# 3. 开始循环处理
#
# ★ 单个文件失败【不再中止整批】
#
#   踩过的坑（2026-10-17）：26 个视频跑了一整夜，第3步处理到第 3 个时
#   遇到一个损坏的 wav（上一批磁盘写满导致的半截文件），
#   torchcodec 抛 RuntimeError 直接把整个脚本干掉了 ——
#   后面 23 个明明正常的文件一个都没处理。
#
#   批量任务的正确做法：单个失败记下来、跳过、继续下一个，
#   最后汇总报告。真正致命的问题（比如模型没加载）在循环外就会失败，
#   不需要靠循环内抛异常来兜。
_ok_files = []
_fail_files = []

emit("stage", "start", 2)
_total = len(audio_files)
for _idx, audio_path in enumerate(audio_files, 1):
    emit("file", _idx, _total, os.path.basename(audio_path))
    print(f"\n处理音频: {audio_path}")

    file_obj = Path(audio_path)
    basename = file_obj.stem
    vad_log_path = os.path.join(config["path"]["vad"], f"{basename}.srt")

    try:
        # 4. 执行 VAD 识别
        with Heartbeat(f"人声检测 {basename}"):
            vad_result = vad_pipeline(audio_path)

        # 5. 生成 SRT
        # 用项目自带的 srt_util 写时间轴
        _rows = []
        for idx, (segment, _, _) in enumerate(vad_result.itertracks(yield_label=True), start=1):
            _rows.append((float(segment.start), float(segment.end), f"Speech_{idx}"))
        write_srt(vad_log_path, _rows)

        # 额外写一份 JSON —— 下一版 b_分离人声.py 要按这些段做「只分离语音」，
        # 直接从 srt 反解析时间戳太脆，存一份结构化的更稳。
        import json as _json
        _segs = [(float(a), float(b)) for a, b in
                 [(s.start, s.end) for s, _, _ in vad_result.itertracks(yield_label=True)]]
        _json_path = os.path.join(config["path"]["vad"], f"{basename}.vad.json")
        with open(_json_path, "w", encoding="utf-8") as _f:
            _json.dump({"segments": _segs}, _f, ensure_ascii=False)
        _total = sum(b - a for a, b in _segs)
        print(f"VAD段已导出: {basename}.vad.json（{len(_segs)} 段 / {_total:.1f} 秒）")
        if not _segs:
            print("  [注意] 没有检测到人声（可能是赠送的 BGM / 纯静音），第3步会跳过")

        print(f"VAD记录写入完成: {vad_log_path}")
        _ok_files.append(basename)

    except Exception as _e:
        print(f"  [跳过] {basename} 处理失败: {type(_e).__name__}: {_e}")
        print(f"         可能原因：音频文件损坏（磁盘满时写一半）、或格式不支持")
        _fail_files.append((basename, f"{type(_e).__name__}: {str(_e)[:120]}"))
        continue

# 6. 汇总
emit("stage", "done", 2)
print("\n" + "=" * 60)
print(f"VAD 处理完毕：成功 {len(_ok_files)} 个，失败 {len(_fail_files)} 个")
if _fail_files:
    print("-" * 60)
    for _bn, _why in _fail_files:
        print(f"  ✗ {_bn}")
        print(f"      {_why}")
    print("-" * 60)
    print("  这些文件的 srt 没有生成，第4步会跳过它们。")
    print("  想重跑的话：检查源视频是否完整，或删除后重新处理。")
print("=" * 60)

if not _ok_files:
    print("[失败] 没有任何音频通过 VAD")
    sys.exit(1)