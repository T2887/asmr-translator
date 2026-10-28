"""v1.7：只分离 VAD 圈出的语音段。

输入：0pre\\*.flac（原始音频）+ 2vad\\*.vad.json（VAD 段）
输出：1audio\\<name>_(Vocals)_...wav（完整长度，静音处补零）

流程（每个文件）：
  ① 读 VAD 段
  ② 按段拼接原始音频 → 临时 concat.flac
  ③ 分离 concat.flac → 人声
  ④ 把人声按段填回原长度（静音补零）
  ⑤ 删除临时文件

为什么这样能快：
  实测语音只占音频的 15%~42%，而分离耗时和音频长度严格成正比。
  12 分钟视频：分离 65s → 9.45s（省 85%）
  30 分钟音声：分离 122s → 52s（省 57%）

为什么填回原长度：
  Whisper 需要完整的上下文，而且下游的分段识别依赖原始时间轴。
  填回去之后时间轴天然正确，识别环节不用做任何映射。
"""
import glob
import json
import os
import sys
import tempfile
import time

# ★ 内置 ffmpeg 优先（不能靠联网下载，也不能靠系统 PATH）
_ff_dir = os.path.dirname(sys.executable)
_ff_dir = os.path.join(_ff_dir, "ffmpeg")
if os.path.isfile(os.path.join(_ff_dir, "ffmpeg.exe")):
    os.environ["PATH"] = _ff_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import soundfile as sf
import torch
from audio_separator.separator import Separator

from settings import config
from events import emit

# ── 心跳：让界面能看到"还在跑" ──
import threading as _ht
import time as _hb_time


class Heartbeat:
    def __init__(self, label):
        self.label = label
        self._stop = _ht.Event()
        self._t0 = _hb_time.time()
        self._thread = _ht.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(3):
            print(f"  [等待] {self.label} 已用 {_hb_time.time() - self._t0:.0f} 秒 ...", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False


# ── 磁盘检查（保留原有防御）──
import shutil as _shutil_disk

_GB = 1024 ** 3
_PER_FILE_GB = 2.7
_SAFETY = 1.3
# 容差：剩余空间和预估需求只差一点点时不判死。
# 实测踩过 —— 界面打印「需要 3.5 GB，当前可用 3.5 GB」却当场中止，
# 因为底下是 3.51 对 3.48，只差 30 MB；四舍五入到一位小数后用户
# 根本看不出差别，只会觉得程序无缘无故不让跑。
# 0.3 GB 的容差足够吸收「刚好卡在临界点」的情况，而真正的空间不足
# （差几百 MB 以上）照样拦得住。
_TOLERANCE_GB = 0.3

try:
    _n_files = len([
        f for f in os.listdir(config["path"]["pre"])
        if f.lower().endswith((".flac", ".wav", ".mp3"))
    ]) or 1
    _need_gb = _n_files * _PER_FILE_GB * _SAFETY
    _work_root = os.path.abspath(".")
    _free_gb = _shutil_disk.disk_usage(_work_root).free / _GB
    print(f"磁盘检查: 待处理 {_n_files} 个，预计需要 {_need_gb:.2f} GB，当前可用 {_free_gb:.2f} GB")
    if _free_gb + _TOLERANCE_GB < _need_gb:
        print("=" * 60)
        print("[中止] 磁盘空间不足")
        print(f"       需要约 {_need_gb:.2f} GB，可用 {_free_gb:.2f} GB"
              f"（还差 {_need_gb - _free_gb:.2f} GB）")
        print("=" * 60)
        sys.exit(1)
except SystemExit:
    raise
except Exception as _e:
    print(f"  [警告] 磁盘检查失败（{_e}），继续执行")


# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════
def load_vad_segments(flac_path):
    """读同名的 .vad.json。

    返回值区分三种情况（很重要，别再合并成 None）：
      None   → 没有 VAD 文件（漏跑第2步？）→ 退回整段分离兜底
      []     → VAD 文件存在但 0 段语音（纯 BGM / 全静音）→ 直接跳过
      非空   → 正常语音段
    """
    base = os.path.splitext(os.path.basename(flac_path))[0]
    p = os.path.join(config["path"]["vad"], base + ".vad.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        segs = [(float(a), float(b)) for a, b in d.get("segments", [])]
        return segs  # 空列表 = 确认没有人声
    except Exception as e:
        print(f"  [警告] 读 VAD 段失败（{e}），退回整段分离")
        return None


def read_audio(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data, sr


def write_flac(path, data, sr):
    sf.write(path, data, sr, subtype="PCM_24")


def concat_by_segments(data, sr, segs):
    """按段拼接。返回 (拼接后的数组, 每段的实际长度样本数)"""
    chunks = []
    lens = []
    total = len(data)
    for s, e in segs:
        a = int(s * sr)
        b = min(int(e * sr), total)
        if b <= a:
            lens.append(0)
            continue
        chunk = data[a:b]
        chunks.append(chunk)
        lens.append(len(chunk))
    if not chunks:
        return None, []
    return np.concatenate(chunks, axis=0), lens


def expand_with_sr(vocals, segs, lens, sr, total_len, ch):
    out = np.zeros((total_len, ch), dtype="float32")
    pos = 0
    for (s, e), n in zip(segs, lens):
        if n <= 0:
            continue
        a = int(s * sr)
        b = min(a + n, total_len)
        n2 = min(b - a, len(vocals) - pos)
        if n2 <= 0:
            break
        chunk = vocals[pos:pos + n2]
        if chunk.shape[1] != ch:
            if chunk.shape[1] == 1:
                chunk = np.repeat(chunk, ch, axis=1)
            else:
                chunk = chunk[:, :ch]
        out[a:a + n2] = chunk
        pos += n2
    return out


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
extensions = (".flac", ".wav", ".mp3")
pattern = os.path.join(config["path"]["pre"], "*.*")
audio_files = [f for f in sorted(glob.glob(pattern)) if f.lower().endswith(extensions)]

if not audio_files:
    print(f"在 {config['path']['pre']} 中没有找到音频文件")
    sys.exit(1)

print(f"待分离: {len(audio_files)} 个文件")
print(f"正在加载模型: {config['model']['sep']}")

separator = Separator(
    model_file_dir=config["path"]["model"],
    # ★ 输出到临时目录，不能直接写 1audio。
    #   原来写 1audio 时，分离拼接音频产生的 concat_xxx_(Instrumental)/(Vocals)
    #   也会落在 1audio 里 —— 实测跑一个文件就多出 2 个 6.6 MB 的垃圾文件，
    #   下游第 4 步会把 Instrumental 也当成人声处理。
    #   真正要保留的只有下面填回原长度后写的那一个。
    output_dir=tempfile.gettempdir(),
    output_format="wav",
    use_autocast=True,
    vr_params={
        "batch_size": 1,
        "window_size": 512,
        "enable_autocast": True,
        "overlap": 4,
        "pitch_shift": 0,
        "shifts": 2,
    },
)

with Heartbeat(f"加载模型 {config['model']['sep']}"):
    separator.load_model(model_filename=config["model"]["sep"])

print("开始批量处理 ...")

_ok = []
_fail = []
_skip = []  # VAD 0 段：纯 BGM / 全静音，没人声可分离
_tmp_files = []

emit("stage", "start", 3)
_total = len(audio_files)
for _idx, flac in enumerate(audio_files, 1):
    emit("file", _idx, _total, os.path.basename(flac))
    base = os.path.splitext(os.path.basename(flac))[0]
    print(f"\n处理: {os.path.basename(flac)}")

    segs = load_vad_segments(flac)
    if segs is None:
        print("  [退回] 没有 VAD 段信息，整段分离")
        with Heartbeat(f"人声分离 {base}"):
            try:
                out = separator.separate(flac)
                # ★ 整段分离的输出在【临时目录】（见上面 output_dir 的注释），
                #   必须自己搬进 1audio。原来这里直接 _ok.append 就 continue，
                #   结果 1audio 一个文件都没有 —— 下面的产出检查会报
                #   「人声分离没有产出任何文件」，退出码 1，整批中止。
                #   实测踩过：第3步跑了 49 分钟后失败。
                outs = out if isinstance(out, (list, tuple)) else [out]
                voc = [str(f) for f in outs
                       if "Vocals" in os.path.basename(str(f))]
                if not voc:
                    raise RuntimeError("分离没有产出 Vocals 文件")
                voc_path = voc[0] if os.path.isabs(voc[0]) else os.path.join(
                    tempfile.gettempdir(), voc[0])
                if not os.path.isfile(voc_path):
                    raise RuntimeError(f"找不到分离结果: {voc_path}")
                stem = os.path.splitext(os.path.basename(flac))[0]
                dst = os.path.join(
                    config["path"]["audio"],
                    f"{stem}_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav")
                _shutil_disk.copyfile(voc_path, dst)
                _tmp_files.append(voc_path)
                _ok.append(base)
                print(f"  整段分离完成（已放入 1audio）: {os.path.basename(dst)}")
            except Exception as e:
                print(f"  [失败] {type(e).__name__}: {e}")
                _fail.append((base, str(e)[:120]))
        continue

    if not segs:
        # VAD 文件存在但 0 段语音：赠送的 BGM / 全静音，本来就没有人声。
        # 直接跳过，千万别退回整段分离 —— 那会白跑几十分钟再产出一条静音轨。
        print("  [跳过] VAD 0 段：没有人声（赠送的 BGM / 纯静音），无需分离")
        _skip.append(base)
        continue

    total_sec = sum(b - a for a, b in segs)
    # ★ v1.22.2：有效语音太短时不要送进分离模型。
    #   BS-RoFormer 处理 1~2 秒的短音频会崩：
    #   "The size of tensor a (0) must match the size of tensor b (76293)"。
    #   阈值取 2.0 秒（实测 1.74 秒必崩；2 秒以上一般能跑）。
    MIN_SPEECH_SEC = 2.0
    if total_sec < MIN_SPEECH_SEC:
        print(f"  [跳过] 有效语音只有 {total_sec:.1f}s（< {MIN_SPEECH_SEC:.0f}s），"
              f"不送分离模型（短音频会崩），继续下一个")
        _skip.append(base)
        continue
    try:
        data, sr = read_audio(flac)
        dur = len(data) / sr
        print(f"  VAD: {len(segs)} 段 / {total_sec:.1f}s（占 {total_sec / dur * 100:.1f}%）")

        merged, lens = concat_by_segments(data, sr, segs)
        if merged is None or len(merged) == 0:
            print("  [跳过] 没有可拼接的语音段")
            _fail.append((base, "VAD 段全为空"))
            continue

        # 写临时拼接文件
        tmp = os.path.join(tempfile.gettempdir(), f"_concat_{base}.flac")
        write_flac(tmp, merged, sr)
        _tmp_files.append(tmp)
        print(f"  拼接: {len(merged) / sr:.1f}s → {os.path.basename(tmp)}")

        with Heartbeat(f"人声分离 {base}"):
            voc_files = separator.separate(tmp)

        voc = [f for f in voc_files if "Vocals" in f]
        if not voc:
            # 分离没产出人声（短音频/模型限制）→ 跳过这个文件，别中止整批
            print("  [跳过] 分离没有产出人声（短音频/模型限制），继续下一个")
            _skip.append(base)
            continue
        # 分离结果现在落在临时目录
        vt = voc[0]
        voc_path = vt if os.path.isabs(vt) else os.path.join(tempfile.gettempdir(), vt)
        if os.path.isfile(voc_path):
            _tmp_files.append(voc_path)
        # Instrumental 也一并删掉（我们只要人声）
        for f in voc_files:
            if "Instrumental" in f:
                it = f if os.path.isabs(f) else os.path.join(tempfile.gettempdir(), f)
                if os.path.isfile(it):
                    _tmp_files.append(it)

        vd, vsr = read_audio(voc_path)
        if vsr != sr:
            import numpy as _np
            idx = _np.linspace(0, len(vd) - 1, int(len(vd) / vsr * sr))
            vd = vd[idx.astype(int)]
        print(f"  人声: {len(vd) / sr:.1f}s")

        full = expand_with_sr(vd, segs, lens, sr, len(data), data.shape[1])

        # 命名保持和原来一致（下游靠文件名找）
        stem = os.path.splitext(os.path.basename(flac))[0]
        dst = os.path.join(config["path"]["audio"],
                           f"{stem}_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav")
        sf.write(dst, full, sr, subtype="PCM_16")
        print(f"  填回原长度: {len(full) / sr:.1f}s → {os.path.basename(dst)}")
        _ok.append(base)

        # 删掉临时拼接文件
        try:
            if os.path.isfile(voc_path) and tmp.lower() in voc_path.lower():
                pass
        except Exception:
            pass

    except Exception as e:
        # 单个文件分离异常（模型限制/文件损坏）→ 记为失败，但不中止整批。
        # 只有「整批一个都没产出 且 存在失败」时才让退出码变 1（文件末尾判断）。
        print(f"  [失败] 分离异常（{type(e).__name__}: {e}），继续下一个")
        _fail.append((base, f"{type(e).__name__}: {str(e)[:120]}"))

# 清理临时文件
for t in _tmp_files:
    try:
        os.remove(t)
    except Exception:
        pass

emit("stage", "done", 3)
print("\n" + "=" * 60)
print(f"人声分离完成：成功 {len(_ok)} 个，失败 {len(_fail)} 个，"
      f"跳过 {len(_skip)} 个（无语音）")
if _fail:
    for b, why in _fail:
        print(f"  ✗ {b}: {why}")
print("=" * 60)

# 产出检查
_produced = [f for f in os.listdir(config["path"]["audio"])
             if f.lower().endswith((".wav", ".flac", ".mp3"))]
if not _produced:
    if _fail:
        # 整批一个都没产出，而且有真错误（磁盘满/模型崩/文件损坏）
        # → 这才是需要中止的情况，让用户看到。
        print("[失败] 人声分离没有产出任何文件，且存在处理错误：")
        for b, why in _fail:
            print(f"  ✗ {b}: {why}")
        sys.exit(1)
    # 全部是「无语音/短音频」跳过 → 正常结束，流水线继续
    print(f"[跳过] 本批没有可分离的人声（跳过 {len(_skip)} 个），流水线继续")
    sys.exit(0)
print(f"产出 {len(_produced)} 个文件")
