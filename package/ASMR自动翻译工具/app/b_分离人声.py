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
def _memory_status():
    """读 Win32 GlobalMemoryStatusEx，返回 (可用物理, 还可提交) 字节数。

    读不到返回 (0, 0)。不依赖 psutil（env 里没装 psutil）。

    ★ 为什么必须看「还可提交」而不是「可用物理」：
      2026-10-31 实测（26:55 素材、RTX 5070）分离阶段
        WorkingSet 峰值 3.09 GB
        Commit     峰值 7.06 GB   ← 是工作集的 2.3 倍
      PyTorch/CUDA 进程会大量 reserve（CUDA 上下文、权重缓冲、
      中间 tensor），这些**不驻留物理内存但占提交额度**。
      判定能不能跑必须用提交额度：Windows 上 malloc 失败
      （DefaultCPUAllocator: not enough memory）几乎都是
      提交量到达 物理内存 + 页文件 的上限，而不是物理内存不够。
    """
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
            # ullAvailPageFile 就是「还能再提交多少」，这是硬上限
            return int(st.ullAvailPhys), int(st.ullAvailPageFile)
    except Exception:
        pass
    return 0, 0


def available_ram_bytes():
    """返回系统当前可用物理内存（字节），读不到返回 0。"""
    return _memory_status()[0]


def available_commit_bytes():
    """返回系统当前还能提交的内存（字节），读不到返回 0。"""
    return _memory_status()[1]


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


def concat_by_segments_partial(path, sr_hint, segs, total_len):
    """concat_by_segments 的分段读版本：不整段载入音频。

    原来 read_audio 一次性把整段读进内存（2 小时 48k 立体声 float32
    就是 1.3 GB），但这里其实只用得到 VAD 圈出的那 15%~40%。
    soundfile 支持 start/stop 局部读，按段读再拼接，峰值内存只和
    【语音总长】有关，和文件总长无关。

    返回 (拼接数组, 每段样本数)，与 concat_by_segments 一致。
    """
    chunks = []
    lens = []
    for s, e in segs:
        a = int(s * sr_hint)
        b = min(int(e * sr_hint), total_len)
        if b <= a:
            lens.append(0)
            continue
        chunk, _sr = sf.read(path, start=a, stop=b, dtype="float32",
                             always_2d=True)
        lens.append(len(chunk))
        chunks.append(chunk)
    if not chunks:
        return None, []
    return np.concatenate(chunks, axis=0), lens


def write_vocals_expanded(dst, segs, lens, sr, total_len, n_ch, voc_path):
    """单批填回（保留旧签名，内部走多批核心）。"""
    write_vocals_batches(dst, [(segs, lens, voc_path)], sr, total_len, n_ch)
    return total_len


def write_vocals_batches(dst, batches, sr, total_len, n_ch):
    """把人声按 VAD 段填回原长度并【逐块写出】，不在内存里攒整段。

    batches: [(该批段, 该批 lens, 该批人声 wav 路径), ...]，按时间顺序。

    原来 expand_with_sr 会 np.zeros((total_len, ch)) —— 又是一份
    和整段音频同样大的数组（2 小时 = 1.3 GB），加上 read 回来的
    人声、sf.write 内部转换，三份叠着走。

    这里改成：预开 PCM_16 输出，按段「读人声片段 → 写静音间隙 →
    写本段」流式推进。峰值内存只和【单批语音长】有关，和文件总长无关。

    语义与 expand_with_sr 完全一致：
      · lens[i] == 0 的段跳过（不写内容，间隙自然由静音补齐）
      · 段间重叠时后段覆盖前段已写内容（seek 回去重写）
      · 声道数不符时复制或截断
    批与批之间不重叠（plan_batches 保证），每批人声游标独立归零。
    """
    written = 0   # 输出里已写到的样本位置
    with sf.SoundFile(dst, mode="w", samplerate=sr, channels=n_ch,
                      subtype="PCM_16", format="WAV") as of:
        for segs_i, lens_i, voc_path in batches:
            info = sf.info(voc_path)
            v_total = info.frames
            if info.samplerate != sr:
                # 人声文件只含【该批】语音段，整读重采样的开销只随批语音长
                # 增长，可以接受 —— 要避免的是整段原音频（随文件总长增长）。
                vd, vsr = sf.read(voc_path, dtype="float32", always_2d=True)
                _idx = np.linspace(0, len(vd) - 1,
                                   int(len(vd) / vsr * sr)).astype(np.int64)
                vd = vd[_idx]
                v_avail = len(vd)
                vf = None
            else:
                vd = None
                v_avail = v_total
                vf = sf.SoundFile(voc_path, "r")
            pos = 0   # 该批人声里的读取游标
            try:
                for (s, e), n in zip(segs_i, lens_i):
                    if n <= 0:
                        continue
                    a = int(s * sr)
                    b = min(a + n, total_len)
                    n2 = min(b - a, v_avail - pos)
                    if n2 <= 0:
                        break
                    if vf is None:
                        chunk = vd[pos:pos + n2]
                    else:
                        vf.seek(pos)
                        chunk = vf.read(n2, dtype="float32", always_2d=True)
                    pos += n2
                    if chunk.shape[1] != n_ch:
                        if chunk.shape[1] == 1:
                            chunk = np.repeat(chunk, n_ch, axis=1)
                        else:
                            chunk = chunk[:, :n_ch]
                    chunk = np.ascontiguousarray(chunk)
                    # 间隙补静音（先 seek 回已写末尾，再顺序写，避免跳跃写）
                    if a > written:
                        of.seek(written)
                        of.write(np.zeros((a - written, n_ch), dtype="float32"))
                        written = a
                    # 重叠时 a < written：seek 回去覆盖，语义同旧版。
                    # plan_batches 保证重叠段同批，跨批之间不会发生覆盖。
                    of.seek(a)
                    of.write(chunk)
                    written = max(written, a + len(chunk))
            finally:
                if vf is not None:
                    vf.close()
        # 尾部补静音到原长度
        if written < total_len:
            of.seek(written)
            of.write(np.zeros((total_len - written, n_ch), dtype="float32"))
    return total_len


def plan_batches(segs, avail_bytes):
    """按可用提交额度决定分几批分离。

    ★ v1.30.2 实测拟合（RTX 5070 同机、流式化之后，分离净增提交内存
      vs 语音总长）：
        语音   56.5s → 净增 5.70 GB
        语音  534.5s → 净增 6.76 GB
      斜率 ≈ 2.2 MB/s（语音），基线 ≈ 5.5 GB。
      文件总长项（旧公式 1.2 MB/s × dur）已被「局部读 + 流式写」消掉。

    进到这里时基线（模型 + CUDA 上下文）已经付掉了，批预算只算增量：
        每批语音上限 = (还可提交 - 0.8 GB 安全余量) × 0.85 / 2.2 MB/s
    额度充足时返回单批 —— 行为与不切完全一致。

    重叠的 VAD 段强制同批：跨批重叠会让「后段覆盖前段」的语义失效。

    ASMR_SEP_BATCH_SEC 环境变量可强制批上限（测试多批路径用）。
    """
    # ★ v1.30.2 修正：这里的 avail 是【文件处理时】的额度，而模型加载
    #   （5.5 GB 基线）在批开始之前就已经付掉了 —— 预算不能再扣一次基线，
    #   否则可用 7 GB 的机器会被算成只能放 0.7 GB 语音，明明分批能救
    #   却被误杀。每批真正要预留的是：
    #       该批语音 × 2.2 MB/s（分离内部张量，实测斜率）
    #       + 0.5 GB 安全余量（拼接数组、flac 编解码缓冲）
    #   批下限 60 秒：再切小每批固定开销占比过高，而且 BS-RoFormer
    #   对 1~2 秒级音频本身会崩（见 MIN_SPEECH_SEC 注释）。
    per_sec = 2.2 * 1024 * 1024
    cap_sec = 0.0
    if avail_bytes:
        cap_sec = max(0.0, avail_bytes - int(0.5 * _GB)) / per_sec
        cap_sec = max(cap_sec, 60.0)
    try:
        _forced = float(os.environ.get("ASMR_SEP_BATCH_SEC", "0"))
        if _forced > 0:
            cap_sec = _forced if cap_sec <= 0 else min(cap_sec, _forced)
    except ValueError:
        pass
    total_speech = sum(e - a for a, e in segs)
    if cap_sec <= 0 or total_speech <= cap_sec:
        return [segs]
    batches, cur, cur_len = [], [], 0.0
    for s, e in segs:
        L = e - s
        overlap = bool(cur) and s < cur[-1][1]
        if cur and not overlap and cur_len + L > cap_sec:
            batches.append(cur)
            cur, cur_len = [], 0.0
        cur.append((s, e))
        cur_len += L
    if cur:
        batches.append(cur)
    return batches

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
        # ── 读原音频信息（不整段载入）──────────────────────────
        # ★ v1.30.2：原来 read_audio 一次性读整段（2 小时 48k 立体声
        #   float32 ≈ 1.3 GB 提交内存），但真正用到的只有 VAD 圈出的
        #   15%~40%。改成只读 header，拼接时用 sf.read(start,stop)
        #   局部读。配合下面「流式填回」，分离阶段的内存峰值不再
        #   随文件总长增长（实测 33 分钟素材：净增 7.07 → 约 6 GB，
        #   2 小时素材从 14 GB 降到 6 GB 出头）。
        _ainfo = sf.info(flac)
        sr = _ainfo.samplerate
        total_len = _ainfo.frames
        n_ch = _ainfo.channels
        dur = total_len / sr
        print(f"  VAD: {len(segs)} 段 / {total_sec:.1f}s（占 {total_sec / dur * 100:.1f}%）")

        # ── v1.27.2：分离前先查可用内存 ────────────────────────
        # ★ v1.27.4：按「提交内存」判定，不再按物理内存。
        #   Windows 上 malloc 失败几乎都是提交额度到顶，物理不足只换页。
        # ★ v1.30.2：流式化之后内存不再随【文件总长】增长，只随【语音总长】：
        #     5.5 GB 固定（模型 + CUDA 上下文）+ 语音秒数 × 2.2 MB/s
        #   （同机实测：语音 56.5s 净增 5.70 GB，语音 534.5s 净增 6.76 GB）
        #   额度不够整跑时，plan_batches 会按额度分批 —— 每批峰值只和
        #   该批语音长有关。只有连【一批最小 60 秒】都放不下时才报错跳过。
        _phys, _commit = _memory_status()
        # 只看提交额度。物理不足 → 系统换页（只是慢）；提交不足 → 直接 malloc 失败。
        _avail = _commit or _phys
        _batches = plan_batches(segs, _avail)
        _max_batch = max(sum(e - a for a, e in b) for b in _batches)
        if _avail:
            # 基线已付，这里只算单批增量 + 安全余量
            _need = int(_max_batch * 2.2 * 1024 * 1024 + 0.5 * _GB)
            if _avail < _need:
                print(f"  [失败] 内存不足：需要约 {_need / _GB:.1f} GB 提交额度，"
                      f"当前只能再拿到 {_avail / _GB:.1f} GB")
                print(f"        （{dur / 60:.0f} 分钟音频、其中语音 {total_sec / 60:.1f} 分钟，"
                      f"已按额度切成 {len(_batches)} 批仍放不下）")
                if _commit and _commit < _need:
                    print("        拦你的是「提交内存」而不是物理内存 ——")
                    print("        多半是虚拟内存（页文件）设得太小或被关闭了。")
                    print("        修法：此电脑→属性→高级系统设置→性能设置→高级→"
                          "虚拟内存→改成「系统管理的大小」或手工 32768 MB，重启后再试。")
                print("        也可以改用兼容模式（不需要人声分离，内存需求约 2~3 GB）")
                _fail.append((base, f"内存不足（需 {_need / _GB:.1f} GB，"
                                    f"有 {_avail / _GB:.1f} GB）"))
                continue

        stem = os.path.splitext(os.path.basename(flac))[0]
        dst = os.path.join(config["path"]["audio"],
                           f"{stem}_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav")

        if len(_batches) == 1:
            # ── 单批：行为与旧版一致 ──
            # ★ merged 数组在闸【之后】才分配：闸不过的文件不该再为它
            #   冒 OOM 风险（语音 2 小时 = 1.7 GB float32）。
            merged, lens = concat_by_segments_partial(flac, sr, segs, total_len)
            if merged is None or len(merged) == 0:
                print("  [跳过] 没有可拼接的语音段")
                _fail.append((base, "VAD 段全为空"))
                continue
            tmp = os.path.join(tempfile.gettempdir(), f"_concat_{base}.flac")
            write_flac(tmp, merged, sr)
            _tmp_files.append(tmp)
            del merged
            print(f"  拼接: {sum(lens) / sr:.1f}s → {os.path.basename(tmp)}")

            with Heartbeat(f"人声分离 {base}"):
                voc_files = separator.separate(tmp)

            voc = [f for f in voc_files if "Vocals" in f]
            if not voc:
                # ★ v1.27.2：走到这里不是「短音频」—— 真正短的在上游
                #   total_sec < MIN_SPEECH_SEC 就被拦掉了。返回空列表只可能是
                #   模型内部失败，最常见的是内存不足。必须记为失败，否则
                #   _fail 为空 → 退出码 0 → 界面显示「完成」但没有任何字幕。
                print("  [失败] 分离没有产出人声（常见原因：内存不足或模型加载失败）")
                _fail.append((base, "分离未产出人声（可能内存不足）"))
                continue
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

            _vi = sf.info(voc_path)
            print(f"  人声: {_vi.frames / _vi.samplerate:.1f}s")
            write_vocals_expanded(dst, segs, lens, sr, total_len, n_ch, voc_path)
        else:
            # ── 多批：拼接文件按批写，批批分离，流式填回 ──
            print(f"  内存不足整跑，按额度分 {len(_batches)} 批分离"
                  f"（每批语音约 {sum(e - a for a, e in _batches[0]):.0f}s）")
            batch_args = []
            ok_all = True
            for bi, bsegs in enumerate(_batches):
                blens = []
                bchunks = []
                for s, e in bsegs:
                    a = int(s * sr)
                    b2 = min(int(e * sr), total_len)
                    if b2 <= a:
                        blens.append(0)
                        continue
                    ch_, _sr2 = sf.read(flac, start=a, stop=b2, dtype="float32",
                                        always_2d=True)
                    blens.append(len(ch_))
                    bchunks.append(ch_)
                if not bchunks:
                    batch_args.append((bsegs, blens, None))
                    continue
                bmerged = np.concatenate(bchunks, axis=0)
                del bchunks
                btmp = os.path.join(tempfile.gettempdir(), f"_concat_{base}_b{bi}.flac")
                write_flac(btmp, bmerged, sr)
                _tmp_files.append(btmp)
                del bmerged
                with Heartbeat(f"人声分离 {base}（批 {bi + 1}/{len(_batches)}）"):
                    voc_files = separator.separate(btmp)
                voc = [f for f in voc_files if "Vocals" in f]
                if not voc:
                    print(f"  [失败] 批 {bi + 1}/{len(_batches)} 没有产出人声"
                          f"（常见原因：内存不足）")
                    ok_all = False
                    break
                vt = voc[0]
                voc_path = (vt if os.path.isabs(vt)
                            else os.path.join(tempfile.gettempdir(), vt))
                # 立刻改名：下一批的输入文件名不同，输出不会撞名，但改名更稳
                vdst = os.path.join(tempfile.gettempdir(), f"_voc_{base}_b{bi}.wav")
                _shutil_disk.copyfile(voc_path, vdst)
                _tmp_files.append(voc_path)
                if os.path.isfile(voc_path):
                    _tmp_files.append(vdst)
                for f in voc_files:
                    if "Instrumental" in f:
                        it = f if os.path.isabs(f) else os.path.join(tempfile.gettempdir(), f)
                        if os.path.isfile(it):
                            _tmp_files.append(it)
                batch_args.append((bsegs, blens, vdst))
            if not ok_all:
                _fail.append((base, "分批分离未产出人声（可能内存不足）"))
                continue
            write_vocals_batches(dst, [x for x in batch_args if x[2] is not None],
                                 sr, total_len, n_ch)
            _vi_total = sum(sf.info(x[2]).frames for x in batch_args if x[2])
            print(f"  人声: {_vi_total / sr:.1f}s（{len(batch_args)} 批）")

        print(f"  填回原长度: {total_len / sr:.1f}s → {os.path.basename(dst)}")
        _ok.append(base)
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
