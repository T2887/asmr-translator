"""兼容模式 第2步：Silero VAD + SenseVoice 转写（全 CPU）。

输入：0pre\\*.flac / *.wav / *.mp3 / *.m4a
输出：2vad\\<名字>.json   [{start, end, text}, ...]

为什么不用 pyannote：兼容模式面向没有可用 CUDA 的机器，
pyannote 在 CPU 上很慢；Silero VAD 是 ONNX 小模型，毫秒级。
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
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

PRE = "0pre"
VAD_DIR = "2vad"
MODEL = os.path.join("model", "SenseVoiceSmall")
EXTS = (".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg")

files = sorted(
    f for f in glob.glob(os.path.join(PRE, "*")) if f.lower().endswith(EXTS)
)
if not files:
    print(f"在 {PRE} 中没有找到音频文件")
    sys.exit(1)

print(f"兼容模式：{len(files)} 个音频")
print("加载 Silero VAD + SenseVoice ...", flush=True)
t0 = time.time()
vad = load_silero_vad(onnx=True)
asr = AutoModel(model=MODEL, device="cpu", disable_update=True)
print(f"模型加载完成（{time.time() - t0:.1f}s）", flush=True)

os.makedirs(VAD_DIR, exist_ok=True)
ok = 0
fail = 0
total = len(files)
t_start = time.time()


def merge_segments(segs, gap_sec=0.6):
    """合并间隔很近的段：Silero 对短促语音切得太碎，语义会被切散。"""
    merged = []
    for s in segs:
        if merged and (s["start"] - merged[-1]["end"]) / 16000.0 < gap_sec:
            merged[-1]["end"] = s["end"]
        else:
            merged.append({"start": s["start"], "end": s["end"]})
    return merged


for idx, fp in enumerate(files, 1):
    stem = os.path.splitext(os.path.basename(fp))[0]
    out = os.path.join(VAD_DIR, stem + ".json")
    print(f"\n处理: {os.path.basename(fp)}", flush=True)
    try:
        wav16 = read_audio(fp, sampling_rate=16000)
        segs = get_speech_timestamps(
            wav16, vad, sampling_rate=16000,
            min_speech_duration_ms=300,
            min_silence_duration_ms=600,
            speech_pad_ms=120,
            max_speech_duration_s=10,
        )
        segs = merge_segments(segs)

        data, sr = sf.read(fp, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        rows = []
        for seg in segs:
            a = max(0, int(seg["start"] / 16000.0 * sr))
            b = min(len(data), int(seg["end"] / 16000.0 * sr))
            if b - a < int(sr * 0.15):
                continue
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                sf.write(tmp, data[a:b], sr)
                r = asr.generate(input=tmp, language="ja", use_itn=True)
                text = rich_transcription_postprocess(
                    r[0].get("text", "") if r else ""
                ).strip()
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if text:
                rows.append({
                    "start": round(seg["start"] / 16000.0, 2),
                    "end": round(seg["end"] / 16000.0, 2),
                    "text": text,
                })
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"  转写完成：{len(rows)} 段", flush=True)
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
    print(f"  {pct:3.0f}%|{bar:<20}| {idx}/{total} [{el // 60:02d}:{el % 60:02d}<00:00]", flush=True)

print(f"\n兼容模式转写完成：成功 {ok} 个，失败 {fail} 个")
if ok == 0:
    print("[失败] 没有任何音频转写成功")
    sys.exit(1)
