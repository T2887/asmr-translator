# -*- coding: utf-8 -*-
"""ASMR 专用 VAD 封装（TransWithAI/Whisper-Vad-EncDec-ASMR-onnx，119MB）。

为什么换掉 Silero
──────────────────
Silero 是通用语音检测器，判据偏「有人声频段」；ASMR 里大量低频喘息、
气流、耳语被它当成语音 → 送进 ASR，模型只能硬编句子（「张冠李戴」）。
这个 VAD 用 ~500 小时日文 ASMR 训练，标签含 whispered-speech，
实测（a120.wav 120s / CPU 4 线程）RTF 0.0162 ≈ 61x 实时，不占显存。

两个必须绕开的坑（子代理实测）
────────────────────────────
1. ONNX 输出是 **logits 不是概率**，必须 sigmoid。
   官方 README 的 `predictions[0] > 0.5` 示例是错的，直接抄会全 0/全 1。
2. 对**绝对响度**敏感：整段音乐/极低音量音频会被判「全程无人说话」。
   实测 asmr5m.wav 的 100ms RMS 中位 -54.8 dBFS → 全程静音。
   所以送进来之前必须先做响度归一化（见 normalize_loudness）。

输入要求：16kHz 单声道 float32；内部按 30s 分块，log-mel [1,80,3000]。
本模块不依赖 transformers —— 用 numpy 手搓 Slaney mel（与 whisper 一致）。
"""
import json
import os

import numpy as np

SR = 16000
CHUNK_SAMPLES = 30 * SR          # 30s
N_MELS = 80
N_FFT = 400
HOP = 160                        # 10ms
TARGET_FRAMES = 3000             # 30s @ 100fps

# Slaney mel（whisper 用 slaney 刻度，不是 HTK —— 用错会整体偏移）
def _mel_filters(n_mels=N_MELS, sr=SR, n_fft=N_FFT):
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mels = f / (200.0 / 3)
        logstep = np.log(6.4) / 27.0
        hi = f >= 1000.0
        mels = np.where(hi, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / logstep, mels)
        return mels

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        freqs = 200.0 / 3 * m
        logstep = np.log(6.4) / 27.0
        hi = m >= 15.0
        return np.where(hi, 1000.0 * np.exp(logstep * (m - 15.0)), freqs)

    mels = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz = mel_to_hz(mels)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / max(ce - lo, 1e-10)
        down = (hi - freqs) / max(hi - ce, 1e-10)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


_MEL_FB = _mel_filters()


def log_mel(audio):
    """→ [1, 80, T] float32（whisper 的 log-mel，含动态范围压缩）。"""
    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)))
    n = 1 + (len(audio) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = audio[idx] * np.hanning(N_FFT)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2          # [T, 201]
    mel = spec @ _MEL_FB.T                                     # [T, 80]
    log_spec = np.log10(np.maximum(mel, 1e-10))
    # whisper 的归一化：整体缩放到 [-1,1] 附近，下限截断
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.T[None, :, :].astype(np.float32)           # [1,80,T]


def normalize_loudness(audio, target_dbfs=-20.0, peak_limit=0.99):
    """响度归一化（送 VAD 前必须做）。

    这个 VAD 对绝对电平敏感：整段 -54dBFS 会被判「全程静音」。
    按 100ms 窗 RMS 的中位数对齐到 target_dbfs，并留峰值余量。
    """
    if audio.size == 0:
        return audio
    win = int(0.1 * SR)
    n = len(audio) // win
    if n >= 1:
        rms = np.sqrt((audio[:n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
        med = float(np.sqrt(np.median(rms) ** 2))
    else:
        med = float(np.sqrt((audio ** 2).mean() + 1e-12))
    if med <= 1e-8:
        return audio
    target = 10 ** (target_dbfs / 20.0)
    gain = target / med
    out = audio * gain
    peak = float(np.abs(out).max())
    if peak > peak_limit:
        out = out * (peak_limit / peak)
    return out.astype(np.float32)


class AsmrVad:
    """加载 119MB ONNX，20ms 帧级概率 → 语音段列表。"""

    def __init__(self, model_path, threads=4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        meta_p = os.path.join(os.path.dirname(model_path), "model_metadata.json")
        self.meta = {}
        if os.path.isfile(meta_p):
            try:
                self.meta = json.load(open(meta_p, encoding="utf-8"))
            except Exception:
                pass

    @staticmethod
    def _to_numpy(audio):
        """兼容 silero_vad.read_audio 返回的 torch.Tensor。

        ★ 踩过的坑：light_asr.py 用 silero_vad.read_audio() 读音频，
        它返回的是 torch.Tensor（不是 numpy），直接 .astype() 会
        AttributeError。这里统一转 numpy float32。
        """
        if audio is None:
            return np.zeros(0, dtype=np.float32)
        if not isinstance(audio, np.ndarray):
            try:
                audio = audio.detach().cpu().numpy()
            except Exception:
                audio = np.asarray(audio)
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        return a.ravel()

    def frame_probs(self, audio, normalize=True):
        """→ (probs[每 20ms 一帧], 总时长秒)。内部 sigmoid。"""
        a = self._to_numpy(audio)
        a = normalize_loudness(a) if normalize else a.astype(np.float32)
        probs = []
        for st in range(0, len(a), CHUNK_SAMPLES):
            chunk = a[st:st + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            mel = log_mel(chunk)
            # 补齐/裁到 [1,80,3000]
            if mel.shape[2] < TARGET_FRAMES:
                mel = np.pad(mel, ((0, 0), (0, 0), (0, TARGET_FRAMES - mel.shape[2])))
            else:
                mel = mel[:, :, :TARGET_FRAMES]
            out = self.sess.run(None, {self.in_name: mel})[0]
            logits = np.asarray(out).reshape(-1)
            # ★ 必须 sigmoid（官方 README 示例漏了这步）
            probs.append(1.0 / (1.0 + np.exp(-logits)))
        p = np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)
        return p, len(a) / SR

    def segments(self, audio, threshold=0.5, neg_threshold=None,
                 min_speech=0.25, min_silence=0.2, pad=0.0, normalize=True):
        """→ [(start_sec, end_sec)]。双阈值（滞回）避免抖动切碎。"""
        p, dur = self.frame_probs(audio, normalize=normalize)
        if p.size == 0:
            return []
        neg = threshold - 0.15 if neg_threshold is None else neg_threshold
        frame_dt = dur / p.size
        segs, in_sp, st = [], False, 0.0
        for i, v in enumerate(p):
            t = i * frame_dt
            if not in_sp and v >= threshold:
                in_sp, st = True, t
            elif in_sp and v < neg:
                in_sp = False
                segs.append([st, t])
        if in_sp:
            segs.append([st, dur])
        # 最短语音 / 最短静音合并
        merged = []
        for a, b in segs:
            if b - a < min_speech:
                continue
            if merged and a - merged[-1][1] < min_silence:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        if pad > 0:
            out = []
            for i, (a, b) in enumerate(merged):
                lo = merged[i - 1][1] if i > 0 else 0.0
                hi = merged[i + 1][0] if i + 1 < len(merged) else dur
                out.append((max(lo, a - pad), min(hi, b + pad)))
            return out
        return [(a, b) for a, b in merged]


if __name__ == "__main__":
    import sys
    import time
    import soundfile as sf
    mp = r"H:\software\ASMR翻译器\_ocr\_model_probe\vad\model.onnx"
    wav = sys.argv[1] if len(sys.argv) > 1 else r"H:\software\ASMR翻译器\_haas_test\a120.wav"
    a, sr = sf.read(wav, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        print(f"警告：采样率 {sr} != 16000，结果可能不准")
    vad = AsmrVad(mp)
    t0 = time.time()
    segs = vad.segments(a)
    el = time.time() - t0
    dur = len(a) / sr
    clean = sum(b - x for x, b in segs)
    print(f"{os.path.basename(wav)}  {dur:.1f}s")
    print(f"  语音段 {len(segs)} 个，覆盖 {clean:.1f}s ({clean/dur*100:.1f}%)")
    print(f"  耗时 {el:.2f}s  RTF {el/dur:.4f}  ({dur/max(el,1e-9):.1f}x 实时)")
    for x, b in segs[:8]:
        print(f"    {x:7.2f} ~ {b:7.2f}  ({b-x:.2f}s)")
