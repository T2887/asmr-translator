import os
import torch
# ====================== 配置 ======================
config = {"path": {
    "pre": "0pre",
    "audio": "1audio",
    "vad": "2vad",
    "asr": "3asr",
    "model": "model",
    "ffmpeg": "ffmpeg",
}, "model": {
    "sep": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "vad": "4evergr8/pyannote-segmentation-3.0",
    "asr": "4evergr8/whisper-large-v2-translate-zh-v0.2-st-ct2",
}, "vad": {
    "min_duration_on": 0.3,
    "min_duration_off": 0.1,
    "sample_rate": 16000,
}, "sep": {
    "sample_rate": 44100,
    # ★ 不设 chunk_duration —— 让 audio-separator 内部自己分段。
#
#   原来这里是 1800（30 分钟一块），本意是防止长音频爆显存。
#   但实测踩到大坑：audio-separator 的 AudioChunker 是【无重叠固定切块】，
#   且一旦某块"太短"就判 empty or not valid，然后【放弃整个文件】——
#   不是跳过那一块。
#
#   实例：2 小时视频 = 7205.53 秒，切成 5 块，第 5 块只剩 5.53 秒
#         → ERROR - Audio file chunk_0004.flac is empty or not valid
#         → Failed to process file ...（整个文件）
#         → 1audio 一个文件都没有
#         → 第3步报"没有找到音频文件"（误导，其实第2步就挂了）
#
#   真实视频的时长几乎不会是 1800 的整数倍，所以这个坑【必踩】。
#   而 BS-Roformer 内部本来就有 segment_size 分段机制（见 b 脚本的
#   mdxc_params），外层再切一遍是多余的。
#
#   注：去掉之后是一次性载入整个音频。2 小时 44.1kHz ≈ 700 MB 内存，
#       在 16 GB 以上的机器上没问题。# ====================== 设备与精度动态计算 ======================
}, "device": "cuda" if torch.cuda.is_available() else "cpu"}

# ── 数据目录前缀 ──
#
# 4 个中间目录（0pre/1audio/2vad/3asr）是【写】的，装进 Program Files 后
# 程序目录不可写，Rust 侧会通过 ASMR_DATA_DIR 告诉我们该写到哪。
# model/ ffmpeg 是【读】的，保持在程序目录（相对 cwd），不加前缀。
#
# 不传这个变量时（从命令行直接跑脚本、或便携版）行为完全不变。
_DATA_DIR = os.environ.get("ASMR_DATA_DIR", "").strip()
if _DATA_DIR:
    for _key in ("pre", "audio", "vad", "asr"):
        config["path"][_key] = os.path.join(_DATA_DIR, config["path"][_key])
# ─────────────────────────────────────────────────────────────
# 按【显存大小】自动选精度。
#
# 实测（RTX 5070 12GB，5 分钟音频）：
#   float16      识别阶段占 +3975 MiB   7.44s   字幕准确
#   int8_float16 识别阶段占 +1960 MiB   8.94s   个别词会错（"演技"→"扮演"）
#
# 6 GB 卡（RTX 4060 笔记本之类）用 float16 会 OOM：
#   基线 2.0 GB + 分离 2.6 GB = 4.6 GB（安全）
#   但识别 float16 再加 4.0 GB → 6.6 GB > 6.0 GB → 爆
# 换 int8 后识别只占 2.0 GB，峰值 4.6 GB，能跑。
#
# 阈值定 8 GB：6/8 GB 卡走 int8，12 GB 及以上走 float16。
# ─────────────────────────────────────────────────────────────
if config["device"] == "cuda":
    try:
        _total_mib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    except Exception:
        _total_mib = 0.0

    # 这里是按【实测峰值 + 预留余量】倒推的阈值，不是拍脑袋的 8 GB。
    #
    # 实测（RTX 5070 12GB，5 分钟音频，12GB 卡走 float16）：
    #   VAD      +869 MiB
    #   分离     +2585 MiB
    #   识别     +4265 MiB   ← 最大的一环
    #
    # 要求：任何时刻都要给用户的【其他程序】留至少 1 GB 显存。
    #   float16 需要 = 识别峰值 4265 + 系统基线 3000（保守，按开着浏览器算）
    #                  + 安全余量 1024  =  8289 MiB ≈ 8.1 GB
    #
    # 所以：
    #   6 GB 卡   → int8（峰值约 1960 + 3000 = 4960，余 1184 MiB ✓）
    #   8 GB 卡   → int8（8192 < 8289，余量不够 ✓ 走保守路线）
    #   12 GB 卡  → float16（7265，余 4962 MiB ✓）
    # 标称 8 GB 卡也可能因驱动/固件预留略低于 8192 MiB。总容量资格给约
    # 4% 容差（7.7 GiB）；界面在启动前仍要求至少 5.0 GB 当前空余显存。
    # 因此容差只修正「标称容量误判」，不会放行已经被占满的显卡。
    _NEED_F16_MIB = 7885                  # 约 7.70 GiB
    # 标称 6 GB 的显卡，CUDA 可管理容量可能因驱动/固件预留略低于 6144 MiB；
    # 若硬卡 6144，界面显示 6.0 GB 的 RTX 3060 Laptop 仍可能被误判成 CPU。
    # 5.75 GiB（5888 MiB）只给标称 6 GB 档约 4.2% 容差；启动前界面仍会
    # 实时要求至少 3.0 GB 空余显存，所以不会因为这个容差放行已占满的卡。
    _MIN_GPU_MIB = 5888

    if 0 < _total_mib < _MIN_GPU_MIB:
        config["device"] = "cpu"
        config["compute_type"] = "int8"
        config["lite_mode"] = False
        print(f"[CPU 模式] 检测到显存只有 {_total_mib/1024:.1f} GB，"
              f"低于精简模式约 5.75 GB 的最低要求")
        print(f"           人声分离需要约 2.5 GB 显存，装不下，已退回 CPU")
        print(f"           速度会慢很多（约为 GPU 的 1/50）")
    elif os.environ.get("ASMR_LITE") == "1":
        # 用户在界面上【手动选了「精简模式」】—— 这是明确要求省显存，
        # 不再看显存够不够，一律走 int8。
        #
        # 为什么需要这条分支：上面那套阈值只看【显存大小】，12 GB 的卡
        # 永远走 float16 —— 于是界面上选「精简模式」是个空操作，
        # 实测用户选了之后显存占用纹丝不动（11.4/12.0 GB）。
        # 界面选精简 = 想留显存给别的程序（游戏、浏览器、视频），不是
        # 因为跑不动，所以必须让用户的选择真正生效。
        config["compute_type"] = "int8_float16"
        config["lite_mode"] = True
        _gb = _total_mib / 1024 if _total_mib else 0.0
        print(f"[精简模式] 已按你的选择强制 int8 量化（界面里选的，"
              f"不看显存大小）")
        if _gb:
            print(f"           当前显存共 {_gb:.1f} GB，识别阶段占用约 2.0 GB"
                  f"（float16 时约 4.0 GB）")
        print(f"           代价：识别速度慢约 20%、个别词可能不够准确")
    elif 0 < _total_mib < _NEED_F16_MIB:
        config["compute_type"] = "int8_float16"
        config["lite_mode"] = True
        _gb = _total_mib / 1024
        print(f"[精简模式] 检测到显存 {_gb:.1f} GB，已自动切换到 int8 量化")
        print(f"           原因：float16 峰值需要约 {_NEED_F16_MIB/1024:.1f} GB，"
              f"用它会挤占你其他程序的显存")
        print(f"           代价：识别速度慢约 20%、个别词可能不够准确")
    else:
        config["compute_type"] = "float16"
        config["lite_mode"] = False
else:
    config["device"] = "cpu"
    config["compute_type"] = "int8"
    config["lite_mode"] = False
