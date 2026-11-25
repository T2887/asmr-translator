"""兼容模式 第3步：用内置 llama.cpp（Hy-MT2-1.8B）翻译。

输入：2vad\\<名字>.json（light_asr.py 的产物）
输出：3asr\\<名字>.srt

流程：
  · 启动 llama\\llama-server.exe（CPU，--device none，避免抢显存）
  · 逐段翻译（few-shot + 按作品的术语表 hotwords_<名字>.txt）
  · 时间轴整理后写 SRT
  · 全部完成后关掉 llama-server

词表（固定错译替换）不在这里做 —— 第4步 e_字幕整理.py 会统一处理。
"""
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

from events import emit

# ★ v1.30.4：术语后处理（fixes.txt / fixes_<名>.txt）。
#   prompt 是软约束，几百段里总会冒出几种写法；这一层做确定性替换。
from terms_fix import apply_fixes, apply_pre, load_fixes

# ★ v1.30.3：运行时状态采集（模型吞吐 / CPU 频率 / 占用核数）
#   借鉴安卓版思路（它从 llama-server 的 /metrics 取 tok/s、
#   从 /proc 取 CPU、从 scaling_cur_freq 取频率）。
#   PC 上没有 /proc，用 GetProcessTimes + CallNtPowerInformation 替代。
#   全部异常吞掉 —— 状态显示失败绝不能中断翻译。
try:
    from status_probe import StatusProbe, find_llama_pid
except Exception:  # pragma: no cover
    StatusProbe = None
    find_llama_pid = None

# ── 数据目录前缀 ──
#
# 0pre / 2vad / 3asr 都是【写】的。装进 Program Files 时程序目录不可写，
# Rust 侧会通过 ASMR_DATA_DIR 告诉我们该写到哪；不传时（命令行直跑）行为不变。
_DATA_DIR = os.environ.get("ASMR_DATA_DIR", "").strip()

# ── 日文字幕（用户在界面上勾选的两级开关）──
#
# 主开关 ASMR_JPN_SUB 打开后，日文原文（就是送进翻译的那段 SenseVoice 文本，
# 也就是 2vad\<名字>.json 里的 text）会一起写进字幕；日文不需要模型额外
# 产出，所以打开它【没有任何额外耗时】，也不改提示词、不影响译文质量。
#
# 子开关 ASMR_JPN_SEPARATE 决定日文放哪儿：
#   · 关（默认）→ 写进中文那一份里，一条字幕两行：日文原文在上、中文在下。
#     播放器不用切字幕轨，一眼能对着原文看。
#   · 开 → 中文归中文，**额外**再写一份纯日文的 `<名字>.jpn.srt`，时间轴与
#     中文完全一致。`.jpn` 是播放器认的语言代码（PotPlayer 等按 ISO 639-2
#     找外挂字幕：`.chs` 中文、`.jpn` 日文），两份都会被自动加载、可随时切换。
JPN_SUB = os.environ.get("ASMR_JPN_SUB", "0").strip() == "1"
JPN_SEPARATE = os.environ.get("ASMR_JPN_SEPARATE", "0").strip() == "1"
# 日文并进中文那一份（行内双语）
JPN_INLINE = JPN_SUB and not JPN_SEPARATE


def _p(name):
    return os.path.join(_DATA_DIR, name) if _DATA_DIR else name

VAD_DIR = _p("2vad")
ASR_DIR = _p("3asr")
LLAMA = os.path.join("llama", "llama-server.exe")
# ── 兼容模式的翻译模型（llama.cpp 的 GGUF）──
#
# 默认用混元翻译专用模型 Hy-MT2-1.8B：同一批 12 句实测它 12/12 全对，
# 口语语气和专有名词都比通用小模型稳，CPU 上只慢约 15%（224ms vs 193ms/句）。
#
# 环境变量 ASMR_TRANSLATE_MODEL 可以指定 model\ 下的其它 gguf 文件名。
# 新模型文件不存在时（例如用户还在用旧资源包）自动回退到原来的
# Qwen2.5-1.5B —— 否则老用户一升级就会看到「缺少兼容模式组件」。
MODEL_NEW = "Hy-MT2-1.8B-Q4_K_M.gguf"
MODEL_OLD = "qwen2.5-1.5b-q4km.gguf"
_MODEL_MIN_BYTES = 800 * 1024 * 1024  # 两个正式 GGUF 都 > 900 MB；拦截半截下载


def _valid_model(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= _MODEL_MIN_BYTES
    except OSError:
        return False


def _resolve_model():
    """优先环境变量 → 新模型 → 旧资源包模型；返回绝对路径。

    环境变量既可写文件名（从 model\ 下找），也可写绝对/相对路径。
    明确指定但无效时只警告并继续回退，不让一处配置错误拖垮整批任务。
    """
    picked = os.environ.get("ASMR_TRANSLATE_MODEL", "").strip()
    if picked:
        candidate = picked if os.path.isabs(picked) else (
            picked if os.path.dirname(picked) else os.path.join("model", picked)
        )
        candidate = os.path.abspath(candidate)
        if _valid_model(candidate):
            return candidate
        print(f"[警告] 指定的翻译模型不存在或文件不完整，将自动回退：{candidate}")

    for name in (MODEL_NEW, MODEL_OLD):
        candidate = os.path.abspath(os.path.join("model", name))
        if _valid_model(candidate):
            if name == MODEL_OLD:
                print("[提示] 新版翻译模型尚未安装，临时使用旧资源包里的 Qwen2.5-1.5B")
            return candidate

    # 交给 main() 的统一组件检查输出路径并退出。
    return os.path.abspath(os.path.join("model", MODEL_NEW))


MODEL = _resolve_model()

# ── ASMR 领域 LoRA 适配器（v1.30.2）──────────────────────────
# 用官方中文字幕对齐出的 1.2 万对日中句对微调（LoRA r16，Hy-MT2
# 专属适配器）。held-out 作品实测：与官方字幕相似度 35%→42%，
# 逐字相同 3→30 句。
# ★ 只在 GPU（Vulkan）设备上启用：LoRA 的 delta 矩阵乘法在 CPU 上
#   会把翻译拖慢约 2.4 倍，Vulkan 上只慢 24%（收益/代价比合适）。
#   找不到文件 = 老环境包 → 自动不启用，行为与 v1.30.1 完全一致。
LORA_MODEL = "hy-mt2-asmr-lora.gguf"
_LORA_MIN_BYTES = 40 * 1024 * 1024   # 实际 77.6 MB，拦半截复制


def _resolve_lora():
    picked = os.environ.get("ASMR_TRANSLATE_LORA", "").strip()
    if picked == "off":
        return None
    if picked and picked != "on":
        cand = picked if os.path.isabs(picked) else os.path.abspath(
            os.path.join("model", picked))
    else:
        cand = os.path.abspath(os.path.join("model", LORA_MODEL))
    try:
        if os.path.isfile(cand) and os.path.getsize(cand) >= _LORA_MIN_BYTES:
            return cand
    except OSError:
        pass
    return None


LORA_PATH = _resolve_lora()

SYSTEM = (
    "你是日译中字幕翻译。把用户给的日语翻译成简体中文，口语化、句子短促，"
    "只输出中文，不要保留日文，不要解释，译文必须是一行、不要换行。\n"
    "示例：\n"
    "日语：今日はいい天気ですね\n中文：今天天气真不错呢\n"
    "日语：ちょっと待ってください\n中文：请稍等一下\n"
    "日语：そろそろ帰りましょうか\n中文：差不多该回去了吧"
)

# ★ v1.30.3：内建通用术语（从 25793 对官方字幕语料挖掘，跨作品一致）。
#   与 hotwords.txt 是【叠加】关系：内置的这些保证开箱即用，
#   用户/作品的词表（hotwords_<名字>.txt）再往上补人名地名。
#   挖掘口径见 d_语音识别.py 的 _HOTWORDS 注释（同一个分析管线）。
BUILTIN_TERMS = ("肉棒", "小穴", "高潮", "精液", "射精", "口交", "乳交",
                 "摩擦", "抽插", "乳头", "耳朵", "屁股", "按摩")
# 拟声词单列：这类词在 ASMR 里高频，且不能都被译成同一句话
#   （v1 实测「れろれろれろ」被译成「好可爱」——音效被当成评价，
#   这是本次重训专门治的问题，内建提示里点明要「像声音」）
ONOMAT_HINT = (
    "\n拟声词（ぺろ/ちゅ/れろ/咕啾类）要译成对应的中文音效词"
    "（如 舔弄声、啾、咕啾、咻、噗嗤），不要译成「好可爱」「舒服」这类评价词；"
    "同一个拟声词在不同句子可以有不同的中文写法，不要每句都用同一个词。"
)


def _builtin_prompt():
    return ("\n术语统一用：" + "、".join(BUILTIN_TERMS) + ONOMAT_HINT)


def load_glossary(stem):
    """读术语表，返回 [{"src","dst","info"}, ...]。

    ★ v1.30.6（借鉴 GalTransl 的 GPT 字典 / Sakura 的术语表格式）：
      一行三列用 **TAB** 分隔：`日文<TAB>中文<TAB>解释(可省)`
      解释里写角色设定（`name, girl, teacher`），模型才会选对「她/老师」。
      老格式（一行几个词、逗号分隔、无译文）仍然支持 —— 那种按
      「专有名词必须原样保留」处理。
    """
    terms = []
    seen = set()
    for name in (f"hotwords_{stem}.txt", "hotwords.txt"):
        p = os.path.join(".", name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8-sig") as f:
                for ln in f:
                    ln = ln.rstrip("\r\n")
                    if not ln.strip() or ln.lstrip().startswith("#"):
                        continue
                    if "\t" in ln:
                        cols = [c.strip() for c in ln.split("\t")]
                        src = cols[0]
                        dst = cols[1] if len(cols) > 1 else ""
                        info = cols[2] if len(cols) > 2 else ""
                        if src and src not in seen:
                            seen.add(src)
                            terms.append({"src": src, "dst": dst, "info": info})
                    else:
                        for w in ln.replace("，", ",").split(","):
                            w = w.strip()
                            if w and w not in seen:
                                seen.add(w)
                                terms.append({"src": w, "dst": "", "info": ""})
        except OSError:
            pass
    return terms


def _glossary_prompt(text, terms):
    """只把**当前这句日文里出现的**术语拼进 prompt（按句注入）。

    ★ 全局注入的问题：几百个词每次都塞，既吃 token 又干扰模型
      （无关的人名会诱发「擅自补主语」）。实测按句过滤后 prompt 短很多，
      而命中的词反而更受重视。
    """
    if not terms:
        return ""
    hit = []
    for t in terms:
        src = t.get("src") or ""
        if src and src in text:
            hit.append(t)
    if not hit:
        return ""
    pairs, plain = [], []
    for t in hit:
        src, dst, info = t["src"], t.get("dst") or "", t.get("info") or ""
        if dst:
            pairs.append(f"{src}->{dst}" + (f" #{info}" if info else ""))
        else:
            plain.append(src)
    out = ""
    if pairs:
        out += ("\n术语表（按对应关系翻译，保持译名一致）：\n"
                + "\n".join(pairs))
    if plain:
        out += ("\n专有名词（必须原样保留，不要意译）：" + "、".join(plain))
    return out


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def pick_device():
    """兼容模式：优先【非 NVIDIA】的 Vulkan 设备，NVIDIA 只作兜底；都不满足 → CPU。

    【为什么不吃 N 卡（2026-10-20 定）】
      用户原话：「我选了兼容模式，这里为什么还是 5070？兼容模式不应该是
      工作在显存不足 6G 的 N 卡、非 N 独显和核显吗？」。
      对：兼容模式的定位就是「不吃 N 卡」—— 显存留给别的 GPU 任务。
      原来的策略是「谁显存多用谁」，在有 5070 的机器上兼容模式照样把
      模型塞进 5070，和模式说明自相矛盾。

    【为什么还留着 NVIDIA 兜底】
      纯 N 卡且没有核显的机器（不少台式机）上，如果硬性排除 N 卡，
      就只剩 CPU，翻译会慢到没法用。所以顺序是：
        非 N 独显 → 核显 → N 卡（兜底）→ CPU

    显存安全空间（沿用 v1.21 规则）：
      · 1.8B 模型约 1.8 GB（权重 + 上下文），另留 1 GB 余量。
      · 核显共享系统内存，同样要求 ≥ 2.5 GB 空闲。
    """
    # 1.8B 模型约 1.8 GB（权重+上下文），再留约 1 GB 余量。
    DISCRETE_MIN_FREE_MIB = 2560
    INTEGRATED_MIN_FREE_MIB = 2560

    try:
        out = subprocess.run(
            [LLAMA, "--list-devices"],
            capture_output=True, text=True, timeout=60, cwd=".",
        )
        devices = []
        for line in out.stdout.splitlines():
            t = line.strip()
            if not t.startswith("Vulkan"):
                continue
            idx, rest = t.split(":", 1)
            name = rest.split("(")[0].strip()
            m = re.search(r"\((\d+) MiB,\s*(\d+) MiB free\)", rest)
            total = int(m.group(1)) if m else 0
            free = int(m.group(2)) if m else 0
            devices.append((idx, name, total, free))
    except Exception:
        devices = []

    if not devices:
        return "none", "CPU"

    # 核显 / 独显判定：先看名字里的强力信号，名字没信号再看显存大小。
    #
    #   【为什么不只看名字（千问评审 2026-10-20 指出）】
    #     纯关键词会漏判两类真实型号：
    #       · "AMD Radeon 780M"      —— 核显，但名字里既没 RX 也没 Graphics
    #       · "Intel Iris Xe MAX"    —— 独显，但命中 "iris" 会被当成核显
    #     所以加了白名单，并用显存兜底：核显报的是【共享内存】，
    #     total 常常上万 MiB（这台机器核显报 32147 MiB），独显是实打实的容量。
    #
    #   【为什么不直接用千问的「total < 8192 就是独显」】
    #     8 GB 内存的机器，核显可能只报 4~8 GB → 会被误判成独显。
    #     反过来「核显一定报得很大」也不成立。所以显存只在名字没命中时兜底。
    def integrated(name: str, total: int) -> bool:
        low = name.lower()
        # ① 独显强力信号（放在最前，先排除）
        if any(
            k in low
            for k in (
                "rx ", "rx(", "radeon pro", "arc a", "arc b", "arc(tm) a", "arc(tm) b",
                "xe max", "quadro", "geforce", "rtx", "tesla",
            )
        ):
            return False
        # ② 核显强力信号
        if any(
            k in low
            for k in (
                "uhd graphics", "iris", "vega", "radeon(tm) graphics", "radeon graphics",
                "780m", "760m", "740m", "680m", "660m",
            )
        ):
            return True
        # ③ 名字没信号 → 看显存：共享内存才可能报到 16 GB 以上
        return total > 16384

    def nvidia(name: str) -> bool:
        low = name.lower()
        return any(k in low for k in ("nvidia", "geforce", "rtx", "quadro", "tesla"))

    non_nvidia = [d for d in devices if not nvidia(d[1])]

    # 1) 非 N 卡独显（AMD / Intel Arc）—— 独显算力远高于核显，优先用
    #
    #    【顺序为什么是独显在前（2026-10-20 用户纠正）】
    #    用户原话：「如果是 AMD 独显或者 intel 独显这些，还是优先选用独显的」。
    #    之前把核显排第一，本意是「不跟 N 卡抢显存」，但在【AMD / Intel
    #    独显】的机器上就变成了「放着独显不用，让核显慢慢跑」，明显不对。
    #    所以只有「N 卡 vs 核显」的场合才偏向核显；同样是独显，
    #    非 N 的独显该排在核显前面。
    for idx, name, total, free in non_nvidia:
        if not integrated(name, total) and free >= DISCRETE_MIN_FREE_MIB:
            return idx, name
    # 2) 核显（非 N 卡里找）：共享内存够就用
    for idx, name, total, free in non_nvidia:
        if integrated(name, total) and free >= INTEGRATED_MIN_FREE_MIB:
            return idx, name
    # 3) 兜底：这台机器没有非 N 的 Vulkan 设备，只能借 N 卡（总比 CPU 快）
    #
    #    这里要打一行日志：走到这条分支时，界面上「兼容模式 + <N 卡型号>
    #    Vulkan 加速中」看起来很像 bug（用户会问「我不是选了兼容模式吗」），
    #    其实只是这台机器没有核显可用。日志里写清原因，省得以后反复排查。
    for idx, name, _total, free in devices:
        if free >= DISCRETE_MIN_FREE_MIB:
            print(
                f"[设备] 没有找到核显/非 N 卡 Vulkan 设备，"
                f"兼容模式借用 {name}（Vulkan）运行",
                flush=True,
            )
            return idx, name
    # 4) CPU
    print(
        "[设备] 没有可用的 Vulkan 设备（或空闲显存不足），兼容模式用 CPU 运行",
        flush=True,
    )
    return "none", "CPU"


def wait_health(port, timeout=120, proc=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        # llama-server 已经退出（设备不兼容 / 显存不够 / 老驱动的 Vulkan 起不来），
        # 不必再空等到 timeout：这里直接判失败，交给上层回退 CPU 重试。
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def translate(port, text, glossary, bump=0.0, ctx=None, pre_map=None):
    """翻译一句。★ v1.28.5：内部重试 3 次，绝不返回空。

    以前这里只请求一次：超时 / 返回空串就一路抛到上层，上层直接
    `continue` 丢掉 —— 用户看到的就是「有些话根本没翻译、整条不见了」。
    现在改成 3 次重试（间隔递增），仍然失败就抛异常，由上层汇总报错，
    不再悄悄留下一句日文或干脆删掉。

    ★ v1.30.4：新增「循环退化」重试（用户反馈「偶尔一两句不断重复」）。
    检到 _is_runaway 就换更高温度重来；3 次都退化，就取最短的那次
    再做收尾压缩（宁可短，也不要把同一句刷屏交给用户）。bump 用于
    上层发现「与上一句完全相同」时强制换温度。
    """
    last = "未知错误"
    best = ""
    # ★ v1.30.6（借鉴 GalTransl 译前字典）：送进模型前先把日文里的
    #   口齿不清/多写法统一掉；字幕里保留的仍是日文原文。
    send = apply_pre(text, pre_map) if pre_map else text
    for attempt in range(3):
        try:
            temp = min(0.9, 0.1 + bump + 0.2 * attempt)
            raw = _translate_once(port, send, glossary, temp=temp, ctx=ctx)
            if raw:
                # ★ v1.30.6 质量自检：
                #   kana = 原样吐回日文、empty = 空译文 —— 这两种必须重试，
                #   否则用户会看到一句日文或一段空白字幕。
                #   long（比原文长 2.2 倍）只当参考，不作废：中文偶尔确实
                #   比日文长（「ちょっと」→「稍微等一下嘛」），判死会误伤。
                bad = [i for i in _quality_issues(raw, text) if i in ("kana", "empty")]
                if bad:
                    last = "质量自检未过：" + ",".join(bad)
                    time.sleep(0.3)
                    continue
                # ★ 必须在【压缩前】判退化：压缩会把 300 字的刷屏压成 2 句，
                #   之后就看不出来了（v1.30.4）。
                if not _is_runaway(raw, text):
                    return _collapse_repeats(raw)
                last = "译文循环退化"
                if not best or len(raw) < len(best):
                    best = raw
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(0.4 * (attempt + 1))
    if best:
        print(f"  [退化] 3 次都重复，取最短一次并压缩：{best[:30]}…", flush=True)
        return _collapse_repeats(best, max_repeat=1)
    raise RuntimeError(last)


_PUNCT = " ~～、,，？?！!…。"


def _collapse_repeats(s, max_repeat=2):
    """收缩连续重复的同一片段（v1.30.3 引入，v1.30.4 加强）。

    ★ 真实问题（用户实测指出）：一句里出现「嘿嘿~ 嘿嘿~ 嘿嘿~ …」20 多次，
      以及「会一直在一起吗？会一直在一起吗？会一直在一起吗？」整句刷屏。
    根因有两层：ASR 侧日语原文本就重复（喘气/舔舐），翻译模型忠于原文
    逐个音节翻成中文 → 中文里重复膨胀；另有小模型的循环退化。

    字幕不需要把每个音节都写出来：写到 2 次就够，20 多次会把一行撑爆。

    v1.30.4 的两处修正：
      ① 单元上限 6 → 10：整句重复常常是 7~10 字的完整句子。
      ② 数重复时忽略尾部标点（「别说那种话啦，」×2 + 「别说那种话啦」）
         —— 旧逻辑要求一字不差，最后一条没有逗号就数不到 3 次，
         于是整句原样留着。现在按「去掉尾部标点后相同」来数。
    只认完全相同的核心串，不做模糊匹配，避免误伤正常语句。
    """
    if not s:
        return s

    def core(x):
        return x.rstrip(_PUNCT)

    out = s
    for unit_len in range(10, 0, -1):
        i = 0
        res = []
        n_out = len(out)
        while i < n_out:
            if i + unit_len > n_out:
                res.append(out[i:])
                break
            unit = out[i:i + unit_len]
            cu = core(unit)
            if not cu:
                res.append(unit)
                i += unit_len
                continue
            k = 1
            j = i + unit_len
            while j < n_out:
                if core(out[j:j + unit_len]) == cu:
                    k += 1
                    j += unit_len
                else:
                    break
            if k >= 3:
                sep = unit[len(cu):]
                if sep:
                    res.append(sep.join([cu] * max_repeat) + sep)
                else:
                    res.append(cu * max_repeat)
                i = j
            else:
                res.append(unit)
                i += unit_len
        out = "".join(res)
    return out.strip().rstrip("，、, ~～")


def _max_run(x, min_unit=4):
    """最长连续重复：(次数 k, 单元长度 u)。只看 ≥min_unit 字的单元，忽略尾部标点。"""
    n = len(x)
    best_k, best_u = 1, 0
    for u in range(min_unit, n // 2 + 1):
        i = 0
        while i + 2 * u <= n:
            blk = x[i:i + u].rstrip(" ~～、,，？?！!…。")
            k = 1
            while (i + (k + 1) * u <= n
                   and x[i + k * u:i + (k + 1) * u].rstrip(" ~～、,，？!…。") == blk):
                k += 1
            if k >= 2 and k > best_k:
                best_k, best_u = k, u
            i += k * u if k >= 2 else 1
    return best_k, best_u


def _is_runaway(zh, src=""):
    """判断译文是否「循环退化」——即刷屏式重复（v1.30.4-p2）。

    ★ A/B 实测教训（两次，共 324 条真实输出）：
      ① 第一版「只要重复就算退化」把 9/12 条**忠实翻译**误判了 —— 原文本来
         就是「だめ、だめ…」14 遍，中文照译「不行，不行…」是对的。
         而且 --repeat-penalty 1.1 前后都是 9/12，说明惩罚参数只能治真循环，
         治不了「原文就重复」→ 判据必须【相对原文】。
      ② 第二版用「译文/原文 ≥ N 倍」当闸口，但**原文一长比例就上不去**：
         实测用户截图那种「嘛..真的是的..真的是的..×28」（4 字单元）在
         原文 90 字时比例只有 1.96，两道闸都放行 → 漏判。
      ③ 现在改成**先看原文有没有重复**，不再靠比例赌：
         · 原文自己连刷 ≥3 次（だめ×12、ずっと一緒×3）→ 译文重复是忠实的，不判；
         · 原文不重复、译文却把某个 ≥4 字片段连刷 ≥3 次且占全句 ≥55% → 判退化。
      离线验证：312 条真实模型输出里只命中 1 条（「やめて×12 → 停×33」，
      确实该判），300 条 1027 真实台词 0 误判；9 条合成用例 9/9 判对。
      注意：即使不判退化，_collapse_repeats 也会把连续 ≥3 次压成 2 次，是最后兜底。
    """
    s = (zh or "").strip()
    n = len(s)
    if n < 12:
        return False
    sl = len((src or "").strip())
    # ① 长度失控（译文 > 原文 6 倍；日译中正常 0.5~1.2 倍）
    if sl >= 4 and n / sl >= 6.0:
        return True
    # ② 原文自己就在重复 → 译文的重复是忠实的，交给 _collapse_repeats 收敛
    if src and _max_run(src.strip())[0] >= 3:
        return False
    # ③ 原文不重复，译文却连刷 ≥3 次且占全句一半以上
    k, u = _max_run(s)
    if k >= 3 and k * u >= max(12, int(n * 0.55)):
        return True
    return False


def _sanitize_zh(s):
    """切掉译文的控制串尾巴（v1.30.2）。

    LoRA 的回合结束符是 hy 模板的私有 token，llama-server 的采样器
    不一定把它登记成停止词 → 偶发漏出「<|...」或全角变体「＜｜...」。
    日译中字幕里任何 < ／ ＜ 之后都不会是正文，从第一个出现处切掉。
    """
    s = (s or "").strip()
    cut = len(s)
    for mark in ("<", "＜"):
        k = s.find(mark)
        if 0 <= k < cut:
            cut = k
    return s[:cut].strip()


# ── v1.30.4：挂 LoRA 时必须用训练时的模板 ──
#
# 训练脚本 `_lora/train_lora.py` 里是手写 hy 模板：
#   BOS + SYSTEM + <|hy_place▁holder▁no▁3｜> + <|hy_User｜> + 日文 + <|hy_Assistant｜>
# 而 llama-server 的 /v1/chat/completions 会套它自己那份 chat 模板——两者
# 对不上时，挂 LoRA 的模型会吐 `<|<|<|…` 占位符刷屏（实测每句 320 token、
# 10.8 秒，全是垃圾）。改成直接发 /completion 原始 prompt 就正常了：
# 每句 0.24~0.68 秒、24 tok/s，与不挂 LoRA 持平。
_HY_BOS = "<|hy_begin▁of▁sentence｜>"
_HY_SYS_END = "<|hy_place▁holder▁no▁3｜>"
_HY_TURN_END = "<|hy_place▁holder▁no▁2｜>"
_HY_USER = "<|hy_User｜>"
_HY_ASSIST = "<|hy_Assistant｜>"

# ★ v1.30.6（借鉴 GalTransl multiTurn / Sakura 上下文）：把前 N 句的
#   「日文 → 中文」当成对话历史一起喂进去，模型才知道「他/她」指谁、
#   省略的主语是什么。设 ASMR_TRANSLATE_CTX=0 可关闭。
# ★ 默认 0（关闭）：2026-11-21 实测，注入上文会让 system 每句都变，
#   llama.cpp 的前缀缓存全废 → 翻译从 80.9 段/分掉到 25.3 段/分（3.2 倍慢），
#   而 sim 只差 -0.0075（噪声级）。人称确实有修正（「我也没有拒绝权」→
#   「你也没有拒绝权」），但代价太大。想要可以设 ASMR_TRANSLATE_CTX=8 打开。
_CTX_N = int(os.environ.get("ASMR_TRANSLATE_CTX", "0") or 0)
# 上下文注入方式：
#   "hint"（默认）= 把上一句「日文 → 中文」写进 system，仍是单轮 —— 实测
#           多轮对话格式会让这个 1.8B + LoRA 严重跑偏（40 句 A/B：
#           sim 0.4653→0.4040，per-句 11 好 25 差，还慢 2.5 倍）。
#           原因是 LoRA 只在单轮格式上训过，没见过多轮历史。
#   "turns" = 拼成多轮对话（GalTransl 的做法），本模型不适用，留作实验。
_CTX_STYLE = os.environ.get("ASMR_TRANSLATE_CTX_STYLE", "hint").strip().lower()

# ★ v1.30.6（借鉴 GalTransl 自动化找错）：译文质量自检。
#   1) 残留日文：译文里还有平假名/片假名 = 这句根本没翻
#      （实测 Hy-MT2 偶尔原样吐回日文，且非空，旧逻辑查不出来）
#   2) 超长：中文一般比日文短，译文比原文长 2.2 倍以上多半是
#      模型开始自我解释 / 把生词注释也吐出来了
_KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")


def _quality_issues(zh, ja):
    """返回问题代码列表；空列表表示通过。"""
    out = []
    core = (zh or "").strip(_PUNCT + " \t")
    if not core:
        out.append("empty")
    if _KANA.search(zh or ""):
        out.append("kana")
    if ja and core and len(core) > max(12, len(ja) * 2.2):
        out.append("long")
    return out


def _raw_completion(port, system, text, temp, ctx=None):
    """按训练时的 hy 模板发原始 prompt（不带 chat 模板）。

    ★ v1.30.6：ctx 是前 N 句的 (日文, 中文) 列表，按训练时的多轮格式
    拼在 system 之后 —— 这就是「上下文翻译」，让它知道前文在说什么。
    """
    parts = [f"{_HY_BOS}{system}{_HY_SYS_END}"]
    for cja, czh in ((ctx or ()) if _CTX_STYLE == "turns" else ()):
        parts.append(f"{_HY_USER}{cja}{_HY_ASSIST}{czh}{_HY_TURN_END}")
    parts.append(f"{_HY_USER}{text}{_HY_ASSIST}")
    prompt = "".join(parts)
    payload = {
        "prompt": prompt,
        "temperature": temp,
        # ★ v1.28.5：128 对长句会被硬截断（译文半截但不为空，更隐蔽）
        "n_predict": 320,
        "stop": [_HY_TURN_END, "<|hy_", "＜｜"],
        "stream": False,
        # ★ 同一条 system 前缀会被 llama.cpp 缓存，实测省掉一半时间
        #   （4.46 s → 2.19 s / 5 句）
        "cache_prompt": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return d.get("content", "")


def _translate_once(port, text, glossary, temp=0.1, ctx=None):
    system = SYSTEM + _builtin_prompt()
    # ★ v1.30.6：术语按句注入（只注入这句话里出现的词）
    system += _glossary_prompt(text, glossary)
    # ★ v1.30.6：上下文参考（默认只给上一句，单轮格式，实测比多轮稳）
    if ctx and _CTX_STYLE != "turns":
        ja0, zh0 = ctx[-1]
        if ja0 and zh0:
            system += ("\n上一句（仅供理解上下文，不要重复它的内容）："
                       + ja0[:60] + " → " + zh0[:60])
    if LORA_PATH:
        # 挂 LoRA：走手写 hy 模板（否则输出退化，见上面的说明）
        return _sanitize_zh(_raw_completion(port, system, text, temp, ctx=ctx))
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": temp,
        # ★ v1.28.5：128 对长句会被硬截断（译文半截但不为空，更隐蔽）
        "max_tokens": 320,
        "stream": False,
    }
    if LORA_PATH:
        # 双保险：让采样器在私有模板 token 前就停（server 认字面串）
        payload["stop"] = ["<|hy_", "＜｜"]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return _sanitize_zh(d["choices"][0]["message"]["content"])


def fmt_time(t):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# 双重兜底：旧版 light_asr 生成的 JSON 里可能已有 SenseVoice 事件/情绪
# emoji；模型也可能原样照抄。最终 `.chs.srt` / `.jpn.srt` 一律不保留。
_SENSEVOICE_MARKS = str.maketrans("", "", "🎼👏😀😭🤧😷😊😔😡😰🤢😮❓")


def _keep_marks(jp, zh):
    """按原文补回译文丢掉的情绪符号（…♡♪♥）。

    为什么需要（v1.30.6 实测，145,423 条 ASMR 语料统计）：
      · 76.8% 的参考译文含省略号「…」（111,661 次），是 ASMR 最主要的停顿标记
      · 句尾最高频的不是「吧/了」，而是「♡」（13,310 次，超过「吧」12,408）
      · 但模型**完全学不会**这两个符号：120 条 val 上参考含 ♡ 14%、♪ 11%，
        模型输出 0~1%；省略号靠 prompt 硬要求能到 20%，但参考只 11% —— 加多了。
    所以不让模型学，改成后处理按原文补：原文有几个就补几个，不新增。

    规则：
      · 「…」按原文计数补到句尾（原文 2 个、译文 1 个，补 1 个）
      · 「♡」「♪」「♥」原文有、译文没有 → 补到句尾
      · 只在句尾补，不往句子中间插（避免破坏语序）
    """
    if not jp or not zh:
        return zh

    # ── 省略号：原文计数（兼容「…」「...」「‥」）──────────────
    n_jp = jp.count("…") + jp.count("...") + jp.count("•••")
    if n_jp <= 0 and ("‥" in jp or "・・" in jp):
        n_jp = jp.count("‥") + jp.count("・・")
    n_zh = zh.count("…")
    if n_jp > n_zh:
        # 去掉句尾已有的标点再补（「好的。」+「…」→「好的…」）
        base = zh.rstrip()
        while base and base[-1] in "。．.!！?？、,，":
            base = base[:-1]
        zh = base + "…" * (n_jp - n_zh)

    # ── 爱心/音符：原文有、译文没有才补（不重复）────────────
    tail = zh.rstrip()
    for ch in ("♡", "♥", "♪"):
        if ch in jp and ch not in zh:
            # 句尾是句号之类就先剥掉
            while tail and tail[-1] in "。．.!！?？、,，":
                tail = tail[:-1]
            tail = tail + ch
    return tail if tail != zh.rstrip() else zh


def _one_line(s):
    """清掉非台词标记并压成单行。

    SRT 一条字幕里除了我们故意加的那一个换行，不能再有换行，否则字幕块
    会被切断（模型偶尔会吐多行、日文里也可能带换行）。
    """
    return " ".join(str(s).translate(_SENSEVOICE_MARKS).split())


def tidy_rows(rows):
    """时长上限 + 不压到下一句 + 最少 0.4 秒（与 e_字幕整理.py 一致）。"""
    out = []
    for i, r in enumerate(rows):
        a = float(r["start"])
        b = float(r["end"])
        # 行内双语是按两行的字数撑时长（日文那一行也要读完）
        n = len(r.get("zh", ""))
        if JPN_INLINE:
            n = max(n, len(r.get("jp", "")))
        # ★ v1.28.3：原来 0.9n+0.8 —— 数的是【中文字数】，而语音长度由日文
        #   音节决定，中文普遍更短 → 字幕收得太早（「话还没说完就消失」）。
        #   放宽到 1.8n+3.5，并且上下文已有「不压到下一句」的兜底。
        cap = a + max(2.5, 1.8 * n + 3.5)
        b = min(b, cap)
        # ★ v1.28.4：不再无条件压到下一句开始。只要离得开，就保底显示
        #   1.2 秒（人眼读完一行中文至少要这么久），不够才压。
        if i + 1 < len(rows):
            nxt = float(rows[i + 1]["start"])
            if nxt > a:
                b = min(b, nxt)
                b = max(b, min(a + 1.2, nxt))
        b = max(b, a + 1.0)
        out.append({**r, "start": a, "end": b})
    return out


# ── 测速结论与界面的握手 ────────────────────────────────────────────
#
# 界面弹窗问「要不要换成纯 CPU」时，脚本必须**真的停下来等** ——
# 否则弹窗还没等人点，翻译已经用核显跑起来了，用户的选择等于没用。
# 握手方式：界面把结论写进 {数据目录}\_bench_choice.txt（cpu / vulkan），
# 脚本每 200ms 看一次文件；超时就自己选更快的那个。
BENCH_CHOICE_FILE = "_bench_choice.txt"
# 等待秒数必须**长于**界面的倒计时（main.rs 的 BENCH_WAIT_SEC = 15 秒）。
# 两边一样长时实测会打平：脚本先判超时，界面还没把结论写下来，
# 用户点什么都来不及生效。留 7 秒余量给「界面决定 → 脚本读取」这段链路。
BENCH_WAIT_DEFAULT = 22.0


def _bench_choice_path():
    return os.path.join(os.getcwd(), BENCH_CHOICE_FILE)


def _read_bench_wait():
    """等待用户决定的秒数（默认 20，环境变量 ASMR_BENCH_WAIT 可改）。"""
    try:
        v = float((os.environ.get("ASMR_BENCH_WAIT", "") or "22").strip())
    except ValueError:
        v = BENCH_WAIT_DEFAULT
    return max(0.0, min(180.0, v))


def _reset_bench_choice():
    """开新一轮测速前清掉上一轮的结论文件（必须在 emit 之前调）。"""
    try:
        os.remove(_bench_choice_path())
    except OSError:
        pass


def _await_bench_choice(wait_sec):
    """等界面把用户的选择写进 _bench_choice.txt。

    返回 "cpu" / "vulkan" / ""（超时或没等）。等待期间每 5 秒在日志里
    提示还剩多久，让用户知道软件在等他而不是卡死了。
    """
    path = _bench_choice_path()
    if wait_sec <= 0:
        return ""
    t0 = time.time()
    _next_note = 5.0
    while True:
        elapsed = time.time() - t0
        if elapsed >= wait_sec:
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                v = (f.read() or "").strip().lower()
        except OSError:
            v = ""
        if v.split(":")[0] in ("cpu", "vulkan"):
            return v          # 形如 cpu / cpu:auto（超时兜底）/ vulkan
        if elapsed >= _next_note:
            print("[测速] 等待你的选择…还剩 %d 秒（不选就自动用更快的那个）"
                  % int(wait_sec - elapsed), flush=True)
            _next_note += 5.0
        time.sleep(0.2)


def main():
    # ★ 只取第2步（light_asr.py）产出的 <名>.json。
    #   同一个文件如果先跑过全量模式，2vad\ 里还会留着 pyannote 的
    #   <名>.vad.json —— 它的结构是 {"segments": [[a,b],...]}，
    #   不是 [{start,end,text}]。原来一起 glob 进来后会先遍历到字典的
    #   键名 'segments'，然后 r.get("text") 抛
    #   AttributeError: 'str' object has no attribute 'get'。
    files = [f for f in sorted(glob.glob(os.path.join(VAD_DIR, "*.json")))
             if not f.endswith(".vad.json")]
    if not files:
        print(f"在 {VAD_DIR} 中没有找到转写结果（先跑第2步）")
        sys.exit(1)
    if not os.path.isfile(LLAMA) or not os.path.isfile(MODEL):
        print(f"[失败] 缺少兼容模式组件：{LLAMA} 或 {MODEL}")
        sys.exit(1)
    print(f"[模型] 兼容模式翻译用：{os.path.basename(MODEL)}")

    port = free_port()

    def start_server(device, use_lora):
        cmd = [LLAMA, "--model", MODEL, "--host", "127.0.0.1", "--port", str(port),
               "--device", device, "-c", "4096", "-np", "1", "--no-webui",
               # ★ v1.30.3：开 /metrics，界面靠它显示实时 tok/s。
               #   安卓版就是这么做的（--metrics + Prometheus 文本解析），
               #   用户能立刻看出「模型是不是在干活」，而不是盯着进度条猜。
               "--metrics",
               # ★ v1.30.6：反重复从 repeat-penalty 换成 DRY（实测 repeat-penalty 无效）。
               #
               #   依据（2026-11-23 在评测机上用同一模型、6 条退化用例实测）：
               #     repeat-penalty 1.1  短句×12 → 214 字（反而更长）、长原文×20 → 240 字（没动）
               #     DRY 0.8              同 6 条全部压到 16~35 字，正常句输出一字未变
               #   原因：repeat-penalty 惩罚「token 出现过」，模型刷屏时在变花样
               #   （停下 → 停下吧 → 停下呀），每次都是新组合，抓不住；
               #   DRY 惩罚 n-gram 序列，检测到「停下，停下」这种重复才压。
               #   质量影响（120 条 val 实测）：f1 完全相同，sim 差 0.0015（噪声级）。
               #   allowed-length 2：允许「嘿嘿」「嗯嗯」这种两字内的正常重复。
               "--dry-multiplier", "0.8",
               "--dry-base", "1.75",
               "--dry-allowed-length", "2",
               "--dry-penalty-last-n", "64",
               "--log-verbosity", "0"]
        if use_lora and LORA_PATH:
            # 加载失败宁可起不来走下面的 CPU 回退，也不静默丢增强
            cmd += ["--lora", LORA_PATH]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=".",
            # CREATE_NO_WINDOW：否则翻译时会在屏幕上闪一个黑框
            creationflags=0x08000000,
        )

    dev, dev_name = pick_device()

    # ★ v1.30.3：开跑前测速（纯 CPU vs Vulkan），选更快的那个。
    #
    #   起因（用户实测指出「感觉比之前慢」）：1.8B 这种小模型在核显上
    #   反而更慢 —— 每算一段都要把模型从内存搬过去，搬运开销超过核显
    #   省下的时间。实测本机（Ryzen 7 7700 + Radeon 核显）：
    #     纯 CPU  1388 ms/句
    #     核显    2752 ms/句   ← 慢 2 倍
    #   用内置的三句日常日语做基准（不读用户素材），6~15 秒完成，
    #   相对整批任务可忽略，但可能省掉一半以上时间。
    #
    #   ASMR_NO_BENCH=1 可跳过测速（调试/不想等的时候用）。
    if dev != "none" and os.environ.get("ASMR_NO_BENCH", "").strip() != "1":
        try:
            from bench_device import bench_devices, make_prompt
            print("[测速] 正在比较纯 CPU 与核显的速度…", flush=True)
            _r = bench_devices(LLAMA, MODEL, LORA_PATH, None)
            _prefer = _r.get("prefer") or ""
            _need = bool(_r.get("need_prompt"))
            _reset_bench_choice()
            emit("bench", _prefer,
                 f"{_r.get('cpu_ms') or 0:.0f}",
                 f"{_r.get('vulkan_ms') or 0:.0f}",
                 "1" if _need else "0",
                 _r.get("advice", ""))
            print(f"[测速] {_r.get('advice', '')}", flush=True)
            # ★ 需要打扰用户时：原地点等，等他自己点，或者等到超时。
            #   llama-server 还没启动，所以这里等不影响任何已完成的活。
            if _need:
                _wait = _read_bench_wait()
                if _wait > 0:
                    print(f"[测速] 等待界面上的选择（最多 {int(_wait)} 秒）…",
                          flush=True)
                _choice = _await_bench_choice(_wait)
                _picked = _choice.split(":")[0]
                _by_ui = bool(_choice) and not _choice.endswith(":auto")
                if _picked == "cpu":
                    dev, dev_name = "none", "CPU"
                    print("[测速] " + ("已按你的选择改用纯 CPU，本次直接生效"
                                        if _by_ui else
                                        "等待超时，自动改用更快的纯 CPU"),
                          flush=True)
                elif _picked == "vulkan":
                    print("[测速] " + ("已按你的选择继续用核显"
                                        if _by_ui else
                                        "等待超时，自动采用更快的核显"),
                          flush=True)
                elif _prefer == "cpu":
                    dev, dev_name = "none", "CPU"
                    print("[测速] 等待超时，自动改用更快的纯 CPU", flush=True)
                else:
                    print("[测速] 等待超时，自动采用更快的核显", flush=True)
        except Exception as _e:
            print(f"[测速] 跳过（{type(_e).__name__}: {_e}）", flush=True)

    use_lora = bool(LORA_PATH)
    if LORA_PATH:
        print(f"[LoRA] 翻译增强适配器：{'启用（' + dev_name + '）' if use_lora else '本次不启用（ASMR_TRANSLATE_LORA=off）'}",
              flush=True)
    print(f"[设备] 翻译使用：{dev_name}（{dev}）", flush=True)
    print(f"启动内置 llama.cpp（端口 {port}）...", flush=True)
    proc = start_server(dev, use_lora)
    try:
        if not wait_health(port, proc=proc):
            # 选中设备加载失败（显存不够 / OOM / 驱动问题）→ 回退 CPU 重试一次。
            # 千问评审指出：--list-devices 的 free 是瞬时值，不能保证加载时还够。
            print("[回退] 该设备加载失败，改用 CPU 重试", flush=True)
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            dev, dev_name = "none", "CPU"
            # ★ v1.30.4：CPU 也挂 LoRA。2026-11-18 实测（llama-server
            #   --device none + 手写 hy 模板 + cache_prompt）：挂与不挂
            #   都是 ~24 tok/s，每句 0.24~0.68 s，没有额外代价。
            #   旧注释「慢 2.4 倍」是把 Vulkan / CPU 的设备测速混进来了。
            use_lora = bool(LORA_PATH)
            proc = start_server("none", False)
            if not wait_health(port, proc=proc):
                print("[失败] llama-server 启动超时")
                sys.exit(1)
        print("llama-server 就绪", flush=True)

        # ★ v1.30.3：状态采集器（模型吞吐 / GPU 或 CPU 指标）。
        #   必须传【实际推理设备】—— 跑在独显上就显示显卡 SM 频率，
        #   跑在核显上显示核显利用率，纯 CPU 才显示 CPU 频率。
        #   一律显示 CPU 频率是错的：CPU 5.3GHz 满频但核显在慢慢算，
        #   用户会被误导。
        _probe = None
        if StatusProbe is not None:
            try:
                _lpid = find_llama_pid() if find_llama_pid else None
                _probe = StatusProbe(port, dev, _lpid)
                print(f"[状态] 采集已启用（设备={dev} llama pid={_lpid}）",
                      flush=True)
            except Exception as _e:
                print(f"[状态] 采集不可用（{type(_e).__name__}）", flush=True)
        print(
            "[字幕] 日文字幕："
            + (
                "单独成文件（额外写一份 <名字>.jpn.srt）"
                if JPN_SEPARATE
                else "并进中文（每条字幕两行：日文在上、中文在下）"
                if JPN_SUB
                else "关闭（只写中文）"
            ),
            flush=True,
        )

        os.makedirs(ASR_DIR, exist_ok=True)
        total = len(files)
        ok = 0
        total_failed = 0
        t_start = time.time()
        for idx, fp in enumerate(files, 1):
            stem = os.path.splitext(os.path.basename(fp))[0]
            rows = json.load(open(fp, encoding="utf-8"))
            glossary = load_glossary(stem)
            # ★ v1.30.4：错译 → 标准译 的替换表（可缺省，缺省时什么都不改）
            fixes = load_fixes(stem)
            print(f"\n翻译: {os.path.basename(fp)}（{len(rows)} 段）", flush=True)

            done = []
            failed = []
            # ★ v1.30.3：段进度（与 light_asr 同口径，借鉴安卓版设计）。
            #   翻译是最慢的一步（每段约 1~2 秒 × 几百段），界面只有
            #   「第3步」时用户完全不知道还要多久。这里每 20 段打点：
            #   已译/总数 + 每分钟几段 + 预计剩余。
            _n_all = len([1 for r in rows if (r.get("text") or "").strip()])
            _t_tr0 = time.time()
            _n_done = 0
            _prev_zh, _prev_src = "", ""
            # ★ v1.30.6：上下文窗口（前 N 句的 ja/zh 对）
            _ctx = []
            for r in rows:
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                # ★ v1.28.5：失败绝不写日文原文，也绝不静默丢条。
                #   内部已重试 3 次；这一层再收集起来整批复试一次。
                #   仍然失败的会在末尾明确列出并让本步退出码非 0，
                #   用户能一眼看出「这次不全」，而不是拿到一份混着日文的字幕。
                try:
                    zh = translate(port, text, glossary, ctx=_ctx, pre_map=fixes)
                except Exception as e:
                    failed.append({"start": r["start"], "end": r["end"], "jp": text, "err": str(e)})
                    print(f"  [失败] {type(e).__name__}: {e}", flush=True)
                    continue
                # ★ v1.30.4：与上一句一字不差、但日文原文不同 → 多半是
                #   循环退化的产物。换更高温度重译一次；仍然相同就保留
                #   （有些段落原文确实该译成同一句）。
                if len(zh) >= 10 and zh == _prev_zh and text != _prev_src:
                    try:
                        zh2 = translate(port, text, glossary, bump=0.6, ctx=_ctx, pre_map=fixes)
                        if zh2 and zh2 != _prev_zh:
                            print(f"  [去重] 与上一句相同，换温度重译 → {zh2[:30]}", flush=True)
                            zh = zh2
                    except Exception:
                        pass
                done.append({"start": r["start"], "end": r["end"],
                             "zh": _keep_marks(text, zh), "jp": text})
                _prev_zh, _prev_src = zh, text
                if _CTX_N > 0:
                    _ctx.append((text, zh))
                    if len(_ctx) > _CTX_N:
                        _ctx.pop(0)
                print(f"[{float(r['start']):.2f}s -> {float(r['end']):.2f}s]: {zh}", flush=True)
                # ★ 段进度：每 20 段或最后一段
                _n_done += 1
                if _n_done % 20 == 0 or _n_done == _n_all:
                    _el = time.time() - _t_tr0
                    _spm = _n_done / max(0.01, _el) * 60.0
                    _left = (_n_all - _n_done) / max(0.001, _spm) if _spm > 0 else 0
                    # ★ 状态采集：模型 tok/s + CPU 占用 + 实时频率。
                    #   与段进度同频（每 20 段），避免日志刷屏。
                    _st = _probe.poll() if _probe else ""
                    print(f"    翻译 {_n_done}/{_n_all} 段"
                          f"（{_n_done / max(1, _n_all) * 100:.0f}%）"
                          f"  {_spm:.1f} 段/分  剩约 {_left:.0f} 分"
                          + (f"  · {_st}" if _st else ""), flush=True)
                    emit("segs", _n_done, _n_all, round(_spm, 2), int(_left))
                    if _st:
                        emit("stat", _st)

            # 整批复试：刚才多半是 llama-server 瞬时卡住，歇 2 秒再来一轮。
            if failed:
                print(f"  [重试] {len(failed)} 句失败，2 秒后整批复试…", flush=True)
                time.sleep(2.0)
                still = []
                for it in failed:
                    try:
                        zh = translate(port, it["jp"], glossary)
                        done.append({"start": it["start"], "end": it["end"], "zh": zh, "jp": it["jp"]})
                        print(f"[{float(it['start']):.2f}s 重试成功]: {zh}", flush=True)
                    except Exception as e:
                        still.append((it, str(e)))
                failed = still
                done.sort(key=lambda x: float(x["start"]))
            total_failed += len(failed)
            if failed:
                print(f"  [未翻译] {len(failed)} 句（本文件字幕不完整）：", flush=True)
                for it, e in failed[:10]:
                    print(f"    - {it['jp'][:40]}  ({e})", flush=True)

            tidy = tidy_rows(done)
            lines = []
            for i, r in enumerate(tidy, 1):
                zh = _one_line(r["zh"])
                # ★ v1.30.4：术语后处理放在最后一步（压缩/去重之后），
                #   替换表里的写法是「用户看到的标准写法」，不该被前面的
                #   去重逻辑再动一遍。
                zh = apply_fixes(zh, fixes)
                if JPN_INLINE:
                    # 一条字幕两行：日文原文在上、中文在下。两条都用 \n 分隔，
                    # 中间不能再有空行（空行会切断 SRT 块）。
                    jp = _one_line(r.get("jp", ""))
                    body = f"{jp}\n{zh}" if jp else zh
                else:
                    body = zh
                lines.append(
                    f"{i}\n{fmt_time(r['start'])} --> {fmt_time(r['end'])}\n{body}\n"
                )
            out = os.path.join(ASR_DIR, stem + ".srt")
            with open(out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"  写入 {len(lines)} 条 -> {out}", flush=True)

            if JPN_SUB and JPN_SEPARATE:
                # 额外一份纯日文，时间轴与中文完全一致。
                #   文件名必须带 `.jpn` 语言代码，播放器才会当成「另一种语言的字幕」
                #   自动加载；写成 `<名字>.jp.srt` 播放器认不出（`.jp` 是国家码，
                #   不是语言码），写成 `<名字>.srt` 又会和某种播放器的默认字幕抢位置。
                jp_lines = []
                for r in tidy:
                    jp = _one_line(r.get("jp", ""))
                    if not jp:
                        continue          # 没识别出日文的段跳过（编号要连续，见下）
                    jp_lines.append(
                        f"{len(jp_lines) + 1}\n"
                        f"{fmt_time(r['start'])} --> {fmt_time(r['end'])}\n{jp}\n"
                    )
                jp_out = os.path.join(ASR_DIR, stem + ".jpn.srt")
                with open(jp_out, "w", encoding="utf-8") as f:
                    f.write("\n".join(jp_lines))
                print(f"  写入 {len(jp_lines)} 条日文字幕 -> {jp_out}", flush=True)
            ok += 1

            pct = idx * 100.0 / total
            bar = "#" * int(pct / 5)
            el = int(time.time() - t_start)
            print(f"  {pct:3.0f}%|{bar:<20}| {idx}/{total} [{el // 60:02d}:{el % 60:02d}<00:00]", flush=True)

        print(f"\n兼容模式翻译完成：成功 {ok} 个")
        if total_failed:
            # ★ v1.28.5：有句子没翻译出来就让本步失败 —— 以前一律 exit 0，
            #   界面显示「完成」，用户却在片子里找不到那几句话。
            print(f"[失败] 共 {total_failed} 句未能翻译，字幕不完整")
            sys.exit(1)
        if ok == 0:
            sys.exit(1)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
