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
    terms = []
    for name in (f"hotwords_{stem}.txt", "hotwords.txt"):
        p = os.path.join(".", name)
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
                        if w and w not in terms:
                            terms.append(w)
        except OSError:
            pass
    return terms


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


def translate(port, text, glossary):
    """翻译一句。★ v1.28.5：内部重试 3 次，绝不返回空。

    以前这里只请求一次：超时 / 返回空串就一路抛到上层，上层直接
    `continue` 丢掉 —— 用户看到的就是「有些话根本没翻译、整条不见了」。
    现在改成 3 次重试（间隔递增），仍然失败就抛异常，由上层汇总报错，
    不再悄悄留下一句日文或干脆删掉。
    """
    last = "未知错误"
    for attempt in range(3):
        try:
            zh = _translate_once(port, text, glossary, temp=0.1 if attempt == 0 else 0.3)
            if zh:
                return zh
            last = "模型返回空"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(last)


def _collapse_repeats(s, max_repeat=2):
    """收缩连续重复的同一片段（v1.30.3）。

    ★ 真实问题（用户实测指出）：一句里出现「嘿嘿~ 嘿嘿~ 嘿嘿~ …」20 多次。
    根因在 ASR 侧：日语原文本就是 `れろれろれろ`、`うんうんうん`、
    `おいおいおい` 这种真实重复（喘气/舔舐的声音就是这样），
    翻译模型忠于原文，逐个音节翻成中文 → 中文里重复膨胀。

    但字幕不需要把每个音节都写出来：
      · 观众看得懂「嘿嘿~ 嘿嘿~」是连续的，写到 2~3 次就足够了
      · 20 多次会把一行字幕撑爆，播放器上根本读不完

    做法：找出长度 1~6 的「同一单元连续重复 ≥3 次」的片段，
    收缩到 max_repeat 次（保留感叹性质，又不刷屏）。
    注意只用【完全相同】的重复，不做模糊匹配，避免误伤正常语句。
    """
    if not s:
        return s
    out = s
    # 从长单元往短单元找，避免「嘿嘿嘿」先被单字规则切坏
    for unit_len in range(6, 0, -1):
        i = 0
        res = []
        while i < len(out):
            unit = out[i:i + unit_len]
            if len(unit) < unit_len:
                res.append(out[i:])
                break
            # 数这一单元连续出现多少次
            n = 1
            j = i + unit_len
            while out[j:j + unit_len] == unit:
                n += 1
                j += unit_len
            if n >= 3:
                res.append(unit * max_repeat)
                # 吃掉重复之间可能的空格/波浪号（「嘿嘿~ 嘿嘿~」）
                i = j
                while i < len(out) and out[i] in " ~～、,，":
                    i += 1
            else:
                res.append(unit)
                i += unit_len
        out = "".join(res)
    return out.strip()


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


def _translate_once(port, text, glossary, temp=0.1):
    system = SYSTEM + _builtin_prompt()
    if glossary:
        system += "\n专有名词（必须原样保留，不要意译）：" + "、".join(glossary)
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
    zh = _sanitize_zh(d["choices"][0]["message"]["content"])
    return _collapse_repeats(zh)


def fmt_time(t):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# 双重兜底：旧版 light_asr 生成的 JSON 里可能已有 SenseVoice 事件/情绪
# emoji；模型也可能原样照抄。最终 `.chs.srt` / `.jpn.srt` 一律不保留。
_SENSEVOICE_MARKS = str.maketrans("", "", "🎼👏😀😭🤧😷😊😔😡😰🤢😮❓")


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

    use_lora = bool(LORA_PATH) and dev != "none"
    if LORA_PATH:
        print(f"[LoRA] 翻译增强适配器：{'启用（' + dev_name + '）' if use_lora else '已有但本次不启用（CPU 太慢）'}",
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
            use_lora = False   # CPU 回退不挂 LoRA（慢 2.4 倍）
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
            for r in rows:
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                # ★ v1.28.5：失败绝不写日文原文，也绝不静默丢条。
                #   内部已重试 3 次；这一层再收集起来整批复试一次。
                #   仍然失败的会在末尾明确列出并让本步退出码非 0，
                #   用户能一眼看出「这次不全」，而不是拿到一份混着日文的字幕。
                try:
                    zh = translate(port, text, glossary)
                except Exception as e:
                    failed.append({"start": r["start"], "end": r["end"], "jp": text, "err": str(e)})
                    print(f"  [失败] {type(e).__name__}: {e}", flush=True)
                    continue
                done.append({"start": r["start"], "end": r["end"], "zh": zh, "jp": text})
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
