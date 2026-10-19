"""兼容模式 第3步：用内置 llama.cpp（Qwen2.5-1.5B）翻译。

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

VAD_DIR = "2vad"
ASR_DIR = "3asr"
LLAMA = os.path.join("llama", "llama-server.exe")
MODEL = os.path.join("model", "qwen2.5-1.5b-q4km.gguf")

SYSTEM = (
    "你是日译中字幕翻译。把用户给的日语翻译成简体中文，口语化、句子短促，"
    "只输出中文，不要保留日文，不要解释。\n"
    "示例：\n"
    "日语：今日はいい天気ですね\n中文：今天天气真不错呢\n"
    "日语：ちょっと待ってください\n中文：请稍等一下\n"
    "日语：そろそろ帰りましょうか\n中文：差不多该回去了吧"
)


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
    """优先独显（但要留足显存安全空间），显存不够就退核显，最后 CPU。

    安全空间规则（v1.21）：
      · 独显：空闲显存 ≥ 3 GB 才用。1.5B 模型约 1.5 GB（权重 + 上下文），
        留 1.5 GB 余量，避免把用户其他程序挤掉（显存不足会触发驱动重置）。
      · 核显：共享系统内存，要求 ≥ 3 GB 空闲。
      · 都不满足 → CPU。
    """
    # 1.5B 模型约 1.5 GB（权重+上下文），再留约 1 GB 余量。
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

    def integrated(name: str) -> bool:
        low = name.lower()
        return any(k in low for k in ("uhd graphics", "iris", "vega", "radeon(tm) graphics"))

    # 1) 独显：空闲显存够才用
    for idx, name, _total, free in devices:
        if not integrated(name) and free >= DISCRETE_MIN_FREE_MIB:
            return idx, name
    # 2) 核显：共享内存够才用
    for idx, name, _total, free in devices:
        if integrated(name) and free >= INTEGRATED_MIN_FREE_MIB:
            return idx, name
    # 3) CPU
    return "none", "CPU"


def wait_health(port, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def translate(port, text, glossary):
    system = SYSTEM
    if glossary:
        system += "\n专有名词（必须原样保留，不要意译）：" + "、".join(glossary)
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"].strip()


def fmt_time(t):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def tidy_rows(rows):
    """时长上限 + 不压到下一句 + 最少 0.4 秒（与 e_字幕整理.py 一致）。"""
    out = []
    for i, r in enumerate(rows):
        a = float(r["start"])
        b = float(r["end"])
        n = len(r.get("zh", ""))
        cap = a + max(1.0, 0.9 * n + 0.8)
        b = min(b, cap)
        if i + 1 < len(rows):
            nxt = float(rows[i + 1]["start"])
            if nxt > a:
                b = min(b, nxt)
        b = max(b, a + 0.4)
        out.append({**r, "start": a, "end": b})
    return out


def main():
    files = sorted(glob.glob(os.path.join(VAD_DIR, "*.json")))
    if not files:
        print(f"在 {VAD_DIR} 中没有找到转写结果（先跑第2步）")
        sys.exit(1)
    if not os.path.isfile(LLAMA) or not os.path.isfile(MODEL):
        print(f"[失败] 缺少兼容模式组件：{LLAMA} 或 {MODEL}")
        sys.exit(1)

    port = free_port()

    def start_server(device):
        return subprocess.Popen(
            [LLAMA, "--model", MODEL, "--host", "127.0.0.1", "--port", str(port),
             "--device", device, "-c", "4096", "-np", "1", "--no-webui",
             "--log-verbosity", "0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=".",
            # CREATE_NO_WINDOW：否则翻译时会在屏幕上闪一个黑框
            creationflags=0x08000000,
        )

    dev, dev_name = pick_device()
    print(f"[设备] 翻译使用：{dev_name}（{dev}）", flush=True)
    print(f"启动内置 llama.cpp（端口 {port}）...", flush=True)
    proc = start_server(dev)
    try:
        if not wait_health(port):
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
            proc = start_server("none")
            if not wait_health(port):
                print("[失败] llama-server 启动超时")
                sys.exit(1)
        print("llama-server 就绪", flush=True)

        os.makedirs(ASR_DIR, exist_ok=True)
        total = len(files)
        ok = 0
        t_start = time.time()
        for idx, fp in enumerate(files, 1):
            stem = os.path.splitext(os.path.basename(fp))[0]
            rows = json.load(open(fp, encoding="utf-8"))
            glossary = load_glossary(stem)
            print(f"\n翻译: {os.path.basename(fp)}（{len(rows)} 段）", flush=True)

            done = []
            for r in rows:
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                try:
                    zh = translate(port, text, glossary)
                except Exception as e:
                    print(f"  [跳过] {type(e).__name__}: {e}", flush=True)
                    continue
                if not zh:
                    continue
                done.append({"start": r["start"], "end": r["end"], "zh": zh})
                print(f"[{float(r['start']):.2f}s -> {float(r['end']):.2f}s]: {zh}", flush=True)

            tidy = tidy_rows(done)
            lines = []
            for i, r in enumerate(tidy, 1):
                lines.append(
                    f"{i}\n{fmt_time(r['start'])} --> {fmt_time(r['end'])}\n{r['zh']}\n"
                )
            out = os.path.join(ASR_DIR, stem + ".srt")
            with open(out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"  写入 {len(lines)} 条 -> {out}", flush=True)
            ok += 1

            pct = idx * 100.0 / total
            bar = "#" * int(pct / 5)
            el = int(time.time() - t_start)
            print(f"  {pct:3.0f}%|{bar:<20}| {idx}/{total} [{el // 60:02d}:{el % 60:02d}<00:00]", flush=True)

        print(f"\n兼容模式翻译完成：成功 {ok} 个")
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
