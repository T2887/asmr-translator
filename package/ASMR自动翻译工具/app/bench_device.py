# -*- coding: utf-8 -*-
"""启动测速：纯 CPU vs Vulkan 核显，自动选更快的那个。

背景（实测数据）
────────────────
本机（Ryzen 7 7700 + Radeon 核显）跑 Hy-MT2-1.8B Q4 + LoRA：
  · 纯 CPU  （--device none）    : 1388 ms/句
  · 核显    （--device Vulkan1） : 2531 ms/句（且随句子变长差距扩大）
核显慢了接近 2 倍。

原因：1.8B 小模型在核显上，每个 token 都要在 CPU 和核显间搬权重，
而核显的计算单元本身也不强（实测核显 100% 占用、独显 13% 闲置）。
内核显反而比纯 CPU 慢 —— 这与「GPU 一定更快」的直觉相反，
所以不能靠猜，必须在任务开始时【实测】。

设计
────
1. 取待翻译文本里 3 句中等长度的做样本（代表性最好）
2. 分别用纯 CPU / Vulkan 各跑一遍，记录每句耗时
3. 返回结论：更快的是哪个、快多少、是否值得提醒用户
4. 不改变用户的显式选择，只提供证据让用户决定

★ 时间成本：3 句 × 2 设备 ≈ 6~15 秒。相对整个任务（十几分钟到几小时）
  可以忽略，而收益是可能省掉一半以上的时间。
"""
import json
import os
import subprocess
import time
import urllib.request

PC = r"D:\ASMR翻译器\package\ASMR自动翻译工具"

# ★ 内置测速文本（正常日常日语，不带任何题材倾向）
#
# 为什么不再从待翻译素材里取样（原设计）：
#   · 素材内容可能很特殊（ASMR 台词的句子结构、长度分布都很偏），
#     拿它测速会把「题材特性」混进「设备性能」里
#   · 而且用户看到的第一个动作就是测速，从自己的素材里抽句子
#     容易让人误解「程序在读我的内容」
#   · 内置文本长度分布可控（短/中/长各一句），测速结果更稳定可复现
#
# 选词原则：日常、干净、无敏感内容的普通日语句子，
# 长度分别约 12 / 28 / 45 字，覆盖真实翻译的典型区间。
BENCH_TEXTS = [
    "今日はいい天気ですね。散歩でもしませんか。",
    "昨日は友達と駅の近くの喫茶店で、二時間くらい話しました。",
    "この本は先月から読み始めたのですが、思っていたより面白くて、"
    "毎晩寝る前に少しずつ読んでいます。",
]

# 测速用的系统提示 —— 与生产完全一致，否则测出来不代表真实速度
SYSTEM = ("你是日译中字幕翻译。把用户给的日语翻译成简体中文，口语化、句子短促，"
          "只输出中文，不要保留日文，不要解释，译文必须是一行、不要换行。")


def _pick_samples(texts=None, n=3):
    """返回内置测速文本（不再从用户素材取样）。

    texts 参数保留但忽略 —— 调用方可能还在传素材文本，
    为了不破坏调用点也不报错。
    """
    return list(BENCH_TEXTS[:n])


def _bench_device(llama, model, lora, device, samples, port,
                  max_tokens=80, timeout=120):
    """在指定设备上跑一遍样本，返回 (平均毫秒/句, 启动耗时, 错误信息)。"""
    cmd = [llama, "--model", model, "--host", "127.0.0.1", "--port", str(port),
           "--device", device, "-c", "4096", "-np", "1", "--no-webui",
           "--log-verbosity", "0"]
    if lora and os.path.isfile(lora):
        cmd += ["--lora", lora]
    t_start = time.time()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, cwd=PC,
                             creationflags=0x08000000)
    except Exception as e:
        return None, 0.0, f"启动失败：{e}"
    ready = False
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2)
                ready = True
                break
            except Exception:
                if p.poll() is not None:
                    return None, 0.0, "进程提前退出（设备不可用）"
                time.sleep(1)
        if not ready:
            return None, 0.0, "启动超时"

        t_load = time.time() - t_start
        times = []
        for s in samples:
            body = {"messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": s}],
                    "temperature": 0.0, "max_tokens": max_tokens,
                    "stream": False}
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    json.load(r)
            except Exception as e:
                return None, t_load, f"推理失败：{type(e).__name__}"
            times.append((time.time() - t0) * 1000.0)
        return (sum(times) / len(times)), t_load, None
    finally:
        try:
            p.terminate()
            p.wait(timeout=8)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        time.sleep(0.5)


def bench_devices(llama, model, lora, samples, base_port=11090):
    """测纯 CPU 与 Vulkan，返回结论 dict。

    → {"cpu_ms": float|None, "vulkan_ms": float|None,
       "prefer": "cpu"|"vulkan"|None, "ratio": float,
       "advice": str, "need_prompt": bool}
    """
    if not samples:
        # ★ 没传样本就用内置测速文本（v1.30.3 起默认走这里）。
        #   之前调用方传 None 就直接判「没有可用样本」，测速被跳过 ——
        #   能用内置文本，不该依赖调用方传什么。
        samples = _pick_samples()
    if not samples:
        return {"prefer": None, "advice": "没有可用样本，跳过测速",
                "need_prompt": False, "cpu_ms": None, "vulkan_ms": None,
                "ratio": 1.0}

    cpu_ms, cpu_load, cpu_err = _bench_device(
        llama, model, lora, "none", samples, base_port)
    vk_ms, vk_load, vk_err = _bench_device(
        llama, model, lora, "Vulkan1", samples, base_port + 1)

    res = {"cpu_ms": cpu_ms, "vulkan_ms": vk_ms,
           "cpu_load": cpu_load, "vulkan_load": vk_load,
           "cpu_err": cpu_err, "vulkan_err": vk_err,
           "prefer": None, "ratio": 1.0, "advice": "",
           "need_prompt": False}

    if cpu_ms is None and vk_ms is None:
        res["advice"] = "两种设备都测失败，保持当前设置"
        return res
    if vk_ms is None:
        res["prefer"] = "cpu"
        res["advice"] = f"AMD 核显不可用（{vk_err}），改用纯 CPU"
        res["need_prompt"] = False     # 没得选，直接切
        return res
    if cpu_ms is None:
        res["prefer"] = "vulkan"
        res["advice"] = f"纯 CPU 不可用（{cpu_err}），保持核显"
        return res

    res["ratio"] = vk_ms / cpu_ms if cpu_ms > 0 else 1.0
    if res["ratio"] >= 1.15:
        # 纯 CPU 明显更快（>15% 差距才提示，避免测量噪声干扰）
        res["prefer"] = "cpu"
        res["need_prompt"] = True
        # ★ 措辞保守：只说「这次测出来」的数，不承诺倍数结论
        res["advice"] = (
            f"测速结果：纯 CPU 约 {cpu_ms:.0f} ms/句，核显约 {vk_ms:.0f} ms/句，"
            f"这次测下来纯 CPU 更快")
    elif res["ratio"] <= 0.87:
        res["prefer"] = "vulkan"
        res["advice"] = (
            f"测速结果：核显约 {vk_ms:.0f} ms/句，纯 CPU 约 {cpu_ms:.0f} ms/句，"
            f"这次测下来核显更快")
    else:
        res["prefer"] = "vulkan"      # 差距不大，按默认（用户选了 Vulkan）
        res["advice"] = (
            f"测速结果：核显约 {vk_ms:.0f} ms/句，纯 CPU 约 {cpu_ms:.0f} ms/句，"
            f"两者相差不大，保持核显")
    return res


def make_prompt(advice, cpu_ms, vk_ms, ratio):
    """生成给用户看的弹窗文案（简洁、说清原因、给选择）。

    ★ 措辞刻意保守（用户要求）：
      · 不写「全程可省 XX%」这种承诺 —— 测速用的是三句标准文本，
        实际语料长短分布不同，节省比例会浮动（实测本机：标准文本差
        4.3 倍，完整 588 段语料差 2 倍）。写死一个百分比等于给了
        一个大概率不成立的承诺。
      · 改成「可能省下不少时间」这种模糊但诚实的表述。
      · 保留具体毫秒数 —— 那是测出来的事实，用户自己能算比例。

    用户要求：简洁说明为什么 CPU 更快 + 询问是否切换。
    """
    return {
        "title": "核显比纯 CPU 更慢",
        "body": (
            f"{advice}。\n\n"
            f"为什么：这个模型参数不多，核显每算一点都要把模型数据从内存"
            f"搬过去，搬运的开销超过了核显本身省下的时间。模型越大核显才越"
            f"占优势，这种小模型往往是纯 CPU 更快。\n\n"
            f"要转换成纯 CPU 推理吗？切换后可能省下不少时间，"
            f"具体取决于你的素材长度分布。"
        ),
        "switch_label": "换成纯 CPU（更快）",
        "keep_label": "继续用核显",
    }


if __name__ == "__main__":
    # 自测：用 2vad 里的真实文本跑一遍
    p = os.path.join(PC, "2vad")
    texts = []
    for f in sorted(os.listdir(p))[:1]:
        if f.endswith(".json"):
            for r in json.load(open(os.path.join(p, f), encoding="utf-8")):
                t = (r.get("text") or "").strip()
                if t:
                    texts.append(t)
    samples = _pick_samples(texts)
    print(f"样本 {len(samples)} 句：")
    for s in samples:
        print(f"  {s[:40]}")
    r = bench_devices(
        os.path.join(PC, "llama", "llama-server.exe"),
        os.path.join(PC, "model", "Hy-MT2-1.8B-Q4_K_M.gguf"),
        os.path.join(PC, "model", "hy-mt2-asmr-lora.gguf"),
        samples)
    print(json.dumps({k: v for k, v in r.items()}, ensure_ascii=False, indent=1))
    if r.get("need_prompt"):
        print(json.dumps(make_prompt(r["advice"], r["cpu_ms"],
                                     r["vulkan_ms"], r["ratio"]),
                         ensure_ascii=False, indent=1))
