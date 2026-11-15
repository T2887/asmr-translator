# -*- coding: utf-8 -*-
"""运行时状态采集：按【实际推理设备】选正确的指标源。

★ 核心原则（用户强调）：模型跑在哪，就显示哪的性能指标。
  之前一律显示 CPU 频率是错的 —— 模型在 GPU 上跑时 CPU 频率
  既不代表瓶颈、还会误导（CPU 5.3GHz 满频，用户以为很快，
  实际是核显在慢慢算）。

四种情况的指标来源：

  ① 纯 CPU（--device none）
     频率：CallNtPowerInformation ProcessorInformation → 各核 CurrentMhz
     占用：GetProcessTimes 两次采样 → 占用核数

  ② NVIDIA 独显（CUDA）
     频率：nvidia-smi clocks.sm / clocks.max.sm（单位 MHz）
     占用：nvidia-smi utilization.gpu（%）
     显存：memory.used / memory.total
     ★ 这些是「显卡自己的指标」，不是 CPU 的。

  ③ AMD/Intel 核显（Vulkan，共享内存，无独立显存）
     频率：拿不到单卡频率（核显与 CPU 同芯片，共享时钟域），
           显示 CPU 频率 + 明确的 GPU 利用率
     占用：Windows GPU Engine 计数器（能按 pid 过滤出本进程）
     显存：共享内存，用「专用 GPU 内存」+ 共享值展示意义不大，
           这里只显示利用率，不编造显存数字

  ④ 模型吞吐 tok/s（三种情况都有）
     数据源：llama-server /metrics 的
       llamacpp:prompt_tokens_total / llamacpp:tokens_predicted_total
     两次采样做差 ÷ 墙钟差。
     注意指标名可能带 llamacpp: 前缀（新版），也可能不带（旧版）。

所有探测都吞异常 —— 状态显示失败绝不影响主流程。
"""
import ctypes
import ctypes.wintypes as wt
import os
import re
import subprocess
import time
import urllib.request

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_powrprof = ctypes.WinDLL("powrprof", use_last_error=True)


# ── CPU 频率（核显/纯 CPU 场景用）────────────────────────────
class PROCESSOR_POWER_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("Number", wt.ULONG), ("MaxMhz", wt.ULONG),
        ("CurrentMhz", wt.ULONG), ("MhzLimit", wt.ULONG),
        ("MaxIdleState", wt.ULONG), ("CurrentIdleState", wt.ULONG),
    ]


def cpu_freqs():
    """→ [CurrentMhz, ...]。失败返回 []。"""
    try:
        n = os.cpu_count() or 1
        buf = (PROCESSOR_POWER_INFORMATION * n)()
        r = _powrprof.CallNtPowerInformation(
            11, None, 0, ctypes.byref(buf), ctypes.sizeof(buf))
        if r != 0:
            return []
        return [int(buf[i].CurrentMhz) for i in range(n)
                if 0 < buf[i].CurrentMhz < 20000]
    except Exception:
        return []


def proc_cpu_seconds(pid):
    """进程累计 CPU 秒数。失败 None。"""
    try:
        h = _k32.OpenProcess(0x0400 | 0x0010, False, pid)
        if not h:
            return None
        try:
            c, e, k, u = (wt.FILETIME(), wt.FILETIME(),
                          wt.FILETIME(), wt.FILETIME())
            if not _k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e),
                                        ctypes.byref(k), ctypes.byref(u)):
                return None
            t = ((k.dwHighDateTime << 32 | k.dwLowDateTime)
                 + (u.dwHighDateTime << 32 | u.dwLowDateTime))
            return t / 1e7
        finally:
            _k32.CloseHandle(h)
    except Exception:
        return None


# ── NVIDIA 独显指标 ────────────────────────────────────────────
def nvidia_stats():
    """→ dict(sm_mhz, max_sm_mhz, util, mem_used, mem_total) 或 None。

    ★ nvidia-smi 在 PATH 里通常没有，要用绝对路径或 where 找。
      实测本机在 C:\\Windows\\System32\\nvidia-smi.exe。
    """
    exe = None
    for cand in (r"C:\Windows\System32\nvidia-smi.exe",
                 r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if os.path.isfile(cand):
            exe = cand
            break
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=clocks.sm,clocks.max.sm,utilization.gpu,"
                   "memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, errors="replace", timeout=8)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        parts = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
        if len(parts) < 5:
            return None
        # 某些驱动给 [N/A]，要能容忍
        def num(i):
            try:
                return float(parts[i])
            except (ValueError, IndexError):
                return None
        out = {"sm_mhz": num(0), "max_sm_mhz": num(1), "util": num(2),
               "mem_used": num(3), "mem_total": num(4), "temp": num(5)}
        # sm 频率拿不到就认为不可用（有些虚拟化环境屏蔽）
        return out if out["sm_mhz"] else None
    except Exception:
        return None


# ── 核显利用率（Windows GPU Engine 计数器）─────────────────────
def gpu_engine_util(pid=None):
    """→ 该 pid（或全部）的 GPU 引擎利用率百分比。失败 None。

    核显没有 nvidia-smi；Windows 的 GPU Engine 计数器能用，
    而且 instance name 里带 pid，可以精确到本进程。
    """
    try:
        ps = ("Get-Counter '\\GPU Engine(*)\\Utilization Percentage' "
              "-ErrorAction SilentlyContinue | "
              "Select-Object -ExpandProperty CounterSamples | "
              "Where-Object { $_.CookedValue -gt 0 } | "
              "Select-Object InstanceName, CookedValue | "
              "ConvertTo-Json -Compress")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, errors="replace",
                           timeout=12)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        import json as _json
        data = _json.loads(r.stdout)
        if isinstance(data, dict):
            data = [data]
        total = 0.0
        for item in data:
            name = str(item.get("InstanceName", ""))
            val = float(item.get("CookedValue", 0) or 0)
            if pid is not None:
                m = re.search(r"pid_(\d+)_", name)
                if not m or int(m.group(1)) != pid:
                    continue
            # 所有引擎类型累加（3d/videodecode/compute 等），
            # 但同一物理引擎的多实例会重复计算，这里取最大值更稳妥
            total = max(total, val)
        return total if total > 0 else 0.0
    except Exception:
        return None


# ── llama-server 吞吐 ─────────────────────────────────────────
def metric_value(text, name):
    """取指标值，兼容带 llamacpp: 前缀和不带两种写法。"""
    for pat in (rf"^{re.escape(name)}\s+([0-9.eE+-]+)",
                rf"^[a-z_]+:{re.escape(name)}\s+([0-9.eE+-]+)"):
        m = re.search(pat, text, re.M)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return -1.0


class StatusProbe:
    """按设备类型采集状态。device 决定显示哪套指标。"""

    def __init__(self, port, device="none", llama_pid=None):
        self.port = port
        self.device = (device or "none").lower()
        self.llama_pid = llama_pid
        # 判定设备类型
        self.is_nvidia = self.device.startswith("cuda")
        self.is_vulkan = self.device.startswith("vulkan")
        self.is_cpu = self.device in ("none", "", "cpu")
        self._p_prev = self._d_prev = -1.0
        self._t_prev = 0.0
        self._cpu_prev = None
        self._cpu_t_prev = 0.0
        self._freq_prev = []
        self._freq_t_prev = 0.0

    def _fetch_metrics(self):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/metrics", timeout=2) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            return None

    def poll(self):
        """→ 一行状态文本。内容随设备类型而变。"""
        parts = []
        now = time.time()

        # ① 模型吞吐（三种设备通用）
        txt = self._fetch_metrics()
        if txt:
            p = metric_value(txt, "prompt_tokens_total")
            d = metric_value(txt, "tokens_predicted_total")
            if p >= 0 and d >= 0:
                restarted = (self._p_prev >= 0 and
                             (p < self._p_prev or d < self._d_prev))
                if self._t_prev > 0 and now > self._t_prev and not restarted:
                    dt = now - self._t_prev
                    parts.append(
                        f"模型 写{(d - self._d_prev) / dt:.1f}"
                        f"/读{(p - self._p_prev) / dt:.0f} t/s")
                else:
                    parts.append("模型 启动中" if restarted else "模型 空闲")
                self._p_prev, self._d_prev, self._t_prev = p, d, now

        # ② 设备指标（按实际推理设备选源）
        if self.is_nvidia:
            st = nvidia_stats()
            if st:
                # 独显：显示【显卡】SM 频率 + 利用率 + 显存
                if st.get("max_sm_mhz"):
                    parts.append(
                        f"GPU {st['sm_mhz']:.0f}"
                        f"/{st['max_sm_mhz']:.0f} MHz")
                elif st.get("sm_mhz"):
                    parts.append(f"GPU {st['sm_mhz']:.0f} MHz")
                if st.get("util") is not None:
                    parts.append(f"占用 {st['util']:.0f}%")
                if st.get("mem_used") and st.get("mem_total"):
                    parts.append(
                        f"显存 {st['mem_used'] / 1024:.1f}"
                        f"/{st['mem_total'] / 1024:.1f} GB")
            else:
                parts.append("GPU 指标不可用")

        elif self.is_vulkan:
            # 核显：没有独立显存、与 CPU 共享频率域，所以
            #   · 显示 GPU 利用率（真实的核显负载）
            #   · 频率回落到 CPU 频率（核显与 CPU 同芯片，这是唯一能拿到的）
            #   明确标「核显」以免误读成独显。
            u = gpu_engine_util(self.llama_pid) if self.llama_pid else None
            if u is not None:
                parts.append(f"核显 占用 {u:.0f}%")
            else:
                parts.append("核显 占用不可用")
            if now - self._freq_t_prev > 3.0:
                fr = cpu_freqs()
                if fr:
                    self._freq_prev = fr
                    self._freq_t_prev = now
            if self._freq_prev:
                avg = sum(self._freq_prev) / len(self._freq_prev) / 1000.0
                parts.append(f"CPU {avg:.2f} GHz")

        else:
            # 纯 CPU：显示 CPU 频率 + 占用核数
            if self.llama_pid:
                cs = proc_cpu_seconds(self.llama_pid)
                if cs is not None:
                    if self._cpu_prev is not None and now > self._cpu_t_prev:
                        dt = now - self._cpu_t_prev
                        cores = (cs - self._cpu_prev) / dt
                        if 0 <= cores < 64:
                            parts.append(f"CPU {cores:.1f} 核")
                    self._cpu_prev, self._cpu_t_prev = cs, now
            if now - self._freq_t_prev > 3.0:
                fr = cpu_freqs()
                if fr:
                    self._freq_prev = fr
                    self._freq_t_prev = now
            if self._freq_prev:
                avg = sum(self._freq_prev) / len(self._freq_prev) / 1000.0
                hi = max(self._freq_prev) / 1000.0
                parts.append(f"{avg:.2f}/{hi:.2f} GHz")

        return " · ".join(parts) if parts else "运行状态 —"


def find_llama_pid():
    """找 llama-server 进程 pid。"""
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe",
             "/FO", "CSV", "/NH"],
            capture_output=True, text=True, errors="replace", timeout=10)
        for line in r.stdout.splitlines():
            f = [x.strip('"') for x in line.split('","')]
            if len(f) >= 2 and f[1].isdigit():
                return int(f[1])
    except Exception:
        pass
    return None


if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "none"
    pid = find_llama_pid()
    print(f"设备={dev}  llama_pid={pid}")
    print(f"NVIDIA 指标: {nvidia_stats()}")
    print(f"GPU Engine(全部): {gpu_engine_util()}")
    p = StatusProbe(int(os.environ.get("ASMR_PORT", "18099")), dev, pid)
    for i in range(3):
        print(f"  [{i}] {p.poll()}")
        time.sleep(1)
