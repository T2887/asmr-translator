"""极简 SRT 读写 —— 用来替换 pysrt。

为什么不用 pysrt：
  pysrt 是 GPLv3。GPLv3 有传染性：只要项目里 import 了它，
  整个项目就必须以 GPLv3 发布，不能选 MIT。

  而我们在这里只用到 pysrt 的两件事：
    ① 生成 SRT 文本文件
    ② 读出 VAD 的 SRT 时间轴
  自己写 60 行就够，而且没有任何第三方依赖。

对外接口（刻意做得和 pysrt 的常用子集对应，方便替换）：
    write_srt(path, rows)      rows = [(start_sec, end_sec, text), ...]
    read_srt(path)             -> [(start_sec, end_sec, text), ...]

时间精度：毫秒（SRT 标准就是这个精度）。
"""

import io
import re

# SRT 时间轴：00:00:01,234 --> 00:00:05,678
_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _fmt_time(sec: float) -> str:
    """秒 → 00:00:01,234"""
    if sec < 0:
        sec = 0.0
    total_ms = int(round(sec * 1000))
    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_time(h, m, s, ms) -> float:
    """00:00:01,234 的四个字段 → 秒（浮点）"""
    # 兼容两位/三位毫秒
    ms_s = ms.ljust(3, "0") if len(ms) < 3 else ms
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms_s) / 1000.0


def write_srt(path, rows):
    """写 SRT。

    rows = [(start_sec, end_sec, text), ...]
    text 里如果有换行，会被保留成多行字幕（SRT 允许）。

    写文件用 UTF-8 无 BOM、换行统一成 \\n —— 和之前 pysrt 的行为一致，
    也避免 Windows 上出现 \\r\\n 导致某些播放器显示异常。
    """
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, (a, b, text) in enumerate(rows, 1):
            t = (text or "").strip()
            if not t:
                continue
            # 保证结束时间不小于开始时间
            if b < a:
                b = a
            f.write(f"{i}\n")
            f.write(f"{_fmt_time(a)} --> {_fmt_time(b)}\n")
            f.write(t + "\n\n")


def read_srt(path):
    """读 SRT，返回 [(start_sec, end_sec, text), ...]。

    容错策略：
      · 找不到时间轴的行跳过
      · 多条文本行拼成一个 text（用空格连接）
      · 空文件返回空列表，不抛异常
    """
    out = []
    try:
        raw = io.open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return out

    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    n = len(lines)
    while i < n:
        m = _TIME_RE.search(lines[i])
        if not m:
            i += 1
            continue
        a = _parse_time(m.group(1), m.group(2), m.group(3), m.group(4))
        b = _parse_time(m.group(5), m.group(6), m.group(7), m.group(8))
        # 收集后续非空行作为文本
        i += 1
        buf = []
        while i < n and lines[i].strip():
            buf.append(lines[i].strip())
            i += 1
        text = " ".join(buf)
        if text:
            out.append((a, b, text))
    return out
