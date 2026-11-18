# -*- coding: utf-8 -*-
"""术语后处理：把译文里跑偏的译名按词表改回标准写法（v1.30.4）。

── 为什么需要这一层 ──
翻译层（`light_translate.py`）已经用 prompt 要求「专有名词必须原样保留」，
但 prompt 是软约束：同一个名字在几百段里还是会出现几种写法
（渚沙 / 渚砂 / 长濑沙），拟声词也偶尔被译成「好可爱」这类评价词。
这里补一层**确定性替换** —— 只改词表里明确写出的映射，其它一个字都不动。

── 词表放哪 ──
和 `hotwords.txt` 同一个目录（程序工作目录）：
  fixes.txt              对所有作品生效（通用错译，例如「好可爱=咕啾」）
  fixes_<音频名>.txt     只对某个音频生效（人名、角色名）
`<音频名>` = `0pre` 里的文件名去掉扩展名，例如 `official01.wav` →
`fixes_official01.txt`。

── 格式 ──
```
#  一行一条「错误写法=标准写法」，井号开头是注释，空行忽略
渚沙=凪咲
长濑沙=凪咲
好可爱=咕啾
```

── 注意 ──
· 左侧越长越先替换：「凪咲」不会被短词提前吃掉。
· 替换是纯文本、幂等的，同一段做两遍结果不变。
· 只做「错误写法 → 标准写法」，不做增删；所以宁可少写，也别写宽泛的词
  （例如别写「声音=声响」，会把正常句子也改掉）。
"""
import os

# 单条映射最长允许多长：超过这个长度的基本不可能是一次误译，更可能是
# 把整句塞进来了 —— 整句替换最容易误伤，直接忽略更安全。
_MAX_BAD_LEN = 24


def _read_file(path):
    """读一个词表文件，返回 [(错误写法, 标准写法), ...]；读不到返回 []。"""
    out = []
    if not os.path.isfile(path):
        return out
    try:
        # utf-8-sig：Windows 记事本存的词表常带 BOM，不处理的话第一条会带
        # 不可见字符，永远匹配不上（这个坑在 hotwords 上踩过一次）。
        with open(path, encoding="utf-8-sig") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                bad, good = ln.split("=", 1)
                bad, good = bad.strip(), good.strip()
                if not bad or bad == good:
                    continue
                if len(bad) > _MAX_BAD_LEN:
                    continue
                out.append((bad, good))
    except OSError:
        return []
    return out


def load_fixes(stem=""):
    """按作品加载替换表：先通用 fixes.txt，再 fixes_<stem>.txt（后者优先）。"""
    pairs = _read_file("fixes.txt")
    if stem:
        pairs += _read_file(f"fixes_{stem}.txt")
    # 去重（后者覆盖前者）+ 长词优先，避免「凪」先被短映射吃掉。
    merged = {}
    for bad, good in pairs:
        merged[bad] = good
    return sorted(merged.items(), key=lambda kv: -len(kv[0]))


def apply_fixes(text, fixes):
    """把 text 里出现的错误写法替换成标准写法；fixes 为空时原样返回。"""
    if not text or not fixes:
        return text
    for bad, good in fixes:
        if bad in text:
            text = text.replace(bad, good)
    return text


def _self_test():  # pragma: no cover - 手动自测用
    pairs = [("长濑沙", "凪咲"), ("渚沙", "凪咲"), ("好可爱", "咕啾")]
    s = "渚沙说好可爱，长濑沙也这么说"
    print(apply_fixes(s, sorted(pairs, key=lambda kv: -len(kv[0]))))
    assert apply_fixes(s, sorted(pairs, key=lambda kv: -len(kv[0]))) == \
        "凪咲说咕啾，凪咲也这么说"


if __name__ == "__main__":
    _self_test()
