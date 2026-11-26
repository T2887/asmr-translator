# -*- coding: utf-8 -*-
"""术语字典：译前替换 + 译后替换 + 条件替换（v1.30.6）。

── 为什么需要这一层 ──
翻译层（`light_translate.py`）已经用 prompt 要求「专有名词必须原样保留」，
但 prompt 是软约束：同一个名字在几百段里还是会出现几种写法
（渚沙 / 渚砂 / 长濑沙），拟声词也偶尔被译成「好可爱」这类评价词。
这里补一层**确定性替换** —— 只改词表里明确写出的映射，其它一个字都不动。

★ v1.30.6 借鉴 GalTransl 的字典体系，从「只有译后替换」扩成三类：

  1. 译后替换（原来就有）   `错写法=对写法`
     作用在**中文译文**上。

  2. 译前替换（新增）       `pre:日文A=日文B`
     作用在**日文原文**上，翻译之前就改。
     用途：口齿不清的统一（`ぺろぺろ` / `れろれろ` 归一到同一个词，
     减少模型对同一个音有几种理解）、同一意思的多种日文写法先归一，
     省得在术语表里写一堆同义条目。

  3. 条件替换（新增）       `错写法=对写法|条件词`
     只有当**译文里出现「条件词」**时才替换；加叹号反过来：
     `错写法=对写法|!条件词` 表示「译文里没有条件词时才替换」。
     用途：治「一旦写了就全局误伤」的问题 ——
     例如 `乳=奶|!乳头` 只在没出现「乳头」时才替换「乳」。

── 词表放哪 ──
和 `hotwords.txt` 同一个目录（程序工作目录）：
  fixes.txt              对所有作品生效（通用错译）
  fixes_<音频名>.txt     只对某个音频生效（人名、角色名）
`<音频名>` = `0pre` 里的文件名去掉扩展名。

── 格式汇总 ──
```
# 译后替换
渚沙=凪咲
好可爱=咕啾
# 译前替换（改日文原文）
pre:ぺろぺろ=れろれろ
# 条件替换
乳=奶|!乳头
```

── 注意 ──
· 左侧越长越先替换：「凪咲」不会被短词提前吃掉。
· 替换是纯文本、幂等的，同一段做两遍结果不变。
· 只做「错误写法 → 标准写法」，不做增删；宁可少写，别写宽泛的词。
"""
import os
import re

# 单条映射最长允许多长：超过这个长度的基本不可能是一次误译，更可能是
# 把整句塞进来了 —— 整句替换最容易误伤，直接忽略更安全。
_MAX_BAD_LEN = 24


def _parse_line(ln):
    """解析一行，返回 (kind, cond_neg, cond_kw, bad, good) 或 None。

    kind: "post" / "pre"
    """
    ln = ln.strip()
    if not ln or ln.startswith("#") or "=" not in ln:
        return None
    left, right = ln.split("=", 1)
    left, right = left.strip(), right.strip()

    kind = "post"
    if left.lower().startswith("pre:"):
        kind = "pre"
        left = left[4:].strip()

    # 条件：右侧 `译文|条件词`
    cond_neg = False
    cond_kw = ""
    if "|" in right:
        right, cond = right.split("|", 1)
        right, cond = right.strip(), cond.strip()
        if cond.startswith("!"):
            cond_neg = True
            cond = cond[1:].strip()
        cond_kw = cond

    if not left or left == right:
        return None
    if len(left) > _MAX_BAD_LEN:
        return None
    return kind, cond_neg, cond_kw, left, right


def _read_file(path):
    """读一个词表文件，返回解析后的条目列表；读不到返回 []。"""
    out = []
    if not os.path.isfile(path):
        return out
    try:
        # utf-8-sig：Windows 记事本存的词表常带 BOM，不处理的话第一条会带
        # 不可见字符，永远匹配不上（这个坑在 hotwords 上踩过一次）。
        with open(path, encoding="utf-8-sig") as f:
            for ln in f:
                it = _parse_line(ln)
                if it:
                    out.append(it)
    except OSError:
        return []
    return out


def load_fixes(stem=""):
    """按作品加载字典，返回 {"post": [...], "pre": [...], "cond": [...], "raw": [...]}。

    - post: [(错, 对)]           译后无条件
    - pre:  [(日A, 日B)]         译前
    - cond: [(条件词, 取反, 错, 对)] 译后带条件
    - raw:  全部条目（给调用方自己判断）
    """
    items = _read_file("fixes.txt")
    if stem:
        items += _read_file(f"fixes_{stem}.txt")

    # 去重：后面的覆盖前面的（fixes_<stem> 优先），长词优先
    seen_post, seen_pre, seen_cond = {}, {}, {}
    for kind, cond_neg, cond_kw, bad, good in items:
        if kind == "pre":
            seen_pre[bad] = good
        elif cond_kw:
            # 同一个 bad 只保留最后一条条件（简化，避免多条条件打架）
            seen_cond[bad] = (cond_kw, cond_neg, good)
        else:
            seen_post[bad] = good

    post = sorted(seen_post.items(), key=lambda kv: -len(kv[0]))
    pre = sorted(seen_pre.items(), key=lambda kv: -len(kv[0]))
    cond = sorted(
        [(kw, neg, bad, good) for bad, (kw, neg, good) in seen_cond.items()],
        key=lambda it: -len(it[2]),
    )
    return {"post": post, "pre": pre, "cond": cond, "raw": items}


def apply_pre(ja_text, fixes):
    """译前：对**日文原文**做替换。fixes 为空/无 pre 时原样返回。"""
    if not ja_text or not fixes:
        return ja_text
    pre = fixes.get("pre") if isinstance(fixes, dict) else None
    if not pre:
        return ja_text
    for bad, good in pre:
        if bad in ja_text:
            ja_text = ja_text.replace(bad, good)
    return ja_text


def _replace_outside(text, bad, good, protect):
    """把 bad 替换成 good，但跳过 protect 内部（保护更长的专名不被切开）。

    「乳=奶|乳头」的语义是：句子里有「乳头」这个更具体的词时，
    替换「乳」不能把「乳头」也改成「奶头」—— 所以按 protect 切开，
    只在保护段之外替换。
    """
    if not protect or protect not in text:
        return text.replace(bad, good)
    return protect.join(seg.replace(bad, good) for seg in text.split(protect))


def apply_fixes(text, fixes):
    """译后：把 text 里出现的错误写法替换成标准写法。

    执行顺序：**条件替换先于无条件替换**（条件更具体，先落地，
    并且替换时保护条件词本身）。

    兼容两种入参：
      - list[(错, 对)]         旧格式（v1.30.4 的返回值）
      - dict（load_fixes 新返回值）
    """
    if not text or not fixes:
        return text
    if isinstance(fixes, dict):
        post = fixes.get("post") or []
        cond = fixes.get("cond") or []
    else:
        post, cond = fixes, []

    for kw, neg, bad, good in cond:
        if not kw:
            continue
        hit = kw in text
        if (hit and not neg) or ((not hit) and neg):
            if bad in text:
                text = _replace_outside(text, bad, good, kw if not neg else "")

    for bad, good in post:
        if bad in text:
            text = text.replace(bad, good)
    return text


def _self_test():  # pragma: no cover - 手动自测用
    old = [("长濑沙", "凪咲"), ("渚沙", "凪咲"), ("好可爱", "咕啾")]
    s = "渚沙说好可爱，长濑沙也这么说"
    got = apply_fixes(s, sorted(old, key=lambda kv: -len(kv[0])))
    assert got == "凪咲说咕啾，凪咲也这么说", got

    fx = {
        "post": [],
        "pre": [("ぺろぺろ", "れろれろ")],
        "cond": [("乳头", False, "乳", "奶")],   # 含「乳头」时才替换「乳」
    }
    assert apply_pre("ぺろぺろする", fx) == "れろれろする"
    assert apply_fixes("乳头和乳", fx) == "乳头和奶"
    assert apply_fixes("只有乳", fx) == "只有乳"   # 不含条件词 → 不替换
    print("terms_fix self-test OK")


if __name__ == "__main__":
    _self_test()
