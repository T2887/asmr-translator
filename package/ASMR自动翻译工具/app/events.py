"""结构化事件总线（v1.17）。

Rust 侧只认这一种前缀行，用来拿阶段/进度/字幕信息；
人类可读的日志照旧打印，两边互不影响。

协议：一行以 ``@@EV@@ `` 开头，字段用 ``|`` 分隔，第一个字段是事件名：

    @@EV@@ sub|38.20|41.80|好高兴
    @@EV@@ progress|0.42|31
    @@EV@@ file|3|14|名字.flac
    @@EV@@ stage|start|5
    @@EV@@ stage|done|5
    @@EV@@ tidy|before=171.0|after=146.2|shrunk=14|n=49

为什么不用 JSON：
  1) 不引入 serde_json，离线编译不受影响；
  2) 文本里可能带 ``|``，但文本永远是最后一个字段（Rust 侧 splitn 限制），
     不会被切坏；
  3) 解析只有几行，出问题时能直接肉眼读懂。

约定：
  · 事件行只写 stdout，不写 stderr；
  · 文本字段里的换行/回车会被换成空格 —— 一条事件必须只占一行；
  · 脚本可以没有事件（旧版本 Rust 也能靠原始日志跑），
    所以 emit 失败不应该影响主流程。
"""

PREFIX = "@@EV@@ "


def emit(kind, *fields):
    """输出一条结构化事件。

    kind 固定为小写 ASCII；fields 会转成字符串并按顺序拼接。
    """
    try:
        parts = [kind]
        for f in fields:
            s = str(f).replace("\r", " ").replace("\n", " ")
            parts.append(s)
        print(PREFIX + "|".join(parts), flush=True)
    except Exception:
        # 事件只是"附加信息"，绝不能让字幕流程因为打印失败而中断。
        pass
