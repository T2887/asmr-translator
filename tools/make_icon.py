"""生成项目图标（赤陶橙圆角方 + 纸白「三根声波竖条 + 一条字幕线」）。

图标是纯代码画的，没有任何第三方素材 —— 这样图标可以和项目一起
用 MIT 发布，不用附带别人的许可证，也不存在"素材来源不明"的问题。

用法：
    python tools/make_icon.py            写 app/assets/icon.png + icon.ico
    python tools/make_icon.py --preview  只出预览图到 tools/preview/

尺寸策略：每个尺寸单独重绘（4x 超采样再缩小），而不是从 1024 直接缩 ——
小图边缘才不糊。ico 里 16..256 共 9 个尺寸，各画一遍再打包。
"""

import os
import sys
from PIL import Image, ImageDraw
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_APP = os.path.join(ROOT, "app", "assets")
OUT_PREVIEW = os.path.join(ROOT, "tools", "preview")

# 赤陶橙渐变（左上浅、右下深），和界面的强调色同源
C1 = (0xD4, 0x71, 0x4F)
C2 = (0xB1, 0x50, 0x2F)
PAPER = (0xFA, 0xF9, 0xF5)  # 纸白

# 几何比例（相对边长的分数）
R_CORNER = 0.196
BAR_W = 0.078
BAR_DX = 0.117          # 中间条到左右条的横向偏移
BAR_H = (0.240, 0.420, 0.240)
BAR_CY = 0.440
LINE_BOX = (0.300, 0.720, 0.700, 0.780)   # x0, y0, x1, y1
LINE_R = 0.030


def gradient(size):
    """左上到右下的线性渐变。"""
    w = h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / w * 0.45 + yy / h * 0.55)
    t = np.clip((t - 0.10) / 0.80, 0, 1)[..., None]
    a = np.array(C1, dtype=np.float32)
    b = np.array(C2, dtype=np.float32)
    rgb = a + (b - a) * t
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def draw_icon(size, ss=4):
    """按目标尺寸绘制；ss 是超采样倍数。"""
    s = size * ss
    canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # 1) 圆角方 + 渐变（用 mask 裁）
    mask = Image.new("L", (s, s), 0)
    md = ImageDraw.Draw(mask)
    pad = round(0.012 * s)
    md.rounded_rectangle(
        [pad, pad, s - 1 - pad, s - 1 - pad],
        radius=round(R_CORNER * s),
        fill=255,
    )
    canvas.paste(gradient(s).convert("RGBA"), (0, 0), mask)

    d = ImageDraw.Draw(canvas)
    white = PAPER + (255,)

    # 2) 三根声波竖条
    bw = BAR_W * s
    dx = BAR_DX * s
    cy = BAR_CY * s
    for i, hh in enumerate(BAR_H):
        x = s / 2 + (i - 1) * dx
        h = hh * s
        d.rounded_rectangle(
            [x - bw / 2, cy - h / 2, x + bw / 2, cy + h / 2],
            radius=bw / 2,
            fill=white,
        )

    # 3) 字幕线
    x0, y0, x1, y1 = [v * s for v in LINE_BOX]
    d.rounded_rectangle([x0, y0, x1, y1], radius=LINE_R * s, fill=white)

    return canvas.resize((size, size), Image.LANCZOS)


def main():
    preview_only = "--preview" in sys.argv
    os.makedirs(OUT_PREVIEW, exist_ok=True)
    if not preview_only:
        os.makedirs(OUT_APP, exist_ok=True)

    big = draw_icon(1024, ss=2)
    big.save(os.path.join(OUT_PREVIEW, "icon-1024.png"))
    if not preview_only:
        big.save(os.path.join(OUT_APP, "icon.png"))

    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    imgs = [draw_icon(n) for n in sizes]
    if not preview_only:
        # ico 里每个尺寸都是单独重画的，不是从 1024 缩
        imgs[-1].save(
            os.path.join(OUT_APP, "icon.ico"),
            format="ICO",
            sizes=[(n, n) for n in sizes],
            append_images=imgs[:-1],
        )

    # 尺寸检查图：16..64 各用 NEAREST 放大 8 倍，看会不会糊成一团
    tiles = [draw_icon(n).resize((n * 8, n * 8), Image.NEAREST) for n in (16, 24, 32, 48, 64)]
    wsum = sum(t.width for t in tiles) + 20 * (len(tiles) - 1)
    sheet = Image.new("RGBA", (wsum, max(t.height for t in tiles)), (240, 238, 230, 255))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0), t)
        x += t.width + 20
    sheet.save(os.path.join(OUT_PREVIEW, "icon-sizes.png"))

    print("done", "preview only" if preview_only else "written to app/assets")


if __name__ == "__main__":
    main()
