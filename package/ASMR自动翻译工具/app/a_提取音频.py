import sys
import ffmpeg_downloader
import os
import glob
from pathlib import Path
import subprocess
from settings import config
from events import emit

# ★ 优化：ffmpeg 装过就别再装。
#   原来每次都跑 `python -m ffmpeg_downloader install`，实测 0.56 秒/次，
#   三个脚本加起来 1.7 秒纯属白跑（而且它会联网检查更新）。
#   改成先看 ffmpeg 在不在 PATH 上，不在才装。
# ★ 优先用随包分发的 ffmpeg（env\ffmpeg\ffmpeg.exe）。
#
#   原来这里只看【系统 PATH】里有没有 ffmpeg，而发布包没有内置 ——
#   于是目标机器上没装过 ffmpeg 的话，会走下面的 ffmpeg_downloader
#   联网下载约 200 MB。两个后果：
#     · 首次运行必须联网（断网 / 网络不通 → 第 1 步直接失败）
#     · 界面上看不出它在下载，用户以为程序卡住了
#   现在包里带了 ffmpeg.exe（静态链接，单文件），先把它塞进 PATH。
#   下面的下载逻辑保留作兜底 —— 万一包里的文件被人删了。
import pathlib as _ff_pl
_ff_dir = _ff_pl.Path(sys.executable).parent / "ffmpeg"
if (_ff_dir / "ffmpeg.exe").is_file():
    os.environ["PATH"] = str(_ff_dir) + os.pathsep + os.environ.get("PATH", "")

ffmpeg_downloader.add_path()
import shutil as _shutil
if _shutil.which("ffmpeg") is None:
    subprocess.run(
        [sys.executable, "-m", "ffmpeg_downloader", "install", "8.0@full-shared"],
        input="y\n",
        text=True
    )
    ffmpeg_downloader.add_path()

pre_dir = config["path"]["pre"]      # "0pre"
# 支持的文件类型（v1.7.2：原来是 ("mp4", "mkv") 两种，
# 用户把 FLAC 音声拖进来时会被当成"不是视频"跳过，第1步扫描 0pre 得到 0 个、
# 退出码 1、整批中止 —— 但纯音频本来就是这个工具的主要用途之一。）
VIDEO_EXTS = ("mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "m4v", "ts", "mpg", "mpeg")
AUDIO_EXTS = ("flac", "wav", "mp3", "m4a", "aac", "ogg", "opus", "wma")
_ALL_EXTS = VIDEO_EXTS + AUDIO_EXTS


def collect_inputs():
    """
    要处理的视频从哪来。两种方式：

      1) 【新】读工作目录下的 _input.txt —— 里面是主程序写好的【源文件绝对路径】。
         这样就不用把几 GB 的视频复制到 0pre 了：视频留在原地，
         我们只把提取出来的 flac 写进 0pre。

         为什么不再走"复制到 0pre"：视频常常在别的盘（比如 H:\\迅雷下载），
         而程序装在 D:\\ —— 跨盘做不了硬链接，只能真复制，一次几 GB、
         要好几分钟。直接读源文件就没这个问题。

      2) 【旧】没有 _input.txt 时，退回原来的行为：扫 0pre 目录里的视频。
         保留这条是为了兼容（单独手动跑这个脚本、或者处理已经放进
         0pre 的文件时会用到）。
    """
    list_path = "_input.txt"
    if os.path.isfile(list_path):
        with open(list_path, "r", encoding="utf-8") as f:
            entries = [ln.strip() for ln in f if ln.strip()]
        kept = []
        for e in entries:
            if not os.path.isfile(e):
                print(f"跳过（文件不存在）: {e}")
                continue
            if not e.lower().endswith(_ALL_EXTS):
                print(f"跳过（不支持的格式）: {e}")
                continue
            kept.append(e)
        if kept:
            print(f"读到 {len(kept)} 个文件")
            return kept

    pattern = os.path.join(pre_dir, "*.*")
    found = [f for f in glob.glob(pattern) if f.lower().endswith(_ALL_EXTS)]
    print(f"扫描 {pre_dir}（{len(found)} 个）")
    return found


files = collect_inputs()

if not files:
    print("没有找到可提取的视频文件")
    sys.exit(1)

emit("stage", "start", 1)
_total = len(files)
for _idx, file_path in enumerate(files, 1):
    emit("file", _idx, _total, Path(file_path).name)
    src = Path(file_path)

    # ★ 关键改动：flac 一律输出到 0pre，【不写在源文件旁边】。
    #
    #   原写法是 file.with_suffix(".flac")，意思是"视频在哪 flac 就写在哪"。
    #   以前视频总是先被复制进 0pre，所以 flac 落在 0pre —— 没毛病。
    #   现在源文件可能就留在你自己的下载目录里，那样会把中间产物丢进
    #   你的目录。所以这里明确写到 0pre。
    #   （b_分离人声.py 也只认 0pre，正好对上。）
    output = Path(pre_dir) / (src.stem + ".flac")

    print(f"提取中: {src.name} -> {output.name}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vn",
        "-acodec", "flac",
        "-ac", "2",
        str(output)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

emit("stage", "done", 1)
print("全部处理完成")
