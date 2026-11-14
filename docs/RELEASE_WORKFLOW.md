# 打包发布流程（固定步骤，照抄即可）

> 适用于 PC 版（Rust + egui）。写于 v1.30.2 发布后，每一步都是实际执行过的。
> 历史文档见 `docs/RELEASE.md`（用户视角）、`docs/GIT_WORKFLOW.md`（提交规范）、`交接说明.md`。

---

## 0. 两个包的结构（先搞清要重打哪个）

| 包 | iss 文件 | 内容 | 频率 |
|---|---|---|---|
| **b1 主程序包** `ASMR-Translator-Setup_v<版本>.exe` | `E:\picture\deepseek-harness\default-workspace\_candidate\b1_main.iss` | exe、app\*.py、assets、llama\（88MB）、hotwords、**model\hy-mt2-asmr-lora.gguf**（77MB，v1.30.2 起） | 几乎每次发版 |
| **b2 运行环境包** `2_运行环境_安装包_v<版本>.exe` + 4 个 .bin 分卷 | 同目录 `b2_runtime.iss` | env\（5.9GB）+ model\ 其余模型（约 6GB） | 只在 Python 环境/模型变化时重打 |

判断规则：
- 只改了 `app\*.py`、Rust 界面、llama\、LoRA → **只打 b1**（用户走「检查更新」装的就是 b1）
- 改了 `env\`（装包/升级依赖）、模型权重（BS-RoFormer / whisper / Haas-ja / Hy-MT2 主模型）→ 重打 b2 + 同步网盘 + 改 `version.json` 的 `runtime` 提示
- LoRA 类小模型刻意放 b1 不放 b2：老用户升级一次就能吃到（iss 里两边都有通配时记得在 b2 加 `Excludes`，防重复打包）

## 1. 版本号三处同步（漏一处就是事故）

| 位置 | 字段 |
|---|---|
| `H:\software\ASMR翻译器\app\src\config.rs` | `pub const APP_VERSION`（`_proto_egui` 经 `#[path]` 共用此文件） |
| `_candidate\b1_main.iss` | `#define AppVer "<版本>"` |
| `_candidate\b2_runtime.iss` | `#define AppVer "<版本>"`（哪怕这次不重打 b2 也同步改，防下次漏） |

版本策略：**一个未发布的批次只 +1**（bug 修第三位，功能第二位）。别每改一个文件就升一版。

## 2. 改动源文件的固定流向

```
D:\ASMR翻译器\package\ASMR自动翻译工具\        ← 唯一直接运行/测试的地方
   └─ 改完必须复制到 → H:\software\ASMR翻译器\package\ASMR自动翻译工具\   ← git 仓库镜像
```

两边都改或直接只改镜像都是坑。发布前 `diff -q` 逐文件核对本次改动。

## 3. 发版命令序列（在 Git-Bash / ops_bash 下）

```bash
# 3.1 构建 + 测试（cargo 不在 PATH，用绝对路径）
cd "H:/software/ASMR翻译器/_proto_egui"
"$USERPROFILE/.cargo/bin/cargo.exe" test   --release --offline   # 必须全过
"$USERPROFILE/.cargo/bin/cargo.exe" build  --release --offline   # 约 1 分钟

# 3.2 部署新 exe 到发布目录
cp "H:/software/ASMR翻译器/_proto_egui/target/release/asmr_proto_egui.exe" \
   "D:/ASMR翻译器/package/ASMR自动翻译工具/ASMR自动翻译工具.exe"

# 3.3 启动一次冒烟（可选但推荐）：确认窗口标题带新版本号
"D:/ASMR翻译器/package/ASMR自动翻译工具/ASMR自动翻译工具.exe" &   # 看完关掉

# 3.4 打主程序包（ISCC 在用户级 AppData，不在 Program Files）
"/c/Users/Dayje/AppData/Local/Programs/Inno Setup 6/ISCC.exe" \
   "E:/picture/deepseek-harness/default-workspace/_candidate/b1_main.iss"
# 产物 → D:\ASMR翻译器\package\安装包\ASMR-Translator-Setup_v<版本>.exe

# 3.5 安装冒烟：静默装到临时目录，跑一遍关键脚本，再卸载
SETUP="D:/ASMR翻译器/package/安装包/ASMR-Translator-Setup_v<版本>.exe"
"$SETUP" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="D:\asmr_install_test"
# 在 D:\asmr_install_test 里用 PC 的 env python 跑被改过的步骤脚本，验证日志行/产物
#   注意：安装目录没有 env\（它在 b2），测试用 D:\...\env\python.exe，cwd=安装目录
"D:/asmr_install_test/unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES; rm -rf D:/asmr_install_test
```

## 4. version.json 更新（禁止手改，用脚本重建）

`H:\software\ASMR翻译器\version.json` 是更新检查的 feed。手改 JSON 极易截断/引号事故（v1.30.2 踩过），固定用脚本：

```bash
cd "H:/software/ASMR翻译器" && PYTHONIOENCODING=utf-8 \
 "D:/ASMR翻译器/package/ASMR自动翻译工具/env/python.exe" - << 'EOF'
import json, subprocess
raw = subprocess.run(['git','show','HEAD:version.json'], capture_output=True).stdout.decode('utf-8-sig')
d = json.loads(raw)
d['version'] = '<版本>'
d['url']  = f'https://github.com/T2887/asmr-translator/releases/download/v<版本>/ASMR-Translator-Setup_v<版本>.exe'
d['page'] = 'https://github.com/T2887/asmr-translator/releases/tag/v<版本>'
d['notes'] = '''<给用户看的更新说明，写法见第 6 节>'''
json.dump(d, open('version.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
json.load(open('version.json',encoding='utf-8')); print('JSON 合法 OK')
EOF
```

要点：`runtime_patch` 字段（Haas-ja 补丁 365MB）只在补丁本身变化时改；`runtime.note` 是环境包体积提示；改完必须回读验证 JSON 合法。

## 5. 提交 → 标签 → 发布

```bash
cd "H:/software/ASMR翻译器"
git add app/src/config.rs version.json package/ASMR自动翻译工具/app/<本次改的文件...>
#    ↑ 按具体路径 add！git add -A 会误吞 _androidsetup(865MB)/_ndk(5GB)/_sv(894MB)/_lora 等
git commit -m "chore: v<版本> <一句话>"
git tag -a v<版本> -m "v<版本> <一句话>"
git push origin main v<版本>          # GitHub: T2887/asmr-translator

# Release（notes 用临时 md，别塞长参数）
gh release create v<版本> --title "v<版本> <标题>" --notes-file <(echo "…")

# 上传资产：文件名必须纯 ASCII（gh 会把中文名 Assets 剥成 1_.exe）
gh release upload v<版本> "D:/ASMR翻译器/package/安装包/ASMR-Translator-Setup_v<版本>.exe"
# 源码包
cd "H:/software/ASMR翻译器" && git archive --format=zip v<版本> > "D:/ASMR翻译器/package/安装包/asmr-translator-<版本>-src.zip"
gh release upload v<版本> "D:/ASMR翻译器/package/安装包/asmr-translator-<版本>-src.zip"
gh release view v<版本> --json assets --jq '.assets[] | .name + " " + (.size|tostring)'   # 核对

# Gitee（更新 feed 的真身）：独立仓库目录，分支是 main
cd "H:/software/ASMR翻译器/_gitee_push"
git config credential.helper wincred
cp "H:/software/ASMR翻译器/version.json" .
git add version.json && git -c user.name=T2887 -c user.email=t2887@users.noreply.github.com commit -m "chore: v<版本>"
git push origin main

# 线上验证（Gitee raw 会 302，必须跟重定向）
curl -sL --max-time 30 "https://gitee.com/aaa15/asmr-translator/raw/main/version.json" | head -3
curl -sI --max-time 30 "https://github.com/T2887/asmr-translator/releases/download/v<版本>/ASMR-Translator-Setup_v<版本>.exe" | head -1   # 302=正常
```

**网盘不在自动化范围**：b2 重打过才需要更新百度/夸克网盘的运行环境（手动），version.json 里的网盘链接不用动；b1 的新版靠 GitHub「检查更新」直达。

## 6. 给用户的 notes 写法（version.json / Release 页）

- 零术语：说「提交额度/虚拟内存」之前先想用户不懂——写成「内存不够时会自动分批处理，不再直接报错」
- 包体积变化要主动解释（例：「本次更新包约 90 MB，比之前大 70 MB，是因为装入了翻译增强」），否则用户以为中毒
- 不宣传旧版本缺陷、不写内部基准数字；「慢」字出现在收益旁边时要配「纯 CPU 自动保持原速度」这类兜底
- 条列 3~5 条，一条一行，改动大的放前面

## 7. 踩坑清单（每条都真实发生过）

1. **`git add -A` 禁用了**——仓库根下有多个巨型实验目录，永远按路径 add
2. **gh 中文名资产会被剥名**——上传文件名一律 ASCII
3. **Gitee 分支叫 main**（不是 master）；push 用 ops_bash，pwsh Start-Process 会被安全网误拦
4. **Gitee raw 返回 302**——验证 feed 必须 `curl -L`
5. **version.json 别手改**，用第 4 节脚本（缩进 indent=1、UTF-8 无 BOM 读）
6. **安装目录没有 env\python.exe**——安装冒烟用发布目录的 env python，cwd 指到安装目录
7. **b2 的 `model\*` 通配会把新塞进 b1 的小模型再打一遍**——加 Excludes（见 b2_runtime.iss 的 `hy-mt2-asmr-lora.gguf`）
8. **Inno 不保留符号链接**——HF 缓存 blobs\ 必须排除（b2 已排除，动了 model 目录时别去掉）
9. **装 Program Files 会被 iss 里 [Code] 拦**——这是特性不是 bug，脚本目录可写才行
10. **改完 .py 先 `env/python.exe -m py_compile` 再打包装**——安装冒烟是最后防线，不是第一防线
11. **Light ASR/翻译脚本是「cwd 相对路径」设计**——任何测试都要 `cd` 到含 0pre/2vad 的目录再跑；别设 ASMR_DATA_DIR 指到临时目录（models 找不到）
12. **HF 离线**：本地跑脚本带 `HF_HUB_OFFLINE=1`（huggingface.co 不可达，会卡 10060 超时）
13. **一次未发布批次只 +1 个版本号**——多处改动攒够再升版，别按文件升

## 8. 快速自检（发布前 30 秒过一遍）

- [ ] config.rs / b1.iss / b2.iss 三处版本号一致
- [ ] D 盘改动已复制进 H 盘仓库镜像（diff 过）
- [ ] cargo test 全过，安装包静默装-跑-卸通过
- [ ] 被改的流水线脚本在**安装目录**里真实跑通（日志行/产物核对）
- [ ] version.json 脚本重建 + JSON 回读合法
- [ ] GitHub Release 资产名与大小核对；Gitee feed 跟重定向后是新版本
- [ ] 上一个已发布 Release 没被误删（历史版本用户可能还没升上来）
