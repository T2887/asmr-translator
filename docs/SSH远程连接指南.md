# 远程连接指南：翼龙 15 Pro / CT5

> 更新：2026-11-22 · 两台笔记本都是 Windows 11 + OpenSSH Server
> 用途：DSH 关掉之后，靠 SSH 直接控制两台机器跑 ASR / 评测 / 传文件

---

## 一、两台机器的身份

| | **翼龙** | **CT5** |
|---|---|---|
| 别名（rsh 用） | `翼龙` | `CT5` |
| 主机名 | `AAA20` | `DESKTOP-G5LS7AG` |
| IP | `192.168.31.193` | `192.168.31.204` |
| 用户名 | `Dayje` | `AAA` |
| 密码 | `Taku001115` | `Taku001115` |
| 域名前缀 | `aaa20\` | `desktop-g5ls7ag\` |
| 端口 | 22 | 22 |
| 内存 | 31.3 GB | 31.8 GB |
| CPU | AMD Ryzen 7 8845H（16 逻辑核） | Intel i5-9300H（8 逻辑核） |
| GPU | RTX 4060 Laptop 8GB + Radeon 780M | GTX 1660 Ti + UHD 630 |
| ASR 实测 RTF | **0.132**（8 线程） | **0.248**（8 线程） |
| 翻译器位置 | `D:\Program Files\ASMR自动翻译工具\` | `D:\Program Files\ASMR自动翻译工具\` |
| 评测机 | `D:\_eval\`（llama.cpp + 模型） | 未部署 |
| ASR 工作目录 | `D:\_asr\` | `D:\_asr\` |
| 账户是否管理员 | **是**（能直接改防火墙） | 是 |

两台都装了翻译器（Python 3.11.8 + Haas-ja 模型 + llama-server），所以**不需要再装 Python 环境**。

---

## 二、怎么连（三种方式）

### 1. 命令行直接连（最简单，手动操作时用）

```bash
ssh Dayje@192.168.31.193      # 翼龙
ssh AAA@192.168.31.204        # CT5
```

密码 `Taku001115`。第一次连会问 `yes/no`，输 `yes`。

### 2. 我用 `rsh.py` 跑远程命令（自动化首选）

工具在 `H:\_tools\rsh.py`，主机配置在 `H:\_tools\hosts.json`：

```bash
# 用生产环境的 python（已带 paramiko 5.1.0）
P="D:/ASMR翻译器/package/ASMR自动翻译工具/env/python.exe"

# 跑一条命令
"$P" -X utf8 H:/_tools/rsh.py 翼龙 "hostname; (Get-Date)"

# 长任务给足超时（毫秒）
"$P" -X utf8 H:/_tools/rsh.py --timeout 1800000 翼龙 "长命令"

# 直接给 IP 也可以（不用别名）
"$P" -X utf8 H:/_tools/rsh.py --host 192.168.31.193 --user Dayje --pw Taku001115 "命令"
```

### 3. 用 `rpush.py` 传文件（单文件或整个目录）

```bash
P="D:/ASMR翻译器/package/ASMR自动翻译工具/env/python.exe"

# 单个文件
"$P" -X utf8 H:/_tools/rpush.py 翼龙 "H:/_asr_worker/asr_worker.py" "D:/_asr/asr_worker.py"

# 整个目录
"$P" -X utf8 H:/_tools/rpush.py CT5 "H:/_asr_worker" "D:/_asr"
```

---

## 三、关键坑：Windows 防火墙（一定要看）

**症状**：SSH 突然连不上，`TimeoutError: [WinError 10060]`。但机器 ping 得通、135 端口也通。

**原因**：Windows 自带那条 `OpenSSH SSH Server (sshd)` 防火墙规则只在 `Domain` / `Private` 网络类型下生效。一旦 Windows 把当前网络判定成 `Public`（换网络、重启、连了新 WiFi 都会发生），这条规则就静默失效。

**修法**（在目标机器上，管理员 PowerShell）：

```powershell
New-NetFirewallRule -DisplayName "SSH-22-In" -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow -Profile Any
```

关键在于 `-Profile Any` —— 不区分网络类型，永久有效。

**两台都已加过这条规则**（2026-11-22）。如果哪天又断了，先查这个：

```powershell
Get-NetFirewallRule -DisplayName "SSH-22-In" | Select-Object DisplayName,Enabled,Action
(Get-Service sshd).Status
```

正常应该是 `Enabled: True / Action: Allow` 且 `sshd: Running`。

---

## 四、故障排查顺序

按这个顺序查，能快速定位：

```bash
# 1. 机器在不在网？扫全段（22/135/445）
#    只有 135 通 = 开机了但 sshd 或防火墙有问题
#    22 通     = 正常
#    全不通    = 关机 / 休眠 / 断网

# 2. sshd 服务起没起（机器通了 135 但 22 不通时先查这个）
ssh ... "Get-Service sshd"

# 3. 防火墙规则在不在
ssh ... "Get-NetFirewallRule -DisplayName 'SSH-22-In'"
```

### 已知的几种「机器不在网」情况及处理

| 现象 | 原因 | 处理 |
|---|---|---|
| 全段扫不到，`Test-Connection` 也不通 | 休眠 / 关机 / WiFi 断 | 去那台机器按电源键唤醒，检查 WiFi |
| 135 通但 22 不通 | 防火墙规则失效（网络被判成 Public） | 补 `SSH-22-In` 规则 |
| 22 通但登录失败 | 密码错 / 账户被锁 | 检查 `hosts.json` 里的密码 |
| 命令能连但中文乱码 | PowerShell 输出 GBK | `rsh.py` 已内置 UTF-8 处理，直接用别手动改 |

### 兜底手段

如果 SSH 彻底不通了：**重新打开那台机器上的 DSH 窗口**，用窗口打字让它自检修复。DSH 不要卸载，它是唯一的手动通道。

---

## 五、常用操作速查

```bash
P="D:/ASMR翻译器/package/ASMR自动翻译工具/env/python.exe"

# 看两台是不是活着
for t in 翼龙 CT5; do "$P" -X utf8 H:/_tools/rsh.py --timeout 30 $t "hostname"; done

# 看 ASR 进度（结果 json 个数）
"$P" -X utf8 H:/_tools/rsh.py 翼龙 "(Get-ChildItem D:\_asr\out\*.json).Count"

# 跑评测（翼龙，120 条约 53 秒）
"$P" -X utf8 H:/_tools/rsh.py 翼龙 "cd D:\_eval; python eval_remote.py val120.json out.json"

# 停 ASR（计划任务 + 进程一起杀）
"$P" -X utf8 H:/_tools/rsh.py CT5 "schtasks /End /TN AsrShard; Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*asr_worker*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
```

---

## 六、重要提醒

1. **`rsh.py` 跑长任务要给 `--timeout`**，默认 120 秒太短，ASR 单文件可能要几分钟。
2. **SSH 会话结束会杀掉它起的子进程**。要跑长任务，用计划任务：
   ```
   schtasks /Create /TN 任务名 /TR "命令行" /SC ONCE /ST 00:00 /RL HIGHEST /F
   schtasks /Run /TN 任务名
   ```
3. **本机不能用密码以外的认证方式**：`~/.ssh` 被 DSH 安全策略当作敏感路径（`secret.home.ssh`）拦截，读写不了，所以只能走密码。
4. **两台机器都不要关机或休眠**，一旦断了就得人去现场。翼龙是笔记本，合盖会休眠 —— 建议插电并设成「合盖不睡眠」。
5. 本机 IP 是 `192.168.31.160`，素材 HTTP 服务就在这台（`H:\_asr_run\serve.py`，端口 8765）。

---

## 七、文件清单

| 路径 | 作用 |
|---|---|
| `H:\_tools\hosts.json` | 主机别名、IP、用户名、密码 |
| `H:\_tools\rsh.py` | 远程执行命令（paramiko） |
| `H:\_tools\rpush.py` | 远程传文件 / 目录（sftp） |
| `H:\_tools\probe_llama.py` | 翼龙评测机健康检查 |
| `H:\_tools\eval_remote.py` | 翼龙纯标准库评测脚本 |
| `H:\_tools\start_eval_server.bat` | 翼龙 llama-server 启动脚本 |
