# A 卡（AMD）加速：PyTorch 官方 Windows ROCm 轮子 —— 实测说明

> 记录时间：2026-11-17 夜。所有数据来自本机实测，非推断。
> 相关文档：[DirectML可行性结论.md](DirectML可行性结论.md)（ONNX/DirectML 路线，结论是不可用）

## 一句话结论

**能装、能 import、能枚举到 A 卡**——PyTorch 官方 Windows ROCm 轮子是真的；但**必须按显卡的 gfx 架构选对包**，装错架构会报一个极具误导性的错（`hipErrorInvalidImage`，看起来像驱动坏了）。

## 一、架构 → 索引对照（2026-11-17 实测索引条目数）

索引根：`https://rocm.nightlies.amd.com/v2/`（正式）与 `https://rocm.nightlies.amd.com/v2-staging/`（较新但有波动）。

| 架构键 | 显卡 | v2 正式 | v2-staging | 说明 |
|---|---|---|---|---|
| `gfx120X-all` | RDNA4：RX 9060/9070/9070XT（gfx1200/1201） | 1393 个 cp311 win 轮子 | 2311 | **覆盖最好** |
| `gfx110X-all` | RDNA3：7900XTX/7800XT/7700XT/780M（1100/1101/1103） | 910 | 2220 | |
| `gfx110X-dgpu` | 同上（独显专用细分） | 267 | 161 | |
| `gfx103X-all` | RDNA2 全家 | 0 | 1061 | 只用 staging |
| `gfx103X-dgpu` | RDNA2 独显（gfx1030 等） | 0 | 840 | 只用 staging |
| `gfx101X-dgpu` | RDNA1：5700XT 等 | 0 | 1833 | 只用 staging |
| `gfx1151` | Strix Halo APU | 898 | 2454 | |
| `gfx1150` / `1152` / `1153` | RDNA3.5 APU（890M 一类及更新） | 0 | 2364 / … | 只用 staging |
| `gfx90X-dcgpu` `gfx94X-dcgpu` `gfx950-dcgpu` `gfx908/90a/906/900` | CDNA 数据中心卡 | — | — | 与消费级无关 |

要点：**「卡更新」不等于「用不了」**——RDNA4 反而是覆盖最全的一档。真正用不了的情况只有两种：
1. 架构比索引里最新的一档还新（目前不存在）；
2. **同一架构族里的边缘型号**，例如核显：`gfx103X-dgpu` 包里只有 `gfx1030` 一类的独显 kernel（在本机 `torch-...gfx103X` 轮子里 grep 架构串，只见到 `gfx1030`，**没见到本机核显的 `gfx1036`**）。

## 二、安装命令

```bash
# RDNA4（9060/9070/9070XT）
pip install --pre torch torchaudio torchvision rocm[libraries] \
  --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/

# RDNA3（7900XTX/7800XT/7700XT/780M）
pip install --pre torch torchaudio torchvision rocm[libraries] \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/

# RDNA2（6700XT 等独显）
pip install --pre torch torchaudio torchvision rocm[libraries] \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/

# Strix Halo / 其它 APU
--index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151/
```

实测体积（gfx120X 全套）：torch 271.8MB + `rocm-sdk-core` 722.7MB + `rocm-sdk-libraries` 666MB + `rocm-sdk-devel` 1.21GB ≈ **3GB 安装完成 5.1GB**（devel 可选，仅为 `rocm[devel]` 需要）。

## 三、验收命令（装完立刻跑）

```python
import torch
print(torch.__version__, torch.version.hip, torch.version.cuda)   # hip 有值、cuda 为 None = ROCm 版
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_arch_list())                                  # 必须包含你的 gfx 架构
a = torch.randn(512, 512, device="cuda"); b = torch.randn(512, 512, device="cuda")
print((a @ b).sum())                                               # 真正的 kernel 执行测试
```

`get_arch_list()` 不含你的架构 → 装错包了，换索引。

## 四、踩过的坑（都是实测）

1. **装错架构 → `AcceleratorError: CUDA error: device kernel image is invalid`（`hipErrorInvalidImage`）**。这个错完全不提架构，极易误判成「ROCm 在 N 卡不可用」或「驱动坏了」。本机就是：核显是 `gfx103x`，装了 `gfx120X`。
2. **轮子 URL 不是项目子目录**：索引页 href 是 `../<wheel>`，正确地址是
   `https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/<wheel>`
   而**不是** `.../gfx103X-dgpu/rocm-sdk-core/<wheel>`（后者返回 S3 `NoSuchKey`）。
3. **pip 在网络抖动时会从头重下**：`rocm-sdk-core` 722MB 下到一半超时，pip 重试即归零。改 `curl -C -` 断点续传（`--retry`、`--connect-timeout 30`）稳定得多。
4. **走代理会更慢**：实测同一文件直连 88KB/s、走 Clash 7897 端口 13.7KB/s。下载这批轮子应当**直连或把域名加进直连规则**。
5. **并发连接无效**：同一 IP 同时开多条 Range 请求，除第一条外全部返回 0 字节。
6. `rocm` 元包是 `sdist`（会现场构建），`rocm-sdk-*` 是预编译 wheel。

## 五、对程序的影响（待改）

我们的 `detect_device()`（Rust 端 `pipeline.rs`）只看 `torch.cuda.is_available()` 与显存数值，遇到 A 卡核显会打印：

```
GPU:AMD Radeon(TM) Graphics|23.9|0.0|10.3
```

即**把核显的共享内存当成 23.9GB 显存**，全量模式（门槛 8GB）会被放行然后崩。需要按厂商/架构分支：

- 识别厂商与 `gfx` 架构；
- AMD + `gfx10xx/11xx` 的**核显**：显存按共享算，不能用于全量判定的「8GB」门槛；
- AMD 独显：走 ROCm 分支，且必须核对 `get_arch_list()` 是否覆盖本卡。

## 六、未验证项（诚实标注）

- **核显（gfx1036）能否真正执行**：`gfx103X-dgpu` 包里 grep 不到 `gfx1036` 架构串，预期会继续报 `hipErrorInvalidImage`；真机端到端验证进行中（见下）。
- 真 A 卡（RDNA3/RDNA4）上的 BS-RoFormer 端到端速度与听感：本机没有 A 卡，无法代验。
- `rocm[devel]` 是否必需：本次先不装 devel 尝试，若 import 失败再补。

## 七、实测结果（2026-11-18 凌晨，本机 AMD Radeon 核显 gfx1036）

### 7.1 `-dgpu` 包：本机核显跑不了

```
torch 2.13.0a0+rocm7.13.0a20260421  hip 7.13.26154
device  AMD Radeon(TM) Graphics  gcnArchName = gfx1036
arch_list ['gfx1030','gfx1031','gfx1032','gfx1034']      ← 无 gfx1036
matmul  AcceleratorError: CUDA error: device kernel image is invalid (hipErrorInvalidImage)
HSA_OVERRIDE_GFX_VERSION=10.3.0 在 Windows 上无效
```

### 7.2 `-all` 包：核显能跑，且数值正确

换成 `https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/`（torch 2.9.1+rocm7.14.0a20260612、rocm_sdk_core 744.9MB、rocm_sdk_libraries_gfx103x_all 235.3MB）后：

```
torch 2.9.1+rocm7.14.0a20260612  hip 7.14.60850
arch_list ['gfx1030','gfx1031','gfx1032','gfx1033','gfx1034','gfx1035','gfx1036']   ← 含 gfx1036
GPU matmul OK
```

BS-RoFormer 端到端（同一段真实音频、8.0s chunk = 352800 样本 @44.1k 立体声）：

| 设备 | 单帧耗时 | 相对实时 |
|---|---|---|
| CPU（torch 2.9，同 venv） | 671.9 s | 84× 慢于实时 |
| **核显 gfx1036（HIP）** | **40.9 s** | 5.1× 慢于实时（比 CPU 快 16.4×） |
| 参考：RTX 5070 CUDA fp32（生产 env） | 1.03 s | 7.8× 快于实时 |

数值一致性：`max|Δ| = 1.053e-05`、`mean|Δ| = 1.493e-06`、相对误差 **1.4e-05**（信号峰值 7.46e-01）→ **与 CPU 结果等价，能算对**。

### 7.3 结论

1. **ROCm Windows 轮子在 A 卡上可用**，PyTorch 侧无需改一行代码（`device="cuda"` 即 HIP）。
2. **必须装对架构包**：同是 gfx103X，`-dgpu` 只编了独显架构，APU 需要 `-all`。
3. **核显（APU）能跑但太慢**：比 CPU 快 16×，但仍比实时慢 5×，做 ASMR 分离没有实用价值；**A 卡提速的真实价值在独显**（RDNA3/RDNA4 的算力是这块 12CU 核显的十几倍）。
4. 安装器该做的事：检测厂商 + `gcnArchName` → 选 `gfx…-all`/`gfx…-dgpu` → 核对 `get_arch_list()`。
5. ⚠️ 换了 ROCm 版 torch（2.9.1）后，`torchaudio`/`torchvision` 也要跟着装对应 ROCm 版本，否则 ABI 不匹配。

## 八、复现命令（本机实测过的完整流程）

```bash
# 1) 建 venv（Python 3.11）
python -m venv H:/_probe_rocm/venv103
H:/_probe_rocm/venv103/Scripts/python.exe -m pip install wheel
H:/_probe_rocm/venv103/Scripts/python.exe -m pip install --no-deps einops beartype rotary-embedding-torch omegaconf pyyaml

# 2) 下三个轮子（直连更快；断了用 curl -C - 续传）
B=https://rocm.nightlies.amd.com/v2-staging/gfx103X-all
curl -sL -C - -O $B/torch-2.9.1%2Brocm7.14.0a20260612-cp311-cp311-win_amd64.whl
curl -sL -C - -O $B/rocm_sdk_core-7.14.0a20260612-py3-none-win_amd64.whl
curl -sL -C - -O $B/rocm_sdk_libraries_gfx103x_all-7.14.0a20260612-py3-none-win_amd64.whl
curl -sL -O $B/rocm-7.14.0a20260612.tar.gz          # 提供 rocm_sdk 模块

# 3) 本地装（--no-deps，避免 pip 从 PyPI 拉 CUDA 版 torch）
pip install --no-index --find-links . --no-deps torch-*.whl rocm_sdk_core-*.whl rocm_sdk_libraries_*-all*.whl
pip install --no-build-isolation --no-deps rocm-*.tar.gz

# 4) 验收
python -c "import torch;print(torch.cuda.get_arch_list(), torch.cuda.is_available())"
```

探针脚本：`H:\_probe_rocm\rocm_e2e.py`（CPU/HIP 波形比对）、`H:\_probe_rocm\asesep\`（剥离出的 BS-RoFormer 最小包）。
