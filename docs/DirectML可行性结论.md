# BS-RoFormer 走 ONNX + DirectML 的可行性结论

> 结论先行：**DirectML 这条路不可用（数值错误，且无法在本机修掉）**。
> A 卡加速请走 **PyTorch 官方 Windows ROCm 轮子**（`torch+rocm` nightly），
> 分离模型原样就能跑 GPU，工程代价比自造 ONNX+DML 管线低得多。
> 实测日期 2026-11-17，验证机 RTX 5070 12GB（本机没有 A 卡，DML 只能间接验证）。

## 1. 起因

用户提问「兼容模式/分离能不能用 A 卡加速」。当时结论「不能用 A 卡加速」的准确含义是
**最快的链路是 CUDA 专属**：

| 层 | 组件 | Windows 上的 A 卡现状 |
|---|---|---|
| ① 分离 BS-RoFormer | PyTorch | 官方轮子只有 CUDA / CPU；ROCm 需装 AMD nightly（见 §5） |
| ② 识别 faster-whisper | CTranslate2 | 官方轮子只有 CUDA，HIP 要自编译 |
| ③ 翻译 llama.cpp | Vulkan | **A 卡本来可用**，兼容模式已经在用 |

第 ① 层当时的候选方案就是「导出 ONNX，用 DirectML 跑」（跨厂商 API，N/A 卡通吃）。

## 2. 做了什么

不改主程序，探针全部放在 `D:\ASMR翻译器\_probe_dml\`：

- `export_net.py`：包一层 `NetOnly` 只导出网络本体（STFT/ISTFT 留在图外），
  输入 real 谱 `(1,2050,801,2)`，输出 mask `(1,1,2050,801,2)`，opset 17。
- `bench_torch.py` / `dml_bench.py`：四种后端的单帧（8.0s 音频）耗时。
- `e2e_parity.py`：同一段输入分别在 CPU EP 与 DML 上跑，比波形。
- `locate_dml.py` / `locate_dml2.py`：用 `onnx.utils.extract_model` 逐点导子图，二分定位错误区间。
- `tiny_repro.py` / `concat62.py` / `bands.py`：隔离小图复现（验证是否算子级问题）。

## 3. 实测数据

**导出**：参数量 159.8 M（fp32 609MB），ONNX 615 MB，5617 节点 / 25 类算子
（Constant 1863、Mul 596、Add 378、MatMul 354、Squeeze 230、Reshape 218、Cast 207、
Slice 192、Div 159、Unsqueeze 159、Split 136、Shape 135、Concat 123、Transpose 122、
ReduceL2 111、Clip 111、Expand 111、Sigmoid 86、Sqrt 72、Tanh 62、Cos 48、Neg 48、
Sin 48、Softmax 24、Erf 24）。
**没有 STFT / complex / RNN 这类 DirectML 已知不支持的算子。**

**耗时（单帧 = 8.0 秒音频）**

| 后端 | 耗时 | 相对 CPU |
|---|---|---|
| torch CUDA fp32 | 1026 ms | 20× |
| torch CUDA fp16 | **379 ms** | 55× |
| **ORT DirectML 1.24.4** | **1158~1241 ms** | **约 19×** |
| ORT CPU EP | 23218 ms | 1× |
| torch CPU | 20972 ms | 1× |

**数值一致性（关键）**

| 组合 | 波形 max\|Δ\| | 结论 |
|---|---|---|
| ORT CPU EP vs torch | 2.241e-08 | 位级一致，导出正确 |
| mask 层 CPU EP vs torch | 5.1e-06 | 正确 |
| **ORT DirectML vs CPU EP** | **1.631e-02（mask 层 5.540）** | **错**，且参考信号幅度只有 1.88e-04 → 偏差是信号的 87 倍 |
| DirectML 连跑两次 | 0.000e+00 | **稳定地错**，不是随机噪声 |

## 4. 根因定位（已缩小到区间，未定位到单一算子）

逐点导出子图对比（`locate_dml2.py`）：idx 2400~5414 之间 12 个探针全部正常
（rel 1.9e-07 ~ 1.6e-06），**只有最后 `Transpose_25`(perm=[0,1,3,2,4]) → 输出 `mask` 错
（rel 1.700）**。再往上游查（`tail_check.py` / `tail2.py`）：
`/mask_estimators.0/Concat_output_0`、`/Unsqueeze_output_0`、`/Concat_24_output_0`、
`/Reshape_49_output_0` **全部已经是 max|Δ| = 5.540**，也就是错误在「62 个频带块
Concat 之后」就已经存在。

而隔离小图全部正确（max|Δ| = 0.000e+00）：纯 5 维 Transpose、Reshape+Transpose、
Reshape+Identity+Transpose、Reshape+Cast+Transpose；62/32/16/8/4/2 输入的
`Concat(axis=-1)`。抽查 7 个频带块输出则都错（band0 max 2.736、band10 1.3e-02、
band61 5.540）。

**结论：不是某个算子单独算错，而是 DirectML 在这张大图上的内核融合 / 内存布局问题**
（同一算子换个截断点结果就变；隔离小图全对、集成大图全错）。
已试过：优化级别（含 DISABLE_ALL）、`enable_mem_pattern=False`、动态图融合、
fp32 常量，误差一字不差。

## 5. 推荐路线：PyTorch 官方 Windows ROCm 轮子

AMD 的 nightly 索引确实提供 **Windows x86_64 轮子**（2026-11 核实）：

- `https://rocm.nightlies.amd.com/v2/gfx120X-all/`（2805 条目，RDNA4：9060/9070/9070XT）
- `https://rocm.nightlies.amd.com/v2/gfx110X-all/`（2212 条目，RDNA3：7900XTX/7800XT/7700XT/780M）
- 其它：Strix Halo `gfx1151`、RDNA2 `v2-staging/gfx103X-dgpu/`

索引里确有 `torch-2.9.1+rocm7.13.0a20260424-cp311-cp311-win_amd64.whl`，
**我们的 env 是 Python 3.11.8，版本对得上**（现有 torch 2.11.0+cu128 会降级到 2.9.1）。

```bat
pip install --pre torch torchaudio torchvision rocm[devel] ^
  --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/
```

- RDNA4 → `gfx120X-all`；RDNA3/780M → `gfx110X-all`；按显卡架构选，选错装上也跑不了。
- 注意事项：nightly 预发布；MIOpen 在部分架构不稳，建议 `set MIOPEN_FIND_MODE=FAST`；
  可能需要 `rocm-sdk init`；已知问题见 https://github.com/ROCm/TheRock/issues/808 。
- 工程做法：安装器里加设备检测分支 —— N 卡装 cu128，A 卡装 rocm，其它保持 CPU。
  这样 BS-RoFormer 代码一行不改就能在 A 卡上跑 GPU。

## 6. 如果将来要重新评估 DirectML

只有满足以下之一才值得再试，否则别浪费工时：

1. `onnxruntime-directml` 版本大跳（当前 1.24.4），重新跑 `e2e_parity.py` 看 DML 的
   `max|Δ|` 是否降到 1e-04 以下；
2. 有人证明 BS-RoFormer 的 ONNX 在 DML 上能出正确结果（那是我们导出姿势的问题）；
3. 出现「分块推理」方案（把 62 个频带拆成独立小模型分别跑 DML，每块 1.2s×62 太慢，不现实）。

## 7. 复现方式

```bash
cd "D:/ASMR翻译器/_probe_dml"
D:/ASMR翻译器/package/ASMR自动翻译工具/env/python.exe -X utf8 export_net.py     # 导出 + 算子统计
D:/ASMR翻译器/package/ASMR自动翻译工具/env/python.exe -X utf8 bench_torch.py    # torch 三档基准
D:/ASMR翻译器/package/ASMR自动翻译工具/env/python.exe -X utf8 dml_bench.py      # DML 会话与耗时
D:/ASMR翻译器/package/ASMR自动翻译工具/env/python.exe -X utf8 e2e_parity.py     # 波形一致性（会打印 DML 偏差）
D:/ASMR翻译器/package/ASMR自动翻译工具/env/python.exe -X utf8 locate_dml2.py    # 子图二分定位
```

坑（写进经验）：

- `sys.path.insert` 必须在**第一次 import onnxruntime 之前**，否则 venv 里的 CUDA 版
  先被导入，后续再 insert 是空操作 —— 表现是 `providers` 里没有 `DmlExecutionProvider`，
  而且只打一行 warn，很容易被漏掉。
- 探针脚本不要叫 `bisect.py` 这类标准库同名文件（会遮蔽 `bisect`，引发
  `random`/`tempfile` 循环导入失败）。
- 用 heredoc（`python - <<PY`）+ 中文路径注入 `sys.path` 会静默失败，探针要写成 `.py` 文件跑。
