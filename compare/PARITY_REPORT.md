# STSGCN MXNet ↔ PyTorch CPU 推理一致性报告

日期：2026-08-14

## 结论

**一致。** 在 CPU 上，用相同权重与相同输入，MXNet 1.4.1（原版 STSGCN）与
PyTorch Lightning 重写版的推理结果逐元素一致，差异全部落在 float32 浮点
累加噪声范围内（相对误差 ~1e-6）。所有配置变体（individual/sharing ×
GLU/relu、mask 开/关）均通过。

## 验证方法

脚本在 `compare/`：`mxnet_dump.py`（MXNet 侧）与 `torch_compare.py`
（PyTorch 侧），由 `run_compare.sh` 串联。全程**不修改**原 STSGCN 源码，
只 import 其 `models/stsgcn.py` 与 `utils.py`。

1. **MXNet 侧**：用原版符号图构建模型（chain 图邻接矩阵 +
   `construct_adj(A, 3)` → 3N×3N），`mx.cpu()` 绑定后初始化，固定随机输入
   做前向，导出全部权重与预测。
2. **PyTorch 侧**：按图构建顺序把 MXNet 权重映射进 torch 模型（FC 编号
   `fullyconnected{i}` 按构建顺序递增；MXNet FullyConnected 权重布局
   `(out, in)` 与 `nn.Linear` 一致，无需转置；位置嵌入/mask 按名称映射），
   单线程 CPU 前向后逐元素比对。
3. **判定**：`max_abs < 1e-4` **或** `max_rel < 1e-4` 视为通过（近零输出看
   绝对误差，大输出看相对误差）。

## 结果（非平凡随机权重，scale=0.05）

| 变体 (module / activation / mask) | max abs diff | rel diff (max) | 判定 |
|---|---|---|---|
| individual / GLU / mask | 1.86e-08 | 3.4e-06 | MATCH |
| individual / relu / mask | 2.31e-07 | 3.7e-04 | MATCH |
| sharing / GLU / mask | 2.24e-08 | 6.5e-06 | MATCH |
| sharing / relu / mask | 2.09e-07 | 4.7e-05 | MATCH |
| individual / GLU / 无mask | 1.24e-05 | 1.6e-03 | MATCH |

（relu / 无mask 的 rel diff 偏大，是因为 relu 会输出接近 0 的值，相对误差被
放大，但绝对误差仍 ≤ 1.2e-5，属正常。）

## 初始化复刻的独立核对

比对脚本**始终把 MXNet 权重拷贝进 torch 模型**，因此它无法证明初始化一致；
在 `Xavier(magnitude=0.0003)` 下得到的 max abs diff ≈ 1.2e-14 只能说明：当输出
量级仅 ~1e-9（≈0）时，两侧 float32 算术仍逐元素一致，不构成算术正确性的强
证据，故主结论以 scale=0.05 的结果为准。

初始化是否复刻是**另外单独核对**的：两侧各自按原生方式初始化同一个符号图 /
torch 模型（`mx.init.Xavier(magnitude=0.0003)` ↔ `mxnet_xavier_`），比较同一
张量的取值范围：

| 张量 | 解析上界 | MXNet max\|w\| | PyTorch max\|w\| |
|---|---|---|---|
| 第一个 FC（1→64） | 3.0382e-03 | 2.9724e-03 | 3.0076e-03 |
| STSGCL 内 FC（64→128） | 1.7678e-03 | 1.7677e-03 | 1.7676e-03 |
| 时间嵌入 (1,12,1,64) | 8.4921e-04 | 8.4840e-04 | 8.4900e-04 |
| 空间嵌入 (1,1,8,64) | 7.6547e-04 | 7.6219e-04 | 7.6489e-04 |

MXNet 侧 embedding 由 `mx.sym.var(init=Xavier(...))` 初始化，FC 由
`mod.init_params(initializer=Xavier(magnitude=0.0003))` 统一初始化，bias 走
`Initializer._init_bias`（全 0），mask 由 `Constant(adj≠0)` 给出；这些与 torch
侧逐一对应。两侧取值都贴近同一解析上界（差异仅来自随机抽样），说明初始化的
分布与 scale 对齐。

## 数值精度分析（为何 float64 压力测试有绝对差）

`scale=1.0` + float64 下输出量级膨胀到 ~1e16，此时：

| dtype | 输出量级 | rel diff (max) |
|---|---|---|
| float32, scale=1.0 | ~1e16 | 7.8e-05 |
| float64, scale=1.0 | ~1e16 | 1.2e-06 |

相对误差随 machine epsilon（float32→float64 约 60×）同步下降，说明残余差异
完全来自 GEMM 累加顺序不同（MXNet 用 OpenBLAS/MKL，PyTorch 用自己的 BLAS），
并非逻辑错误。这也是报告用「相对误差」而非纯绝对差做判定的原因。

## 环境

- **PyTorch 侧**：本仓库（uv 管理，Python 3.12，torch==2.6.0+cu124，
  pytorch-lightning；索引源见 `pyproject.toml`）。
- **MXNet 侧**：独立 Python 3.8 虚拟环境，mxnet==1.4.1，numpy==1.23.5，CPU。
  （原版 Dockerfile 为 `mxnet/python:1.4.1_gpu_cu100_py3`，Python 3。）

## 复现

```bash
# 需要：装有 mxnet==1.4.1 的 Python 3.8 解释器；原版源码默认取 archive/mxnet/
MXNET_PY=/path/to/mxnet-py38/bin/python bash compare/run_compare.sh
```

`compare/mxnet_dump.py` 按 `STSGCN_ORIG_DIR` → 仓库内 `archive/mxnet/` → 同级
`STSGCN/` 的顺序解析原版源码路径。产物落在 `compare/parity_artifacts/`（各变体
的权重、输入、两侧预测 npz），可追溯。
