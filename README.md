# eagle3-tree-sparse-flashattn

基于 **PyTorch + Triton** 的「前缀稠密 + 树结构稀疏」FlashAttention 前向原型实现。

这个项目的目标不是“再写一个 attention”，而是回答一个更工程化的问题：

> 在推理阶段（尤其是 speculative decoding / tree decoding）里，如何让 Tree Attention 不再用稠密矩阵硬算所有无效分支，而是只算真正可见的 token 对？

---

## 项目解决的问题

在树状推理场景中（公共前缀 + 多分支后缀），传统做法通常是：

1. 先算完整 `Q x K` 分数矩阵；
2. 再用大 mask 把无效分支置为 `-inf`。

这会带来三类核心浪费：

- **算力浪费**：大量 FLOPs 花在本就互不可见的分支配对上；
- **带宽浪费**：读取了不需要的 K/V 缓存；
- **显存压力**：中间 `scores` 和大尺寸 mask 成本高。

本项目通过「**块稀疏 + FlashAttention 在线 softmax**」把问题改写为：

- Prefix 区域保持稠密高吞吐；
- Tree 区域只遍历有效块（CSR）；
- MIXED 块使用 bitset 掩码按位过滤。

结果是：在保留计算语义的前提下，尽可能把无效计算从“算完再丢”变为“从源头跳过”。

---

## 目录结构

- `tree_sparse_flashattn.py`
  - `TreeSparseMetadata`：树稀疏块元数据结构。
  - `_pack_block_mask_to_int32`：把块内布尔掩码打包为 int32 bitset。
  - `prepare_tree_sparse_metadata`：将 `tree_mask` 压缩为 CSR + tri-state（FULL/MIXED）。
  - `block_sparse_tree_fwd_kernel`：Triton 前向 kernel（Dense Prefix + Sparse Tree）。

> 当前仓库聚焦算子原型与元数据协议，不包含完整 benchmark.py 与训练/反向流程。

---

## 这份代码在做什么？

### 1) 稀疏元数据构建（Python 侧）

`prepare_tree_sparse_metadata` 会把二维 `tree_mask` 按 `BLOCK_M x BLOCK_N` 切块，并将每个块分为：

- `ZERO`：全 0（不进入 CSR）；
- `FULL`：全 1（不需要块内 bitmask）；
- `MIXED`：部分 1（需要 bitset）。

然后输出 CSR 元数据：

- `row_ptr`：每个 query block 的 nnz 范围；
- `col_idx`：有效 tree block 的列号；
- `blk_type`：FULL 或 MIXED；
- `mixed_mask_payload`：所有 MIXED bitset 串接；
- `mixed_mask_offset`：**与 nnz_blocks 对齐**的偏移（FULL 位置填 0，占位对齐）。

### 2) Triton Kernel 前向（GPU 侧）

`block_sparse_tree_fwd_kernel` 中每个程序实例处理一个 query block，分两阶段：

- **Phase 1: Dense Prefix**
  - 遍历 `[0, prefix_len)` 的 KV block；
  - 使用在线 softmax 累加（`m_i`, `l_i`, `acc`）。
- **Phase 2: Sparse Tree**
  - 通过 `row_ptr/col_idx` 只遍历该 query block 的有效 tree blocks；
  - 若为 MIXED block，则从 `mixed_mask_payload` 解码 bitset 做块内掩码。

并且已加入数值稳定防护：

- 越界位置 `scores -> -inf`；
- 对 `scores == -inf` 的概率项强制 `p=0`；
- 对 `-inf - (-inf)` 的 `alpha` 路径进行保护，避免 NaN 污染。

---

## 设计思路（Why it works）

### 1) 动静分离（Phase Split）

Prefix 是公共上下文，密集且规律；Tree 是分叉结构，天然稀疏。将两者拆开处理，才能同时兼顾吞吐与稀疏跳过。

### 2) 块级稀疏（Block-Sparsity + CSR）

GPU 更擅长规则块计算。用块级 CSR 而不是 token 级散乱索引，能减少调度与访存碎片，并直接跳过 ZERO 块。

### 3) 位级掩码压缩（Bitset for MIXED）

MIXED 块若传完整布尔矩阵会增加带宽压力。bitset 将 `BLOCK_M*BLOCK_N` 的布尔信息压缩到 `int32` payload，在 kernel 内位运算解包，I/O 更友好。

---

## 与传统 Tree Attention 的区别

| 维度 | 传统 Dense + Mask | 本项目 Block-Sparse FlashAttention |
| --- | --- | --- |
| 计算路径 | 先算全量，再 mask 无效 | 仅遍历 CSR 标记的有效块 |
| 中间开销 | 大 `scores` / 大 mask | 在线 softmax，避免完整中间矩阵 |
| 无效分支成本 | 仍参与 matmul | 物理层面跳过 ZERO 块 |
| MIXED 粒度 | 通常整块或逐元素 bool | bitset 压缩 + 位运算恢复 |
| 适配场景 | 通用但浪费 | Tree decoding / speculative decoding 更有优势 |

一句话概括：

- 传统：**先把所有矿都挖一遍，再扔掉废石**；
- 本实现：**拿着 CSR 地图，只挖有矿的块**。

---

## 元数据结构说明

`TreeSparseMetadata` 字段：

- `row_ptr` (`int32`)：CSR 行指针，形状 `[n_q_blocks + 1]`。
- `col_idx` (`int32`)：tree KV block 列索引，形状 `[nnz_blocks]`。
- `blk_type` (`int8`)：块类型（FULL/MIXED），形状 `[nnz_blocks]`。
- `mixed_mask_payload` (`int32`)：所有 MIXED block 位图串接。
- `mixed_mask_offset` (`int32`)：**按 nnz_blocks 对齐**的 payload 起始偏移。
- `n_q_blocks` / `n_tree_blocks`：块数量。
- `block_m` / `block_n`：块大小。
- `prefix_len`：前缀长度（Prefix 不进入 CSR）。

### Mixed 位图编码

- 将 `[BLOCK_M, BLOCK_N]` 按 row-major 拉平；
- 每 32bit 打包为 1 个 `int32`；
- `bit i` 对应平铺后索引 `i` 的位置。

---

## 快速开始

> 下述示例演示元数据构建流程。完整 end-to-end 调用还需补充 kernel launch 包装（grid、stride、dtype/device 检查）。

```python
import torch
from tree_sparse_flashattn import prepare_tree_sparse_metadata

q_len = 128
prefix_len = 32
BLOCK_M, BLOCK_N = 64, 64

# 示例 tree-only mask
# 真实业务可替换为“祖先可见、跨分支不可见”的树结构掩码
tree_mask = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool))

meta = prepare_tree_sparse_metadata(
    tree_mask=tree_mask,
    prefix_len=prefix_len,
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
)

print(meta.row_ptr.shape, meta.col_idx.shape, meta.blk_type.shape)
print(meta.mixed_mask_payload.shape, meta.mixed_mask_offset.shape)
```

---

## 正确性验证与性能评测（建议流程）

虽然当前仓库未内置 `benchmark.py`，但建议按以下方式搭建：

1. **PyTorch 基线**：实现 dense+mask 版本（作为 reference）。
2. **正确性**：同一输入下比较 Triton 输出与基线，检查 `max_abs_err` / `mean_abs_err`。
3. **性能**：固定 batch / head / seq，分别测 Triton 与基线 latency，统计 speedup。
4. **覆盖场景**：
   - prefix 长短变化；
   - 树分支稀疏度变化；
   - 不同 BLOCK 配置与 head_dim。

建议验收阈值（可按任务调整）：

- `max_abs_err <= 1e-3`（FP16/BF16 常见目标）；
- 稀疏度越高，速度优势越明显。

---

## Kernel 形状与约束（当前版本）

- 预期 grid：`(n_q_blocks, batch_size * num_heads)`。
- Q/K/V/O 采用 4D stride 显式传参语义。
- `kv_h_idx = h_idx // gqa_group_size` 支持 GQA 映射。
- `BLOCK_D: tl.constexpr` 需要与 head_dim 一致。

---

## 已知限制

当前仓库还不包含：

- backward kernel；
- 完整 Python API（一键调用包装）；
- 自动调参（autotune）与多 block 策略搜索；
- 开箱即用 benchmark/correctness 脚本。

项目定位是“核心算子+协议原型”，适合你在此基础上继续工程化。

---

## 依赖

- Python 3.9+
- PyTorch（CUDA 版本）
- Triton

```bash
pip install torch triton
```

---

## 后续可扩展方向

- 支持 causal / padding / sliding-window 与 tree mask 组合；
- 增加 per-row 稀疏统计与负载均衡；
- 对接现有 FlashAttention API 以便无缝替换；
- 补齐 benchmark 与 correctness 套件（含自动报告）。

