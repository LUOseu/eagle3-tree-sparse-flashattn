# eagle3-tree-sparse-flashattn

基于 **PyTorch + Triton** 的「前缀稠密 + 树结构稀疏」FlashAttention 前向原型实现。该仓库重点展示了：

- 如何将树掩码压缩为适合 GPU kernel 读取的块级 CSR 元数据；
- 如何在同一个 Triton kernel 内分两阶段执行注意力（dense prefix / sparse tree）；
- 如何对部分填充（mixed）块使用位图（bitset）做细粒度 mask。

> 当前代码是 forward kernel skeleton（原型），适合作为二次开发基础。

---

## 目录结构

- `tree_sparse_flashattn.py`
  - `TreeSparseMetadata`：树稀疏块元数据结构。
  - `_pack_block_mask_to_int32`：把块内布尔掩码打包为 int32 bitset。
  - `prepare_tree_sparse_metadata`：将 `tree_mask` 压缩为 CSR + tri-state（FULL/MIXED）。
  - `block_sparse_tree_fwd_kernel`：Triton 前向 kernel（两阶段 softmax 累加）。

---

## 设计概览

### 1) 两阶段注意力

kernel 按每个 query block 执行：

1. **Phase 1: Dense Prefix**
   - 对 `[0, prefix_len)` 的 KV 区间按 `BLOCK_N` 全量遍历。
2. **Phase 2: Sparse Tree**
   - 仅遍历 CSR 中该 query block 对应的非零 tree blocks。
   - 若块类型为 `MIXED_BLOCK`，在块内再按位图屏蔽无效元素。

两阶段都复用在线 softmax 归一化变量（`m_i`, `l_i`, `acc`）。

### 2) Tri-state 块类型

当前实现显式定义：

- `FULL_BLOCK = 0`：块内全部有效。
- `MIXED_BLOCK = 1`：块内部分有效，需读取 bitset。

`ZERO_BLOCK` 不单独存储（CSR 中天然省略）。

### 3) CSR 元数据

对于 `n_q_blocks` 个 query blocks：

- `row_ptr: [n_q_blocks + 1]`
- `col_idx: [nnz_blocks]`
- `blk_type: [nnz_blocks]`

其中 `row_ptr[qb]:row_ptr[qb+1]` 是第 `qb` 个 query block 的有效 tree block 列表区间。

---

## 元数据结构说明

`TreeSparseMetadata` 字段：

- `row_ptr` (`int32`)：CSR 行指针。
- `col_idx` (`int32`)：tree KV block 列索引。
- `blk_type` (`int8`)：块类型（FULL/MIXED）。
- `mixed_mask_payload` (`int32`)：所有 mixed block 的位图 payload 拼接。
- `mixed_mask_offset` (`int32`)：每个 mixed block 在 payload 内的起始偏移。
- `n_q_blocks` / `n_tree_blocks`：块数量。
- `block_m` / `block_n`：块大小。
- `prefix_len`：前缀长度（不在 CSR 中表达）。

### Mixed 位图编码

- 先将 `[BLOCK_M, BLOCK_N]` 掩码按 row-major 拉平；
- 每 32 bit 打包成 1 个 `int32`；
- `bit i` 对应平铺后的第 `i` 个位置。

---

## 快速开始

> 下述示例演示元数据构建流程。完整 end-to-end 调用还需你补充 kernel launch 封装（grid、strides、dtype/device 检查等）。

```python
import torch
from tree_sparse_flashattn import prepare_tree_sparse_metadata

q_len = 128
prefix_len = 32
BLOCK_M, BLOCK_N = 64, 64

# tree-only 区域掩码（示例：下三角）
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

## Kernel 形状与约束（当前版本）

- 预期 grid：`(n_q_blocks, batch_size * num_heads)`。
- Q/K/V/O 采用 4D contiguous 语义（通过 stride 显式传入）。
- `kv_h_idx = h_idx // gqa_group_size` 实现 GQA 头映射。
- 使用 `BLOCK_D: tl.constexpr`，要求与实际 head_dim 匹配。

### 目前建议

1. 先在固定形状（如 `BLOCK_M=BLOCK_N=64`）下验证数值正确性；
2. 增加 Python wrapper：
   - 参数合法性检查；
   - 自动计算 launch grid；
   - 将 metadata 张量移动到 CUDA；
3. 对比 dense attention 结果，做误差与性能基线。

---

## 已知限制

当前仓库还不包含：

- 反向传播 kernel；
- 完整 Python API（一键调用封装）；
- 单元测试 / 基准脚本；
- 自动调参（autotune）与多配置 block 策略。

此外，`MIXED_BLOCK` 分支内的 payload 索引逻辑在大规模场景下通常需要更严格的“mixed 块局部编号”映射设计（当前代码偏原型表达）。

---

## 依赖

- Python 3.9+
- PyTorch（CUDA 版本）
- Triton

示例安装（按你的 CUDA 环境选择对应版本）：

```bash
pip install torch triton
```

---

## 后续可扩展方向

- 引入 per-row 稀疏统计与负载均衡策略；
- 增加 causal / padding / sliding-window 与 tree mask 的组合规则；
- 与现有 FlashAttention 实现对齐接口，支持无缝替换；
- 增加 benchmark（吞吐、显存、时延）与 correctness test。

