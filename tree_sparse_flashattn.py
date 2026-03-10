from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


FULL_BLOCK = 0
MIXED_BLOCK = 1


@dataclass
class TreeSparseMetadata:
    """Compressed tree-block metadata for sparse verification attention.

    Attributes:
        row_ptr: CSR row pointer over Q blocks, shape [n_q_blocks + 1].
        col_idx: CSR column indices for tree KV blocks, shape [nnz_blocks].
        blk_type: Block type per nnz block, FULL(0) or MIXED(1), shape [nnz_blocks].
        mixed_mask_payload: Packed bitset payload for MIXED blocks, flattened int32.
        mixed_mask_offset: Starting int32 offset (into payload) for each MIXED block.
        n_q_blocks: Number of Q blocks.
        n_tree_blocks: Number of tree KV blocks.
        block_m: Q block size.
        block_n: KV block size.
        prefix_len: Prefix length (dense phase, not represented in CSR).
    """

    row_ptr: torch.Tensor
    col_idx: torch.Tensor
    blk_type: torch.Tensor
    mixed_mask_payload: torch.Tensor
    mixed_mask_offset: torch.Tensor
    n_q_blocks: int
    n_tree_blocks: int
    block_m: int
    block_n: int
    prefix_len: int


def _pack_block_mask_to_int32(mask_2d: torch.Tensor) -> torch.Tensor:
    """Pack boolean [BLOCK_M, BLOCK_N] mask into int32 bitset payload.

    Bit order is row-major over flattened mask. Bit i corresponds to
    flat_mask[i] where i = r * BLOCK_N + c.
    """

    flat = mask_2d.reshape(-1).to(torch.bool)
    num_bits = flat.numel()
    words = (num_bits + 31) // 32

    out = torch.zeros(words, dtype=torch.int32)
    idx = torch.nonzero(flat, as_tuple=False).flatten()
    if idx.numel() == 0:
        return out

    word_idx = torch.div(idx, 32, rounding_mode="floor")
    bit_idx = idx % 32
    bitvals = (1 << bit_idx).to(torch.int32)
    out.scatter_add_(0, word_idx, bitvals)
    return out


def prepare_tree_sparse_metadata(
    tree_mask: torch.Tensor,
    prefix_len: int,
    BLOCK_M: int,
    BLOCK_N: int,
) -> TreeSparseMetadata:
    """Generate CSR + tri-state metadata for block-sparse tree attention.

    Args:
        tree_mask: [q_len, q_len] bool/int tensor over *tree-only* region.
        prefix_len: Prefix length; dense and excluded from CSR.
        BLOCK_M: Q block height.
        BLOCK_N: KV block width (for tree part).

    Returns:
        TreeSparseMetadata with CSR structure and packed MIXED masks.
    """

    if tree_mask.ndim != 2 or tree_mask.shape[0] != tree_mask.shape[1]:
        raise ValueError("tree_mask must be shape [q_len, q_len].")
    if BLOCK_M <= 0 or BLOCK_N <= 0:
        raise ValueError("BLOCK_M and BLOCK_N must be positive.")

    mask_bool = tree_mask.to(torch.bool).contiguous()
    q_len = mask_bool.shape[0]
    n_q_blocks = (q_len + BLOCK_M - 1) // BLOCK_M
    n_tree_blocks = (q_len + BLOCK_N - 1) // BLOCK_N

    row_ptr = [0]
    col_idx = []
    blk_type = []

    mixed_offsets = []
    payload_chunks = []
    payload_cursor = 0

    for qb in range(n_q_blocks):
        q0 = qb * BLOCK_M
        q1 = min(q0 + BLOCK_M, q_len)
        row_nnz = 0

        for tb in range(n_tree_blocks):
            k0 = tb * BLOCK_N
            k1 = min(k0 + BLOCK_N, q_len)
            local = mask_bool[q0:q1, k0:k1]

            if not bool(local.any()):
                continue  # ZERO block

            total = local.numel()
            ones = int(local.sum().item())

            col_idx.append(tb)
            row_nnz += 1

            if ones == total:
                blk_type.append(FULL_BLOCK)
            else:
                blk_type.append(MIXED_BLOCK)

                padded = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.bool)
                padded[: (q1 - q0), : (k1 - k0)] = local

                packed = _pack_block_mask_to_int32(padded)
                payload_chunks.append(packed)
                mixed_offsets.append(payload_cursor)
                payload_cursor += packed.numel()

        row_ptr.append(row_ptr[-1] + row_nnz)

    row_ptr_t = torch.tensor(row_ptr, dtype=torch.int32)
    col_idx_t = torch.tensor(col_idx, dtype=torch.int32)
    blk_type_t = torch.tensor(blk_type, dtype=torch.int8)

    if payload_chunks:
        mixed_payload_t = torch.cat(payload_chunks, dim=0).to(torch.int32)
        mixed_offset_t = torch.tensor(mixed_offsets, dtype=torch.int32)
    else:
        mixed_payload_t = torch.empty((0,), dtype=torch.int32)
        mixed_offset_t = torch.empty((0,), dtype=torch.int32)

    return TreeSparseMetadata(
        row_ptr=row_ptr_t,
        col_idx=col_idx_t,
        blk_type=blk_type_t,
        mixed_mask_payload=mixed_payload_t,
        mixed_mask_offset=mixed_offset_t,
        n_q_blocks=n_q_blocks,
        n_tree_blocks=n_tree_blocks,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        prefix_len=prefix_len,
    )


@triton.jit
def block_sparse_tree_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    row_ptr_ptr,
    col_idx_ptr,
    blk_type_ptr,
    mixed_mask_payload_ptr,
    mixed_mask_offset_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    sm_scale,
    batch_size,
    num_heads,
    num_kv_heads,
    q_len,
    kv_len,
    prefix_len,
    n_q_blocks,
    n_tree_blocks,
    gqa_group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Block-sparse tree FlashAttention forward kernel skeleton.

    Expected launch grid: (n_q_blocks, batch_size * num_heads)
    program_id(0): q_block index
    program_id(1): flattened (batch, head)
    """

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b_idx = pid_bh // num_heads
    h_idx = pid_bh % num_heads
    kv_h_idx = h_idx // gqa_group_size  # GQA mapping

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        q_ptr
        + b_idx * stride_qb
        + h_idx * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < BLOCK_D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # -------------------------
    # Phase 1: Dense prefix
    # -------------------------
    n_prefix_blocks = tl.cdiv(prefix_len, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)

    for pb in range(0, n_prefix_blocks):
        kv_start = pb * BLOCK_N
        n_idx = kv_start + offs_n

        k_ptrs = (
            k_ptr
            + b_idx * stride_kb
            + kv_h_idx * stride_kh
            + n_idx[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + b_idx * stride_vb
            + kv_h_idx * stride_vh
            + n_idx[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )

        k = tl.load(k_ptrs, mask=(n_idx[None, :] < prefix_len), other=0.0)
        v = tl.load(v_ptrs, mask=(n_idx[:, None] < prefix_len), other=0.0)

        scores = tl.dot(q, k) * sm_scale

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # -------------------------
    # Phase 2: Sparse tree
    # -------------------------
    row_start = tl.load(row_ptr_ptr + pid_m)
    row_end = tl.load(row_ptr_ptr + pid_m + 1)

    for ridx in range(row_start, row_end):
        tree_block_idx = tl.load(col_idx_ptr + ridx)
        kv_start = prefix_len + tree_block_idx * BLOCK_N
        n_idx = kv_start + offs_n

        k_ptrs = (
            k_ptr
            + b_idx * stride_kb
            + kv_h_idx * stride_kh
            + n_idx[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + b_idx * stride_vb
            + kv_h_idx * stride_vh
            + n_idx[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )

        k = tl.load(k_ptrs, mask=(n_idx[None, :] < kv_len), other=0.0)
        v = tl.load(v_ptrs, mask=(n_idx[:, None] < kv_len), other=0.0)

        scores = tl.dot(q, k) * sm_scale

        block_kind = tl.load(blk_type_ptr + ridx)
        if block_kind == MIXED_BLOCK:
            local_r = tl.arange(0, BLOCK_M)[:, None]
            local_c = tl.arange(0, BLOCK_N)[None, :]
            flat_idx = local_r * BLOCK_N + local_c
            word_idx = flat_idx // 32
            bit_idx = flat_idx % 32

            mix_idx = tl.load(mixed_mask_offset_ptr + ridx)
            words = tl.load(mixed_mask_payload_ptr + mix_idx + word_idx)
            local_mask = ((words >> bit_idx) & 1) == 1
            scores = tl.where(local_mask, scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]

    o_ptrs = (
        o_ptr
        + b_idx * stride_ob
        + h_idx * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    o_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < BLOCK_D)
    tl.store(o_ptrs, acc.to(tl.float16), mask=o_mask)
