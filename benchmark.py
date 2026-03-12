import math
import importlib
import importlib.util

import torch
import torch.nn.functional as F
import triton

from tree_sparse_flashattn import prepare_tree_sparse_metadata, block_sparse_tree_fwd_kernel


def torch_baseline_tree_attention(q, k, v, tree_mask, prefix_len):
    """Standard PyTorch implementation to verify correctness (Exact Match)."""
    _, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape
    gqa_group = num_heads // num_kv_heads

    k_expanded = k.repeat_interleave(gqa_group, dim=1)
    v_expanded = v.repeat_interleave(gqa_group, dim=1)

    scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

    full_mask = torch.zeros((q_len, kv_len), dtype=torch.bool, device=q.device)
    full_mask[:, :prefix_len] = True
    full_mask[:, prefix_len:] = tree_mask

    scores = scores.masked_fill(~full_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v_expanded)
    return out


def flashattention_baseline_tree_attention(q, k, v, tree_mask, prefix_len):
    """FlashAttention backend baseline (via PyTorch SDPA flash kernel)."""
    _, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape
    gqa_group = num_heads // num_kv_heads

    k_expanded = k.repeat_interleave(gqa_group, dim=1)
    v_expanded = v.repeat_interleave(gqa_group, dim=1)

    full_mask = torch.zeros((q_len, kv_len), dtype=torch.bool, device=q.device)
    full_mask[:, :prefix_len] = True
    full_mask[:, prefix_len:] = tree_mask

    additive_mask = torch.zeros((q_len, kv_len), dtype=q.dtype, device=q.device)
    additive_mask = additive_mask.masked_fill(~full_mask, float("-inf")).unsqueeze(0).unsqueeze(0)

    return F.scaled_dot_product_attention(
        q,
        k_expanded,
        v_expanded,
        attn_mask=additive_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(head_dim),
    )


def _build_flashinfer_paged_kv_cache(k, v, page_size):
    """Convert [B, H_kv, KV_LEN, D] K/V into paged KV cache: [max_pages, 2, P, H_kv, D]."""
    bsz, num_kv_heads, kv_len, head_dim = k.shape
    pages_per_req = (kv_len + page_size - 1) // page_size
    max_pages = bsz * pages_per_req

    kv_cache = torch.zeros(
        (max_pages, 2, page_size, num_kv_heads, head_dim),
        dtype=k.dtype,
        device=k.device,
    )

    for b in range(bsz):
        req_base = b * pages_per_req
        for p in range(pages_per_req):
            start = p * page_size
            end = min((p + 1) * page_size, kv_len)
            valid = end - start
            kv_cache[req_base + p, 0, :valid] = k[b, :, start:end, :].transpose(0, 1)
            kv_cache[req_base + p, 1, :valid] = v[b, :, start:end, :].transpose(0, 1)

    return kv_cache, pages_per_req


def _prepare_flashinfer_baseline3(q, k, v, prefix_len, page_size=16, workspace_bytes=128 * 1024 * 1024):
    """
    Prepare FlashInfer decode benchmark artifacts once, and return:
    - run_fn: function that runs only wrapper.run(...)
    - out_tree_layout: output converted to [B, H, Q, D]

    Important for fair benchmarking:
      1) Do not include plan() in timed loop.
      2) Do not over-assign page indices per query. Use only visible pages per query.
    """
    flashinfer = importlib.import_module("flashinfer")

    bsz, num_qo_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape

    kv_cache, pages_per_req = _build_flashinfer_paged_kv_cache(k, v, page_size)
    q_decode = q.permute(0, 2, 1, 3).contiguous().reshape(bsz * q_len, num_qo_heads, head_dim)

    visible_kv = torch.empty(bsz * q_len, dtype=torch.int32, device=q.device)
    for b in range(bsz):
        for t in range(q_len):
            row = b * q_len + t
            visible_kv[row] = min(prefix_len + t + 1, kv_len)

    pages_per_query = (visible_kv + page_size - 1) // page_size
    indptr = torch.zeros(bsz * q_len + 1, dtype=torch.int32, device=q.device)
    indptr[1:] = torch.cumsum(pages_per_query, dim=0)

    total_page_refs = int(indptr[-1].item())
    indices = torch.empty(total_page_refs, dtype=torch.int32, device=q.device)
    last_page_len = ((visible_kv - 1) % page_size) + 1

    for b in range(bsz):
        req_base = b * pages_per_req
        for t in range(q_len):
            row = b * q_len + t
            n_pages = int(pages_per_query[row].item())
            start = int(indptr[row].item())
            indices[start : start + n_pages] = torch.arange(
                req_base,
                req_base + n_pages,
                device=q.device,
                dtype=torch.int32,
            )

    workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace)
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )

    run_fn = lambda: wrapper.run(q_decode, kv_cache)
    out = run_fn().reshape(bsz, q_len, num_qo_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return run_fn, out


def test_lossless_and_acceleration():
    B = 2
    H = 32
    H_KV = 8
    HEAD_DIM = 128
    PREFIX_LEN = 1024
    Q_LEN = 64
    KV_LEN = PREFIX_LEN + Q_LEN

    BLOCK_M = 16
    BLOCK_N = 16

    torch.manual_seed(42)
    device = torch.device("cuda")

    q = torch.randn(B, H, Q_LEN, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(B, H_KV, KV_LEN, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(B, H_KV, KV_LEN, HEAD_DIM, dtype=torch.float16, device=device)

    tree_mask = torch.tril(torch.ones(Q_LEN, Q_LEN, dtype=torch.bool, device=device))
    branch_dropout = torch.rand(Q_LEN, Q_LEN, device=device) > 0.6
    tree_mask = tree_mask & ~branch_dropout
    tree_mask.fill_diagonal_(True)

    meta = prepare_tree_sparse_metadata(tree_mask.cpu(), PREFIX_LEN, BLOCK_M, BLOCK_N)
    meta.row_ptr = meta.row_ptr.to(device)
    meta.col_idx = meta.col_idx.to(device)
    meta.blk_type = meta.blk_type.to(device)
    meta.mixed_mask_payload = meta.mixed_mask_payload.to(device)
    meta.mixed_mask_offset = meta.mixed_mask_offset.to(device)

    o_triton = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    grid = (meta.n_q_blocks, B * H)
    block_sparse_tree_fwd_kernel[grid](
        q,
        k,
        v,
        o_triton,
        meta.row_ptr,
        meta.col_idx,
        meta.blk_type,
        meta.mixed_mask_payload,
        meta.mixed_mask_offset,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o_triton.stride(0),
        o_triton.stride(1),
        o_triton.stride(2),
        o_triton.stride(3),
        sm_scale,
        B,
        H,
        H_KV,
        Q_LEN,
        KV_LEN,
        PREFIX_LEN,
        meta.n_q_blocks,
        meta.n_tree_blocks,
        H // H_KV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
        num_stages=2,
    )

    o_torch = torch_baseline_tree_attention(q, k, v, tree_mask, PREFIX_LEN)
    o_flash = flashattention_baseline_tree_attention(q, k, v, tree_mask, PREFIX_LEN)

    max_diff_torch = (o_triton - o_torch).abs().max().item()
    max_diff_flash = (o_triton - o_flash).abs().max().item()
    print(f"Max Difference between PyTorch and Triton: {max_diff_torch:.6f}")
    print(f"Max Difference between FlashAttention and Triton: {max_diff_flash:.6f}")
    if max_diff_torch < 1e-3 and max_diff_flash < 1e-3:
        print(" Triton Kernel is LOSSLESS.")
    else:
        print("Precision mismatch.")

    flashinfer_available = importlib.util.find_spec("flashinfer") is not None
    flashinfer_run_fn = None
    if flashinfer_available:
        flashinfer_run_fn, o_flashinfer = _prepare_flashinfer_baseline3(q, k, v, PREFIX_LEN, page_size=16)
        max_diff_flashinfer = (o_triton - o_flashinfer).abs().max().item()
        print(f"Max Difference between FlashInfer Baseline3 and Triton: {max_diff_flashinfer:.6f}")
        print("[Note] FlashInfer Baseline3 is decode-style contiguous KV, so mismatch vs sparse tree mask is expected.")

    print("\nRunning Benchmarks...")
    quantiles = [0.5, 0.2, 0.8]

    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch_baseline_tree_attention(q, k, v, tree_mask, PREFIX_LEN), quantiles=quantiles
    )

    ms_triton, _, _ = triton.testing.do_bench(
        lambda: block_sparse_tree_fwd_kernel[grid](
            q,
            k,
            v,
            o_triton,
            meta.row_ptr,
            meta.col_idx,
            meta.blk_type,
            meta.mixed_mask_payload,
            meta.mixed_mask_offset,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o_triton.stride(0),
            o_triton.stride(1),
            o_triton.stride(2),
            o_triton.stride(3),
            sm_scale,
            B,
            H,
            H_KV,
            Q_LEN,
            KV_LEN,
            PREFIX_LEN,
            meta.n_q_blocks,
            meta.n_tree_blocks,
            H // H_KV,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=HEAD_DIM,
        ),
        quantiles=quantiles,
    )

    ms_flash, _, _ = triton.testing.do_bench(
        lambda: flashattention_baseline_tree_attention(q, k, v, tree_mask, PREFIX_LEN), quantiles=quantiles
    )

    ms_flashinfer = None
    if flashinfer_run_fn is not None:
        ms_flashinfer, _, _ = triton.testing.do_bench(flashinfer_run_fn, quantiles=quantiles)

    print(f"PyTorch Dense+Mask Latency: {ms_torch:.3f} ms")
    print(f"FlashAttention Baseline2 Latency: {ms_flash:.3f} ms")
    if ms_flashinfer is not None:
        print(f"FlashInfer Baseline3 Latency: {ms_flashinfer:.3f} ms")
    print(f"Triton Block-Sparse Latency: {ms_triton:.3f} ms")
    print(f"Speedup: {ms_torch / ms_triton:.2f}x")
    print(f"Speedup vs FlashAttention Baseline2: {ms_flash / ms_triton:.2f}x")
    if ms_flashinfer is not None:
        print(f"Speedup vs FlashInfer Baseline3: {ms_flashinfer / ms_triton:.2f}x")
    else:
        print("FlashInfer Baseline3 skipped: flashinfer is not installed in this environment.")


if __name__ == "__main__":
    test_lossless_and_acceleration()
