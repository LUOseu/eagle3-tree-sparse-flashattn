import torch
import triton
import math
from tree_flash_attn import prepare_tree_sparse_metadata, block_sparse_tree_fwd_kernel

def torch_baseline_tree_attention(q, k, v, tree_mask, prefix_len):
    """
    Standard PyTorch implementation to verify correctness (Exact Match).
    """
    batch_size, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape
    gqa_group = num_heads // num_kv_heads
    
    # Expand K and V for GQA
    k_expanded = k.repeat_interleave(gqa_group, dim=1) # [B, H, kv_len, D]
    v_expanded = v.repeat_interleave(gqa_group, dim=1)
    
    # Calculate scores
    scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Construct dense causal + tree mask
    # [q_len, kv_len]
    full_mask = torch.zeros((q_len, kv_len), dtype=torch.bool, device=q.device)
    full_mask[:, :prefix_len] = True  # Prefix is fully visible
    full_mask[:, prefix_len:] = tree_mask # Tree part
    
    # Apply mask
    scores = scores.masked_fill(~full_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax and multiply V
    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v_expanded)
    return out

def test_lossless_and_acceleration():
    # --- 1. Settings ---
    B = 2
    H = 32
    H_KV = 8
    HEAD_DIM = 128
    PREFIX_LEN = 1024
    Q_LEN = 64 # Tree size
    KV_LEN = PREFIX_LEN + Q_LEN
    
    BLOCK_M = 16
    BLOCK_N = 16
    
    torch.manual_seed(42)
    device = torch.device('cuda')
    
    # --- 2. Initialize Tensors ---
    q = torch.randn(B, H, Q_LEN, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(B, H_KV, KV_LEN, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(B, H_KV, KV_LEN, HEAD_DIM, dtype=torch.float16, device=device)
    
    # Simulate a sparse Tree Mask (e.g., branching structure)
    # Upper triangular is False (causal), plus randomly dropping out different branches
    tree_mask = torch.tril(torch.ones(Q_LEN, Q_LEN, dtype=torch.bool, device=device))
    branch_dropout = torch.rand(Q_LEN, Q_LEN, device=device) > 0.6
    tree_mask = tree_mask & ~branch_dropout 
    # Ensure diagonal is True
    tree_mask.fill_diagonal_(True)

    # --- 3. Preprocessing (Metadata) ---
    meta = prepare_tree_sparse_metadata(tree_mask.cpu(), PREFIX_LEN, BLOCK_M, BLOCK_N)
    meta.row_ptr = meta.row_ptr.to(device)
    meta.col_idx = meta.col_idx.to(device)
    meta.blk_type = meta.blk_type.to(device)
    meta.mixed_mask_payload = meta.mixed_mask_payload.to(device)
    meta.mixed_mask_offset = meta.mixed_mask_offset.to(device)
    
    # --- 4. Run Triton Kernel ---
    o_triton = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    grid = (meta.n_q_blocks, B * H)
    block_sparse_tree_fwd_kernel[grid](
        q, k, v, o_triton,
        meta.row_ptr, meta.col_idx, meta.blk_type, meta.mixed_mask_payload, meta.mixed_mask_offset,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o_triton.stride(0), o_triton.stride(1), o_triton.stride(2), o_triton.stride(3),
        sm_scale, B, H, H_KV, Q_LEN, KV_LEN, PREFIX_LEN,
        meta.n_q_blocks, meta.n_tree_blocks, H // H_KV,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=HEAD_DIM,
        num_warps=4, num_stages=2
    )

    # --- 5. Run PyTorch Baseline ---
    o_torch = torch_baseline_tree_attention(q, k, v, tree_mask, PREFIX_LEN)

    # --- 6. Verification ---
    max_diff = (o_triton - o_torch).abs().max().item()
    print(f"Max Difference between PyTorch and Triton: {max_diff:.6f}")
    if max_diff < 1e-3:
        print(" Triton Kernel is LOSSLESS.")
    else:
        print("Precision mismatch.")

    # --- 7. Benchmarking ---
    print("\nRunning Benchmarks...")
    quantiles = [0.5, 0.2, 0.8]
    
    ms_torch, min_torch, max_torch = triton.testing.do_bench(
        lambda: torch_baseline_tree_attention(q, k, v, tree_mask, PREFIX_LEN), quantiles=quantiles
    )
    
    ms_triton, min_triton, max_triton = triton.testing.do_bench(
        lambda: block_sparse_tree_fwd_kernel[grid](
            q, k, v, o_triton,
            meta.row_ptr, meta.col_idx, meta.blk_type, meta.mixed_mask_payload, meta.mixed_mask_offset,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), o_triton.stride(0), o_triton.stride(1), o_triton.stride(2), o_triton.stride(3),
            sm_scale, B, H, H_KV, Q_LEN, KV_LEN, PREFIX_LEN, meta.n_q_blocks, meta.n_tree_blocks, H // H_KV,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=HEAD_DIM
        ), quantiles=quantiles
    )
    
    print(f"PyTorch Dense+Mask Latency: {ms_torch:.3f} ms")
    print(f"Triton Block-Sparse Latency: {ms_triton:.3f} ms")
    print(f"Speedup: {ms_torch / ms_triton:.2f}x")

if __name__ == "__main__":
    test_lossless_and_acceleration()
