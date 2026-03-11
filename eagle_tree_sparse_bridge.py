"""EAGLE Tree-Sparse FlashAttention bridge (monkey patch version).

该脚本提供“最小侵入式”接入方式：
- 不改你原有 llama3_eagle.py 主体逻辑；
- 通过 monkey patch 覆盖 LlamaFlexAttention / LlamaAttention 的 forward；
- 仅在 verify phase(q_len > 1 且可从 4D attention_mask 提取 tree mask)时触发 Triton 稀疏算子；
- 其他路径完全 fallback 到原实现行为。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton

from tree_sparse_flashattn import block_sparse_tree_fwd_kernel, prepare_tree_sparse_metadata


# -----------------------------
# 1) attention_mask -> tree_mask 提取
# -----------------------------
def extract_tree_mask_from_attention_mask(
    attention_mask: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
    *,
    batch_idx: int = 0,
) -> Optional[torch.Tensor]:
    """从 HF/EAGLE 常见 4D attention_mask 中提取 tree-only bool mask。

    期望输入（verify 常见）:
      attention_mask: [B, 1, Q, KV]
      数值语义: 0 表示可见，-inf 表示屏蔽。

    提取逻辑:
      prefix_len = KV - Q
      tree 部分为最后 Q 列: [:, prefix_len:prefix_len+Q]
      输出 [Q, Q] bool，True=可见，False=屏蔽。
    """
    if attention_mask is None:
        return None
    if q_len <= 1:
        return None
    if attention_mask.ndim != 4:
        return None

    # [B,1,Q,KV]
    if attention_mask.shape[2] < q_len or attention_mask.shape[3] < kv_len:
        return None

    prefix_len = kv_len - q_len
    if prefix_len < 0:
        return None

    # 取一个 batch 的 tree 结构（通常 batch 内结构一致）
    m = attention_mask[batch_idx, 0, :q_len, :kv_len]
    tree_slice = m[:, prefix_len : prefix_len + q_len]
    if tree_slice.shape != (q_len, q_len):
        return None

    if torch.is_floating_point(tree_slice):
        # 常见: 0 / -inf
        tree_mask = torch.isfinite(tree_slice) & (tree_slice >= 0)
    else:
        # 兼容 bool/int mask
        tree_mask = tree_slice.to(torch.bool)

    return tree_mask.contiguous()


# -----------------------------
# 2) Triton launch wrapper
# -----------------------------
def run_tree_sparse_kernel(
    query_states: torch.Tensor,  # [B,Hq,Q,D]
    key_states: torch.Tensor,  # [B,Hkv,KV,D]
    value_states: torch.Tensor,  # [B,Hkv,KV,D]
    tree_mask: torch.Tensor,  # [Q,Q] bool
    *,
    block_m: int,
    block_n: int,
) -> torch.Tensor:
    assert query_states.is_cuda and key_states.is_cuda and value_states.is_cuda
    bsz, num_heads, q_len, head_dim = query_states.shape
    b2, num_kv_heads, kv_len, d2 = key_states.shape
    assert bsz == b2 and head_dim == d2
    assert value_states.shape == key_states.shape

    prefix_len = kv_len - q_len
    if prefix_len < 0:
        raise ValueError(f"Invalid lengths: kv_len={kv_len}, q_len={q_len}")

    meta = prepare_tree_sparse_metadata(
        tree_mask=tree_mask.to(torch.bool).cpu(),
        prefix_len=prefix_len,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )

    row_ptr = meta.row_ptr.to(query_states.device)
    col_idx = meta.col_idx.to(query_states.device)
    blk_type = meta.blk_type.to(query_states.device)
    mixed_payload = meta.mixed_mask_payload.to(query_states.device)
    mixed_offset = meta.mixed_mask_offset.to(query_states.device)

    out = torch.empty_like(query_states)

    sqb, sqh, sqm, sqd = query_states.stride()
    skb, skh, skn, skd = key_states.stride()
    svb, svh, svn, svd = value_states.stride()
    sob, soh, som, sod = out.stride()

    gqa_group_size = num_heads // num_kv_heads
    sm_scale = 1.0 / math.sqrt(head_dim)
    grid = (triton.cdiv(q_len, block_m), bsz * num_heads)

    block_sparse_tree_fwd_kernel[grid](
        query_states,
        key_states,
        value_states,
        out,
        row_ptr,
        col_idx,
        blk_type,
        mixed_payload,
        mixed_offset,
        sqb,
        sqh,
        sqm,
        sqd,
        skb,
        skh,
        skn,
        skd,
        svb,
        svh,
        svn,
        svd,
        sob,
        soh,
        som,
        sod,
        sm_scale,
        bsz,
        num_heads,
        num_kv_heads,
        q_len,
        kv_len,
        prefix_len,
        meta.n_q_blocks,
        meta.n_tree_blocks,
        gqa_group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
    )
    return out


def _fallback_to_original(orig_forward, self, **kwargs):
    return orig_forward(self, **kwargs)


def patch_llama_flex_attention(llama_module, *, block_m: int = 64, block_n: int = 64):
    """为 llama_module.LlamaFlexAttention 注入 tree-sparse bridge。"""
    cls = llama_module.LlamaFlexAttention
    if hasattr(cls, "_orig_forward_tree_sparse_bridge"):
        return

    cls._orig_forward_tree_sparse_bridge = cls.forward

    def _forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # 保持与原 forward 同样的前处理
        bsz, q_len, _ = hidden_states.size()
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        lck = past_seen_tokens // q_len
        if isinstance(self.rotary_emb, llama_module.LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = llama_module.apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = llama_module.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck
            )

        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + q_len, device=hidden_states.device
        )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        key_cache, value_cache = past_key_values.update(
            key_states,
            value_states,
            layer_idx=0,
            cache_kwargs=cache_kwargs,
        )

        # ---- verify 条件触发 ----
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        kv_len = key_cache.shape[-2]
        tree_mask = extract_tree_mask_from_attention_mask(attention_mask, q_len, kv_len)
        use_tree_sparse = (q_len > 1) and (tree_mask is not None)

        if use_tree_sparse:
            attn_output = run_tree_sparse_kernel(
                query_states=query_states.contiguous(),
                key_states=key_cache,
                value_states=value_cache,
                tree_mask=tree_mask,
                block_m=block_m,
                block_n=block_n,
            )
        else:
            # 原 flex attention 路径
            seq_lengths = attention_mask.sum(dim=-1)
            seq_lengths -= lck
            if q_len <= 128:
                create_block_mask_func = llama_module.create_block_mask
                flex_attention_func = llama_module.flex_attention
            else:
                create_block_mask_func = llama_module.compile_friendly_create_block_mask
                flex_attention_func = llama_module.compile_friendly_flex_attention

            block_mask = create_block_mask_func(
                mask_mod=llama_module.generate_eagle3_mask(
                    seq_lengths=seq_lengths,
                    Q_LEN=q_len,
                    KV_LEN=kv_len,
                    lck=lck,
                ),
                B=bsz,
                H=1,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=query_states.device,
            )
            attn_output = flex_attention_func(
                query=query_states,
                key=key_cache,
                value=value_cache,
                block_mask=block_mask,
                enable_gqa=True,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output

    cls.forward = _forward


def patch_llama_attention(llama_module, *, block_m: int = 64, block_n: int = 64):
    """为 llama_module.LlamaAttention 注入 tree-sparse bridge。"""
    cls = llama_module.LlamaAttention
    if hasattr(cls, "_orig_forward_tree_sparse_bridge"):
        return

    cls._orig_forward_tree_sparse_bridge = cls.forward

    def _forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # 无 cache_hidden 时，直接走原实现（prefill/普通decode）
        if cache_hidden is None:
            return _fallback_to_original(
                cls._orig_forward_tree_sparse_bridge,
                self,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        # cache_hidden 分支（verify 可能发生）重写，保持原流程但在条件满足时替换注意力算子
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        lck = len(cache_hidden[0])
        if isinstance(self.rotary_emb, llama_module.LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = llama_module.apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = llama_module.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck
            )

        # 注意: LlamaAttention 原逻辑会先 repeat_kv 后再入 cache_hidden
        key_states = llama_module.repeat_kv(key_states, self.num_key_value_groups)
        value_states = llama_module.repeat_kv(value_states, self.num_key_value_groups)

        cache_hidden[0] = cache_hidden[0] + [key_states]
        cache_hidden[1] = cache_hidden[1] + [value_states]

        # 拼接成连续 KV（必要条件）
        full_k = torch.cat(cache_hidden[0], dim=2).contiguous()  # [B,H,KV,D]
        full_v = torch.cat(cache_hidden[1], dim=2).contiguous()
        kv_len = full_k.shape[-2]

        tree_mask = extract_tree_mask_from_attention_mask(attention_mask, q_len, kv_len)
        use_tree_sparse = (q_len > 1) and (tree_mask is not None)

        if use_tree_sparse:
            # 由于此分支已经 repeat_kv，当前把 H 当作 kv_heads（gqa=1）喂给 kernel
            attn_output = run_tree_sparse_kernel(
                query_states=query_states.contiguous(),
                key_states=full_k,
                value_states=full_v,
                tree_mask=tree_mask,
                block_m=block_m,
                block_n=block_n,
            )
        else:
            # fallback 原 LlamaAttention(cache_hidden) 逻辑
            k0 = cache_hidden[0][0]
            v0 = cache_hidden[1][0]

            attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            l_total = len(cache_hidden[0])
            for i in range(1, l_total):
                ki = cache_hidden[0][i]
                attn_weightsi = (query_states * ki).sum(-1) / math.sqrt(self.head_dim)
                attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            attn_weights0 = attn_weights[..., :q_len]
            attn_output = torch.matmul(attn_weights0, v0)
            for i in range(1, l_total):
                vi = cache_hidden[1][i]
                attn_weightsi = attn_weights[..., q_len + i - 1]
                attn_output = attn_output + attn_weightsi[..., None] * vi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output

    cls.forward = _forward


def inject_tree_sparse_bridge(llama_module, *, block_m: int = 64, block_n: int = 64):
    """一键注入两个分支：LlamaFlexAttention + LlamaAttention。"""
    patch_llama_flex_attention(llama_module, block_m=block_m, block_n=block_n)
    patch_llama_attention(llama_module, block_m=block_m, block_n=block_n)
