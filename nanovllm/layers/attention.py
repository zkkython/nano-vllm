import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kv_kernel(
    src_ptr,
    src_stride,
    cache_ptr,
    cache_stride,
    slot_mapping_ptr,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    src_offsets = idx * src_stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < D
    val = tl.load(src_ptr + src_offsets, mask=mask)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * cache_stride + tl.arange(0, BLOCK_SIZE)
    tl.store(cache_ptr + cache_offsets, val, mask=mask)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    _, _, v_head_dim = value.shape
    k_D = num_heads * head_dim
    v_D = num_heads * v_head_dim

    assert key.stride(-1) == 1 and value.stride(-1) == 1
    # print(
    #     f"key shape {key.shape}, value shape {value.shape}, {key.stride(1)}, {head_dim}, {value.stride(1)}, {v_head_dim}"
    # )
    # 对于 MLA，head_dim 可能不同，这里分别检查
    assert key.stride(1) == head_dim
    assert value.stride(1) == v_head_dim

    # k_cache 和 v_cache 的 stride(1) 代表一个 block 中单个 token 占用的空间
    k_cache_stride = k_cache.stride(1)
    v_cache_stride = v_cache.stride(1)
    # print(f"slot numel: {slot_mapping.numel()}, N {N}")
    assert slot_mapping.numel() == N

    # 分别存储 K 和 V
    k_block_size = triton.next_power_of_2(k_D)
    store_kv_kernel[(N,)](
        key,
        key.stride(0),
        k_cache,
        k_cache_stride,
        slot_mapping,
        k_D,
        BLOCK_SIZE=k_block_size,
    )

    v_block_size = triton.next_power_of_2(v_D)
    store_kv_kernel[(N,)](
        value,
        value.stride(0),
        v_cache,
        v_cache_stride,
        slot_mapping,
        v_D,
        BLOCK_SIZE=v_block_size,
    )


"""
切分TP后的Attention
所以num_heads 不是原有的 num_heads，而是 num_heads // tp_size
num_kv_heads 也不是原有的 num_kv_heads，而是 num_kv_heads // tp_size

"""


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        v_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_dim = v_head_dim or head_dim
        self.k_cache = self.v_cache = torch.tensor([])
        # print(f"attention heads {num_heads}, kv heads {num_kv_heads}")

    def _repeat_kv_heads(self, x: torch.Tensor):
        return x[:, : self.num_kv_heads].repeat(
            1, self.num_heads // self.num_kv_heads, 1
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim).contiguous()
        v = v.reshape(-1, self.num_kv_heads, self.v_head_dim).contiguous()

        # Flash Attention requires head_dim(K) == head_dim(V)
        # For MLA, we pad V to match K's head_dim if they differ
        if self.head_dim != self.v_head_dim:
            v = torch.nn.functional.pad(v, (0, self.head_dim - self.v_head_dim))

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # MLA 场景下 K 和 V 的 head_dim 可能不同，且 cache 可能有 padding
        if k_cache.numel():
            k_cache = k_cache[..., : self.head_dim]
        if v_cache.numel():
            # For MLA, use the full allocated 192 instead of sliced 128 to match K
            v_cache = v_cache[..., : self.head_dim]

        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache

            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:  # decode

            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )

        # Slice output if we padded V
        if self.head_dim != self.v_head_dim:
            o = o[..., : self.v_head_dim]

        o = o.reshape(-1, self.num_heads * self.v_head_dim)
        return o
