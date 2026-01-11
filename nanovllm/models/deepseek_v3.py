import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton import Config
from typing import Tuple


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


fp8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c


import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PretrainedConfig

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.rotary_embedding import get_rope
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


class DeepSeekV3Config(PretrainedConfig):
    model_type = "deepseek_v3"

    def __init__(
        self,
        vocab_size: int = 102400,
        hidden_size: int = 2048,
        intermediate_size: int = 10944,
        moe_intermediate_size: int = 1408,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        # MoE config
        n_routed_experts: int = 64,
        n_shared_experts: int = 2,
        num_experts_per_tok: int = 6,
        n_group: int = 1,
        topk_group: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.001,
        seq_aux: bool = True,
        # MLA config
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        # RoPE config
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        # Normalization
        rms_norm_eps: float = 1e-6,
        # Other
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # yarn specific
        original_max_position_embeddings: int = 4096,
        rope_factor: float = 40,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # MoE
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # RoPE
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Norm
        self.rms_norm_eps = rms_norm_eps
        # Other
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # YARN
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_factor = rope_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        super().__init__(**kwargs)


# Use VocabParallelEmbedding from nano-vllm instead


# Use nano-vllm's Linear layers
# Keep FP8 quantization functions for potential future use
def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


# Use RMSNorm from nano-vllm layers


# Use get_rope from nano-vllm for rotary embeddings


class DeepSeekV3MLA(nn.Module):
    """Multi-head Latent Attention for DeepSeek-V3.

    Adapted to nano-vllm's architecture using flash-attention and kv-cache management.
    """

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        tp_size = dist.get_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        # Q projection with optional LoRA
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=config.attention_bias,
            )
        else:
            self.wq_a = ReplicatedLinear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.wq_b = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=config.attention_bias,
            )

        # KV projection with LoRA
        self.wkv_a = ReplicatedLinear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.wo = RowParallelLinear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        self.scaling = self.qk_head_dim**-0.5
        if config.max_position_embeddings > config.original_max_position_embeddings:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.scaling = self.scaling * mscale * mscale

        # Rotary embedding
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # Use simplified attention without custom kv-cache (nano-vllm handles this)
        from nanovllm.layers.attention import Attention

        # For MLA, we'll handle kv cache differently - store compressed kv representation
        # This is a simplified version, full MLA cache compression can be added later
        self.attn = Attention(
            self.num_local_heads,
            self.qk_head_dim,  # Use full qk_head_dim
            self.scaling,
            self.num_local_heads,  # Assume same num_kv_heads for now
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Q projection
        if self.q_lora_rank == 0:
            q = self.wq(hidden_states)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))

        # Reshape q: (total_tokens, num_local_heads * qk_head_dim) -> (total_tokens, num_local_heads, qk_head_dim)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)

        # Split q into nope and rope parts
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection
        kv_a_out = self.wkv_a(hidden_states)
        kv, k_pe = kv_a_out.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # Apply RoPE to position-encoded parts
        # Expand k_pe to match num_heads for RoPE application
        k_pe = k_pe.unsqueeze(1).expand(-1, self.num_local_heads, -1).contiguous()
        q_pe, k_pe = self.rotary_emb(
            positions,
            q_pe.view(-1, self.qk_rope_head_dim),
            k_pe.view(-1, self.qk_rope_head_dim),
        )
        q_pe = q_pe.view(-1, self.num_local_heads, self.qk_rope_head_dim)
        k_pe = k_pe.view(-1, self.num_local_heads, self.qk_rope_head_dim)

        # Project KV through second stage
        kv = self.kv_norm(kv)
        kv_b_out = self.wkv_b(kv)
        kv_b_out = kv_b_out.view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_b_out.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Combine nope and rope parts
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # Reshape for attention: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
        q = q.view(-1, self.num_local_heads * self.qk_head_dim)
        k = k.view(-1, self.num_local_heads * self.qk_head_dim)
        v = v.view(-1, self.num_local_heads * self.v_head_dim)

        # Note: This is simplified - ideally we'd use the compressed kv representation
        # For now, treat as standard MHA with larger head dimensions
        # TODO: Implement efficient kv cache compression

        # Apply attention (nano-vllm's Attention handles kv-cache)
        attn_output = self.attn(q, k, v)

        # Output projection
        output = self.wo(attn_output)
        return output


class DeepSeekV3MLP(nn.Module):
    """Standard MLP for dense layers."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class DeepSeekV3Gate(nn.Module):
    """Gating network for MoE routing."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.topk = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.scoring_func = config.scoring_func
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        # DeepSeek-V3 may have bias in some configurations
        if config.hidden_size == 7168:
            self.bias = nn.Parameter(torch.empty(config.n_routed_experts))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gate logits
        scores = F.linear(x, self.weight, self.bias)

        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()

        original_scores = scores

        # Group-wise top-k if n_group > 1
        if self.n_group > 1:
            scores = scores.view(x.size(0), self.n_group, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_group, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)

        # Select top-k experts
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)

        if self.scoring_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights.type_as(x), indices


class DeepSeekV3Expert(nn.Module):
    """Single expert MLP."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.gate_proj = ReplicatedLinear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.up_proj = ReplicatedLinear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.down_proj = ReplicatedLinear(
            config.moe_intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class DeepSeekV3MoE(nn.Module):
    """Mixture of Experts layer with shared experts."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = dist.get_world_size()
        assert config.n_routed_experts % tp_size == 0

        self.n_routed_experts = config.n_routed_experts
        self.n_local_experts = config.n_routed_experts // tp_size
        self.num_experts_per_tok = config.num_experts_per_tok

        # Expert partitioning across TP ranks
        tp_rank = dist.get_rank()
        self.expert_start_idx = tp_rank * self.n_local_experts
        self.expert_end_idx = self.expert_start_idx + self.n_local_experts

        # Gate
        self.gate = DeepSeekV3Gate(config)

        # Routed experts (only instantiate local experts)
        self.experts = nn.ModuleList(
            [
                (
                    DeepSeekV3Expert(config)
                    if self.expert_start_idx <= i < self.expert_end_idx
                    else None
                )
                for i in range(self.n_routed_experts)
            ]
        )

        # Shared experts
        self.shared_experts = (
            DeepSeekV3MLP(config) if config.n_shared_experts > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.view(-1, self.hidden_size)

        # Routing
        weights, indices = self.gate(x)

        # Process routed experts
        y = torch.zeros_like(x)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()

        for i in range(self.expert_start_idx, self.expert_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        # Shared experts
        if self.shared_experts is not None:
            shared_out = self.shared_experts(x)
        else:
            shared_out = 0

        # All-reduce routed expert outputs across TP ranks
        if dist.get_world_size() > 1:
            dist.all_reduce(y)

        output = (y + shared_out).view(original_shape)
        return output


class DeepSeekV3DecoderLayer(nn.Module):
    """DeepSeek-V3 decoder layer with MLA attention and MoE/MLP."""

    def __init__(self, layer_id: int, config: DeepSeekV3Config):
        super().__init__()
        self.self_attn = DeepSeekV3MLA(config)

        # Use MLP for first layer, MoE for remaining layers
        # Note: Original code uses n_dense_layers, but config uses num_hidden_layers
        # Assuming first layer is dense
        if layer_id == 0:
            self.mlp = DeepSeekV3MLP(config)
        else:
            self.mlp = DeepSeekV3MoE(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        # MLP/MoE
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class DeepSeekV3Model(nn.Module):
    """DeepSeek-V3 model without language modeling head."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.padding_idx = (
            config.pad_token_id if hasattr(config, "pad_token_id") else None
        )
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                DeepSeekV3DecoderLayer(layer_id, config)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepSeekV3ForCausalLM(nn.Module):
    """DeepSeek-V3 model for causal language modeling."""

    # Weight mapping for nano-vllm loader
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),  # May need adjustment based on actual weights
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        self.config = config
        self.model = DeepSeekV3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # Tie weights if specified
        if hasattr(config, "tie_word_embeddings") and config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
