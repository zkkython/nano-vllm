from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.attention import Attention
from nanovllm.models.models_mapping import register_model
from transformers import PretrainedConfig


class DeepseekV3Config(PretrainedConfig):
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
        quantization: Optional[str] = None,
        routed_scaling_factor: float = 1.0,
        topk_method: str = "greedy",
        first_k_dense_replace: int = 3,
        norm_topk_prob: bool = False,
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
        self.quantization = quantization
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob

        super().__init__(**kwargs)


class DeepseekV3MLP(nn.Module):
    def __init__(
        self, config: DeepseekV3Config, intermediate_size: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quantization=getattr(config, "quantization", None),
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quantization=getattr(config, "quantization", None),
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quantization=getattr(config, "quantization", None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = getattr(config, "scoring_func", "softmax")

        # Routing weight
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size))
        )
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.n_routed_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (num_tokens, hidden_size)
        logits = F.linear(hidden_states.float(), self.weight.float())

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            scores = logits.sigmoid()

        # Apply correction bias if exists (used in DeepSeek-V3 for load balancing)
        if hasattr(self, "e_score_correction_bias"):
            scores_for_choice = scores + self.e_score_correction_bias
        else:
            scores_for_choice = scores

        # Group-wise routing
        if self.n_group > 1:
            # Grouping logic
            scores_for_choice = scores_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            if self.config.topk_method == "greedy":
                group_scores = scores_for_choice.max(dim=-1).values
            else:  # "sum"
                group_scores = scores_for_choice.topk(2, dim=-1).values.sum(dim=-1)

            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(-1, self.n_routed_experts)
            )
            scores_for_choice = scores_for_choice.reshape(-1, self.n_routed_experts)
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )

        # Use original scores for weights
        topk_weights = scores.gather(1, topk_indices)

        if self.config.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator

        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.type_as(hidden_states), topk_indices


class DeepseekV3Expert(nn.Module):
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.gate_proj = ReplicatedLinear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
            quantization=getattr(config, "quantization", None),
        )
        self.up_proj = ReplicatedLinear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
            quantization=getattr(config, "quantization", None),
        )
        self.down_proj = ReplicatedLinear(
            config.moe_intermediate_size,
            config.hidden_size,
            bias=False,
            quantization=getattr(config, "quantization", None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class DeepseekV3MoE(nn.Module):
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0

        self.n_routed_experts = config.n_routed_experts
        self.n_local_experts = self.n_routed_experts // self.tp_size
        self.expert_start_idx = self.tp_rank * self.n_local_experts
        self.expert_end_idx = self.expert_start_idx + self.n_local_experts

        self.gate = DeepseekV3TopkRouter(config)

        # Shared experts
        if config.n_shared_experts > 0:
            shared_intermediate_size = (
                config.moe_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = DeepseekV3MLP(
                config=config,
                intermediate_size=shared_intermediate_size,
            )
        else:
            self.shared_experts = None

        # Routed experts
        self.experts = nn.ModuleDict()
        for i in range(self.n_local_experts):
            expert_id = self.expert_start_idx + i
            self.experts[str(expert_id)] = DeepseekV3Expert(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # Shared experts path
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = 0

        # Router
        topk_weights, topk_indices = self.gate(hidden_states)

        # Routed experts path
        # Note: This is a simple implementation. In production, we'd use fused kernels.
        # However, following the "framework logic" and "deepseek_v3.py" pattern:
        final_hidden_states = torch.zeros_like(hidden_states)

        # Flatten for processing
        flat_indices = topk_indices.flatten()
        flat_weights = topk_weights.flatten()

        # Count tokens per expert
        # Use bincount for now, can be optimized
        counts = torch.bincount(flat_indices, minlength=self.n_routed_experts).tolist()

        for i in range(self.n_local_experts):
            global_id = self.expert_start_idx + i
            if counts[global_id] == 0:
                continue

            expert = self.experts[str(global_id)]
            # Find tokens routed to this expert
            idx, top_pos = torch.where(topk_indices == global_id)

            expert_input = hidden_states[idx]
            expert_output = expert(expert_input)

            # Weighted add
            final_hidden_states.index_add_(
                0, idx, expert_output * topk_weights[idx, top_pos, None]
            )

        # All-reduce across TP ranks for routed experts
        if self.tp_size > 1:
            dist.all_reduce(final_hidden_states)

        output = (final_hidden_states + shared_output).view(orig_shape)
        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, config: DeepseekV3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        # For DeepSeek-V3, we typically use the default parameters for initialization
        # unless specialized Yarn RoPE is required.
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(
            config, device
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: DeepseekV3Config,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        base = getattr(config, "rope_theta", 10000.0)
        dim = config.qk_rope_head_dim
        attention_factor = 1.0

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: [num_tokens]
        # x: [num_tokens, ..., dim]
        inv_freq_expanded = (
            self.inv_freq[None, :].float().to(x.device)
        )
        position_ids_expanded = position_ids[:, None].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (position_ids_expanded @ inv_freq_expanded)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV3Attention(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1

        self.num_heads = config.num_attention_heads
        assert self.num_heads % self.tp_size == 0
        self.num_local_heads = self.num_heads // self.tp_size

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        # Q projection
        if self.q_lora_rank is None or self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(
                config.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=config.attention_bias,
                quantization=getattr(config, "quantization", None),
            )
        else:
            self.wq_a = ReplicatedLinear(
                config.hidden_size,
                config.q_lora_rank,
                bias=config.attention_bias,
                quantization=getattr(config, "quantization", None),
            )
            self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.wq_b = ColumnParallelLinear(
                config.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quantization=getattr(config, "quantization", None),
            )

        # KV projection (MLA)
        self.wkv_a = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
            quantization=getattr(config, "quantization", None),
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quantization=getattr(config, "quantization", None),
        )

        # Output projection
        self.wo = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quantization=getattr(config, "quantization", None),
        )

        self.scaling = self.qk_head_dim**-0.5
        
        # RoPE
        self.rotary_emb = DeepseekV3RotaryEmbedding(config)
        
        # Attention Kernel Wrapper
        self.attn = Attention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            self.num_local_heads,  # MLA effectively uses GQA/MQA style or full heads
            v_head_dim=self.v_head_dim,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Q projection
        if self.q_lora_rank is None or self.q_lora_rank == 0:
            q = self.wq(hidden_states)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection
        kv_a_out = self.wkv_a(hidden_states)
        kv, k_pe = kv_a_out.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # RoPE application
        # Expand k_pe to match heads for rotary_emb if needed,
        # but MLA usually has a single shared k_pe across heads
        k_pe = k_pe.unsqueeze(1).expand(-1, self.num_local_heads, -1).contiguous()

        cos, sin = self.rotary_emb(q_pe, positions)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        q_pe = q_pe.view(-1, self.num_local_heads, self.qk_rope_head_dim)
        k_pe = k_pe.view(-1, self.num_local_heads, self.qk_rope_head_dim)

        # Second stage KV projection
        kv = self.kv_norm(kv)
        kv_b_out = self.wkv_b(kv)
        kv_b_out = kv_b_out.view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_b_out.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Combine
        q = torch.cat([q_nope, q_pe], dim=-1).reshape(
            -1, self.num_local_heads * self.qk_head_dim
        )
        k = torch.cat([k_nope, k_pe], dim=-1).reshape(
            -1, self.num_local_heads * self.qk_head_dim
        )
        v = v.reshape(-1, self.num_local_heads * self.v_head_dim)

        # Attention
        attn_output = self.attn(q, k, v)

        # Output projection
        output = self.wo(attn_output)
        return output


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input Norm
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(positions, hidden_states)

        # Post Attention Norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP / MoE
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@register_model("deepseek_v3")
class DeepseekV3ForCausalLM(nn.Module):
    # Weight mapping for nano-vllm loader
    packed_modules_mapping = {
        "q_proj": ("wq", "q"),  # if q_lora_rank is 0
        "q_a_proj": ("wq_a", 0),
        "q_b_proj": ("wq_b", 0),
        "kv_a_proj_with_mqa": ("wkv_a", 0),
        "kv_b_proj": ("wkv_b", 0),
        "gate_proj": ("gate_proj", 0),
        "up_proj": ("up_proj", 1),
    }

    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.model = DeepseekV3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(
        self, config, model_path: str, load_partial_layers: int | None = None
    ):
        from nanovllm.weight_mappings.deepseek_v3_weight_mapping import (
            build_deepseek_v3_weight_mappings,
            build_deepseek_v3_expert_mappings,
        )
        from nanovllm.utils.weight_loader import WeightLoader
        import torch.distributed as dist

        weight_mappings = build_deepseek_v3_weight_mappings(
            num_hidden_layers=config.num_hidden_layers,
            q_lora_rank=config.q_lora_rank,
            n_shared_experts=config.n_shared_experts,
        )

        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        tp_rank = dist.get_rank() if dist.is_initialized() else 0

        n_local_experts = config.n_routed_experts // tp_size
        expert_start_idx = tp_rank * n_local_experts
        expert_end_idx = expert_start_idx + n_local_experts

        expert_mappings = build_deepseek_v3_expert_mappings(
            num_hidden_layers=config.num_hidden_layers,
            router_expert_start_idx=expert_start_idx,
            router_expert_end_idx=expert_end_idx,
        )
        weight_mappings.update(expert_mappings)

        loader = WeightLoader(
            model=self,
            config=config,
            model_path=model_path,
            load_partial_layers=load_partial_layers,
        )
        return loader.load_weights_from_safetensors(weight_mappings)


class DeepseekV3Model(nn.Module):
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(config, layer_idx)
                for layer_idx in range(
                    getattr(config, "load_partial_layers", None)
                    or config.num_hidden_layers
                )
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
