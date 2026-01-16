import torch
from torch import nn
import torch.distributed as dist
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.weight_loader import WeightLoader, WeightMapping
import torch.nn.functional as F


class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # 32 // 4 = 8
        self.tp_size = dist.get_world_size()
        assert config.num_attention_heads % self.tp_size == 0
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        # 128
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = self.num_attention_heads // self.tp_size
        # 如果配置中的num_key_value_heads 比tp_size，则每个rank拥有自己的kv头
        if self.num_key_value_heads >= self.tp_size:
            assert self.num_key_value_heads % self.tp_size == 0
            self.num_kv_heads = self.num_key_value_heads // self.tp_size
        else:
            # qwen3 30b moe 模型，num_kv_heads=4 < 单机tp8
            self.num_kv_heads = 1

        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_attention_heads * self.head_dim,  # 32 * 128 = 4096
            bias=config.attention_bias,
            transpose=True,
        )

        if self.num_key_value_heads >= self.tp_size:
            self.is_kv_replicated = False
            self.k_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.num_key_value_heads * self.head_dim,  # 4 * 128 = 512
                bias=config.attention_bias,
                transpose=True,
            )
            self.v_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.num_key_value_heads * self.head_dim,  # 4 * 128 = 512
                bias=config.attention_bias,
                transpose=True,
            )
        else:
            self.is_kv_replicated = True
            self.k_proj = ReplicatedLinear(
                input_size=self.hidden_size,
                output_size=self.num_key_value_heads * self.head_dim,  # 4 * 128 = 512
                bias=config.attention_bias,
                transpose=True,
            )
            self.v_proj = ReplicatedLinear(
                input_size=self.hidden_size,
                output_size=self.num_key_value_heads * self.head_dim,  # 4 * 128 = 512
                bias=config.attention_bias,
                transpose=True,
            )
        self.o_proj = RowParallelLinear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            transpose=True,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states shape: (seq1_len+seq2_len+..+seq_bs_len, hidden_size=1024)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        if self.is_kv_replicated:
            # 假设 tp=8, heads=4, 则 rank 0,1 取 head 0; rank 2,3 取 head 1...
            head_idx = dist.get_rank() // (self.tp_size // self.num_key_value_heads)
            k = k.view(-1, self.num_key_value_heads, self.head_dim)[
                :, head_idx : head_idx + 1, :
            ]
        v = self.v_proj(hidden_states)
        if self.is_kv_replicated:
            # 假设 tp=8, heads=4, 则 rank 0,1 取 head 0; rank 2,3 取 head 1...
            head_idx = dist.get_rank() // (self.tp_size // self.num_key_value_heads)
            v = v.view(-1, self.num_key_value_heads, self.head_dim)[
                :, head_idx : head_idx + 1, :
            ]
        # q_by_head shape: (seq1_len+seq2_len+..+seq_bs_len, 2, 128)
        q_by_head = q.view(
            -1, self.num_heads, self.head_dim
        )  # tp8的时候单rank的num_heads = 16/tp_size = 2
        # q_by_head 经过RMSNorm归一化 shape: (seq1_len+seq2_len+..+seq_bs_len, 2, 128)
        q_by_head = self.q_norm(q_by_head)
        # q矩阵shape: (seq1_len+seq2_len+..+seq_bs_len, 256)
        q = q_by_head.view(q.shape)
        # k_by_head shape: (seq1_len+seq2_len+..+seq_bs_len, 1, 128)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        # k_by_head 经过RMSNorm归一化 shape: (seq1_len+seq2_len+..+seq_bs_len, 1, 128)
        k_by_head = self.k_norm(k_by_head)
        # k矩阵shape: (seq1_len+seq2_len+..+seq_bs_len, 128)
        k = k_by_head.view(k.shape)

        # q，k矩阵进行旋转位置编码，
        q, k = self.rotary_emb(positions, q, k)
        # q,k,v 送入attention计算，输出o shape: (seq1_len+seq2_len+..+seq_bs_len, 256)
        o = self.attn(q, k, v)

        # o_proj: RowParallelLinear, tp8时其weight 是（1024， 256），
        # 所以o @ weight.T = (seq1_len+seq2_len+..+seq_bs_len, 256) @ (256, 1024) = (seq1_len+seq2_len+..+seq_bs_len, 1024)
        # output shape: (seq1_len+seq2_len+..+seq_bs_len, 1024)， o_proj是行并行，在forward中会执行all_reduce动作，来获取全量SUM的结果
        output = self.o_proj(o)
        return output


class Qwen3MoeExperts(nn.ModuleList):
    """
    ModuleList of experts.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        for _ in range(self.num_experts):
            self.append(
                Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # 注意：.nonzero() 会触发 CPU-GPU 同步，不兼容 CUDA Graph 捕获。
        # 目前 MoE 模型在 ModelRunner 中已自动切换到 eager 模式。
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()  # 获取标量值
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(
                -1, hidden_states.shape[-1]
            )
            current_hidden_states = (
                self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        return final_hidden_states


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3MoeExperts(config)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

    def route_tokens_to_experts(self, hidden_states, router_logits):
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        return selected_experts, routing_weights

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size_sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_reshaped)
        selected_experts, routing_weights = self.route_tokens_to_experts(
            hidden_states_reshaped, router_logits
        )
        final_hidden_states = self.experts(
            hidden_states_reshaped, selected_experts, routing_weights
        )
        return final_hidden_states.reshape(batch_size_sequence_length, hidden_dim)


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act == "silu":
            self.act_fn = nn.functional.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
            config=config,
        )
        self.mlp = Qwen3MoeSparseMoeBlock(config=config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config)
                # for _ in range(config.num_hidden_layers)
                for _ in range(
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


class Qwen3MoeForCausalLM(nn.Module):

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.transformers = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.transformers.embed_tokens.weight.data

    def load_weights(self, config, model_path, load_partial_layers: int | None = None):
        weight_mappings = self._build_weight_mappings()
        loader = WeightLoader(
            config=config,
            model_path=model_path,
            model=self,
            load_partial_layers=(
                load_partial_layers
                or getattr(config, "load_partial_layers", None)
                or config.num_hidden_layers
            ),
        )
        return loader.load_weights_from_safetensors(weight_mappings)

    def _build_weight_mappings(self):
        """
        Build wight mapping for Qwen3 MOE
        Embedding/LM Head 使用vocab_parallel, 使用行并行

        Decode layers: 使用row/col 并行
        - Q 使用列并行K,V全量
        - O/Down 使用行并行

        """

        weight_mappings: dict[str, WeightMapping] = {
            # model.embed_tokens.weight：shape = torch.Size([151936, 4096]) QWEN3 DENSE
            # model.embed_tokens.weight：shape = torch.Size([151936, 2048]) QWEN3 30B MOE
            "model.embed_tokens.weight": WeightMapping(
                target_path="transformers.embed_tokens.weight"
            ),
            # lm_head.weight：shape = torch.Size([151936, 4096]) QWEN3 DENSE
            # lm_head.weight：shape = torch.Size([151936, 2048]) QWEN3 30B MOE
            "lm_head.weight": WeightMapping(target_path="lm_head.weight"),
            # model.norm.weight：shape = torch.Size([4096]) QWEN3 DENSE
            # model.norm.weight：shape = torch.Size([2048]) QWEN3 30B MOE
            "model.norm.weight": WeightMapping(target_path="transformers.norm.weight"),
        }

        # decode layers mappings
        layer_nums = getattr(self.config, "num_hidden_layers", 0)
        assert layer_nums > 0
        for layer_id in range(layer_nums):
            """
            model.layers.0.input_layernorm.weight：shape = torch.Size([2048])
            model.layers.0.mlp.experts.0.down_proj.weight：shape = torch.Size([2048, 768])
            model.layers.0.mlp.experts.0.gate_proj.weight：shape = torch.Size([768, 2048])
            model.layers.0.mlp.experts.0.up_proj.weight：shape = torch.Size([768, 2048])
            ...experts 有128个
            model.layers.0.mlp.gate.weight：shape = torch.Size([128, 2048])
            model.layers.0.post_attention_layernorm.weight：shape = torch.Size([2048])
            model.layers.0.self_attn.k_norm.weight：shape = torch.Size([128])
            model.layers.0.self_attn.k_proj.weight：shape = torch.Size([512, 2048])
            model.layers.0.self_attn.o_proj.weight：shape = torch.Size([2048, 4096])
            model.layers.0.self_attn.q_norm.weight：shape = torch.Size([128])
            model.layers.0.self_attn.q_proj.weight：shape = torch.Size([4096, 2048])
            model.layers.0.self_attn.v_proj.weight：shape = torch.Size([512, 2048])
            """
            hf_prefix = f"model.layers.{layer_id}"
            current_prefix = f"transformers.layers.{layer_id}"
            # LayerNorms, input_layernorm, post_attention_layernorm
            weight_mappings[f"{hf_prefix}.input_layernorm.weight"] = WeightMapping(
                target_path=f"{current_prefix}.input_layernorm.weight"
            )
            weight_mappings[f"{hf_prefix}.post_attention_layernorm.weight"] = (
                WeightMapping(
                    target_path=f"{current_prefix}.post_attention_layernorm.weight"
                )
            )
            # model.layers.0.mlp.gate.weight：shape = torch.Size([128, 2048])
            # gate 路由器，门控网络
            weight_mappings[f"{hf_prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{current_prefix}.mlp.gate.weight",
            )

            # attention projection
            for proj in ["q_proj", "k_proj", "v_proj"]:
                if proj == "q_proj":
                    weight_mappings[f"{hf_prefix}.self_attn.{proj}.weight"] = (
                        WeightMapping(
                            target_path=f"{current_prefix}.self_attn.{proj}.weight",
                            transpose=True,
                            dist_strategy="col",
                        )
                    )
                else:
                    weight_mappings[f"{hf_prefix}.self_attn.{proj}.weight"] = (
                        WeightMapping(
                            target_path=f"{current_prefix}.self_attn.{proj}.weight",
                            transpose=True,
                        )
                    )

            for qknorm in ["q_norm", "k_norm"]:
                weight_mappings[f"{hf_prefix}.self_attn.{qknorm}.weight"] = (
                    WeightMapping(
                        target_path=f"{current_prefix}.self_attn.{qknorm}.weight",
                    )
                )

            # output projection
            weight_mappings[f"{hf_prefix}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{current_prefix}.self_attn.o_proj.weight",
                transpose=True,
                dist_strategy="row",
            )
            """
            model.layers.0.mlp.experts.0.down_proj.weight：shape = torch.Size([2048, 768])  
            model.layers.0.mlp.experts.0.gate_proj.weight：shape = torch.Size([768, 2048])
            model.layers.0.mlp.experts.0.up_proj.weight：shape = torch.Size([768, 2048])
            Qwen3 30B 是moe模型，所以不在是纯粹的mlp，而是expers mlp了
            """
            num_experts = getattr(self.config, "num_experts", 0)
            assert num_experts > 0
            for expert_id in range(num_experts):
                weight_mappings[
                    f"{hf_prefix}.mlp.experts.{expert_id}.down_proj.weight"
                ] = WeightMapping(
                    target_path=f"{current_prefix}.mlp.experts.{expert_id}.down_proj.weight",
                )
                weight_mappings[
                    f"{hf_prefix}.mlp.experts.{expert_id}.gate_proj.weight"
                ] = WeightMapping(
                    target_path=f"{current_prefix}.mlp.experts.{expert_id}.gate_proj.weight",
                )
                weight_mappings[
                    f"{hf_prefix}.mlp.experts.{expert_id}.up_proj.weight"
                ] = WeightMapping(
                    target_path=f"{current_prefix}.mlp.experts.{expert_id}.up_proj.weight",
                )

        return weight_mappings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.transformers(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
