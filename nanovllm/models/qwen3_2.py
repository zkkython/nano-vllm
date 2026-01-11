import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMulSplit
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.weight_loader import WeightLoader, WeightMapping


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads  # 16
        assert self.total_num_heads % tp_size == 0
        self.num_heads = (
            self.total_num_heads // tp_size
        )  # 16/tp_size 如果是8，那么num_heads = 16/8 = 2
        self.total_num_kv_heads = num_kv_heads  # 8
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = (
            self.total_num_kv_heads // tp_size
        )  # 8/tp_size 如果是8，那么num_kv_heads = 8/8 = 1
        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 128
        self.q_size = self.num_heads * self.head_dim  # 2 * 128 = 256
        self.kv_size = self.num_kv_heads * self.head_dim  # 1 * 128 = 128
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=False,
            transpose=True,
        )
        self.k_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_kv_heads * self.head_dim,
            bias=False,
            transpose=True,
        )
        self.v_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_kv_heads * self.head_dim,
            bias=False,
            transpose=True,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,  # 16 * 128 = 2048
            hidden_size,  # 1024
            bias=False,
            transpose=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states shape: (seq1_len+seq2_len+..+seq_bs_len, hidden_size=1024)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
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


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # self.gate_up_proj = MergedColumnParallelLinear(
        #     hidden_size,
        #     [intermediate_size] * 2,
        #     bias=False,
        # )
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, transpose=True
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, transpose=True
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            transpose=True,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMulSplit()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act_fn(gate, up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,  # 1024
            num_heads=config.num_attention_heads,  # 16
            num_kv_heads=config.num_key_value_heads,  # 8
            max_position=config.max_position_embeddings,  # 40960
            rms_norm_eps=config.rms_norm_eps,  # 1e-06
            qkv_bias=getattr(config, "attention_bias", False),  # false
            head_dim=getattr(config, "head_dim", None),  # 128
            rope_theta=getattr(config, "rope_theta", 1000000),  # 1000000
            rope_scaling=getattr(config, "rope_scaling", None),  # null
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
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


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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


class Qwen3ForCausalLM(nn.Module):

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.transformers = Qwen3Model(config)
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
                getattr(config, "load_partial_layers", None)
                if load_partial_layers is None
                else load_partial_layers
            ),
        )
        return loader.load_weights_from_safetensors(weight_mappings)

    def _build_weight_mappings(self):
        """
        Build wight mapping for Qwen3
        Embedding/LM Head 使用vocab_parallel, 使用行并行

        Decode layers: 使用row/col 并行
        - Q/K/V/Gate/Up 使用列并行
        - O/Down 使用行并行

        """

        weight_mappings: dict[str, WeightMapping] = {
            # model.embed_tokens.weight：shape = torch.Size([151936, 4096])
            "model.embed_tokens.weight": WeightMapping(
                target_path="transformers.embed_tokens.weight"
            ),
            # lm_head.weight：shape = torch.Size([151936, 4096])
            "lm_head.weight": WeightMapping(target_path="lm_head.weight"),
            # model.norm.weight：shape = torch.Size([4096])
            "model.norm.weight": WeightMapping(target_path="transformers.norm.weight"),
        }

        # decode layers mappings
        layer_nums = getattr(self.config, "num_hidden_layers", 0)
        assert layer_nums > 0
        for layer_id in range(layer_nums):
            """
            model.layers.18.input_layernorm.weight：shape = torch.Size([4096])
            model.layers.18.mlp.down_proj.weight：shape = torch.Size([4096, 12288])
            model.layers.18.mlp.gate_proj.weight：shape = torch.Size([12288, 4096])
            model.layers.18.mlp.up_proj.weight：shape = torch.Size([12288, 4096])
            model.layers.18.post_attention_layernorm.weight：shape = torch.Size([4096])
            model.layers.18.self_attn.k_norm.weight：shape = torch.Size([128])
            model.layers.18.self_attn.k_proj.weight：shape = torch.Size([1024, 4096])
            model.layers.18.self_attn.o_proj.weight：shape = torch.Size([4096, 4096])
            model.layers.18.self_attn.q_norm.weight：shape = torch.Size([128])
            model.layers.18.self_attn.q_proj.weight：shape = torch.Size([4096, 4096])
            model.layers.18.self_attn.v_proj.weight：shape = torch.Size([1024, 4096])
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

            # attention projection
            for proj in ["q_proj", "k_proj", "v_proj"]:
                weight_mappings[f"{hf_prefix}.self_attn.{proj}.weight"] = WeightMapping(
                    target_path=f"{current_prefix}.self_attn.{proj}.weight",
                    transpose=True,
                    dist_strategy="col",
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

            # MLP, gate, up 是列并行， down 是行并行
            for proj in ["gate_proj", "up_proj"]:
                hf_key = f"{hf_prefix}.mlp.{proj}.weight"
                target = f"{current_prefix}.mlp.{proj}.weight"
                weight_mappings[hf_key] = WeightMapping(
                    target_path=target,
                    transpose=True,
                    dist_strategy="col",
                )
            weight_mappings[f"{hf_prefix}.mlp.down_proj.weight"] = WeightMapping(
                target_path=f"{current_prefix}.mlp.down_proj.weight",
                transpose=True,
                dist_strategy="row",
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
