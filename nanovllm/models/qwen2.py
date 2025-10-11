from torch import nn
from transformers import Qwen2Config
import torch
import torch.distributed as dist
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope


class Qwen2Attention(nn.Module):

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

        self.qkv_proj = QKVParallelLinear(
            hidden_size,  # 1024
            self.head_dim,  # 128
            self.total_num_heads,  # 16
            self.total_num_kv_heads,  # 8
            bias=qkv_bias,  # False
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,  # 16 * 128 = 2048
            hidden_size,  # 1024
            bias=False,
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
        # 单rank上的qkv:  (seq1_len+seq2_len+..+seq_bs_len, hidden_size=1024) * (hidden_size, 512) = (seq1_len+seq2_len+..+seq_bs_len, 512)
        qkv = self.qkv_proj(hidden_states)
        # 切分拆解出q,k,v的矩阵，
        # 每个rank上面的q矩阵shape: (seq1_len+seq2_len+..+seq_bs_len, 256),
        # k矩阵shape: (seq1_len+seq2_len+..+seq_bs_len, 128),
        # v矩阵shape: (seq1_len+seq2_len+..+seq_bs_len, 128)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # q，k矩阵进行旋转位置编码，
        q, k = self.rotary_emb(positions, q, k)
        # q,k,v 送入attention计算，输出o shape: (seq1_len+seq2_len+..+seq_bs_len, 256)
        o = self.attn(q, k, v)

        # o_proj: RowParallelLinear, tp8时其weight 是（1024， 256），
        # 所以o @ weight.T = (seq1_len+seq2_len+..+seq_bs_len, 256) @ (256, 1024) = (seq1_len+seq2_len+..+seq_bs_len, 1024)
        # output shape: (seq1_len+seq2_len+..+seq_bs_len, 1024)， o_proj是行并行，在forward中会执行all_reduce动作，来获取全量SUM的结果
        output = self.o_proj(o)
        return output


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,  # 1024
            num_heads=config.num_attention_heads,  # 16
            num_kv_heads=config.num_key_value_heads,  # 8
            max_position=config.max_position_embeddings,  # 40960
            rms_norm_eps=config.rms_norm_eps,  # 1e-06
            qkv_bias=(
                True
                if config.model_type == "qwen2"
                else getattr(config, "attention_bias", False)
            ),  # false
            head_dim=getattr(config, "head_dim", None),  # 128
            rope_theta=getattr(config, "rope_theta", 1000000),  # 1000000
            rope_scaling=getattr(config, "rope_scaling", None),  # null
        )
        self.mlp = Qwen2MLP(
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


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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


class Qwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

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
