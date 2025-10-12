import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
from transformers import Qwen3Config

from nanovllm_jax.layers.activation import SiluAndMul
from nanovllm_jax.layers.attention import Attention
from nanovllm_jax.layers.layernorm import RMSNorm
from nanovllm_jax.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm_jax.layers.rotary_embedding import get_rope
from nanovllm_jax.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 attention layer."""
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    max_position: int = 4096 * 32
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-06
    qkv_bias: bool = False
    rope_theta: float = 1000000
    rope_scaling: Optional[Tuple] = None
    tp_size: int = 1
    tp_rank: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Calculate derived values without modifying existing attributes
        total_num_heads = self.num_heads
        assert total_num_heads % self.tp_size == 0
        num_heads_per_partition = total_num_heads // self.tp_size
        
        total_num_kv_heads = self.num_kv_heads
        assert total_num_kv_heads % self.tp_size == 0
        num_kv_heads_per_partition = total_num_kv_heads // self.tp_size
        
        head_dim = self.head_dim or self.hidden_size // total_num_heads
        q_size = num_heads_per_partition * head_dim
        kv_size = num_kv_heads_per_partition * head_dim
        scaling = head_dim ** -0.5

        # Calculate output size for QKV projection
        qkv_output_size = (total_num_heads + 2 * total_num_kv_heads) * head_dim
        
        self.qkv_proj = QKVParallelLinear(
            input_size=self.hidden_size,
            output_size=qkv_output_size,
            head_size=head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=self.qkv_bias,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            dtype=self.dtype
        )
        
        self.o_proj = RowParallelLinear(
            input_size=total_num_heads * head_dim,
            output_size=self.hidden_size,
            bias=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            dtype=self.dtype
        )
        
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=self.max_position,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        
        self.attn = Attention(
            num_heads=num_heads_per_partition,
            head_dim=head_dim,
            scale=scaling,
            num_kv_heads=num_kv_heads_per_partition,
            dtype=self.dtype
        )
        
        self.q_norm = RMSNorm(head_dim, eps=self.rms_norm_eps, dtype=self.dtype)
        self.k_norm = RMSNorm(head_dim, eps=self.rms_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        positions: jnp.ndarray,
        hidden_states: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass of Qwen3 attention."""
        # Calculate derived values
        total_num_heads = self.num_heads
        num_heads_per_partition = total_num_heads // self.tp_size
        total_num_kv_heads = self.num_kv_heads
        num_kv_heads_per_partition = total_num_kv_heads // self.tp_size
        head_dim = self.head_dim or self.hidden_size // total_num_heads
        q_size = num_heads_per_partition * head_dim
        kv_size = num_kv_heads_per_partition * head_dim
        
        # Compute QKV projections
        qkv = self.qkv_proj(hidden_states)
        
        # Split into Q, K, V
        q, k, v = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)
        
        # Reshape for attention heads
        q_by_head = q.reshape(-1, num_heads_per_partition, head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.reshape(q.shape)
        
        k_by_head = k.reshape(-1, num_kv_heads_per_partition, head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.reshape(k.shape)
        
        # Apply rotary embedding
        q, k = self.rotary_emb(positions, q, k)
        
        # Apply attention
        o = self.attn(q, k, v)
        
        # Output projection
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):
    """Qwen3 MLP layer."""
    hidden_size: int
    intermediate_size: int
    hidden_act: str
    tp_size: int = 1
    tp_rank: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.intermediate_size * 2,
            output_sizes=[self.intermediate_size] * 2,
            bias=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            dtype=self.dtype
        )
        
        self.down_proj = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            dtype=self.dtype
        )
        
        assert self.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of Qwen3 MLP."""
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer."""
    config: Qwen3Config
    tp_size: int = 1
    tp_rank: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = Qwen3Attention(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            max_position=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            qkv_bias=getattr(self.config, "attention_bias", False),
            head_dim=getattr(self.config, "head_dim", None),
            rope_theta=getattr(self.config, "rope_theta", 1000000),
            rope_scaling=getattr(self.config, "rope_scaling", None),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            dtype=self.dtype
        )
        
        self.mlp = Qwen3MLP(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            dtype=self.dtype
        )
        
        self.input_layernorm = RMSNorm(
            self.config.hidden_size, 
            eps=self.config.rms_norm_eps,
            dtype=self.dtype
        )
        self.post_attention_layernorm = RMSNorm(
            self.config.hidden_size, 
            eps=self.config.rms_norm_eps,
            dtype=self.dtype
        )

    def __call__(
        self,
        positions: jnp.ndarray,
        hidden_states: jnp.ndarray,
        residual: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of Qwen3 decoder layer."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        # Note: No post_attention_layernorm here in the original PyTorch implementation
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 model."""
    config: Qwen3Config
    tp_size: int = 1
    tp_rank: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            dtype=self.dtype
        )
        
        self.norm = RMSNorm(
            self.config.hidden_size, 
            eps=self.config.rms_norm_eps,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass of Qwen3 model."""
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        
        # Create layers dynamically in compact method
        for i in range(self.config.num_hidden_layers):
            layer = Qwen3DecoderLayer(
                config=self.config,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                dtype=self.dtype
            )
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 for causal language modeling."""
    config: Qwen3Config
    tp_size: int = 1
    tp_rank: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = Qwen3Model(
            config=self.config,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            dtype=self.dtype
        )
        
        self.lm_head = ParallelLMHead(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            dtype=self.dtype
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray,
        compute_logits: bool = True,
    ) -> jnp.ndarray:
        """Forward pass of Qwen3ForCausalLM."""
        hidden_states = self.model(input_ids, positions)
        
        if compute_logits:
            # Compute logits from hidden states
            logits = self.lm_head(hidden_states)
            return logits
        else:
            return hidden_states

    def compute_logits(
        self,
        hidden_states: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute logits from hidden states."""
        # Use the lm_head from setup
        logits = self.lm_head(hidden_states)
        return logits
