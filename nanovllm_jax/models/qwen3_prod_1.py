from typing import Optional
import jax
from jax import numpy as jnp
from flax import nnx
from transformers import PretrainedConfig
import logging

from nanovllm_jax.layers.embed_head import Embed

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


def _build_rope_cos_sin(seq_len: int, head_dim: int, base: float, dtype=jnp.float32):
    """Build RoPE cos/sin tables with shape [seq_len, head_dim//2]."""
    assert head_dim % 2 == 0
    # Generate inverse frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    # Generate position indices
    t = jnp.arange(seq_len, dtype=dtype)
    # Compute freqs: [seq_len, head_dim/2]
    freqs = jnp.einsum("i,j->ij", t, inv_freq)
    # Compute cos and sin
    cos = jnp.cos(freqs)  # [seq_len, head_dim/2]
    sin = jnp.sin(freqs)  # [seq_len, head_dim/2]
    return cos, sin


def _apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply RoPE to tensor x of shape [batch, seq, heads, head_dim].

    Args:
        x: [batch, seq, heads, head_dim]
        cos: [seq, head_dim//2]
        sin: [seq, head_dim//2]
    """
    # x: [batch, seq, heads, head_dim]
    # Split x into two halves along the last dimension
    x1, x2 = jnp.split(x, 2, axis=-1)  # Each: [batch, seq, heads, head_dim/2]

    # cos/sin: [seq, head_dim/2] -> broadcast to [1, seq, 1, head_dim/2]
    cos = cos[None, :, None, :]  # [1, seq, 1, head_dim/2]
    sin = sin[None, :, None, :]  # [1, seq, 1, head_dim/2]

    # Apply rotation
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    return jnp.concatenate([y1, y2], axis=-1)


class RMSNorm(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x_f = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)
        x_norm = x_f * jax.lax.rsqrt(var + self.eps)
        x_norm = x_norm.astype(self.dtype) * self.weight.value
        return x_norm.astype(orig_dtype)


class Linear(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        use_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.use_bias = use_bias

        w_shape = (out_features, in_features)
        self.weight = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(0), w_shape, dtype)
            * (1.0 / jnp.sqrt(in_features))
        )
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_features,), dtype=dtype))
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.weight.value
        y = x @ w.T
        if self.bias is not None:
            y = y + self.bias.value
        return y.astype(self.dtype)


class SelfAttention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.config = config
        self.dtype = dtype

        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", hidden_size // self.num_heads)

        # RoPE and KV cache configuration
        self.max_position = getattr(config, "max_position_embeddings", 4096)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        # Use nnx.Variable to store mutable KV cache
        self.k_cache = nnx.Variable(None)
        self.v_cache = nnx.Variable(None)

        self.q_proj = Linear(
            hidden_size,
            self.num_heads * self.head_dim,
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.k_proj = Linear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.v_proj = Linear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.o_proj = Linear(
            hidden_size, hidden_size, dtype=dtype, rngs=rngs, use_bias=False
        )

        self.q_norm = RMSNorm(
            hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )

        self.k_norm = RMSNorm(
            hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )

    def __call__(
        self, hidden_states: jax.Array, use_cache: bool = False, is_decode: bool = False
    ) -> jax.Array:
        """
        Args:
            hidden_states: [batch, seq, hidden] for prefill, or [batch, 1, hidden] for decode
            use_cache: whether to use/update KV cache
            is_decode: if True, only process last token and append to cache
        """
        # hidden_states: [batch, seq, hidden]
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 归一化
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Apply RoPE to Q and K
        if is_decode and use_cache and self.k_cache.value is not None:
            # Decode: use past_len as position offset
            past_len = self.k_cache.value.shape[1]
            cos, sin = _build_rope_cos_sin(
                past_len + seq_len, self.head_dim, self.rope_theta, dtype=jnp.float32
            )
            # Only take the positions for current tokens
            cos = cos[past_len : past_len + seq_len, :]
            sin = sin[past_len : past_len + seq_len, :]
        else:
            # Prefill: positions 0..seq_len-1
            cos, sin = _build_rope_cos_sin(
                seq_len, self.head_dim, self.rope_theta, dtype=jnp.float32
            )

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # KV cache management (store BEFORE head expansion)
        if use_cache:
            if is_decode and self.k_cache.value is not None:
                # Append current k/v to cache (before repeat)
                k = jnp.concatenate([self.k_cache.value, k], axis=1)
                v = jnp.concatenate([self.v_cache.value, v], axis=1)
            # Store/update cache using nnx.Variable (original kv_heads, not repeated)
            self.k_cache.value = k
            self.v_cache.value = v

        # Expand KV heads if using GQA/MQA (AFTER caching)
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat_factor, axis=2)
            v = jnp.repeat(v, repeat_factor, axis=2)

        q = jnp.transpose(q, (0, 2, 1, 3))  # [b, h, t, d]
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # DEBUG: Print shapes in decode mode
        if is_decode and use_cache:
            import sys

            # print(
            #     f"[DEBUG Attention] q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}",
            #     file=sys.stderr,
            # )

        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # Causal mask: query can only attend to key positions <= query position
        kv_len = k.shape[2]
        q_len = q.shape[2]
        if is_decode:
            # Decode: q_len=1, kv_len=past_len+1, no mask needed (all positions visible)
            causal_mask = jnp.ones((q_len, kv_len), dtype=bool)
        else:
            # Prefill: standard causal mask
            causal_mask = jnp.tril(jnp.ones((q_len, kv_len), dtype=bool))

        attn_scores = jnp.where(causal_mask, attn_scores, -1e9)

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)

        context = jnp.transpose(context, (0, 2, 1, 3))  # [b, t, h, d]
        context = context.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        output = self.o_proj(context)

        # DEBUG
        if is_decode and use_cache:
            import sys

            # print(
            #     f"[DEBUG Attention Output] output shape: {output.shape}, expected: [{batch_size}, {seq_len}, {hidden_size}]",
            #     file=sys.stderr,
            # )

        return output


class MLP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.config = config
        self.dtype = dtype

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = Linear(
            hidden_size, intermediate_size, dtype=dtype, rngs=rngs, use_bias=False
        )
        self.up_proj = Linear(
            hidden_size, intermediate_size, dtype=dtype, rngs=rngs, use_bias=False
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, dtype=dtype, rngs=rngs, use_bias=False
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        act = jax.nn.silu(gate) * up
        out = self.down_proj(act)
        return out


class Qwen3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.config = config
        self.layer_id = layer_id
        self.dtype = dtype

        eps = getattr(config, "rms_norm_eps", 1e-6)
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.self_attn = SelfAttention(config, dtype=dtype, rngs=rngs)
        self.mlp = MLP(config, dtype=dtype, rngs=rngs)

    def __call__(
        self, hidden_states: jax.Array, use_cache: bool = False, is_decode: bool = False
    ) -> jax.Array:
        # Self-attention block
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, use_cache=use_cache, is_decode=is_decode)
        x = x + residual

        # MLP block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual

        return x


class Qwen3Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        # logger.info(f"Initializing Qwen3Model with config: {config}")
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        self.layers = nnx.List(
            [
                Qwen3DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    rngs=rngs,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=eps, dtype=dtype)

    def __call__(
        self, input_ids: jax.Array, use_cache: bool = False, is_decode: bool = False
    ) -> jax.Array:
        # input_ids: [batch, seq]
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, use_cache=use_cache, is_decode=is_decode
            )

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def sample(
    logits: jax.Array,
    temperature: float = 1.0,
    top_k: int = 50,
    rng_key: jax.Array = None,
) -> jax.Array:
    """
    Sample next token from logits.

    Args:
        logits: [batch, vocab_size] logits for next token prediction
        temperature: sampling temperature (0.0 = greedy, 1.0 = no change, <1.0 = sharper, >1.0 = flatter)
        top_k: only sample from top k tokens (0 = disabled, full distribution)
        rng_key: JAX random key for sampling (if None, uses key based on hash of logits)

    Returns:
        next_token: [batch] sampled token ids
    """
    batch_size = logits.shape[0]

    if temperature == 0.0:
        # Greedy: directly argmax without temperature scaling
        return jnp.argmax(logits, axis=-1)

    # Apply temperature scaling
    scaled_logits = logits / temperature

    if top_k > 0:
        # Keep only top k logits
        top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, top_k)
        # Sample from top k
        probs = jax.nn.softmax(top_k_logits, axis=-1)
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        next_token_idx = jax.random.categorical(
            rng_key, jnp.log(probs + 1e-10), axis=-1
        )
        next_token = top_k_indices[jnp.arange(batch_size), next_token_idx]
    else:
        # Sample from full distribution
        probs = jax.nn.softmax(scaled_logits, axis=-1)
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        next_token = jax.random.categorical(rng_key, jnp.log(probs + 1e-10), axis=-1)

    return next_token


class ParallelLMHead(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.rngs = rngs

        self.weight = nnx.Param(
            init_fn(jax.random.PRNGKey(0), (vocab_size, hidden_size), dtype)
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = hidden_states.astype(self.dtype)
        return hidden_states @ self.weight.value.T


class Qwen3ForCausalLM(nnx.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        # logger.info(f"Initializing Qwen3ForCausalLM with config: {config}")
        self.transformers = Qwen3Model(config, dtype=self.dtype, rngs=rngs)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, dtype=self.dtype, rngs=rngs
        )

    def __call__(
        self, input_ids: jax.Array, use_cache: bool = False, is_decode: bool = False
    ) -> jax.Array:
        hidden_states = self.transformers(
            input_ids, use_cache=use_cache, is_decode=is_decode
        )
        logits = self.lm_head(hidden_states)
        return logits

    def clear_kv_cache(self):
        """Clear KV cache in all layers."""
        for layer in self.transformers.layers:
            layer.self_attn.k_cache.value = None
            layer.self_attn.v_cache.value = None

    def generate(
        self,
        input_ids: jax.Array,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int = 151643,
    ) -> jax.Array:
        """
        Simple greedy/sampling generation loop with KV cache.

        Args:
            input_ids: [batch, seq_len] prompt token ids
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (1.0 = no change, <1.0 = sharper, >1.0 = flatter)
            top_k: only sample from top k tokens (0 = disabled)
            eos_token_id: stop when this token is generated

        Returns:
            generated_ids: [batch, seq_len + num_generated] including prompt
        """
        batch_size = input_ids.shape[0]

        # Clear any existing KV cache before starting
        self.clear_kv_cache()

        # Prefill: process full prompt
        logits = self(input_ids, use_cache=True, is_decode=False)  # [batch, seq, vocab]

        # Sample next token from last position
        next_token = sample(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            rng_key=jax.random.PRNGKey(0),
        )

        generated_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)

        # Decode loop
        for step in range(max_new_tokens - 1):
            # Check if all sequences hit EOS
            if jnp.all(next_token == eos_token_id):
                break

            # Decode: process only last token
            logits = self(
                next_token[:, None], use_cache=True, is_decode=True
            )  # [batch, 1, vocab]

            next_token = sample(
                logits[:, 0, :],
                temperature=temperature,
                top_k=top_k,
                rng_key=jax.random.PRNGKey(step + 1),
            )

            generated_ids = jnp.concatenate(
                [generated_ids, next_token[:, None]], axis=1
            )

        return generated_ids
