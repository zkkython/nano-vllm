import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Dict, Any


class Attention(nn.Module):
    """Multi-head attention with KV cache support."""
    num_heads: int
    head_dim: int
    scale: float
    num_kv_heads: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.k_cache = None
        self.v_cache = None

    def store_kvcache(self, key: jnp.ndarray, value: jnp.ndarray, slot_mapping: jnp.ndarray):
        """Store key-value cache."""
        if self.k_cache is not None and self.v_cache is not None:
            # Store k, v in cache at specified slots
            self.k_cache = self.k_cache.at[slot_mapping].set(key)
            self.v_cache = self.v_cache.at[slot_mapping].set(value)

    def __call__(
        self, 
        q: jnp.ndarray, 
        k: jnp.ndarray, 
        v: jnp.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> jnp.ndarray:
        """Forward pass of attention mechanism."""
        # Input shape: (num_tokens, hidden_size) where num_tokens = batch_size * seq_len
        # Reshape to (num_tokens, num_heads, head_dim)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        
        num_tokens = q.shape[0]

        if context is not None:
            # Store in KV cache if provided
            if 'slot_mapping' in context:
                self.store_kvcache(k, v, context['slot_mapping'])

            # Use cached k, v if available
            if self.k_cache is not None and self.v_cache is not None:
                k = self.k_cache
                v = self.v_cache

        # Handle different number of heads for Q and KV
        # q: (num_tokens, num_heads, head_dim)
        # k, v: (num_tokens, num_kv_heads, head_dim)
        
        # For multi-query attention, we need to repeat k and v to match q
        if self.num_kv_heads != self.num_heads:
            # Repeat k and v to match the number of query heads
            repeat_factor = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat_factor, axis=1)  # (num_tokens, num_heads, head_dim)
            v = jnp.repeat(v, repeat_factor, axis=1)  # (num_tokens, num_heads, head_dim)
        
        # Transpose to (num_heads, num_tokens, head_dim)
        q = jnp.transpose(q, (1, 0, 2))  # (num_heads, num_tokens, head_dim)
        k = jnp.transpose(k, (1, 0, 2))  # (num_heads, num_tokens, head_dim)
        v = jnp.transpose(v, (1, 0, 2))  # (num_heads, num_tokens, head_dim)
        
        # Compute scores: (num_heads, num_tokens, num_tokens)
        scores = jnp.einsum('hqd,hkd->hqk', q, k) * self.scale
        
        # Apply causal mask
        causal_mask = jnp.tril(jnp.ones((num_tokens, num_tokens)))
        scores = jnp.where(causal_mask == 0, -jnp.inf, scores)
        
        # Softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention to values: (num_heads, num_tokens, head_dim)
        o = jnp.einsum('hqk,hvd->hqd', attn_weights, v)
        
        # Reshape back to (num_tokens, num_heads * head_dim)
        o = jnp.transpose(o, (1, 0, 2))  # (num_tokens, num_heads, head_dim)
        return o.reshape(num_tokens, self.num_heads * self.head_dim)


class FlashAttention(nn.Module):
    """Flash Attention implementation for JAX."""
    num_heads: int
    head_dim: int
    scale: float
    num_kv_heads: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.k_cache = None
        self.v_cache = None

    def __call__(
        self, 
        q: jnp.ndarray, 
        k: jnp.ndarray, 
        v: jnp.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> jnp.ndarray:
        """Flash attention forward pass."""
        # For now, use standard attention
        # In a full implementation, this would use optimized flash attention
        return Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            dtype=self.dtype
        )(q, k, v, context)
