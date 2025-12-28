import jax.numpy as jnp
from functools import lru_cache
from flax import nnx


def apply_rotary_emb(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    """Apply rotary embedding to input tensor."""
    cos = jnp.expand_dims(cos, -2)
    sin = jnp.expand_dims(sin, -2)
    x1, x2 = jnp.split(x.astype(jnp.bfloat16), 2, axis=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return jnp.concatenate((y1, y2), axis=-1).astype(x.dtype)


class RotaryEmbedding(nnx.Module):
    """Rotary Position Embedding (RoPE) using Flax NNX."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

        assert self.rotary_dim == self.head_size

        # Build cos/sin cache
        inv_freq = 1.0 / (
            self.base
            ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.bfloat16) / self.rotary_dim)
        )
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.bfloat16)
        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        # Store as nnx.Variable (immutable cache)
        self.cos_sin_cache = nnx.Variable(jnp.concatenate((cos, sin), axis=-1))

    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply rotary embedding to query and key tensors."""
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.value[positions]
        cos, sin = jnp.split(cos_sin, 2, axis=-1)

        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).reshape(key_shape)

        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    dtype: jnp.dtype = jnp.bfloat16,
    rope_scaling: dict | None = None,
):
    """Get cached rotary embedding instance."""
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, dtype)
    return rotary_emb
