import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    hidden_size: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.weight = self.param(
            'weight',
            nn.initializers.ones,
            (self.hidden_size,),
            self.dtype
        )

    def rms_forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """RMS normalization forward pass."""
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        x = x.astype(orig_dtype) * self.weight
        return x

    def add_rms_forward(self, x: jnp.ndarray, residual: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """RMS normalization with residual connection."""
        orig_dtype = x.dtype
        x = x.astype(jnp.float32) + residual.astype(jnp.float32)
        residual = x.astype(orig_dtype)
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        x = x.astype(orig_dtype) * self.weight
        return x, residual

    def __call__(self, x: jnp.ndarray, residual: Optional[jnp.ndarray] = None):
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
