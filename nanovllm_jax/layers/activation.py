import jax
import jax.numpy as jnp
from flax import linen as nn


class SiluAndMul(nn.Module):
    """SiLU activation with element-wise multiplication."""
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x, y = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(x) * y
