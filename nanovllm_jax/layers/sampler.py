import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class Sampler(nn.Module):
    """Sampling layer for token generation."""
    
    def __call__(
        self, 
        logits: jnp.ndarray, 
        temperatures: Optional[jnp.ndarray] = None,
        rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """Sample tokens from logits."""
        if rng is None:
            rng = jax.random.PRNGKey(42)  # Use a different seed for variety
        
        if temperatures is not None and temperatures.size > 0:
            # Apply temperature scaling
            # Reshape temperatures to match logits batch dimension
            temperatures = jnp.reshape(temperatures, (-1, 1))
            # Only apply to the batch size that matches
            batch_size = min(logits.shape[0], temperatures.shape[0])
            logits = logits.at[:batch_size].set(
                logits[:batch_size] / temperatures[:batch_size]
            )
        
        # Sample from the distribution
        token_ids = jax.random.categorical(
            rng, 
            logits, 
            axis=-1
        )
        
        return token_ids
