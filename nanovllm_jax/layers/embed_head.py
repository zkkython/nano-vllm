import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class VocabParallelEmbedding(nn.Module):
    """Vocabulary parallel embedding layer."""
    vocab_size: int
    hidden_size: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedding = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.hidden_size),
            self.dtype
        )

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Embed input token IDs."""
        return self.embedding[input_ids]


class ParallelLMHead(nn.Module):
    """Parallel language modeling head."""
    vocab_size: int
    hidden_size: int
    dtype: jnp.dtype = jnp.float32
    tied_weights: Optional[jnp.ndarray] = None

    def setup(self):
        # Initialize weight parameter
        self.weight = self.param(
            'weight',
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.hidden_size),
            self.dtype
        )

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Compute logits from hidden states."""
        return jnp.dot(hidden_states, self.weight.T)
