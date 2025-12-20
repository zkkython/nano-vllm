import jax
from jax import numpy as jnp
from flax import nnx
from transformers import PretrainedConfig
import logging
logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()

class Qwen3ForCausalLM(nnx.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info(f"Initializing Qwen3ForCausalLM with config: {config}")
 