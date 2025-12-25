from tkinter.constants import NONE
from typing import Optional
import jax
from jax import numpy as jnp
from flax import nnx
from transformers import PretrainedConfig
import logging

from nanovllm_jax.layers.embed_head import Embed
logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class QWen3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        pass


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
        logger.info(f"Initializing Qwen3Model with config: {config}")
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        self.layers = nnx.data(
            [
                QWen3DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    rngs=rngs,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # self.norm = RMSNorm(
        #     config.hidden_size,
        #     epsilon=config.rms_norm_eps,
        #     param_dtype=dtype,
        #     scale_init=nnx.with_partitioning(init_fn, (None,)),
        #     rngs=rngs,
        # )


class LogitsProcessor(nnx.Module):
    def __init__(self, vocab_size, mesh):
        self.vocab_size = vocab_size
        self.mesh = mesh


class ParallelLMHead(nnx.Module):
    def __init__(self, vocab_size, hidden_size, rngs: nnx.Rngs | None = None):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rngs = rngs
        ...


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
        logger.info(f"Initializing Qwen3ForCausalLM with config: {config}")
        self.transformers = Qwen3Model(config, dtype=self.dtype, rngs=rngs)
        if not getattr(self.config, "tie_word_embeddings", True):
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, rngs=rngs)
        self.logits_processor = LogitsProcessor(config.vocab_size, self.mesh)
