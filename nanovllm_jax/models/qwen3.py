import jax
from flax import nnx
from transformers import PretrainedConfig
class Qwen3ForCausalLM(nnx.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ):
        ...