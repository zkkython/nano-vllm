from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.llama import LLamaForCausalLM

MODELS_MAPPING = {
    "qwen3": Qwen3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "llama": LLamaForCausalLM,
}
