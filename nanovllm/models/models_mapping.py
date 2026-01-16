from nanovllm.models.qwen3 import Qwen3ForCausalLM

# from nanovllm.models.qwen3_2 import Qwen3ForCausalLM as Qwen3_2ForCausalLM
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.llama import LLamaForCausalLM
from nanovllm.models.deepseek_v3 import DeepSeekV3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM

MODELS_MAPPING = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "llama": LLamaForCausalLM,
    "deepseek_v3": DeepSeekV3ForCausalLM,
}
