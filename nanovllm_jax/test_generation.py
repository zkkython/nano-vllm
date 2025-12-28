"""测试 Qwen3 生成质量"""

import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoConfig

from nanovllm_jax.configs.model_config import ModelConfig
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.utils.weight_utils import WeightLoader
from nanovllm_jax.utils.test_weight_emd_lm import TestQwen3NNXForward

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

devices = jax.devices()
mesh = jax.sharding.Mesh(devices[: min(4, len(devices))], ("tensor",))
model = Qwen3ForCausalLM(config=hf_config, dtype=jnp.bfloat16, rngs=None, mesh=mesh)

model_config = ModelConfig(model_path=MODEL_PATH, trust_remote_code=True)
helper = TestQwen3NNXForward()
helper.hf_config = hf_config
weight_mappings = helper._build_weight_mappings()
loader = WeightLoader(
    model=model, model_config=model_config, mesh=mesh, dtype=jnp.bfloat16
)
loader.load_weights_from_safetensors(weight_mappings)

print("\n" + "=" * 70)
print("测试生成 (Greedy Decoding, temperature=0)")
print("=" * 70)

test_prompts = [
    "北京是中国的首都，",
    "你是什么模型",
    "哪个城市最美",
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids_jax = jnp.array(input_ids.numpy(), dtype=jnp.int32)

    # 使用 greedy decoding (temperature=0)
    output = model.generate(
        input_ids=input_ids_jax, max_new_tokens=20, temperature=0.0, top_k=1
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    print(f"Output tokens: {output[0].tolist()}")

print("\n" + "=" * 70)
print("测试生成 (Sampling, temperature=0.7)")
print("=" * 70)

for prompt in test_prompts[:3]:  # 只测试前两个
    print(f"\nPrompt: {prompt}")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids_jax = jnp.array(input_ids.numpy(), dtype=jnp.int32)

    output = model.generate(
        input_ids=input_ids_jax, max_new_tokens=20, temperature=0.7, top_k=0
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")

print("\n完成！")
