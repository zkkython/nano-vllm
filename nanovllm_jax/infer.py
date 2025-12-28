import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer, AutoConfig

from nanovllm_jax.configs.model_config import ModelConfig
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.utils.weight_utils import WeightLoader, WeightMapping
from nanovllm_jax.utils.test_weight_emd_lm import (
    TestQwen3NNXForward,
)  # 复用里面的 _build_weight_mappings

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

# 1. 加载 tokenizer 和 HF config
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 2. 构建 mesh 和 nnx 模型
devices = jax.devices()
mesh = jax.sharding.Mesh(devices[: min(4, len(devices))], ("tensor",))

model = Qwen3ForCausalLM(
    config=hf_config,
    dtype=jnp.bfloat16,
    rngs=None,
    mesh=mesh,
)

# 3. 构建 ModelConfig + WeightLoader + WeightMapping
model_config = ModelConfig(model_path=MODEL_PATH, trust_remote_code=True)

# 复用测试类里的 _build_weight_mappings 实现（核心逻辑已经在那里）
helper = TestQwen3NNXForward()
helper.hf_config = hf_config  # 手动塞进去，供 _build_weight_mappings 使用
weight_mappings = helper._build_weight_mappings()

loader = WeightLoader(
    model=model,
    model_config=model_config,
    mesh=mesh,
    dtype=jnp.bfloat16,
)

loader.load_weights_from_safetensors(weight_mappings)

# 验证权重是否加载：检查 embedding 的范数
embed_weight = model.transformers.embed_tokens.embedding.value
print(f"\nEmbedding weight shape: {embed_weight.shape}")
print(f"Embedding weight dtype: {embed_weight.dtype}")
# 检查前几个token的embedding范数
for i in [0, 1, 100, 1000]:
    norm = float(jnp.linalg.norm(embed_weight[i]))
    print(f"  Token {i} embedding norm: {norm:.4f}")

print("=" * 80)
print("Qwen3-0.6B JAX/nnx 模型已加载真实权重，开始推理测试")
print("=" * 80)

# 4. 测试单步 Prefill（验证模型基本功能）
print("\n[测试1] 单步 Prefill - 验证模型前向")
prompt = "你好，介绍一下你自己。"
input_ids = tokenizer.encode(prompt, return_tensors="np")
input_ids = jnp.array(input_ids, dtype=jnp.int32)

logits = model(input_ids, use_cache=False, is_decode=False)  # [batch, seq, vocab]
next_token = int(jnp.argmax(logits[0, -1]))
print(f"Prompt: {prompt}")
print(f"Next token (greedy): {next_token} -> '{tokenizer.decode([next_token])}'")

# 5. 测试 Prefill + Decode 循环（真正的多步生成）
print("\n[测试2] Prefill + Decode 循环 - 生成多个 token")
prompt2 = "北京是中国的首都，"
input_ids2 = tokenizer.encode(prompt2, return_tensors="np")
input_ids2 = jnp.array(input_ids2, dtype=jnp.int32)

print(f"Prompt: {prompt2}")
print("开始生成...")

# 使用 model.generate() 自动完成 Prefill + Decode
# 先用 greedy (temperature=0) 测试，避免采样随机性
generated_ids = model.generate(
    input_ids2,
    max_new_tokens=20,
    temperature=0.0,  # Greedy decoding
    top_k=0,  # 禁用 top-k
    eos_token_id=tokenizer.eos_token_id or 151643,
)

generated_text = tokenizer.decode(generated_ids[0].tolist())
print(f"\n生成结果:\n{generated_text}")

print("\n" + "=" * 80)
print("推理测试完成！")
print("=" * 80)
