"""验证 JAX 模型权重是否正确加载，对比 HF 官方模型输出"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from nanovllm_jax.configs.model_config import ModelConfig
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.utils.weight_utils import WeightLoader
from nanovllm_jax.utils.test_weight_emd_lm import TestQwen3NNXForward

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

print("=" * 80)
print("对比 JAX 模型 vs HF 官方模型的输出")
print("=" * 80)

# 1. 加载 HF 官方模型
print("\n[1] 加载 HF 官方 Qwen3-0.6B 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
hf_model = hf_model.to("cpu")  # 确保在 CPU 上
hf_model.eval()
print("✓ HF 模型加载完成")

# 2. 加载 JAX 模型并导入权重
print("\n[2] 加载 JAX nnx Qwen3 模型并导入真实权重...")
devices = jax.devices()
mesh = jax.sharding.Mesh(devices[: min(4, len(devices))], ("tensor",))

jax_model = Qwen3ForCausalLM(
    config=hf_config,
    dtype=jnp.bfloat16,
    rngs=None,
    mesh=mesh,
)

model_config = ModelConfig(model_path=MODEL_PATH, trust_remote_code=True)
helper = TestQwen3NNXForward()
helper.hf_config = hf_config
weight_mappings = helper._build_weight_mappings()

loader = WeightLoader(
    model=jax_model,
    model_config=model_config,
    mesh=mesh,
    dtype=jnp.bfloat16,
)
loader.load_weights_from_safetensors(weight_mappings)
print("✓ JAX 模型权重加载完成")

# 3. 测试相同输入
print("\n[3] 使用相同输入进行前向推理...")
test_prompt = "你好"
input_ids = tokenizer.encode(test_prompt, return_tensors="pt")
print(f"测试输入: '{test_prompt}'")
print(f"Token IDs: {input_ids.tolist()}")

# HF 前向
with torch.no_grad():
    hf_outputs = hf_model(input_ids)
    hf_logits = hf_outputs.logits[0, -1, :]  # Last token logits
    hf_top5 = torch.topk(hf_logits, 5)

print("\n[HF 模型] Top-5 预测:")
for i, (score, token_id) in enumerate(zip(hf_top5.values, hf_top5.indices)):
    token_text = tokenizer.decode([token_id.item()])
    print(
        f"  {i+1}. Token {token_id.item()}: '{token_text}' (score: {score.item():.4f})"
    )

# JAX 前向
jax_input_ids = jnp.array(input_ids.numpy(), dtype=jnp.int32)
jax_logits = jax_model(jax_input_ids, use_cache=False, is_decode=False)
jax_logits_last = jax_logits[0, -1, :]  # Last token logits
jax_top5_indices = jnp.argsort(jax_logits_last)[-5:][::-1]
jax_top5_scores = jax_logits_last[jax_top5_indices]

print("\n[JAX 模型] Top-5 预测:")
for i, (token_id, score) in enumerate(zip(jax_top5_indices, jax_top5_scores)):
    token_text = tokenizer.decode([int(token_id)])
    print(f"  {i+1}. Token {int(token_id)}: '{token_text}' (score: {float(score):.4f})")

# 4. 检查权重是否一致
print("\n[4] 检查 Embedding 权重是否一致...")
hf_embed_weight = hf_model.model.embed_tokens.weight.data  # [vocab, hidden]
jax_embed_weight = jax_model.transformers.embed_tokens.embedding.value  # nnx Param

# 选择几个 token 检查
test_token_ids = [1, 100, 1000, 10000]
print(f"检查 Token IDs: {test_token_ids}")
for tid in test_token_ids:
    hf_vec = hf_embed_weight[tid].cpu().numpy()
    jax_vec = jnp.array(jax_embed_weight[tid])
    diff = jnp.abs(hf_vec - jax_vec).max()
    print(f"  Token {tid}: max diff = {float(diff):.6f}")

print("\n" + "=" * 80)
print("验证完成！")
print("=" * 80)
