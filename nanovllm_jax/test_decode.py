"""快速测试 Decode 功能"""

import os
import sys

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

print("✓ 模型加载完成\n")

# 测试：Greedy decode，只生成5个token
prompt = "请介绍你自己，"
input_ids = tokenizer.encode(prompt, return_tensors="np")
input_ids = jnp.array(input_ids, dtype=jnp.int32)

print(f"Prompt: {prompt}")
print(f"Input IDs: {input_ids.tolist()}\n")

print("=" * 60)
print("使用 Greedy Decoding (temperature=0) 生成 5 个 token")
print("=" * 60)

# 先手动测试 prefill 采样
print("\n[调试] 测试 Prefill 阶段的采样...")
model.clear_kv_cache()
logits_prefill = model(input_ids, use_cache=True, is_decode=False)
print(f"Logits shape: {logits_prefill.shape}")
print(f"Last token logits shape: {logits_prefill[:, -1, :].shape}")
next_token_manual = jnp.argmax(logits_prefill[0, -1, :])
print(
    f"next_token_manual type: {type(next_token_manual)}, shape: {next_token_manual.shape}, value: {int(next_token_manual)}"
)
print(
    f"Prefill 采样到的 token: {int(next_token_manual)} -> '{tokenizer.decode([int(next_token_manual)])}'"
)

# 测试 generate 内部逻辑
print("\n[调试] 模拟 generate() 内部的采样逻辑...")
next_token_logits = logits_prefill[:, -1, :] / 0.0  # temperature = 0.0 会导致除0！
print(
    f"After temperature division: min={float(jnp.min(next_token_logits))}, max={float(jnp.max(next_token_logits))}, has_inf={bool(jnp.any(jnp.isinf(next_token_logits)))}"
)
next_token_test = jnp.argmax(next_token_logits, axis=-1)
print(f"Argmax result: shape={next_token_test.shape}, value={int(next_token_test[0])}")
print()

generated_ids = model.generate(
    input_ids,
    max_new_tokens=30,
    temperature=0.6,
    top_k=0,
    eos_token_id=tokenizer.eos_token_id or 151643,
)

generated_text = tokenizer.decode(generated_ids[0].tolist())
print(f"\n生成结果:\n{generated_text}\n")

# 手动逐步 decode 验证
print("=" * 60)
print("手动逐步 Decode 验证")
print("=" * 60)

model.clear_kv_cache()

# Prefill
print(f"\n[Prefill] 输入: {prompt}")
logits = model(input_ids, use_cache=True, is_decode=False)
next_token = int(jnp.argmax(logits[0, -1]))
print(f"  -> Token {next_token}: '{tokenizer.decode([next_token])}'")

tokens = [next_token]

# Decode 4步
for step in range(30):
    # 检查 KV cache 状态
    cache_len = (
        model.transformers.layers[0].self_attn.k_cache.value.shape[1]
        if model.transformers.layers[0].self_attn.k_cache.value is not None
        else 0
    )
    print(f"\n[Before Decode {step+1}] KV cache length: {cache_len}")

    logits = model(
        jnp.array([[next_token]], dtype=jnp.int32), use_cache=True, is_decode=True
    )
    next_token = int(jnp.argmax(logits[0, 0]))
    tokens.append(next_token)

    new_cache_len = model.transformers.layers[0].self_attn.k_cache.value.shape[1]
    print(
        f"[Decode {step+1}] Token {next_token}: '{tokenizer.decode([next_token])}', KV cache now: {new_cache_len}"
    )

full_text = prompt + tokenizer.decode(tokens)
print(f"\n完整生成:\n{full_text}\n")
