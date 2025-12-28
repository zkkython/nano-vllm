"""对比 JAX 和 HF 的 Prefill 输出"""

import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from nanovllm_jax.configs.model_config import ModelConfig
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.utils.weight_utils import WeightLoader
from nanovllm_jax.utils.test_weight_emd_lm import TestQwen3NNXForward

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
hf_model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    .to("cpu")
    .eval()
)
hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

devices = jax.devices()
mesh = jax.sharding.Mesh(devices[: min(4, len(devices))], ("tensor",))
jax_model = Qwen3ForCausalLM(config=hf_config, dtype=jnp.bfloat16, rngs=None, mesh=mesh)

model_config = ModelConfig(model_path=MODEL_PATH, trust_remote_code=True)
helper = TestQwen3NNXForward()
helper.hf_config = hf_config
weight_mappings = helper._build_weight_mappings()
loader = WeightLoader(
    model=jax_model, model_config=model_config, mesh=mesh, dtype=jnp.bfloat16
)
loader.load_weights_from_safetensors(weight_mappings)

print("\n" + "=" * 80)
print("测试 Prefill 输出")
print("=" * 80)

test_prompts = [
    "北京是中国的首都，",
    "人工智能",
    "今天天气很好",
]

for prompt in test_prompts:
    print(f"\n{'=' * 80}")
    print(f"Prompt: {prompt}")
    print("=" * 80)

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids_jax = jnp.array(input_ids.numpy(), dtype=jnp.int32)

    print(f"Input IDs: {input_ids[0].tolist()}")
    print(f"Sequence length: {input_ids.shape[1]}")

    # HF forward
    print("\n[HF Prefill]")
    with torch.no_grad():
        hf_output = hf_model(input_ids, use_cache=True, return_dict=True)
        hf_logits = hf_output.logits  # [batch, seq, vocab]
        hf_cache = hf_output.past_key_values

    print(f"HF logits shape: {hf_logits.shape}")
    print(f"HF last token logits[:10]: {hf_logits[0, -1, :10].float().numpy()}")
    hf_top5 = torch.topk(hf_logits[0, -1], k=5)
    print(f"HF top-5 tokens: {hf_top5.indices.tolist()}")
    print(f"HF top-5 values: {hf_top5.values.float().numpy()}")

    # JAX forward
    print("\n[JAX Prefill]")
    jax_model.clear_kv_cache()
    jax_logits = jax_model(input_ids_jax, use_cache=True, is_decode=False)

    print(f"JAX logits shape: {jax_logits.shape}")
    print(
        f"JAX last token logits[:10]: {np.array(jax_logits[0, -1, :10], dtype=np.float32)}"
    )
    jax_top5_indices = jnp.argsort(jax_logits[0, -1])[-5:][::-1]
    jax_top5_values = jax_logits[0, -1, jax_top5_indices]
    print(f"JAX top-5 tokens: {jax_top5_indices.tolist()}")
    print(f"JAX top-5 values: {np.array(jax_top5_values, dtype=np.float32)}")

    # 对比
    print("\n[对比]")
    hf_logits_np = hf_logits[0, -1].float().numpy()
    jax_logits_np = np.array(jax_logits[0, -1], dtype=np.float32)

    diff = np.abs(hf_logits_np - jax_logits_np)
    print(f"Logits diff - max: {diff.max():.6f}, mean: {diff.mean():.6f}")

    # 检查 top-1 是否一致
    hf_top1 = hf_top5.indices[0].item()
    jax_top1 = jax_top5_indices[0].item()

    if hf_top1 == jax_top1:
        print(f"✅ Top-1 token 一致: {hf_top1}")
    else:
        print(f"❌ Top-1 token 不一致: HF={hf_top1}, JAX={jax_top1}")

    # 检查 top-5 重叠
    hf_top5_set = set(hf_top5.indices.tolist())
    jax_top5_set = set(jax_top5_indices.tolist())
    overlap = hf_top5_set & jax_top5_set
    print(f"Top-5 重叠数: {len(overlap)}/5")

    if diff.max() < 1.0:
        print("✅ Prefill logits 基本一致（diff < 1.0）")
    else:
        print(f"⚠️  Prefill logits 差异较大（max diff = {diff.max():.6f}）")

    # 检查 KV cache
    print("\n[KV Cache 检查]")
    hf_k_cache, hf_v_cache = hf_cache[0]
    jax_k_cache = jax_model.transformers.layers[0].self_attn.k_cache.value
    jax_v_cache = jax_model.transformers.layers[0].self_attn.v_cache.value

    print(
        f"HF K cache shape: {hf_k_cache.shape}"
    )  # [batch, num_kv_heads, seq, head_dim]
    print(
        f"JAX K cache shape: {jax_k_cache.shape}"
    )  # [batch, seq, num_kv_heads, head_dim]

    # Transpose JAX cache to HF order
    jax_k_transposed = jnp.transpose(jax_k_cache, (0, 2, 1, 3))

    k_diff = np.abs(
        hf_k_cache.float().numpy() - np.array(jax_k_transposed, dtype=np.float32)
    )
    print(f"K cache diff - max: {k_diff.max():.6f}, mean: {k_diff.mean():.6f}")

    if k_diff.max() < 1.0:
        print("✅ K cache 基本一致（diff < 1.0）")
    else:
        print(f"⚠️  K cache 差异较大（max diff = {k_diff.max():.6f}）")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
