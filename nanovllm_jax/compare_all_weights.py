"""Compare all weights between HuggingFace and JAX models"""

import torch
import jax.numpy as jnp
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig
import jax

from nanovllm_jax.configs.model_config import ModelConfig
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.utils.weight_utils import WeightLoader
from nanovllm_jax.utils.test_weight_emd_lm import TestQwen3NNXForward

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"


def get_hf_all_weights(model):
    """Get all weight names and shapes from HuggingFace model"""
    weights_info = {}
    for name, param in model.named_parameters():
        weights_info[name] = param.shape
    return weights_info


def get_nested_attr(obj, attr_path):
    """Get nested attribute using dot notation path"""
    attrs = attr_path.split(".")
    for attr in attrs:
        if attr.isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    return obj


def compare_weight(hf_weight, jax_weight, name, tolerance=0.01):
    """Compare a single weight tensor"""
    print(f"\n{'='*80}")
    print(f"Comparing: {name}")
    print(f"{'='*80}")

    hf_np = (
        hf_weight.float().numpy() if isinstance(hf_weight, torch.Tensor) else hf_weight
    )
    jax_np = np.array(jax_weight, dtype=np.float32)

    print(f"HF  shape: {hf_np.shape}")
    print(f"JAX shape: {jax_np.shape}")

    # Check shape match
    if hf_np.shape != jax_np.shape:
        print(f"❌ Shape mismatch!")
        return False

    # Show sample values
    if len(hf_np.shape) == 1:
        print(f"HF  values[:10]: {hf_np[:10]}")
        print(f"JAX values[:10]: {jax_np[:10]}")
    else:
        print(f"HF  values[0,:10]: {hf_np.flatten()[:10]}")
        print(f"JAX values[0,:10]: {jax_np.flatten()[:10]}")

    # Calculate difference
    diff = np.abs(hf_np - jax_np)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"\nDifference - max: {max_diff:.6f}, mean: {mean_diff:.6f}")

    if max_diff < tolerance:
        print(f"✅ Weight matches (max diff < {tolerance})")
        return True
    else:
        print(f"❌ Weight mismatch (max diff >= {tolerance})")
        return False


def main():
    print("=" * 80)
    print("Loading HuggingFace model...")
    print("=" * 80)
    hf_model = (
        AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
        .to("cpu")
        .eval()
    )
    hf_config = AutoConfig.from_pretrained(MODEL_PATH)

    print("\n" + "=" * 80)
    print("Getting all HuggingFace weights...")
    print("=" * 80)
    hf_weights_info = get_hf_all_weights(hf_model)

    print(f"\nTotal {len(hf_weights_info)} weights found:")
    for name, shape in sorted(hf_weights_info.items()):
        print(f"{name}: shape = {shape}")

    print("\n" + "=" * 80)
    print("Creating JAX model...")
    print("=" * 80)
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices[: min(4, len(devices))], ("tensor",))
    jax_model = Qwen3ForCausalLM(
        config=hf_config, dtype=jnp.bfloat16, rngs=None, mesh=mesh
    )

    print("\n" + "=" * 80)
    print("Loading weights into JAX model...")
    print("=" * 80)
    model_config = ModelConfig(model_path=MODEL_PATH, trust_remote_code=True)
    helper = TestQwen3NNXForward()
    helper.hf_config = hf_config
    weight_mappings = helper._build_weight_mappings()
    loader = WeightLoader(
        model=jax_model, model_config=model_config, mesh=mesh, dtype=jnp.bfloat16
    )
    loader.load_weights_from_safetensors(weight_mappings)

    print("\n" + "=" * 80)
    print("Starting weight comparison...")
    print("=" * 80)

    # Mapping from HF names to JAX model paths
    weight_mappings_check = {
        # Embedding and LM head
        "model.embed_tokens.weight": "transformers.embed_tokens.embedding.value",
        "lm_head.weight": "lm_head.weight.value",
    }

    # Add layer-specific mappings
    num_layers = hf_config.num_hidden_layers
    for i in range(num_layers):

        layer_mappings = {
            f"model.layers.{i}.input_layernorm.weight": f"transformers.layers.{i}.input_layernorm.weight.value",
            f"model.layers.{i}.post_attention_layernorm.weight": f"transformers.layers.{i}.post_attention_layernorm.weight.value",
            f"model.layers.{i}.self_attn.q_proj.weight": f"transformers.layers.{i}.self_attn.q_proj.weight.value",
            f"model.layers.{i}.self_attn.k_proj.weight": f"transformers.layers.{i}.self_attn.k_proj.weight.value",
            f"model.layers.{i}.self_attn.v_proj.weight": f"transformers.layers.{i}.self_attn.v_proj.weight.value",
            f"model.layers.{i}.self_attn.o_proj.weight": f"transformers.layers.{i}.self_attn.o_proj.weight.value",
            f"model.layers.{i}.self_attn.q_norm.weight": f"transformers.layers.{i}.self_attn.q_norm.weight.value",
            f"model.layers.{i}.self_attn.k_norm.weight": f"transformers.layers.{i}.self_attn.k_norm.weight.value",
            f"model.layers.{i}.mlp.gate_proj.weight": f"transformers.layers.{i}.mlp.gate_proj.weight.value",
            f"model.layers.{i}.mlp.up_proj.weight": f"transformers.layers.{i}.mlp.up_proj.weight.value",
            f"model.layers.{i}.mlp.down_proj.weight": f"transformers.layers.{i}.mlp.down_proj.weight.value",
        }
        weight_mappings_check.update(layer_mappings)

    # Compare all weights
    total_weights = len(weight_mappings_check)
    matched_weights = 0
    failed_weights = []

    with torch.no_grad():
        for hf_name, jax_path in sorted(weight_mappings_check.items()):
            try:
                # Get HF weight
                hf_weight = get_nested_attr(hf_model, hf_name)

                # Get JAX weight
                jax_weight = get_nested_attr(jax_model, jax_path)

                # Compare
                if compare_weight(hf_weight, jax_weight, hf_name):
                    matched_weights += 1
                else:
                    failed_weights.append(hf_name)

            except Exception as e:
                print(f"\n❌ Error comparing {hf_name}: {str(e)}")
                failed_weights.append(hf_name)

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Total weights compared: {total_weights}")
    print(f"Matched weights: {matched_weights}")
    print(f"Failed weights: {len(failed_weights)}")
    print(f"Success rate: {matched_weights/total_weights*100:.2f}%")

    if failed_weights:
        print(f"\nFailed weights list:")
        for name in failed_weights:
            print(f"  - {name}")
    else:
        print("\n🎉 All weights match successfully!")


if __name__ == "__main__":
    main()
