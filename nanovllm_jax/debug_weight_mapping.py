#!/usr/bin/env python3
"""
Debug script to check weight mapping in detail.
"""

import jax
import jax.numpy as jnp
from nanovllm_jax.utils.loader import load_model_weights
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.config import Config

def debug_weight_mapping():
    """Debug weight mapping in detail."""
    print("=== Detailed Weight Mapping Debug ===")
    
    # Load weights
    model_path = "/home/kason/models/qwen06b"
    weights = load_model_weights(model_path)
    
    print(f"Loaded weights structure:")
    print(f"  Top level: {list(weights.keys())}")
    print(f"  Model keys: {list(weights['model'].keys())}")
    print(f"  LM head keys: {list(weights['lm_head'].keys())}")
    
    # Create model
    config = Config(model=model_path, max_model_len=32)
    model = Qwen3ForCausalLM(
        config=config.hf_config,
        tp_size=1,
        tp_rank=0
    )
    
    # Initialize parameters
    input_ids = jnp.zeros((1,), dtype=jnp.int32)
    positions = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), input_ids, positions)
    
    print(f"\nModel params structure:")
    print(f"  Top level: {list(params.keys())}")
    print(f"  Params keys: {list(params['params'].keys())}")
    print(f"  Model params keys: {list(params['params']['model'].keys())}")
    print(f"  LM head params keys: {list(params['params']['lm_head'].keys())}")
    
    # Check embedding weights before merging
    print(f"\nBefore merging:")
    if 'embed_tokens' in params['params']['model']:
        embed_tokens = params['params']['model']['embed_tokens']
        print(f"  Embed tokens keys: {list(embed_tokens.keys())}")
        if 'embedding' in embed_tokens:
            embed_before = embed_tokens['embedding']
            print(f"  Embedding shape: {embed_before.shape}")
            print(f"  Embedding sample: {embed_before[0, :5]}")
        elif 'weight' in embed_tokens:
            embed_before = embed_tokens['weight']
            print(f"  Weight shape: {embed_before.shape}")
            print(f"  Weight sample: {embed_before[0, :5]}")
    
    # Try to merge weights
    def merge_dict(d1, d2):
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                merge_dict(d1[key], value)
            else:
                d1[key] = value
        return d1
    
    merged_params = jax.tree.map(lambda x: x, params)
    merge_dict(merged_params, weights)
    
    # Check embedding weights after merging
    print(f"\nAfter merging:")
    if 'params' in merged_params and 'model' in merged_params['params'] and 'embed_tokens' in merged_params['params']['model']:
        embed_tokens_after = merged_params['params']['model']['embed_tokens']
        print(f"  Embed tokens keys after: {list(embed_tokens_after.keys())}")
        
        if 'embedding' in embed_tokens_after:
            embed_after = embed_tokens_after['embedding']
            print(f"  Embedding shape: {embed_after.shape}")
            print(f"  Embedding sample: {embed_after[0, :5]}")
        elif 'weight' in embed_tokens_after:
            embed_after = embed_tokens_after['weight']
            print(f"  Weight shape: {embed_after.shape}")
            print(f"  Weight sample: {embed_after[0, :5]}")
        
        # Compare with original loaded weights
        original_embed = weights['model']['embed_tokens']['weight']
        print(f"  Original embed sample: {original_embed[0, :5]}")
        if 'embedding' in embed_tokens_after:
            print(f"  Values match: {jnp.allclose(embed_after, original_embed)}")
        elif 'weight' in embed_tokens_after:
            print(f"  Values match: {jnp.allclose(embed_after, original_embed)}")
    
    # Test model forward pass
    print(f"\n=== Testing Model Forward Pass ===")
    
    # Test with simple input
    test_input = jnp.array([1, 2, 3])  # Simple token IDs
    test_positions = jnp.array([0, 1, 2])
    
    try:
        # Test with original params (random)
        output_random = model.apply(params, test_input, test_positions)
        print(f"Random params output shape: {output_random.shape}")
        print(f"Random params sample: {output_random[0, :5]}")
        
        # Test with merged params (loaded weights)
        output_loaded = model.apply(merged_params, test_input, test_positions)
        print(f"Loaded params output shape: {output_loaded.shape}")
        print(f"Loaded params sample: {output_loaded[0, :5]}")
        
        # Check if outputs are different
        print(f"Outputs are different: {not jnp.allclose(output_random, output_loaded)}")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_weight_mapping()
