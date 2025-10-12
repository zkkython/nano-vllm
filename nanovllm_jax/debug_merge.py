#!/usr/bin/env python3
"""
Debug weight merging specifically.
"""

import jax
import jax.numpy as jnp
from nanovllm_jax.utils.loader import load_model_weights
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.config import Config

def debug_merge():
    """Debug weight merging specifically."""
    print("=== Weight Merging Debug ===")
    
    # Load weights
    model_path = "/home/kason/models/qwen06b"
    weights = load_model_weights(model_path)
    
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
    
    print("Loaded weights structure:")
    print(f"  model.embed_tokens keys: {list(weights['model']['embed_tokens'].keys())}")
    print(f"  model.embed_tokens.weight shape: {weights['model']['embed_tokens']['weight'].shape}")
    print(f"  model.embed_tokens.weight sample: {weights['model']['embed_tokens']['weight'][0, :5]}")
    
    print("\nModel params structure:")
    print(f"  params.model.embed_tokens keys: {list(params['params']['model']['embed_tokens'].keys())}")
    print(f"  params.model.embed_tokens.embedding shape: {params['params']['model']['embed_tokens']['embedding'].shape}")
    print(f"  params.model.embed_tokens.embedding sample: {params['params']['model']['embed_tokens']['embedding'][0, :5]}")
    
    # Try manual merge
    print("\n=== Manual Merge Test ===")
    
    # Create a copy
    merged_params = jax.tree.map(lambda x: x, params)
    
    # Manually set the embedding weights
    print("Before manual merge:")
    print(f"  embedding sample: {merged_params['params']['model']['embed_tokens']['embedding'][0, :5]}")
    
    # Direct assignment
    merged_params['params']['model']['embed_tokens']['embedding'] = weights['model']['embed_tokens']['weight']
    
    print("After manual merge:")
    print(f"  embedding sample: {merged_params['params']['model']['embed_tokens']['embedding'][0, :5]}")
    print(f"  matches loaded: {jnp.allclose(merged_params['params']['model']['embed_tokens']['embedding'], weights['model']['embed_tokens']['weight'])}")
    
    # Test forward pass
    print("\n=== Forward Pass Test ===")
    test_input = jnp.array([1, 2, 3])
    test_positions = jnp.array([0, 1, 2])
    
    # Original params
    output_original = model.apply(params, test_input, test_positions)
    print(f"Original output sample: {output_original[0, :5]}")
    
    # Merged params
    output_merged = model.apply(merged_params, test_input, test_positions)
    print(f"Merged output sample: {output_merged[0, :5]}")
    print(f"Outputs different: {not jnp.allclose(output_original, output_merged)}")

if __name__ == "__main__":
    debug_merge()
