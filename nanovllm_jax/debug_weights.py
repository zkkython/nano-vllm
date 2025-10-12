#!/usr/bin/env python3
"""
Debug script to check weight loading and mapping.
"""

import jax.numpy as jnp
from nanovllm_jax.utils.loader import load_model_weights
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.config import Config
from transformers import AutoConfig

def debug_weights():
    """Debug weight loading and mapping."""
    print("=== Weight Loading Debug ===")
    
    # Load weights
    model_path = "/home/kason/models/qwen06b"
    weights = load_model_weights(model_path)
    
    print(f"Loaded weights keys: {list(weights.keys())}")
    
    if 'model' in weights:
        print(f"Model keys: {list(weights['model'].keys())}")
        
        if 'embed_tokens' in weights['model']:
            embed_weights = weights['model']['embed_tokens']['weight']
            print(f"Embedding shape: {embed_weights.shape}")
            print(f"Embedding sample values: {embed_weights[0, :5]}")
    
    if 'lm_head' in weights:
        lm_head_weights = weights['lm_head']['weight']
        print(f"LM head shape: {lm_head_weights.shape}")
        print(f"LM head sample values: {lm_head_weights[0, :5]}")
    
    print("\n=== Model Structure Debug ===")
    
    # Create model and check structure
    config = Config(model=model_path, max_model_len=32)
    model = Qwen3ForCausalLM(
        config=config.hf_config,
        tp_size=1,
        tp_rank=0
    )
    
    # Initialize with dummy inputs
    input_ids = jnp.zeros((1,), dtype=jnp.int32)
    positions = jnp.zeros((1,), dtype=jnp.int32)
    
    params = model.init(jax.random.PRNGKey(0), input_ids, positions)
    
    print(f"Initialized params keys: {list(params.keys())}")
    
    if 'model' in params:
        print(f"Model params keys: {list(params['model'].keys())}")
        
        if 'embed_tokens' in params['model']:
            embed_params = params['model']['embed_tokens']
            print(f"Embed tokens params keys: {list(embed_params.keys())}")
            
            if 'embedding' in embed_params:
                print(f"Embedding param shape: {embed_params['embedding'].shape}")
            elif 'weight' in embed_params:
                print(f"Weight param shape: {embed_params['weight'].shape}")
    
    if 'lm_head' in params:
        lm_head_params = params['lm_head']
        print(f"LM head params keys: {list(lm_head_params.keys())}")
        
        if 'weight' in lm_head_params:
            print(f"LM head weight shape: {lm_head_params['weight'].shape}")
    
    print("\n=== Weight Mapping Check ===")
    
    # Check if we can map the weights
    try:
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
        
        print("✓ Weight merging successful")
        
        # Check if weights were actually applied
        if 'model' in merged_params and 'embed_tokens' in merged_params['model']:
            if 'weight' in merged_params['model']['embed_tokens']:
                merged_embed = merged_params['model']['embed_tokens']['weight']
                original_embed = weights['model']['embed_tokens']['weight']
                print(f"Original embed shape: {original_embed.shape}")
                print(f"Merged embed shape: {merged_embed.shape}")
                print(f"Shapes match: {original_embed.shape == merged_embed.shape}")
                print(f"Values match: {jnp.allclose(original_embed, merged_embed)}")
        
    except Exception as e:
        print(f"✗ Weight merging failed: {e}")

if __name__ == "__main__":
    import jax
    debug_weights()
