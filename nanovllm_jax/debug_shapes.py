#!/usr/bin/env python3
"""
Debug script to understand the actual shapes in the model.
"""

import jax
import jax.numpy as jnp
from nanovllm_jax import LLM, SamplingParams

def debug_shapes():
    print("=== Shape Debugging ===")
    
    # Load model
    print("Loading model...")
    llm = LLM('/home/kason/models/qwen06b', max_model_len=128, gpu_memory_utilization=0.2)
    
    # Get model parameters
    params = llm.model_runner.params
    
    # Test with simple input
    print("\n=== Testing with simple input ===")
    input_ids = jnp.array([[1, 2, 3]])  # (1, 3)
    positions = jnp.array([0, 1, 2])
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Positions shape: {positions.shape}")
    
    # Test embedding
    print("\n=== Testing embedding ===")
    try:
        embed_output = llm.model_runner.model.model.embed_tokens.apply(
            params['params']['model']['embed_tokens'],
            input_ids
        )
        print(f"Embedding output shape: {embed_output.shape}")
    except Exception as e:
        print(f"Embedding failed: {e}")
    
    # Test first layer
    print("\n=== Testing first layer ===")
    try:
        # Get first layer
        layer_params = params['params']['model']['Qwen3DecoderLayer_0']
        
        # Test input layernorm
        input_layernorm_output = llm.model_runner.model.model.Qwen3DecoderLayer_0.input_layernorm.apply(
            layer_params['input_layernorm'],
            embed_output
        )
        print(f"Input layernorm output shape: {input_layernorm_output.shape}")
        
        # Test QKV projection
        qkv_output = llm.model_runner.model.model.Qwen3DecoderLayer_0.self_attn.qkv_proj.apply(
            layer_params['self_attn']['qkv_proj'],
            input_layernorm_output
        )
        print(f"QKV projection output shape: {qkv_output.shape}")
        
        # Test QKV split
        q_size = 2048  # This should be calculated from config
        kv_size = 1024
        q, k, v = jnp.split(qkv_output, [q_size, q_size + kv_size], axis=-1)
        print(f"Q shape: {q.shape}")
        print(f"K shape: {k.shape}")
        print(f"V shape: {v.shape}")
        
    except Exception as e:
        print(f"Layer test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_shapes()
