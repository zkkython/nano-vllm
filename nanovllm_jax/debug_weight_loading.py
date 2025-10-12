#!/usr/bin/env python3
"""
Debug script to check if weights are actually being used.
"""

import jax
import jax.numpy as jnp
from nanovllm_jax import LLM, SamplingParams
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM, Qwen3Config
from nanovllm_jax.utils.loader import load_model_weights_from_safetensors
import numpy as np

def debug_weight_loading():
    print("=== Weight Loading Debug ===")
    
    # Create model config
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        num_hidden_layers=2,  # Use fewer layers for testing
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=3072,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True
    )
    
    # Create model
    model = Qwen3ForCausalLM(config=config)
    
    # Test input
    input_ids = jnp.array([[16, 10, 16]])  # "1+1"
    positions = jnp.array([0, 1, 2])
    
    print("=== Testing with random initialization ===")
    # Initialize with random weights
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, input_ids, positions, compute_logits=True)
    
    # Run forward pass
    logits_random = model.apply(params, input_ids, positions, compute_logits=True)
    print(f"Random logits shape: {logits_random.shape}")
    print(f"Random logits mean: {float(jnp.mean(logits_random)):.6f}")
    print(f"Random logits std: {float(jnp.std(logits_random)):.6f}")
    
    # Check top predictions
    last_token_logits = logits_random[0, -1, :]
    top_5_indices = jnp.argsort(last_token_logits)[-5:]
    print(f"Top 5 random predictions: {top_5_indices}")
    
    print("\n=== Testing with loaded weights ===")
    # Load actual model
    llm = LLM('/home/kason/models/qwen06b', max_model_len=64, gpu_memory_utilization=0.2)
    loaded_params = llm.model_runner.params
    
    # Run forward pass with loaded weights
    logits_loaded = llm.model_runner.model.apply(
        loaded_params, input_ids, positions, compute_logits=True
    )
    print(f"Loaded logits shape: {logits_loaded.shape}")
    print(f"Loaded logits mean: {float(jnp.mean(logits_loaded)):.6f}")
    print(f"Loaded logits std: {float(jnp.std(logits_loaded)):.6f}")
    
    # Check top predictions
    last_token_logits_loaded = logits_loaded[0, -1, :]
    top_5_indices_loaded = jnp.argsort(last_token_logits_loaded)[-5:]
    print(f"Top 5 loaded predictions: {top_5_indices_loaded}")
    
    # Compare logits
    logits_diff = jnp.abs(logits_random - logits_loaded)
    print(f"\nLogits difference:")
    print(f"  Mean difference: {float(jnp.mean(logits_diff)):.6f}")
    print(f"  Max difference: {float(jnp.max(logits_diff)):.6f}")
    
    if jnp.allclose(logits_random, logits_loaded, atol=1e-6):
        print("❌ WARNING: Loaded weights are identical to random weights!")
        print("This suggests weights are not being loaded correctly.")
    else:
        print("✅ Loaded weights are different from random weights.")
    
    # Check specific weight values
    print(f"\n=== Checking specific weights ===")
    
    # Check embedding weights
    if 'params' in loaded_params and 'model' in loaded_params['params']:
        embed_weights = loaded_params['params']['model']['embed_tokens']['embedding']
        print(f"Embedding weights shape: {embed_weights.shape}")
        print(f"Embedding weights mean: {float(jnp.mean(embed_weights)):.6f}")
        print(f"Embedding weights std: {float(jnp.std(embed_weights)):.6f}")
        
        # Check if weights are all zeros or ones (indicating loading failure)
        if jnp.allclose(embed_weights, 0.0):
            print("❌ WARNING: Embedding weights are all zeros!")
        elif jnp.allclose(embed_weights, 1.0):
            print("❌ WARNING: Embedding weights are all ones!")
        else:
            print("✅ Embedding weights look reasonable.")
    
    # Check LM head weights
    if 'lm_head' in loaded_params['params']:
        lm_head_weights = loaded_params['params']['lm_head']['weight']
        print(f"LM head weights shape: {lm_head_weights.shape}")
        print(f"LM head weights mean: {float(jnp.mean(lm_head_weights)):.6f}")
        print(f"LM head weights std: {float(jnp.std(lm_head_weights)):.6f}")

if __name__ == "__main__":
    debug_weight_loading()
