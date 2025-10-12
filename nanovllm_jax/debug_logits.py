#!/usr/bin/env python3
"""
Debug script to check logits quality and model behavior.
"""

import jax
import jax.numpy as jnp
from nanovllm_jax import LLM, SamplingParams
import numpy as np

def debug_logits():
    print("=== Logits Debugging ===")
    
    # Load model
    print("Loading model...")
    llm = LLM('/home/kason/models/qwen06b', max_model_len=64, gpu_memory_utilization=0.2)
    
    # Test with simple input
    print("\n=== Testing with simple input ===")
    input_ids = jnp.array([[16, 10, 16]])  # "1+1" tokens
    positions = jnp.array([0, 1, 2])
    
    print(f"Input IDs: {input_ids}")
    print(f"Positions: {positions}")
    
    # Get model parameters
    params = llm.model_runner.params
    
    # Run model forward pass
    print("\n=== Running model forward pass ===")
    try:
        logits = llm.model_runner.model.apply(
            params,
            input_ids,
            positions,
            compute_logits=True
        )
        
        print(f"Logits shape: {logits.shape}")
        print(f"Logits dtype: {logits.dtype}")
        
        # Check logits statistics
        logits_flat = logits.reshape(-1)
        print(f"Logits statistics:")
        print(f"  Mean: {float(jnp.mean(logits_flat)):.6f}")
        print(f"  Std: {float(jnp.std(logits_flat)):.6f}")
        print(f"  Min: {float(jnp.min(logits_flat)):.6f}")
        print(f"  Max: {float(jnp.max(logits_flat)):.6f}")
        
        # Check for NaN or Inf
        if jnp.any(jnp.isnan(logits)):
            print("❌ WARNING: Logits contain NaN values!")
        if jnp.any(jnp.isinf(logits)):
            print("❌ WARNING: Logits contain Inf values!")
        
        # Check logits for specific tokens
        print(f"\n=== Checking logits for specific tokens ===")
        tokenizer = llm.tokenizer
        
        # Check logits for "2" token
        token_2_id = tokenizer.encode("2")[0]
        print(f"Token '2' ID: {token_2_id}")
        print(f"Logits for token '2': {float(logits[0, -1, token_2_id]):.6f}")
        
        # Check top 10 logits
        last_token_logits = logits[0, -1, :]  # Last token logits
        top_10_indices = jnp.argsort(last_token_logits)[-10:]
        top_10_values = last_token_logits[top_10_indices]
        
        print(f"\nTop 10 logits:")
        for i, (idx, val) in enumerate(zip(top_10_indices, top_10_values)):
            token = tokenizer.decode([int(idx)])
            print(f"  {i+1}. Token {int(idx)} ('{token}'): {float(val):.6f}")
        
        # Check if "2" is in top predictions
        if token_2_id in top_10_indices:
            rank = jnp.where(top_10_indices == token_2_id)[0][0]
            print(f"✅ Token '2' is ranked #{int(rank)+1} in top predictions")
        else:
            print(f"❌ Token '2' is NOT in top 10 predictions")
            
        # Test sampling
        print(f"\n=== Testing sampling ===")
        temperatures = jnp.array([0.0])  # Deterministic sampling
        rng = jax.random.PRNGKey(42)
        
        # Apply temperature
        scaled_logits = logits / temperatures[0]
        
        # Sample
        sampled_tokens = jax.random.categorical(rng, scaled_logits, axis=-1)
        print(f"Sampled tokens: {sampled_tokens}")
        
        # Decode sampled tokens
        for i, token_id in enumerate(sampled_tokens[0]):
            token = tokenizer.decode([int(token_id)])
            print(f"  Token {i}: {int(token_id)} -> '{token}'")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_logits()
