#!/usr/bin/env python3
"""
Detailed weight debugging script to check if weights are correctly loaded and used.
"""

import jax
import jax.numpy as jnp
from nanovllm_jax import LLM, SamplingParams
from nanovllm_jax.utils.loader import load_model_weights_from_safetensors
import numpy as np

def debug_weights():
    print("=== Detailed Weight Debugging ===")
    
    # Load model
    print("Loading model...")
    llm = LLM('/home/kason/models/qwen06b', max_model_len=128, gpu_memory_utilization=0.2)
    
    # Get model parameters
    params = llm.model_runner.params
    print(f"Model parameters loaded: {type(params)}")
    
    # Check embedding weights
    if 'params' in params and 'model' in params['params']:
        model_params = params['params']['model']
        
        # Check embed_tokens
        if 'embed_tokens' in model_params:
            embed_weights = model_params['embed_tokens']['embedding']
            print(f"Embed tokens shape: {embed_weights.shape}")
            print(f"Embed tokens stats: mean={float(jnp.mean(embed_weights)):.6f}, std={float(jnp.std(embed_weights)):.6f}")
            print(f"Embed tokens range: [{float(jnp.min(embed_weights)):.6f}, {float(jnp.max(embed_weights)):.6f}]")
        
        # Check first layer weights
        if 'Qwen3DecoderLayer_0' in model_params:
            layer0 = model_params['Qwen3DecoderLayer_0']
            print(f"\nFirst layer components: {list(layer0.keys())}")
            
            # Check self_attn weights
            if 'self_attn' in layer0:
                attn = layer0['self_attn']
                print(f"Self attention components: {list(attn.keys())}")
                
                if 'qkv_proj' in attn and 'weight' in attn['qkv_proj']:
                    qkv_weight = attn['qkv_proj']['weight']
                    print(f"QKV projection shape: {qkv_weight.shape}")
                    print(f"QKV projection stats: mean={float(jnp.mean(qkv_weight)):.6f}, std={float(jnp.std(qkv_weight)):.6f}")
                
                if 'o_proj' in attn and 'weight' in attn['o_proj']:
                    o_weight = attn['o_proj']['weight']
                    print(f"Output projection shape: {o_weight.shape}")
                    print(f"Output projection stats: mean={float(jnp.mean(o_weight)):.6f}, std={float(jnp.std(o_weight)):.6f}")
            
            # Check MLP weights
            if 'mlp' in layer0:
                mlp = layer0['mlp']
                print(f"MLP components: {list(mlp.keys())}")
                
                if 'gate_up_proj' in mlp and 'weight' in mlp['gate_up_proj']:
                    gate_up_weight = mlp['gate_up_proj']['weight']
                    print(f"Gate-up projection shape: {gate_up_weight.shape}")
                    print(f"Gate-up projection stats: mean={float(jnp.mean(gate_up_weight)):.6f}, std={float(jnp.std(gate_up_weight)):.6f}")
                
                if 'down_proj' in mlp and 'weight' in mlp['down_proj']:
                    down_weight = mlp['down_proj']['weight']
                    print(f"Down projection shape: {down_weight.shape}")
                    print(f"Down projection stats: mean={float(jnp.mean(down_weight)):.6f}, std={float(jnp.std(down_weight)):.6f}")
        
        # Check LM head weights
        if 'lm_head' in params['params']:
            lm_head_weight = params['params']['lm_head']['weight']
            print(f"\nLM head shape: {lm_head_weight.shape}")
            print(f"LM head stats: mean={float(jnp.mean(lm_head_weight)):.6f}, std={float(jnp.std(lm_head_weight)):.6f}")
    
    # Test a simple forward pass
    print("\n=== Testing Forward Pass ===")
    try:
        # Create simple input
        input_ids = jnp.array([[1, 2, 3]])  # Simple token sequence
        positions = jnp.array([0, 1, 2])
        
        # Run model
        logits = llm.model_runner.model.apply(
            llm.model_runner.params,
            input_ids,
            positions,
            compute_logits=True
        )
        
        print(f"Forward pass successful!")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits stats: mean={float(jnp.mean(logits)):.6f}, std={float(jnp.std(logits)):.6f}")
        print(f"Logits range: [{float(jnp.min(logits)):.6f}, {float(jnp.max(logits)):.6f}]")
        
        # Check if logits are reasonable
        if jnp.any(jnp.isnan(logits)):
            print("❌ WARNING: Logits contain NaN values!")
        if jnp.any(jnp.isinf(logits)):
            print("❌ WARNING: Logits contain Inf values!")
        
        # Check logits distribution
        logits_flat = logits.reshape(-1)
        print(f"Logits distribution:")
        print(f"  - Mean: {float(jnp.mean(logits_flat)):.6f}")
        print(f"  - Std: {float(jnp.std(logits_flat)):.6f}")
        print(f"  - Min: {float(jnp.min(logits_flat)):.6f}")
        print(f"  - Max: {float(jnp.max(logits_flat)):.6f}")
        
        # Check if logits are too extreme
        if float(jnp.std(logits_flat)) > 10:
            print("❌ WARNING: Logits have very high variance!")
        if float(jnp.max(jnp.abs(logits_flat))) > 50:
            print("❌ WARNING: Logits have very extreme values!")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_weights()
