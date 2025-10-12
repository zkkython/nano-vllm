#!/usr/bin/env python3
"""
Debug complete weight loading to check if all layers are loaded.
"""

import jax
import jax.numpy as jnp
from nanovllm_jax.utils.loader import load_model_weights
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.config import Config

def debug_complete_weights():
    """Debug complete weight loading."""
    print("=== Complete Weight Loading Debug ===")
    
    # Load weights
    model_path = "/home/kason/models/qwen06b"
    weights = load_model_weights(model_path)
    
    print("Loaded weights structure:")
    print(f"  Top level: {list(weights.keys())}")
    print(f"  Model keys: {list(weights['model'].keys())}")
    
    if 'layers' in weights['model']:
        print(f"  Layers keys: {list(weights['model']['layers'].keys())[:5]}...")  # Show first 5
        if '0' in weights['model']['layers']:
            layer_0 = weights['model']['layers']['0']
            print(f"  Layer 0 keys: {list(layer_0.keys())}")
            if 'self_attn' in layer_0:
                print(f"  Self attn keys: {list(layer_0['self_attn'].keys())}")
    
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
    print(f"  Params keys: {list(params['params'].keys())}")
    print(f"  Model params keys: {list(params['params']['model'].keys())}")
    
    # Check if we have decoder layers
    decoder_layers = [k for k in params['params']['model'].keys() if k.startswith('Qwen3DecoderLayer_')]
    print(f"  Decoder layers: {len(decoder_layers)}")
    
    if decoder_layers:
        layer_0_key = decoder_layers[0]
        layer_0_params = params['params']['model'][layer_0_key]
        print(f"  {layer_0_key} keys: {list(layer_0_params.keys())}")
        
        if 'self_attn' in layer_0_params:
            self_attn_params = layer_0_params['self_attn']
            print(f"  Self attn params keys: {list(self_attn_params.keys())}")
    
    # Test weight merging
    print(f"\n=== Testing Weight Merging ===")
    
    def merge_dict(d1, d2):
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                merge_dict(d1[key], value)
            else:
                d1[key] = value
        return d1
    
    merged_params = jax.tree.map(lambda x: x, params)
    
    # Try to merge all weights
    if 'model' in weights and 'params' in merged_params and 'model' in merged_params['params']:
        # Handle embedding weights
        if 'embed_tokens' in weights['model'] and 'embed_tokens' in merged_params['params']['model']:
            loaded_embed = weights['model']['embed_tokens']
            merged_embed = merged_params['params']['model']['embed_tokens']
            
            if 'weight' in loaded_embed and 'embedding' in merged_embed:
                merged_embed['embedding'] = loaded_embed['weight']
                print("✓ Embedding weights merged")
        
        # Handle layer weights
        if 'layers' in weights['model']:
            print("Attempting to merge layer weights...")
            for layer_idx in range(config.hf_config.num_hidden_layers):
                layer_key = f"Qwen3DecoderLayer_{layer_idx}"
                if layer_key in merged_params['params']['model']:
                    # Try to merge this layer's weights
                    if str(layer_idx) in weights['model']['layers']:
                        layer_weights = weights['model']['layers'][str(layer_idx)]
                        layer_params = merged_params['params']['model'][layer_key]
                        
                        # Merge layer weights
                        merge_dict(layer_params, layer_weights)
                        print(f"✓ Layer {layer_idx} weights merged")
                    else:
                        print(f"✗ Layer {layer_idx} weights not found in loaded weights")
        
        # Handle norm weights
        if 'norm' in weights['model'] and 'norm' in merged_params['params']['model']:
            merge_dict(merged_params['params']['model']['norm'], weights['model']['norm'])
            print("✓ Norm weights merged")
    
    # Test forward pass
    print(f"\n=== Testing Forward Pass ===")
    test_input = jnp.array([1, 2, 3])
    test_positions = jnp.array([0, 1, 2])
    
    try:
        # Original params
        output_original = model.apply(params, test_input, test_positions)
        print(f"Original output shape: {output_original.shape}")
        print(f"Original output sample: {output_original[0, :5]}")
        
        # Merged params
        output_merged = model.apply(merged_params, test_input, test_positions)
        print(f"Merged output shape: {output_merged.shape}")
        print(f"Merged output sample: {output_merged[0, :5]}")
        print(f"Outputs different: {not jnp.allclose(output_original, output_merged)}")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_complete_weights()
