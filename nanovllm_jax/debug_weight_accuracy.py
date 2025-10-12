#!/usr/bin/env python3
"""
Debug script to verify weight loading accuracy by comparing with original model.
"""

import jax
import jax.numpy as jnp
import numpy as np
from nanovllm_jax import LLM, SamplingParams
from nanovllm_jax.utils.loader import load_model_weights_from_safetensors
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def debug_weight_accuracy():
    print("=== Weight Loading Accuracy Debug ===")
    
    # Load JAX model
    print("Loading JAX model...")
    llm_jax = LLM('/home/kason/models/qwen06b', max_model_len=32, gpu_memory_utilization=0.1)
    
    # Load PyTorch model for comparison
    print("Loading PyTorch model...")
    model_pytorch = AutoModelForCausalLM.from_pretrained(
        '/home/kason/models/qwen06b',
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained('/home/kason/models/qwen06b')
    
    # Test input
    input_text = "1+1="
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    print(f"Input text: {input_text}")
    print(f"Input IDs: {input_ids}")
    
    # Get PyTorch model output
    print("\n=== PyTorch Model Output ===")
    with torch.no_grad():
        outputs_pytorch = model_pytorch(input_ids)
        logits_pytorch = outputs_pytorch.logits
        print(f"PyTorch logits shape: {logits_pytorch.shape}")
        print(f"PyTorch logits mean: {logits_pytorch.mean().item():.6f}")
        print(f"PyTorch logits std: {logits_pytorch.std().item():.6f}")
        
        # Get top predictions
        last_token_logits = logits_pytorch[0, -1, :]
        top_5_indices = torch.topk(last_token_logits, 5).indices
        top_5_values = last_token_logits[top_5_indices]
        
        print(f"PyTorch top 5 predictions:")
        for i, (idx, val) in enumerate(zip(top_5_indices, top_5_values)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. Token {idx.item()} ('{token}'): {val.item():.6f}")
        
        # Check token '2'
        token_2_id = tokenizer.encode("2")[0]
        print(f"Token '2' (ID {token_2_id}) logit: {last_token_logits[token_2_id].item():.6f}")
    
    # Get JAX model output
    print("\n=== JAX Model Output ===")
    input_ids_jax = jnp.array(input_ids.numpy())
    positions = jnp.array([0, 1, 2, 3])
    
    logits_jax = llm_jax.model_runner.model.apply(
        llm_jax.model_runner.params,
        input_ids_jax,
        positions,
        compute_logits=True
    )
    
    print(f"JAX logits shape: {logits_jax.shape}")
    print(f"JAX logits mean: {float(jnp.mean(logits_jax)):.6f}")
    print(f"JAX logits std: {float(jnp.std(logits_jax)):.6f}")
    
    # Get top predictions
    last_token_logits_jax = logits_jax[0, -1, :]
    top_5_indices_jax = jnp.argsort(last_token_logits_jax)[-5:]
    top_5_values_jax = last_token_logits_jax[top_5_indices_jax]
    
    print(f"JAX top 5 predictions:")
    for i, (idx, val) in enumerate(zip(top_5_indices_jax, top_5_values_jax)):
        token = tokenizer.decode([int(idx)])
        print(f"  {i+1}. Token {int(idx)} ('{token}'): {float(val):.6f}")
    
    # Check token '2'
    token_2_id = tokenizer.encode("2")[0]
    print(f"Token '2' (ID {token_2_id}) logit: {float(last_token_logits_jax[token_2_id]):.6f}")
    
    # Compare outputs
    print(f"\n=== Comparison ===")
    logits_diff = jnp.abs(logits_jax - jnp.array(logits_pytorch.numpy()))
    print(f"Logits difference:")
    print(f"  Mean difference: {float(jnp.mean(logits_diff)):.6f}")
    print(f"  Max difference: {float(jnp.max(logits_diff)):.6f}")
    
    if jnp.allclose(logits_jax, jnp.array(logits_pytorch.numpy()), atol=1e-3):
        print("✅ Logits are very close (within 1e-3)")
    elif jnp.allclose(logits_jax, jnp.array(logits_pytorch.numpy()), atol=1e-1):
        print("⚠️ Logits are somewhat close (within 1e-1)")
    else:
        print("❌ Logits are significantly different")
    
    # Check specific weight values
    print(f"\n=== Weight Comparison ===")
    
    # Get embedding weights
    jax_embed_weights = llm_jax.model_runner.params['params']['model']['embed_tokens']['embedding']
    pytorch_embed_weights = model_pytorch.model.embed_tokens.weight.data.numpy()
    
    print(f"Embedding weights comparison:")
    print(f"  JAX shape: {jax_embed_weights.shape}")
    print(f"  PyTorch shape: {pytorch_embed_weights.shape}")
    print(f"  Mean difference: {float(jnp.mean(jnp.abs(jax_embed_weights - pytorch_embed_weights))):.6f}")
    
    # Check if shapes match
    if jax_embed_weights.shape == pytorch_embed_weights.shape:
        print("✅ Embedding weight shapes match")
    else:
        print("❌ Embedding weight shapes don't match")
        print(f"  JAX: {jax_embed_weights.shape}")
        print(f"  PyTorch: {pytorch_embed_weights.shape}")

if __name__ == "__main__":
    debug_weight_accuracy()
