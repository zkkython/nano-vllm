#!/usr/bin/env python3
"""
Test script to verify if the model can do simple math.
"""

import jax
import jax.numpy as jnp
from nanovllm_jax import LLM, SamplingParams
from transformers import AutoTokenizer

def test_simple_math():
    print("=== Simple Math Test ===")
    
    # Load model
    print("Loading model...")
    llm = LLM('/home/kason/models/qwen06b', max_model_len=32, gpu_memory_utilization=0.1)
    
    # Test with very simple prompts
    prompts = [
        "1+1=",
        "2+2=", 
        "3+3=",
        "What is 1+1?",
        "Calculate 2+2:"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Test with different temperatures
        for temp in [0.0, 0.1, 0.5]:
            outputs = llm.generate([prompt], SamplingParams(max_tokens=3, temperature=temp))
            print(f"  Temp {temp}: '{outputs[0]['text']}'")
    
    # Test with chat format
    print(f"\n=== Testing with chat format ===")
    tokenizer = llm.tokenizer
    
    # Create proper chat format
    chat_prompt = tokenizer.apply_chat_template([
        {'role': 'user', 'content': 'What is 1+1?'}
    ], tokenize=False)
    
    print(f"Chat prompt: {chat_prompt}")
    
    outputs = llm.generate([chat_prompt], SamplingParams(max_tokens=5, temperature=0.0))
    print(f"Chat output: '{outputs[0]['text']}'")
    
    # Test tokenizer directly
    print(f"\n=== Testing tokenizer ===")
    test_tokens = ['1', '2', '3', '+', '=', 'What', 'is']
    for token in test_tokens:
        encoded = tokenizer.encode(token)
        decoded = tokenizer.decode(encoded)
        print(f"'{token}' -> {encoded} -> '{decoded}'")

if __name__ == "__main__":
    test_simple_math()
