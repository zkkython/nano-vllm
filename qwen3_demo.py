#!/usr/bin/env python3
"""
Qwen3 Demo for JAX-based nano-vllm implementation.

This demo shows how to use the JAX implementation of nano-vllm
with Qwen3 models for text generation.
"""

import argparse
import os
import jax
from nanovllm_jax import LLM, SamplingParams
from transformers import AutoTokenizer


def main(args):
    """Main demo function."""
    print("JAX-based Nano-vLLM Qwen3 Demo")
    print("=" * 40)
    
    # Set JAX device
    print(f"JAX devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")
    
    # Load model
    model_path = os.path.expanduser(args.model)
    print(f"Loading model from: {model_path}")
    
    try:
        llm = LLM(
            model_path, 
            enforce_eager=True, 
            tensor_parallel_size=1,
            max_model_len=args.max_length,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kvcache_block_size=256,  # Must be multiple of 256
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )
    
    # Prepare prompts
    prompts = [
        "1+1=？",
        #"The future of artificial intelligence is",
        "100内的质数有哪些",
    ]
    
    # Apply chat template if available
    try:
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ) if hasattr(tokenizer, 'apply_chat_template') else prompt
            for prompt in prompts
        ]
    except:
        formatted_prompts = prompts
    
    print(f"\nGenerating text for {len(formatted_prompts)} prompts...")
    print("-" * 40)
    
    # Generate text
    try:
        outputs = llm.generate(formatted_prompts, sampling_params)
        
        # Display results
        for i, (prompt, output) in enumerate(zip(formatted_prompts, outputs)):
            print(f"\nPrompt {i+1}:")
            print(f"Input:  {prompt}")
            print(f"Output: {output['text']}")
            print(f"Tokens: {len(output['token_ids'])}")
            print("-" * 40)
        
        print("\n✓ Generation completed successfully!")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 Demo for JAX-based nano-vllm")
    parser.add_argument(
        "--model", 
        type=str, 
        default="/home/kason/models/qwen06b",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization (0.1-0.9)"
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=1024,
        help="Maximum number of batched tokens"
    )
    
    args = parser.parse_args()
    main(args)
