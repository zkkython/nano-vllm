#!/usr/bin/env python3
"""
Example usage of JAX-based nano-vllm implementation.

This example demonstrates how to use the JAX implementation
for text generation with Qwen3 models.
"""

import argparse
import os
import jax
import jax.numpy as jnp
from nanovllm_jax import LLM, SamplingParams
from transformers import AutoTokenizer


def main(args):
    """Main example function."""
    print("JAX-based Nano-vLLM Example")
    print("=" * 40)
    
    # Set JAX device and configuration
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")
    
    # Load model
    model_path = os.path.expanduser(args.model)
    print(f"\nLoading model from: {model_path}")
    
    try:
        # Initialize LLM with JAX backend
        llm = LLM(
            model_path, 
            enforce_eager=True, 
            tensor_parallel_size=1,
            max_model_len=args.max_length,
            max_num_batched_tokens=args.batch_size * args.max_length
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("Note: This is a demo implementation. Full model loading requires actual model weights.")
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
        "Hello, I am Qwen3, a large language model.",
        # "The future of artificial intelligence is",
        # "Explain quantum computing in simple terms:",
        # "Write a short story about a robot learning to paint:",
    ]
    
    # Apply chat template if available
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for prompt in prompts
            ]
        else:
            formatted_prompts = prompts
    except:
        formatted_prompts = prompts
    
    print(f"\nGenerating text for {len(formatted_prompts)} prompts...")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
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
        
        # Performance summary
        total_tokens = sum(len(output['token_ids']) for output in outputs)
        print(f"\nPerformance Summary:")
        print(f"Total prompts: {len(prompts)}")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Average tokens per prompt: {total_tokens / len(prompts):.1f}")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        print("Note: This is a demo implementation. Full functionality requires actual model weights.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX-based nano-vllm example")
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
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    main(args)
