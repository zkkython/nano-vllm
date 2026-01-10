#!/usr/bin/env python
"""简单测试 Chunked Prefill 功能"""
import os
import sys

# 清理可能的 CUDA 缓存
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def main():
    print("=" * 60)
    print("Chunked Prefill Simple Test")
    print("=" * 60)
    
    from nanovllm import LLM, SamplingParams
    
    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n✓ Model path verified: {model_path}")
    print("\n[Step 1] Initializing LLM...")
    print(f"  - max_num_batched_tokens: 128")
    print(f"  - tensor_parallel_size: 1")
    print(f"  - enforce_eager: True")
    
    try:
        llm = LLM(
            model_path,
            max_num_batched_tokens=128,
            tensor_parallel_size=1,
            enforce_eager=True
        )
        print("✓ LLM initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[Step 2] Preparing generation parameters...")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    prompt = "Hello"
    
    print(f"  - Prompt: {prompt!r}")
    print(f"  - Max tokens: 5")
    print(f"  - Temperature: 0.0")
    
    print("\n[Step 3] Generating...")
    try:
        outputs = llm.generate([prompt], sampling_params)
        
        print("\n" + "=" * 60)
        print("✓ Generation completed successfully!")
        print("=" * 60)
        
        for i, output in enumerate(outputs):
            print(f"\n[Output {i+1}]")
            print(f"Text: {output['text']!r}")
            print(f"Tokens: {output['token_ids']}")
        
        print("\n" + "=" * 60)
        print("✓ TEST PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
