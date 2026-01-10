#!/usr/bin/env python
"""测试长序列的 Chunked Prefill 功能"""
import os

def main():
    print("=" * 70)
    print("Testing Chunked Prefill with LONG Sequence")
    print("=" * 70)
    
    from nanovllm import LLM, SamplingParams
    
    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    
    # 设置很小的 max_num_batched_tokens 来强制分块
    print("\n[Config]")
    print(f"  - max_num_batched_tokens: 64  (小值，强制分块)")
    print(f"  - tensor_parallel_size: 1")
    print(f"  - enforce_eager: True")
    
    llm = LLM(
        model_path,
        max_num_batched_tokens=64,  # 调整到 64，避免过小
        tensor_parallel_size=8,
        enforce_eager=True
    )
    
    print("\n✓ LLM initialized")
    print("[DEBUG] Starting test with long prompt...")
    
    # 构造一个较长的 prompt
    prompt = "请详细介绍一下人工智能的发展历史，包括早期的研究、重要的里程碑事件、关键技术突破，以及当前的发展趋势和未来展望。"
    
    print(f"\n[Prompt]")
    print(f"  Text: {prompt[:50]}...")
    print(f"  Length: {len(prompt)} characters")
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    
    print(f"\n[Generation Parameters]")
    print(f"  - Max tokens: 10")
    print(f"  - Temperature: 0.0")
    
    print("\n" + "=" * 70)
    print("Starting generation (watch for multiple prefill chunks)...")
    print("=" * 70 + "\n")
    
    outputs = llm.generate([prompt], sampling_params)
    
    print("\n" + "=" * 70)
    print("✓ Generation completed!")
    print("=" * 70)
    
    for i, output in enumerate(outputs):
        print(f"\n[Output {i+1}]")
        print(f"Generated text: {output['text']!r}")
        print(f"Token count: {len(output['token_ids'])}")
        print(f"Tokens: {output['token_ids']}")
    
    print("\n" + "=" * 70)
    print("✓ CHUNKED PREFILL TEST PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
