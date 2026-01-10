#!/usr/bin/env python
"""调试 chunked prefill 的 slot_mapping 问题"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步执行以捕获确切错误

def main():
    from nanovllm import LLM, SamplingParams
    
    print("Testing with CUDA_LAUNCH_BLOCKING=1...")
    print("=" * 70)
    
    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    
    # 使用适中的 max_num_batched_tokens
    llm = LLM(
        model_path,
        max_num_batched_tokens=64,
        tensor_parallel_size=1,
        enforce_eager=True
    )
    
    print("\n✓ Model loaded\n")
    
    # 使用一个会触发分块的中等长度 prompt
    prompt = "请介绍一下人工智能" * 5  # 重复几次以增加长度
    
    print(f"Prompt length: {len(prompt)} characters\n")
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=3)
    
    try:
        outputs = llm.generate([prompt], sampling_params)
        print("\n" + "=" * 70)
        print("✓ SUCCESS!")
        print(f"Output: {outputs[0]['text']!r}")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
