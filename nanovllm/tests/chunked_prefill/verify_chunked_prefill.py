import os
print("File loaded")
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    print("Starting verification script...")
    # 使用 Qwen3-8B 模型进行测试
    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        return

    print(f"Loading LLM from {model_path}...")
    # 设置一个非常小的 max_num_batched_tokens 来强制触发 chunking
    try:
        # 使用一个较大的值先测试，避免 warmup 问题
        llm = LLM(
            model_path, 
            enforce_eager=True, 
            tensor_parallel_size=1,
            max_num_batched_tokens=128  # 增大一些，先确保基本功能正常
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        import traceback
        traceback.print_exc()
        return

    print("LLM initialized successfully.")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    
    # 使用一个简短的 prompt 先测试
    prompt = "Hello, how are you?"
    
    print(f"Testing with prompt: {prompt!r}")
    
    try:
        outputs = llm.generate([prompt], sampling_params)
        
        for output in outputs:
            print("\nGenerated text:")
            print(output["text"])
            print("\nTest PASSED!")
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
