from nanovllm_jax.llm import LLM, SamplingParams

llm = LLM(
    # "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    max_model_len=2048,
    tensor_parallel_size=1,
)

# Single-prompt test
outputs = llm.generate(
    ["你好，介绍一下你自己。"],
    SamplingParams(temperature=0.7, max_tokens=32),
)
print("Single prompt output:\n", outputs[0]["text"])

# Multi-prompt test (same length prompts to trigger batched prefill/decode)
multi_prompts = [
    "中国的首都是哪里",
    "你好，介绍一下你自己。",
]
multi_outputs = llm.generate(
    multi_prompts,
    SamplingParams(temperature=0.7, max_tokens=32),
)
print("\nMulti-prompt outputs:")
for i, out in enumerate(multi_outputs):
    print(f"Prompt {i}: {multi_prompts[i]}")
    print("Response:", out["text"])
    print("-")
