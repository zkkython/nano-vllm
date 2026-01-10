#!/usr/bin/env python
"""测试禁用 Chunked Prefill 时的 warmup 功能"""


def main():
    print("=" * 70)
    print("Test: Warmup with Chunked Prefill Disabled")
    print("=" * 70)

    from nanovllm import LLM, SamplingParams

    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

    # 禁用 Chunked Prefill
    llm = LLM(
        model_path,
        max_num_batched_tokens=20000,
        enable_chunked_prefill=False,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    # 如果能走到这里说明 warmup 成功
    print("\n✓ Warmup completed successfully without Chunked Prefill!")

    # 测试正常推理
    prompt = "请详细介绍一下人工智能的发展历史，包括早期的研究、重要的里程碑事件、关键技术突破，以及当前的发展趋势和未来展望"
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))
    print(f"\nTest inference:")
    print(f"  Prompt: {prompt}")
    print(f"  Output: {outputs[0]['text']}")

    print("\n" + "=" * 70)
    print("✓ TEST PASSED - Warmup without Chunked Prefill works!")
    print("=" * 70)


if __name__ == "__main__":
    main()
