#!/usr/bin/env python
"""测试启用 Chunked Prefill 时的 warmup 功能"""


def main():
    print("=" * 70)
    print("Test: Warmup with Chunked Prefill Enabled")
    print("=" * 70)

    from nanovllm import LLM, SamplingParams

    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

    # 启用 Chunked Prefill
    llm = LLM(
        model_path,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        chunked_prefill_size=16,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    # 如果能走到这里说明 warmup 成功
    print("\n✓ Warmup completed successfully with Chunked Prefill!")

    # 测试正常推理
    prompt = "请详细介绍一下人工智能的发展历史，包括早期的研究、重要的里程碑事件、关键技术突破，以及当前的发展趋势和未来展望"
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))
    print(f"\nTest inference:")
    print(f"  Prompt: {prompt}")
    print(f"  Output: {outputs[0]['text']}")

    print("\n" + "=" * 70)
    print("✓ TEST PASSED - Warmup with Chunked Prefill works!")
    print("=" * 70)


if __name__ == "__main__":
    main()
