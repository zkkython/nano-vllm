#!/usr/bin/env python
"""
日志配置使用示例

展示如何使用 LogConfig 控制不同模块的日志输出
"""


def example_1_default():
    """示例 1：使用默认配置（从环境变量读取）"""
    print("=" * 70)
    print("Example 1: Default Log Config (from environment variables)")
    print("=" * 70)

    from nanovllm import LLM, SamplingParams

    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

    # 默认配置，从环境变量读取
    # 可以通过设置 NANOVLLM_LOG_LEVEL=DEBUG 来启用所有日志
    llm = LLM(
        model_path,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        chunked_prefill_size=64,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    prompt = "Hello"
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))
    print(f"\nOutput: {outputs[0]['text']}")


def example_2_production():
    """示例 2：生产环境配置（只输出错误）"""
    print("\n" + "=" * 70)
    print("Example 2: Production Config (ERROR level only)")
    print("=" * 70)

    from nanovllm import LLM, SamplingParams
    from nanovllm.log_config import LogConfig, LogLevel

    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

    # 生产环境：只输出错误，屏蔽所有调试信息
    log_config = LogConfig(global_level=LogLevel.ERROR)

    llm = LLM(
        model_path,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        chunked_prefill_size=64,
        enforce_eager=True,
        tensor_parallel_size=1,
        log_config=log_config,
    )

    prompt = "你好"
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))
    print(f"\nOutput: {outputs[0]['text']}")


def example_3_debug_chunked_prefill():
    """示例 3：调试 Chunked Prefill（只开启该模块的调试日志）"""
    print("\n" + "=" * 70)
    print("Example 3: Debug Chunked Prefill Only")
    print("=" * 70)

    from nanovllm import LLM, SamplingParams
    from nanovllm.log_config import LogConfig, LogLevel

    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

    # 全局设置为 ERROR，但 chunked_prefill 模块设置为 DEBUG
    log_config = LogConfig(
        global_level=LogLevel.ERROR,
        chunked_prefill=LogLevel.DEBUG,  # 只调试 chunked prefill
        warmup=LogLevel.INFO,  # warmup 显示基本信息
    )

    llm = LLM(
        model_path,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        chunked_prefill_size=64,
        enforce_eager=True,
        tensor_parallel_size=1,
        log_config=log_config,
    )

    prompt = "请介绍一下人工智能"
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=30))
    print(f"\nOutput: {outputs[0]['text']}")


def example_4_verbose():
    """示例 4：详细调试模式（所有模块都输出 DEBUG）"""
    print("\n" + "=" * 70)
    print("Example 4: Verbose Debug Mode")
    print("=" * 70)

    from nanovllm import LLM, SamplingParams
    from nanovllm.log_config import LogConfig, LogLevel

    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

    # 所有模块都输出 DEBUG 日志
    log_config = LogConfig(global_level=LogLevel.DEBUG)

    llm = LLM(
        model_path,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        chunked_prefill_size=64,
        enforce_eager=True,
        tensor_parallel_size=1,
        log_config=log_config,
    )

    prompt = "测试"
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=3))
    print(f"\nOutput: {outputs[0]['text']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "1":
            example_1_default()
        elif example == "2":
            example_2_production()
        elif example == "3":
            example_3_debug_chunked_prefill()
        elif example == "4":
            example_4_verbose()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python test_log_config.py [1|2|3|4]")
    else:
        print("Available examples:")
        print("  1. Default config (from environment)")
        print("  2. Production config (ERROR only)")
        print("  3. Debug Chunked Prefill only")
        print("  4. Verbose debug mode")
        print("\nUsage: python test_log_config.py [1|2|3|4]")
        print("\nRunning example 3 by default...\n")
        example_3_debug_chunked_prefill()
