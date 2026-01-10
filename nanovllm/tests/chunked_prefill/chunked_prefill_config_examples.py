#!/usr/bin/env python
"""
Chunked Prefill 配置示例

本文件展示如何使用不同的配置来控制 Chunked Prefill 功能
"""

from nanovllm import LLM, SamplingParams

# ============================================================================
# 示例 1: 默认配置（推荐用于大多数场景）
# ============================================================================
def example_1_default():
    """
    默认启用 Chunked Prefill，chunk size 等于 max_num_batched_tokens
    适用于一般场景
    """
    llm = LLM(
        model_path="/path/to/model",
        max_num_batched_tokens=128,
        # enable_chunked_prefill=True (默认值)
        # chunked_prefill_size=None (默认使用 max_num_batched_tokens)
    )
    return llm


# ============================================================================
# 示例 2: 显存严重受限（如 8GB GPU）
# ============================================================================
def example_2_low_memory():
    """
    使用小 chunk size 降低显存峰值
    """
    llm = LLM(
        model_path="/path/to/model",
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        chunked_prefill_size=32,  # 小 chunk size
    )
    return llm


# ============================================================================
# 示例 3: 显存适中（如 16GB GPU，推荐配置）
# ============================================================================
def example_3_balanced():
    """
    平衡显存和性能
    """
    llm = LLM(
        model_path="/path/to/model",
        max_num_batched_tokens=512,
        enable_chunked_prefill=True,
        chunked_prefill_size=128,  # 平衡的 chunk size
    )
    return llm


# ============================================================================
# 示例 4: 显存充足（如 24GB+ GPU）
# ============================================================================
def example_4_high_memory():
    """
    使用大 chunk size 追求高吞吐量
    """
    llm = LLM(
        model_path="/path/to/model",
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        chunked_prefill_size=512,  # 大 chunk size
    )
    return llm


# ============================================================================
# 示例 5: 禁用 Chunked Prefill（传统模式）
# ============================================================================
def example_5_disabled():
    """
    禁用 Chunked Prefill，使用传统的一次性 prefill
    适用于短序列场景或需要最高性能的场景
    """
    llm = LLM(
        model_path="/path/to/model",
        max_num_batched_tokens=4096,  # 必须 >= max_model_len
        enable_chunked_prefill=False,  # 显式禁用
    )
    return llm


# ============================================================================
# 示例 6: 超长序列处理（如文档分析）
# ============================================================================
def example_6_ultra_long():
    """
    处理超长序列，使用小 chunk 多次处理
    """
    llm = LLM(
        model_path="/path/to/model",
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        chunked_prefill_size=64,  # 小 chunk 以处理超长序列
        max_num_seqs=4,  # 允许多序列并行
    )
    return llm


# ============================================================================
# 示例 7: 批量推理场景
# ============================================================================
def example_7_batch_inference():
    """
    批量推理，多个短序列并行处理
    """
    llm = LLM(
        model_path="/path/to/model",
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        chunked_prefill_size=256,
        max_num_seqs=16,  # 允许更多序列并行
    )
    return llm


# ============================================================================
# 配置建议表
# ============================================================================
CONFIGURATION_GUIDE = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    Chunked Prefill 配置建议表                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ 场景                  │ GPU显存 │ enable │ chunk_size │ max_batched │ 说明   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ 超长序列+显存受限    │  8GB    │  True  │    32      │     256     │ 最低显存 ║
║ 一般长序列            │ 16GB    │  True  │   128      │     512     │ 平衡模式 ║
║ 显存充足              │ 24GB+   │  True  │   512      │    2048     │ 高性能   ║
║ 短序列场景            │ Any     │  False │     -      │    4096     │ 传统模式 ║
║ 文档分析              │ 16GB    │  True  │    64      │    1024     │ 超长序列 ║
║ 批量推理              │ 24GB    │  True  │   256      │    1024     │ 多序列   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

关键参数说明：
• enable_chunked_prefill: 
  - True: 启用分块 prefill，允许处理超长序列，降低显存峰值
  - False: 禁用分块，使用传统一次性 prefill，性能最优但显存占用高

• chunked_prefill_size:
  - None: 自动使用 max_num_batched_tokens 作为 chunk size（默认）
  - 32-64: 极小 chunk，适合显存严重受限场景
  - 128-256: 平衡 chunk，适合大多数场景
  - 512+: 大 chunk，适合显存充足追求性能场景

• max_num_batched_tokens:
  - 控制单次推理的最大 token 数量
  - 启用 chunked prefill 时可以小于 max_model_len
  - 禁用时必须 >= max_model_len

性能权衡：
• 小 chunk size: ↓显存峰值, ↑调度开销, ↓吞吐量
• 大 chunk size: ↑显存峰值, ↓调度开销, ↑吞吐量
"""


def main():
    print(CONFIGURATION_GUIDE)
    
    print("\n示例代码:")
    print("=" * 80)
    
    examples = [
        ("默认配置", example_1_default),
        ("显存受限", example_2_low_memory),
        ("平衡配置", example_3_balanced),
        ("显存充足", example_4_high_memory),
        ("禁用分块", example_5_disabled),
        ("超长序列", example_6_ultra_long),
        ("批量推理", example_7_batch_inference),
    ]
    
    for name, func in examples:
        print(f"\n# {name}")
        print(f"# llm = {func.__name__}()")
        import inspect
        source = inspect.getsource(func)
        # 提取 LLM 初始化代码
        lines = source.split('\n')
        in_llm = False
        for line in lines:
            if 'llm = LLM(' in line:
                in_llm = True
            if in_llm:
                if 'return llm' in line:
                    break
                print(line)


if __name__ == "__main__":
    main()
