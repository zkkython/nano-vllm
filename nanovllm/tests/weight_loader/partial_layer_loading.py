#!/usr/bin/env python
"""部分层加载示例：快速调试与验证

演示如何使用 load_partial_layers 功能快速加载模型并验证权重加载逻辑。
PYTHONPATH=/root/mingtong/aiwork/nano-vllm python nanovllm/tests/weight_loader/partial_layer_loading.py
PYTHONPATH=/root/mingtong/aiwork/nano-vllm python nanovllm/tests/weight_loader/partial_layer_loading.py verify
"""

from nanovllm import LLM, SamplingParams


def example_partial_loading():
    """示例：只加载前 2 层进行快速验证"""
    print("=" * 70)
    print("Example: Partial Layer Loading for Quick Debugging")
    print("=" * 70)

    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

    # 只加载前 2 层（layer 0 和 layer 1）
    # 这样可以大幅减少初始化时间和显存占用
    llm = LLM(
        model_path,
        max_num_batched_tokens=25600,
        tensor_parallel_size=1,
        load_partial_layers=2,  # 关键参数：只加载前 2 层
        enforce_eager=True,
    )

    print("\n模型加载完成！")
    print("注意：只加载了前 2 层，推理结果仅供调试参考，不代表真实效果。\n")

    # 简单推理测试
    prompt = "你好"
    print(f"输入: {prompt}")

    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))
    print(f"输出: {outputs[0]['text']}")
    print("\n说明：由于只加载了 2 层，输出通常是无意义的，但可以验证：")
    print("  - 权重加载流程是否正确")
    print("  - 模型 forward 是否能正常运行")
    print("  - 参数形状是否匹配")


def example_verify_weight_shapes():
    """示例：验证部分层的权重是否正确加载"""
    print("\n" + "=" * 70)
    print("Example: Verify Weight Loading with Partial Layers")
    print("=" * 70)

    from transformers import Qwen3Config
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    import torch
    import torch.distributed as dist

    # 单进程环境下的分布式初始化（仅用于测试）
    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

    config = Qwen3Config.from_pretrained(model_path)
    model = Qwen3ForCausalLM(config)

    print(f"\n模型总层数: {config.num_hidden_layers}")
    print("只加载前 3 层...")

    # 只加载前 3 层
    stats = model.load_weights(
        config=config,
        model_path=model_path,
        load_partial_layers=1,
    )
    print("\n部分层加载统计信息:")
    print(f"  总权重数量: {stats['total_weights']}")
    print(f"  成功加载: {stats['loaded_weights']}")
    print(f"  跳过数量: {stats['skipped_weights']}")
    print(f"  已加载层: {stats['loaded_layers']}")
    print(f"  已加载的权重名称列表: {stats['loaded_weight_names']}")
    print(f"  跳过的层: {stats['skipped_layers']}")
    print(f"  跳过的权重名称列表: {stats['skipped_weight_names']}")

    print("\n验证权重加载情况:")

    # 检查已加载层的权重
    for i in range(3):
        layer = model.model.layers[i]
        qkv_weight = layer.self_attn.qkv_proj.weight
        print(
            f"  Layer {i} qkv_proj.weight: shape={qkv_weight.shape}, "
            f"mean={qkv_weight.mean().item():.6f}, "
            f"std={qkv_weight.std().item():.6f}"
        )

    # 检查未加载层的权重（应该是随机初始化的）
    if config.num_hidden_layers > 3:
        print("\n未加载层（保持随机初始化）:")
        for i in range(3, min(5, config.num_hidden_layers)):
            layer = model.model.layers[i]
            qkv_weight = layer.self_attn.qkv_proj.weight
            print(
                f"  Layer {i} qkv_proj.weight: shape={qkv_weight.shape}, "
                f"mean={qkv_weight.mean().item():.6f}, "
                f"std={qkv_weight.std().item():.6f}"
            )

    print("\n说明：")
    print("  - 已加载层的权重均值和标准差应该符合预训练模型的分布")
    print("  - 未加载层的权重保持 PyTorch 默认初始化状态")


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        # 验证权重形状
        example_verify_weight_shapes()
    else:
        # 快速推理测试
        example_partial_loading()
