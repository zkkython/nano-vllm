import torch
import torch.distributed as dist

from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from transformers import AutoConfig
from nanovllm import LLM, SamplingParams

model_path = "/data/hf/Qwen3-30B-A3B-Instruct-2507"
import logging

logging.basicConfig(level=logging.DEBUG)


def check():
    """示例：部分层加载 Qwen3 30b moe."""
    torch.set_default_dtype(torch.bfloat16)
    print("\n" + "=" * 70)
    print("Qwen3 30b moe 部分层加载示例")
    print("=" * 70)

    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    # 使用真实配置

    config = AutoConfig.from_pretrained(model_path)
    config.load_partial_layers = 2
    print(f"qwen3 mode config={config}")
    model = Qwen3MoeForCausalLM(config)

    # 只加载前 2 层进行快速测试
    print(
        f"\n只加载前 {config.load_partial_layers} 层（共 {config.num_hidden_layers} 层）..."
    )
    stats = model.load_weights(
        config=config,
        model_path=model_path,
        load_partial_layers=config.load_partial_layers,
    )

    print("\n权重加载统计:")
    print(f"  总权重数量: {stats['total_weights']}")
    print(f"  成功加载: {stats['loaded_weights']}")
    print(f"  跳过数量: {stats['skipped_weights']}")
    print(f"  已加载层: {stats['loaded_layers']}")
    print(f"  跳过的层: {stats['skipped_layers']}")
    print(f"  已加载的权重名称列表: {stats['loaded_weight_names']}")

    print(
        f"\n跳过的专家权重数量: {len([n for n in stats['skipped_weight_names'] if 'experts' in n])}"
    )

    print("\n✅ 部分层加载完成！节省内存和时间")
    # 检查某一层（例如第 0 层）的 MLP 权重
    layer_id = 0
    gate = model.transformers.layers[layer_id].mlp.gate
    print(f"Weight dtype: {gate.weight.dtype}")
    print("=" * 70)


def qwen_30b_moe_partial_layer_infer():
    # torch.set_default_dtype(torch.bfloat16)
    load_partial_layers = 48
    """示例：只加载前 load_partial_layers 层进行快速验证"""
    print("=" * 70)
    print(f"Example: {load_partial_layers} Partial Layer Loading for Quick Debugging")
    print("=" * 70)

    # 只加载前 2 层（layer 0 和 layer 1）
    # 这样可以大幅减少初始化时间和显存占用
    llm = LLM(
        model_path,
        max_num_batched_tokens=5000,
        max_num_seqs=4,
        gpu_memory_utilization=0.8,
        # quantization="fp8",
        tensor_parallel_size=8,
        load_partial_layers=load_partial_layers,  # 关键参数：只加载前 2 层
        enforce_eager=True,
    )

    print("\n模型加载完成！")
    print(
        "注意：只加载了前 {load_partial_layers} 层，推理结果仅供调试参考，不代表真实效果。\n"
    )

    # 简单推理测试
    prompt = "你好"
    print(f"输入: {prompt}")

    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))
    print(f"输出: {outputs[0]['text']}")
    print("\n说明：由于只加载了 2 层，输出通常是无意义的，但可以验证：")
    print("  - 权重加载流程是否正确")
    print("  - 模型 forward 是否能正常运行")
    print("  - 参数形状是否匹配")


if __name__ == "__main__":
    qwen_30b_moe_partial_layer_infer()
