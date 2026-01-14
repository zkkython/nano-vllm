import torch.distributed as dist

from nanovllm.models.deepseek_v3 import DeepSeekV3ForCausalLM
from nanovllm import LLM, SamplingParams
from transformers import AutoConfig

model_path = "/data/ds-671"


def check():
    """示例：部分层加载 DeepSeek-V3."""
    print("\n" + "=" * 70)
    print("DeepSeek-V3 部分层加载示例")
    print("=" * 70)

    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    # 使用真实配置
    config = AutoConfig.from_pretrained(model_path)
    # print(f'deepseek config: {config}')

    model = DeepSeekV3ForCausalLM(config)

    # 只加载前 2 层进行快速测试
    print(f"\n只加载前 2 层（共 {config.num_hidden_layers} 层）...")
    stats = model.load_weights(
        config=config, model_path=model_path, load_partial_layers=3
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
    print("=" * 70)


def deepseek_v3_partial_layer_infer():
    """示例：只加载前 2 层进行快速验证"""
    print("=" * 70)
    print("Example: Partial Layer Loading for Quick Debugging")
    print("=" * 70)

    # 只加载前 2 层（layer 0 和 layer 1）
    # 这样可以大幅减少初始化时间和显存占用
    llm = LLM(
        model_path,
        max_num_batched_tokens=5000,
        max_num_seqs=4,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=8,
        load_partial_layers=3,  # 关键参数：只加载前 2 层
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


if __name__ == "__main__":
    # check()
    deepseek_v3_partial_layer_infer()
