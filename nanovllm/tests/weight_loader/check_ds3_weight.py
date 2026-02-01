import torch
import torch.distributed as dist

from nanovllm.models.deepseek_v3 import DeepseekV3ForCausalLM
from nanovllm import LLM, SamplingParams
from transformers import AutoConfig

model_path = "/data/ds-671"
import logging

logging.basicConfig(level=logging.INFO)


def check():
    """示例：部分层加载 DeepSeek-V3."""
    torch.set_default_dtype(torch.bfloat16)
    print("\n" + "=" * 70)
    print("DeepSeek-V3 部分层加载示例")
    print("=" * 70)

    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    # 使用真实配置
    config = AutoConfig.from_pretrained(model_path)
    config.quantization = "fp8"

    def print_gpu_memory(label):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(
                f"[MEMORY] {label}: Allocated={allocated:.5f}GB, Reserved={reserved:.5f}GB"
            )

    print_gpu_memory("Before model init")
    model = DeepseekV3ForCausalLM(config)
    print_gpu_memory("After model init (Empty)")

    # 只加载前 2 层进行快速测试
    print(f"\n只加载前 2 层（共 {config.num_hidden_layers} 层）...")
    stats = model.load_weights(
        config=config, model_path=model_path, load_partial_layers=4
    )
    print_gpu_memory("After weight loading")

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
    gate_proj = model.model.layers[layer_id].mlp.gate_proj

    print(f"Weight dtype: {gate_proj.weight.dtype}")
    print(f"Weight scale is None: {gate_proj.weight_scale is None}")
    if gate_proj.weight_scale is not None:
        print(f"Weight scale shape: {gate_proj.weight_scale.shape}")
        print(f"Weight scale example values: {gate_proj.weight_scale.flatten()[:5]}")
    print("=" * 70)


def deepseek_v3_partial_layer_infer():

    print("=" * 70)
    print("Example: Partial Layer Loading for Quick Debugging")
    print("=" * 70)

    # 这样可以大幅减少初始化时间和显存占用
    llm = LLM(
        model_path,
        max_model_len=2000,
        max_num_batched_tokens=2000,
        max_num_seqs=1,
        gpu_memory_utilization=0.88,
        quantization="fp8",
        tensor_parallel_size=8,
        # load_partial_layers=61,
        enforce_eager=True,
    )

    print("\n模型加载完成！")

    # 简单推理测试
    prompt = "请用中文介绍你自己"
    print(f"输入: {prompt}")

    outputs = llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=100))
    for output in outputs:
        print(f"输出: {output['text']}")

    print("\n验证项:")
    print("  - 权重加载流程是否正确")
    print("  - 模型 forward 是否能正常运行")
    print("  - 参数形状是否匹配")


if __name__ == "__main__":
    # check()
    deepseek_v3_partial_layer_infer()
