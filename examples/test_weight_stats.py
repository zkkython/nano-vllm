"""测试权重加载统计信息功能."""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.weight_loader import WeightMapping


def test_weight_loading_stats():
    """测试权重加载统计信息."""
    print("=" * 70)
    print("测试权重加载统计信息功能")
    print("=" * 70)

    # 单进程环境下的分布式初始化（仅用于测试）
    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    # 构造一个小尺寸的 Qwen3 配置
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=5,  # 总共 5 层
        rms_norm_eps=1e-5,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )

    model = Qwen3ForCausalLM(config)

    # 从模型内部构建映射规则
    weight_mappings = model._build_weight_mappings()

    # 根据映射规则构造假的 HF 权重
    param_dict = dict(model.named_parameters())
    tensors: dict[str, torch.Tensor] = {}

    for hf_key, mapping in weight_mappings.items():
        if isinstance(mapping, str):
            target_path = mapping
            transpose_flag = False
        else:
            target_path = mapping.target_path
            transpose_flag = getattr(mapping, "transpose", False)

        param = param_dict.get(target_path)
        if param is None:
            continue

        shape = param.shape
        if transpose_flag and param.dim() == 2:
            hf_shape = (shape[1], shape[0])
        else:
            hf_shape = shape

        numel = hf_shape[0] * hf_shape[1] if len(hf_shape) == 2 else param.numel()
        tensors[hf_key] = torch.arange(numel, dtype=torch.float32).reshape(hf_shape)

    # 保存权重文件
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_file = os.path.join(tmp_dir, "model-00001-of-00001.safetensors")
        save_file(tensors, weights_file)

        print(f"\n创建临时权重文件: {weights_file}")
        print(f"权重文件包含 {len(tensors)} 个权重")

        # 加载前 2 层，并获取统计信息
        print("\n正在加载前 2 层...")
        stats = model.load_weights(
            config=config, model_path=tmp_dir, load_partial_layers=2
        )

        # 输出统计信息
        print("\n" + "=" * 70)
        print("加载统计信息:")
        print("=" * 70)
        print(f"总权重数量:    {stats['total_weights']}")
        print(f"成功加载:      {stats['loaded_weights']}")
        print(f"跳过数量:      {stats['skipped_weights']}")
        print(f"已加载层:      {stats['loaded_layers']}")
        print(f"跳过的层:      {stats['skipped_layers']}")

        print(f"\n已加载的权重名称 (共 {len(stats['loaded_weight_names'])} 个):")
        for i, name in enumerate(stats["loaded_weight_names"][:15], 1):
            print(f"  {i:2d}. {name}")
        if len(stats["loaded_weight_names"]) > 15:
            print(f"  ... 还有 {len(stats['loaded_weight_names']) - 15} 个权重")

        if stats["skipped_weight_names"]:
            print(f"\n跳过的权重名称 (共 {len(stats['skipped_weight_names'])} 个):")
            for i, name in enumerate(stats["skipped_weight_names"][:15], 1):
                print(f"  {i:2d}. {name}")
            if len(stats["skipped_weight_names"]) > 15:
                print(f"  ... 还有 {len(stats['skipped_weight_names']) - 15} 个权重")

        # 验证权重名称的正确性
        print("\n" + "=" * 70)
        print("验证权重名称正确性:")
        print("=" * 70)

        loaded_layer_weights = [
            name for name in stats["loaded_weight_names"] if "model.layers." in name
        ]
        skipped_layer_weights = [
            name for name in stats["skipped_weight_names"] if "model.layers." in name
        ]

        print(f"已加载的层权重: {len(loaded_layer_weights)} 个")
        loaded_layers_from_names = set()
        for name in loaded_layer_weights:
            layer_num = int(name.split(".")[2])
            loaded_layers_from_names.add(layer_num)
        print(f"  涉及的层号: {sorted(loaded_layers_from_names)}")

        print(f"\n跳过的层权重: {len(skipped_layer_weights)} 个")
        skipped_layers_from_names = set()
        for name in skipped_layer_weights:
            layer_num = int(name.split(".")[2])
            skipped_layers_from_names.add(layer_num)
        print(f"  涉及的层号: {sorted(skipped_layers_from_names)}")

        # 断言验证
        assert loaded_layers_from_names == set(
            stats["loaded_layers"]
        ), "已加载层号不匹配"
        assert skipped_layers_from_names == set(
            stats["skipped_layers"]
        ), "跳过层号不匹配"

        print("\n✅ 所有验证通过!")
        print("=" * 70)


if __name__ == "__main__":
    test_weight_loading_stats()
