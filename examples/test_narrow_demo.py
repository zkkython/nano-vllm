import torch
import torch.nn as nn
import torch.distributed as dist
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear


def test_narrow_column_parallel():
    """测试 ColumnParallelLinear 中的 narrow 使用（tp_dim=0）"""
    # 模拟 2 个进程的并行
    tp_size = 2
    tp_rank = 0  # 模拟 rank 0

    # 创建一个模拟的 ColumnParallelLinear 实例
    # 假设 input_size=4, output_size=6
    linear = ColumnParallelLinear(input_size=4, output_size=6)
    linear.tp_size = tp_size
    linear.tp_rank = tp_rank

    # 模拟全局权重，形状 (6, 4)
    loaded_weight = torch.randn(6, 4)
    print(f"全局权重形状: {loaded_weight.shape}")

    # 模拟 weight_loader 中的 narrow
    param_data = linear.weight.data  # 形状 (3, 4)，因为 output_size_per_partition = 6 // 2 = 3
    shard_size = param_data.size(linear.tp_dim)  # 3
    start_idx = linear.tp_rank * shard_size  # 0 * 3 = 0
    extracted_weight = loaded_weight.narrow(linear.tp_dim, start_idx, shard_size)
    print(f"Rank {tp_rank} 提取的分片形状: {extracted_weight.shape}")  # (3, 4)
    print(f"提取的分片内容:\n{extracted_weight}")

    # 验证：分片应该是全局权重的第 0-2 行
    assert torch.allclose(extracted_weight, loaded_weight[0:3, :]), "分片不匹配"


def test_narrow_row_parallel():
    """测试 RowParallelLinear 中的 narrow 使用（tp_dim=1）"""
    # 模拟 2 个进程的并行
    tp_size = 2
    tp_rank = 1  # 模拟 rank 1

    # 创建一个模拟的 RowParallelLinear 实例
    # 假设 input_size=4, output_size=6
    linear = RowParallelLinear(input_size=4, output_size=6)
    linear.tp_size = tp_size
    linear.tp_rank = tp_rank

    # 模拟全局权重，形状 (6, 2)，因为 input_size_per_partition = 4 // 2 = 2
    loaded_weight = torch.randn(6, 4)
    print(f"全局权重形状: {loaded_weight.shape}")

    # 模拟 weight_loader 中的 narrow
    param_data = linear.weight.data  # 形状 (6, 2)
    shard_size = param_data.size(linear.tp_dim)  # 2
    start_idx = linear.tp_rank * shard_size  # 1 * 2 = 2
    extracted_weight = loaded_weight.narrow(linear.tp_dim, start_idx, shard_size)
    print(f"Rank {tp_rank} 提取的分片形状: {extracted_weight.shape}")  # (6, 2)
    print(f"提取的分片内容:\n{extracted_weight}")

    # 验证：分片应该是全局权重的第 2-3 列
    assert torch.allclose(extracted_weight, loaded_weight[:, 2:4]), "分片不匹配"


if __name__ == "__main__":
    print("=== 测试 ColumnParallelLinear 中的 narrow ===")
    test_narrow_column_parallel()
    print("\n=== 测试 RowParallelLinear 中的 narrow ===")
    test_narrow_row_parallel()
    print("\n所有测试通过！narrow 成功提取了指定维度的子张量，用于权重分片。")
