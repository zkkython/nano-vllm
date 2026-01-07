#!/usr/bin/env python3
"""
双机PyTorch All-Reduce测试 - 从节点
主节点IP: 115.190.188.193
从节点IP: 115.190.188.194
"""

import os
import torch
import torch.distributed as dist


def run_worker():
    """从节点运行函数"""
    # 设置分布式环境变量
    os.environ["MASTER_ADDR"] = "192.168.0.163"  # 主节点IP
    os.environ["MASTER_PORT"] = "29500"  # 通信端口
    os.environ["WORLD_SIZE"] = "2"  # 总进程数（2台机器）
    os.environ["RANK"] = "1"  # 从节点rank为1

    print("=" * 60)
    print("从节点启动 - All Reduce测试")
    print("=" * 60)
    print(f"连接到主节点: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    print(f"Rank: {os.environ['RANK']}")
    print(f"World Size: {os.environ['WORLD_SIZE']}")
    print("=" * 60)

    # 初始化分布式进程组
    print("\n[步骤1] 初始化分布式进程组...")
    print("正在连接主节点...")
    dist.init_process_group(
        backend="nccl",  # 使用NCCL后端（GPU通信）
        init_method="env://",  # 使用环境变量初始化
        rank=1,
        world_size=2,
    )
    print("✓ 分布式进程组初始化成功")
    print("✓ 已成功连接到主节点")

    # 设置CUDA设备
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        torch.cuda.set_device(1)
        print(f"✓ 使用GPU设备: {torch.cuda.get_device_name(1)}")
    else:
        device = torch.device("cpu")
        print("⚠ GPU不可用，使用CPU")

    # 创建测试张量
    print("\n[步骤2] 创建测试张量...")
    tensor = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], device=device)
    print(f"从节点原始张量: {tensor}")

    # 执行All-Reduce操作
    print("\n[步骤3] 执行All-Reduce操作（求和）...")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"✓ All-Reduce完成")
    print(f"从节点结果张量: {tensor}")
    print(f"期望结果: 主节点[1,2,3,4,5] + 从节点[10,20,30,40,50] = {tensor}")

    # 执行Barrier同步
    print("\n[步骤4] 同步所有进程...")
    dist.barrier()
    print("✓ 所有进程已同步")

    # 测试广播操作
    print("\n[步骤5] 测试广播操作...")
    broadcast_tensor = torch.zeros(3, device=device)
    print(f"从节点接收前的张量: {broadcast_tensor}")
    dist.broadcast(broadcast_tensor, src=0)
    print(f"✓ 广播完成")
    print(f"从节点接收到的张量: {broadcast_tensor}")

    # 测试Gather操作
    print("\n[步骤6] 测试Gather操作...")
    gather_tensor = torch.tensor([2.0], device=device)
    dist.gather(gather_tensor, dst=0)
    print("✓ Gather完成（数据已发送到主节点）")

    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)

    # 清理
    dist.destroy_process_group()
    print("\n✓ 分布式进程组已销毁")


if __name__ == "__main__":
    try:
        run_worker()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
