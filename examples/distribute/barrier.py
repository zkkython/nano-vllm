"""
Barrier 同步屏障示例
功能：同步所有进程，确保所有进程到达屏障点后才继续执行

使用方法：
# 在节点1上运行 (假设IP: 192.168.1.100)
python barrier.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29500

# 在节点2上运行
python barrier.py --rank 1 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
"""

import torch
import torch.distributed as dist
import argparse
import os
import time


def run_barrier(rank, world_size):
    """执行 barrier 操作"""

    print(f"[Rank {rank}] 开始执行任务...")

    # 模拟不同进程执行不同时长的任务
    if rank == 0:
        print(f"[Rank {rank}] 执行快速任务 (1秒)...")
        time.sleep(1)
    else:
        print(f"[Rank {rank}] 执行慢速任务 (3秒)...")
        time.sleep(3)

    print(f"[Rank {rank}] 任务完成，到达屏障点...")

    # 等待所有进程到达此处
    dist.barrier()

    print(f"[Rank {rank}] 所有进程已同步，继续执行后续任务")

    # 屏障后的任务
    tensor = torch.ones(3) * (rank + 1)
    print(f"[Rank {rank}] 后续任务: 创建张量 {tensor}")

    # 再次同步
    print(f"[Rank {rank}] 到达第二个屏障点...")
    dist.barrier()
    print(f"[Rank {rank}] 第二次同步完成")


def demonstrate_barrier_importance(rank, world_size):
    """演示 barrier 的重要性"""
    print(f"\n[Rank {rank}] === 演示屏障的重要性 ===")

    # 场景：确保所有进程都准备好后再开始计算
    print(f"[Rank {rank}] 准备阶段...")
    if rank == 0:
        time.sleep(0.5)
        print(f"[Rank {rank}] 准备就绪")
    else:
        time.sleep(1.5)
        print(f"[Rank {rank}] 准备就绪")

    # 使用屏障确保所有进程都准备好
    dist.barrier()
    print(f"[Rank {rank}] 所有进程准备就绪，开始同步计算")

    # 同步计算
    tensor = torch.ones(3) * 10
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] 计算结果: {tensor}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Barrier 示例")
    parser.add_argument("--rank", type=int, required=True, help="当前进程的rank")
    parser.add_argument("--world-size", type=int, required=True, help="总进程数")
    parser.add_argument("--master-addr", type=str, required=True, help="主节点IP地址")
    parser.add_argument("--master-port", type=str, default="29500", help="主节点端口")
    parser.add_argument(
        "--backend", type=str, default="nccl", help="通信后端 (nccl/gloo)"
    )

    args = parser.parse_args()

    # 设置环境变量
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)

    # 初始化进程组
    print(f"初始化进程组 - Rank: {args.rank}, World Size: {args.world_size}")
    dist.init_process_group(
        backend=args.backend, rank=args.rank, world_size=args.world_size
    )

    # 如果使用NCCL后端，设置GPU设备
    if args.backend == "nccl":
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
        print(f"[Rank {args.rank}] 使用 GPU {args.rank % torch.cuda.device_count()}")

    # 执行 barrier 操作
    run_barrier(args.rank, args.world_size)

    # 演示 barrier 的重要性
    demonstrate_barrier_importance(args.rank, args.world_size)

    # 清理
    dist.destroy_process_group()
    print(f"[Rank {args.rank}] 完成")


if __name__ == "__main__":
    main()
