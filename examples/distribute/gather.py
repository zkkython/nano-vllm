"""
Gather 集合通信示例
功能：将所有进程的数据收集到一个指定进程

使用方法：
# 在节点1上运行 (假设IP: 192.168.1.100)
python gather.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29500

# 在节点2上运行
python gather.py --rank 1 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
"""

import torch
import torch.distributed as dist
import argparse
import os


def run_gather(rank, world_size):
    """执行 gather 操作"""

    # 每个进程创建一个不同的张量
    tensor = torch.ones(3) * (rank + 1)
    print(f"[Rank {rank}] 发送的张量: {tensor}")

    # 准备接收列表（只有 rank 0 需要）
    if rank == 0:
        # 目标进程需要准备一个列表来接收所有进程的数据
        gather_list = [torch.zeros(3) for _ in range(world_size)]
        print(f"[Rank {rank}] 准备接收列表...")
    else:
        gather_list = None

    # 执行 gather 操作，将数据收集到 rank 0
    print(f"[Rank {rank}] 执行 gather...")
    dist.gather(tensor, gather_list, dst=0)

    # 只有 rank 0 能看到收集的结果
    if rank == 0:
        print(f"[Rank {rank}] Gather 后收集到的数据:")
        for i, t in enumerate(gather_list):
            print(f"  从 Rank {i}: {t}")
    else:
        print(f"[Rank {rank}] 数据已发送到 Rank 0")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Gather 示例")
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

    # 执行 gather 操作
    run_gather(args.rank, args.world_size)

    # 清理
    dist.destroy_process_group()
    print(f"[Rank {args.rank}] 完成")


if __name__ == "__main__":
    main()
