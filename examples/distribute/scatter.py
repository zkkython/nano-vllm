"""
Scatter 集合通信示例
功能：将一个进程的数据分发到所有进程

使用方法：
# 在节点1上运行 (假设IP: 192.168.1.100)
python scatter.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29500

# 在节点2上运行
python scatter.py --rank 1 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
"""

import torch
import torch.distributed as dist
import argparse
import os


def run_scatter(rank, world_size):
    """执行 scatter 操作"""

    # 准备发送列表（只有 rank 0 需要）
    if rank == 0:
        # 源进程准备要分发的数据列表
        scatter_list = [
            torch.tensor([1.0, 2.0, 3.0]),  # 发送给 rank 0
            torch.tensor([4.0, 5.0, 6.0]),  # 发送给 rank 1
        ]
        print(f"[Rank {rank}] 准备分发的数据:")
        for i, t in enumerate(scatter_list):
            print(f"  发送给 Rank {i}: {t}")
    else:
        scatter_list = None

    # 准备接收张量
    output = torch.zeros(3)
    print(f"[Rank {rank}] 接收前的张量: {output}")

    # 执行 scatter 操作
    print(f"[Rank {rank}] 执行 scatter...")
    dist.scatter(output, scatter_list, src=0)

    # 所有进程都会接收到数据
    print(f"[Rank {rank}] Scatter 后接收到的张量: {output}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Scatter 示例")
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

    # 执行 scatter 操作
    run_scatter(args.rank, args.world_size)

    # 清理
    dist.destroy_process_group()
    print(f"[Rank {args.rank}] 完成")


if __name__ == "__main__":
    main()
