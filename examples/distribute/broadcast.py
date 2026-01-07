"""
Broadcast 集合通信示例
功能：将一个进程的数据广播到所有其他进程

使用方法：
# 在节点1上运行 (假设IP: 192.168.1.100)
python broadcast.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29500

# 在节点2上运行
python broadcast.py --rank 1 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
"""

import torch
import torch.distributed as dist
import argparse
import os


def run_broadcast(rank, world_size):
    """执行 broadcast 操作"""

    # 创建一个张量
    if rank == 0:
        # Rank 0 是源节点，初始化数据
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        print(f"[Rank {rank}] 源节点的初始张量: {tensor}")
    else:
        # 其他节点初始化为零
        tensor = torch.zeros(5)
        print(f"[Rank {rank}] 接收节点的初始张量: {tensor}")

    # 从 rank 0 广播数据到所有进程
    print(f"[Rank {rank}] 执行 broadcast...")
    dist.broadcast(tensor, src=0)
    print(f"[Rank {rank}] Broadcast 后的张量: {tensor}")

    # 验证所有进程的张量是否相同
    print(f"[Rank {rank}] 验证: tensor[0] = {tensor[0].item()}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Broadcast 示例")
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

    # 执行 broadcast 操作
    run_broadcast(args.rank, args.world_size)

    # 清理
    dist.destroy_process_group()
    print(f"[Rank {args.rank}] 完成")


if __name__ == "__main__":
    main()
