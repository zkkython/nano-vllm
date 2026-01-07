"""
Reduce-Scatter 集合通信示例
功能：先对所有进程的数据执行规约操作，然后将结果分散到所有进程

使用方法：
# 在节点1上运行 (假设IP: 192.168.1.100)
python reduce_scatter.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29500

# 在节点2上运行
python reduce_scatter.py --rank 1 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
"""

import torch
import torch.distributed as dist
import argparse
import os


def run_reduce_scatter(rank, world_size):
    """执行 reduce_scatter 操作"""

    # 每个进程准备一个输入列表，包含要发送给各个进程的数据
    # 例如：rank 0 发送 [1,1] 给 rank 0，[2,2] 给 rank 1
    input_list = [torch.ones(2) * (rank + 1) * (i + 1) for i in range(world_size)]

    print(f"[Rank {rank}] 输入列表:")
    for i, t in enumerate(input_list):
        print(f"  给 Rank {i}: {t}")

    # 准备输出张量
    output = torch.zeros(2)

    # 执行 reduce_scatter 操作（求和）
    print(f"[Rank {rank}] 执行 reduce_scatter...")
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

    # 每个进程接收对应位置的规约结果
    print(f"[Rank {rank}] Reduce-Scatter 后的结果: {output}")

    # 解释结果
    # Rank 0 接收: sum([rank0发给rank0, rank1发给rank0])
    # Rank 1 接收: sum([rank0发给rank1, rank1发给rank1])


def main():
    parser = argparse.ArgumentParser(description="PyTorch Reduce-Scatter 示例")
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

    # 执行 reduce_scatter 操作
    run_reduce_scatter(args.rank, args.world_size)

    # 清理
    dist.destroy_process_group()
    print(f"[Rank {args.rank}] 完成")


if __name__ == "__main__":
    main()
