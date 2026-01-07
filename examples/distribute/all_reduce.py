"""
All-Reduce 集合通信示例
功能：所有进程对数据执行规约操作（如求和），然后将结果分发给所有进程

使用方法：
# 在节点1上运行 (假设IP: 192.168.1.100)
python all_reduce.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29500

# 在节点2上运行
python all_reduce.py --rank 1 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
"""

import torch
import torch.distributed as dist
import argparse
import os


def run_all_reduce(rank, world_size):
    """执行 all_reduce 操作"""

    # 创建一个张量，每个进程的值不同
    tensor = torch.ones(3) * (rank + 1)
    print(f"[Rank {rank}] 初始张量: {tensor}")

    # All-Reduce: 求和操作
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] All-Reduce 后 (SUM): {tensor}")

    # 再次测试 MAX 操作
    tensor2 = torch.ones(3) * (rank + 1)
    dist.all_reduce(tensor2, op=dist.ReduceOp.MAX)
    print(f"[Rank {rank}] All-Reduce 后 (MAX): {tensor2}")

    # 测试 MIN 操作
    tensor3 = torch.ones(3) * (rank + 1)
    dist.all_reduce(tensor3, op=dist.ReduceOp.MIN)
    print(f"[Rank {rank}] All-Reduce 后 (MIN): {tensor3}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch All-Reduce 示例")
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

    # 执行 all_reduce 操作
    run_all_reduce(args.rank, args.world_size)

    # 清理
    dist.destroy_process_group()
    print(f"[Rank {args.rank}] 完成")


if __name__ == "__main__":
    main()
