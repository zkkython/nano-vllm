"""
Send-Recv 点对点通信示例
功能：进程之间的点对点数据发送和接收

使用方法：
# 在节点1上运行 (假设IP: 192.168.1.100)
python send_recv.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29500

# 在节点2上运行
python send_recv.py --rank 1 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
"""

import torch
import torch.distributed as dist
import argparse
import os


def run_send_recv(rank, world_size):
    """执行 send 和 recv 操作"""

    if rank == 0:
        # Rank 0 发送数据给 Rank 1
        tensor_to_send = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        print(f"[Rank {rank}] 发送张量给 Rank 1: {tensor_to_send}")
        dist.send(tensor=tensor_to_send, dst=1)
        print(f"[Rank {rank}] 数据已发送")

        # Rank 0 接收来自 Rank 1 的数据
        tensor_to_recv = torch.zeros(5)
        print(f"[Rank {rank}] 等待接收来自 Rank 1 的数据...")
        dist.recv(tensor=tensor_to_recv, src=1)
        print(f"[Rank {rank}] 接收到的张量: {tensor_to_recv}")

    elif rank == 1:
        # Rank 1 接收来自 Rank 0 的数据
        tensor_to_recv = torch.zeros(5)
        print(f"[Rank {rank}] 等待接收来自 Rank 0 的数据...")
        dist.recv(tensor=tensor_to_recv, src=0)
        print(f"[Rank {rank}] 接收到的张量: {tensor_to_recv}")

        # Rank 1 发送数据给 Rank 0
        tensor_to_send = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        print(f"[Rank {rank}] 发送张量给 Rank 0: {tensor_to_send}")
        dist.send(tensor=tensor_to_send, dst=0)
        print(f"[Rank {rank}] 数据已发送")


def run_isend_irecv(rank, world_size):
    """执行异步 isend 和 irecv 操作"""
    print(f"\n[Rank {rank}] === 测试异步通信 ===")

    if rank == 0:
        # Rank 0 异步发送和接收
        tensor_to_send = torch.tensor([100.0, 200.0, 300.0])
        tensor_to_recv = torch.zeros(3)

        print(f"[Rank {rank}] 异步发送: {tensor_to_send}")
        send_req = dist.isend(tensor=tensor_to_send, dst=1)

        print(f"[Rank {rank}] 异步接收...")
        recv_req = dist.irecv(tensor=tensor_to_recv, src=1)

        # 等待通信完成
        send_req.wait()
        recv_req.wait()
        print(f"[Rank {rank}] 异步接收完成: {tensor_to_recv}")

    elif rank == 1:
        # Rank 1 异步接收和发送
        tensor_to_send = torch.tensor([111.0, 222.0, 333.0])
        tensor_to_recv = torch.zeros(3)

        print(f"[Rank {rank}] 异步接收...")
        recv_req = dist.irecv(tensor=tensor_to_recv, src=0)

        print(f"[Rank {rank}] 异步发送: {tensor_to_send}")
        send_req = dist.isend(tensor=tensor_to_send, dst=0)

        # 等待通信完成
        recv_req.wait()
        send_req.wait()
        print(f"[Rank {rank}] 异步接收完成: {tensor_to_recv}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Send-Recv 示例")
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

    # 执行同步 send/recv 操作
    print(f"[Rank {args.rank}] === 测试同步通信 ===")
    run_send_recv(args.rank, args.world_size)

    # 执行异步 isend/irecv 操作
    run_isend_irecv(args.rank, args.world_size)

    # 清理
    dist.destroy_process_group()
    print(f"[Rank {args.rank}] 完成")


if __name__ == "__main__":
    main()
