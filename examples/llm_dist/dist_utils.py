import argparse
import os

import torch
import torch.distributed as dist


def init_distributed(rank: int, world_size: int, master_addr: str, master_port: str, backend: str = "nccl") -> None:
    """初始化 PyTorch 分布式环境。

    与 examples/distribute 下的示例保持一致，显式传入 rank / world_size / master_addr / master_port，
    方便在两台节点上用简单命令行启动。
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    print(f"初始化进程组 - Rank: {rank}, World Size: {world_size}")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if backend == "nccl":
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] 使用 GPU {local_rank}")


def parse_args(default_world_size: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM 并行示例通用参数")
    parser.add_argument("--rank", type=int, required=True, help="当前进程的 rank")
    parser.add_argument("--world-size", type=int, default=default_world_size, help="总进程数")
    parser.add_argument("--master-addr", type=str, required=True, help="主节点 IP 地址")
    parser.add_argument("--master-port", type=str, default="29500", help="主节点端口")
    parser.add_argument("--backend", type=str, default="nccl", help="通信后端 (nccl/gloo)")
    return parser.parse_args()


def cleanup() -> None:
    dist.destroy_process_group()
