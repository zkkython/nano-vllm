#!/usr/bin/env python3
"""
多机TP并行启动脚本 - Node2
使用方式: torchrun --nnodes=2 --nproc_per_node=2 --master_addr=115.190.188.193 \
            --master_port=2333 --node_rank=1 node2_launch.py
"""

import os
import torch
import torch.distributed as dist
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def main():
    # 从环境变量获取分布式信息（由torchrun自动设置）
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "115.190.188.193")
    master_port = int(os.environ.get("MASTER_PORT", 2333))

    print(
        f"[Node2] Starting process rank={rank}, local_rank={local_rank}, world_size={world_size}"
    )
    print(f"[Node2] Connecting to master: {master_addr}:{master_port}")
    print(f"[Node2] Current node has 2 GPUs")

    # 只在rank 0上初始化LLM（此时rank 8）
    if rank == 0:
        # rank 0 在此脚本中是全局的rank 8
        # 但由于其他rank上已经初始化了LLM，这里不需要再执行
        print(f"[Node2] Rank {rank} (global rank 8) initialized.")
    else:
        # 其他rank等待
        print(f"[Node2] Rank {rank} waiting for tasks...")


if __name__ == "__main__":
    main()
