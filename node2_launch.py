#!/usr/bin/env python3
"""
多机TP并行启动脚本 - Node2
使用方式: torchrun --nnodes=2 --nproc_per_node=2 --master_addr=115.190.188.193 \
            --master_port=2333 --node_rank=1 node2_launch.py
"""

import os
import torch
import torch.distributed as dist
import time
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def main():
    # 从环境变量获取分布式信息（由torchrun自动设置）
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "115.190.188.193")
    master_port = int(os.environ.get("MASTER_PORT", 2333))

    # 设置 NCCL 环境变量以改善跨节点通信
    # NCCL_SOCKET_IFNAME: 指定用于通信的网络接口
    # NCCL_DEBUG: 打开 NCCL 调试信息
    # NCCL_BLOCKING_WAIT: 设置阻塞等待时间
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # 或者 en0、ens33 等
    os.environ.setdefault("NCCL_DEBUG", "INFO")  # 打开 NCCL 调试消息
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")  # 使用1秒阻塞等待

    print(
        f"[Node2] Starting process rank={rank}, local_rank={local_rank}, world_size={world_size}"
    )
    print(f"[Node2] Connecting to master: {master_addr}:{master_port}")
    print(f"[Node2] Current node has 2 GPUs")
    # Node2上的所有rank都不初始化LLM
    # LLM已经在Node1的rank 0初始化，这里所有rank都等待指令
    print(
        f"[Node2] Rank {rank} (global rank {rank}) initialized."
        f"\n[DEBUG] ModelRunner.loop() will wait for broadcast signals from rank 0"
    )

if __name__ == "__main__":
    main()
