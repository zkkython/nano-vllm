#!/usr/bin/env python3
"""
多机TP并行启动脚本 - Node1
使用方式: torchrun --nnodes=2 --nproc_per_node=8 --master_addr=115.190.188.193 \
            --master_port=2333 --node_rank=0 node1_launch.py
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
        f"[Node1] Starting process rank={rank}, local_rank={local_rank}, world_size={world_size}"
    )
    print(f"[Node1] Connecting to master: {master_addr}:{master_port}")
    print(f"[Node1] Current node has 8 GPUs")

    # 只在rank 0上初始化LLM并执行推理
    if rank == 0:
        # 初始化LLM
        llm = LLM(
            model="/data/Qwen3-8B/Qwen3-8B",
            tensor_parallel_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
        )

        # 示例推理
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        prompts = ["Hello, how are you?", "What is machine learning?"]

        print(f"[Node1] Rank 0 executing inference...")
        outputs = llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            print(f"[Node1] Generated [{i}]: {output['text'][:100]}...")

        print(f"[Node1] Rank 0 finished.")
    else:
        # 其他rank等待主rank的指令
        print(f"[Node1] Rank {rank} waiting for tasks...")
        # 这里可以实现其他rank的逻辑，或者继续等待


if __name__ == "__main__":
    main()
