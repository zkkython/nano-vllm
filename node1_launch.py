#!/usr/bin/env python3
"""
多机TP并行启动脚本 - Node1
使用方式: torchrun --nnodes=2 --nproc_per_node=8 --master_addr=192.168.0.163 \
            --master_port=2333 --node_rank=0 node1_launch.py
"""

import os
import time
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

model_path = "/data/ds-671"


def main():
    # 从环境变量获取分布式信息（由torchrun自动设置）
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "192.168.0.163")
    master_port = int(os.environ.get("MASTER_PORT", 2333))

    # 设置 NCCL 环境变量以改善跨节点通信
    # NCCL_SOCKET_IFNAME: 指定用于通信的网络接口
    # NCCL_DEBUG: 打开 NCCL 调试信息
    # NCCL_BLOCKING_WAIT: 设置阻塞等待时间
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # 或者 en0、ens33 等
    os.environ.setdefault("NCCL_DEBUG", "INFO")  # 打开 NCCL 调试消息
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")  # 使用1秒阻塞等待

    print(
        f"[Node1] Starting process rank={rank}, local_rank={local_rank}, world_size={world_size}"
    )
    print(f"[Node1] Connecting to master: {master_addr}:{master_port}")
    print(f"[Node1] Current node has 8 GPUs")
    # 只在rank 0上初始化LLM并执行推理
    if rank == 0:
        print(f"[Node1] Rank 0 initializing LLM...")
        # 初始化LLM
        llm = LLM(
            model_path,
            max_model_len=2000,
            max_num_batched_tokens=2000,
            max_num_seqs=1,
            gpu_memory_utilization=0.88,
            enforce_eager=True,
            tensor_parallel_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
        )

        # 等待所有Worker rank完成初始化
        # 这个延迟是为了让Worker节点上的其他rank完成ModelRunner初始化
        print(f"[Node1] Rank 0 waiting for all ranks to initialize...")
        time.sleep(3)  # 给Worker rank足够的时间来初始化

        # 示例推理
        sampling_params = SamplingParams(temperature=0.8)
        prompts = ["Hello, how are you?", "What is machine learning?"]

        print(f"[Node1] Rank 0 executing inference...")
        outputs = llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            print(f"[Node1] Generated [{i}]: {output['text'][:100]}...")

        print(f"[Node1] Rank 0 finished.")
    else:
        # 其他rank等待主rank的指令
        print(
            f"[Node1] Rank {rank} waiting for tasks from rank 0..."
            f"\n[DEBUG] ModelRunner.loop() will wait for broadcast signals"
        )


if __name__ == "__main__":
    main()
