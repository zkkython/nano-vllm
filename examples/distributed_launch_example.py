"""
使用torchrun进行分布式启动的示例
"""

import os
import torch
import torch.distributed as dist
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def main():
    # 从环境变量获取分布式配置
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = int(os.getenv("MASTER_PORT", "2333"))

    print(
        f"Starting process: rank={rank}, world_size={world_size}, local_rank={local_rank}"
    )

    # 初始化分布式环境
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        print(f"Distributed environment initialized on rank {rank}")

    # 如果是主进程（rank 0），执行推理任务
    if rank == 0:
        print("Initializing LLM on rank 0...")
        llm = LLM(
            model="/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B",  # 替换为实际模型路径
            tensor_parallel_size=world_size,  # 使用所有可用的GPU
        )

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
        ]

        print("Generating...")
        outputs = llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            print(f"Prompt {i+1}: {prompts[i]}")
            print(f"Generated: {output['text']}")
            print("-" * 50)
    else:
        # 其他进程等待主进程完成
        print(f"Rank {rank} waiting for tasks...")
        # 在实际应用中，这些进程会参与模型计算
        if world_size > 1:
            # 同步等待
            torch.distributed.barrier()

    if world_size > 1:
        dist.destroy_process_group()
        print(f"Rank {rank} finished.")


if __name__ == "__main__":
    main()
