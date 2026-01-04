#!/usr/bin/env python3
"""
自动生成的多机TP并行启动脚本
节点: 1, 世界大小: 10
"""

import os
import torch
import torch.distributed as dist
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

# 设置分布式环境变量
os.environ["MASTER_ADDR"] = "115.190.188.194"
os.environ["MASTER_PORT"] = "2333"
os.environ["RANK"] = "1"
os.environ["WORLD_SIZE"] = "10"

def main():
    print(f"Starting node {os.environ['RANK']}/{os.environ['WORLD_SIZE']}")
    print(f"Connecting to master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    print(f"Current node has 2 GPUs" if 2 else "")
    
    # 初始化LLM
    llm = LLM(
        model="/data/Qwen3-8B/Qwen3-8B",
        tensor_parallel_size=10,
        master_addr="115.190.188.194",
        master_port=2333,
        node_rank=1
    )
    
    # 示例推理
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    prompts = ["Hello, how are you?", "What is machine learning?"]
    
    if 1 == 0:  # 只在主节点执行推理
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            print(f"Generated: {output['text'][:100]}...")
    else:
        # 其他节点等待主节点的指令
        print(f"Node {node_rank} waiting for tasks...")
        # 这里可以实现其他节点的逻辑
    
    print(f"Node {node_rank} finished.")

if __name__ == "__main__":
    main()
