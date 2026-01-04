"""
多机TP并行使用示例
此示例展示如何在多台机器上启动分布式推理
"""

import os
import torch
import torch.distributed as dist
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def run_single_machine_inference():
    """
    单机多卡推理示例
    """
    print("启动单机多卡推理...")
    
    llm = LLM(
        model="/path/to/your/model",  # 替换为实际模型路径
        tensor_parallel_size=8,       # 使用8张卡
        enforce_eager=False
    )
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Generated text: {output['text']}")
        print(f"Token IDs: {output['token_ids'][:10]}...")  # 只显示前10个token ID


def run_multi_machine_inference():
    """
    多机TP并行推理示例
    此函数演示如何配置多机环境
    """
    print("启动多机TP并行推理...")
    
    # 假设我们有2台机器，每台8张卡，总共16张卡
    # 每台机器需要设置相同的master地址和端口
    master_addr = "192.168.1.10"  # 主节点IP地址
    master_port = 2333            # 主节点端口
    tensor_parallel_size = 16     # 总共16张卡
    
    # 根据当前节点的rank来设置不同的配置
    # 这通常通过环境变量或命令行参数传入
    node_rank = int(os.getenv("NODE_RANK", "0"))
    
    print(f"Node rank: {node_rank}")
    print(f"Initializing with master_addr: {master_addr}, port: {master_port}")
    
    llm = LLM(
        model="/path/to/your/model",  # 替换为实际模型路径
        tensor_parallel_size=tensor_parallel_size,
        master_addr=master_addr,
        master_port=master_port,
        node_rank=node_rank,
        enforce_eager=False
    )
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    prompts = [
        "Explain the theory of relativity.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis."
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    if node_rank == 0:  # 只在主节点打印结果
        for output in outputs:
            print(f"Generated text: {output['text']}")
            print(f"Token IDs length: {len(output['token_ids'])}")


def setup_distributed_environment():
    """
    手动设置分布式环境的示例
    这个函数演示如何手动初始化分布式环境
    """
    # 通常这些值通过环境变量传入
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = int(os.getenv("MASTER_PORT", "2333"))
    
    print(f"Setting up distributed environment: rank={rank}, world_size={world_size}")
    print(f"Master: {master_addr}:{master_port}")
    
    # 初始化分布式环境
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )
    
    print(f"Distributed environment initialized on rank {rank}")


if __name__ == "__main__":
    print("nano-vllm 多机TP并行示例")
    print("=" * 50)
    
    # 选择运行模式
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        run_multi_machine_inference()
    else:
        print("运行单机示例...")
        run_single_machine_inference()