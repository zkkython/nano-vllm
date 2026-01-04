#!/usr/bin/env python3
"""
多机TP并行启动脚本
此脚本帮助用户在多台机器上启动分布式推理
"""

import os
import sys
import subprocess
import argparse
from typing import List


def launch_multi_machine_job(
    master_addr: str,
    master_port: int,
    nnodes: int,
    nproc_per_node: int,
    model_path: str,
    script: str,
    script_args: List[str] = None,
):
    """
    启动多机分布式任务
    """
    print(f"Launching multi-machine job:")
    print(f"  Master address: {master_addr}:{master_port}")
    print(f"  Number of nodes: {nnodes}")
    print(f"  Number of processes per node: {nproc_per_node}")
    print(f"  Model path: {model_path}")
    print(f"  Script: {script}")

    # 构建启动命令
    cmd = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        script,
    ]

    if script_args:
        cmd.extend(script_args)

    print(f"Command: {' '.join(cmd)}")

    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching job: {e}")
        sys.exit(1)


def create_launch_script(
    master_addr: str,
    master_port: int,
    node_rank: int,
    world_size: int,
    model_path: str,
    output_file: str,
    nproc_per_node: int = None,
):
    """
    创建一个启动脚本，用于在每台机器上运行

    参数说明：
    - world_size: 总的GPU数量（所有节点的总和）
    - nproc_per_node: 当前节点的GPU数量（如果不指定则为None）
    """
    script_content = f'''#!/usr/bin/env python3
"""
自动生成的多机TP并行启动脚本
节点: {node_rank}, 世界大小: {world_size}
"""

import os
import torch
import torch.distributed as dist
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

# 设置分布式环境变量
os.environ["MASTER_ADDR"] = "{master_addr}"
os.environ["MASTER_PORT"] = "{master_port}"
os.environ["RANK"] = "{node_rank}"
os.environ["WORLD_SIZE"] = "{world_size}"

def main():
    print(f"Starting node {{os.environ['RANK']}}/{{os.environ['WORLD_SIZE']}}")
    print(f"Connecting to master: {{os.environ['MASTER_ADDR']}}:{{os.environ['MASTER_PORT']}}")
    print(f"Current node has {nproc_per_node} GPUs" if {nproc_per_node} else "")
    
    # 初始化LLM
    llm = LLM(
        model="{model_path}",
        tensor_parallel_size={world_size},
        master_addr="{master_addr}",
        master_port={master_port},
        node_rank={node_rank}
    )
    
    # 示例推理
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    prompts = ["Hello, how are you?", "What is machine learning?"]
    
    if {node_rank} == 0:  # 只在主节点执行推理
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            print(f"Generated: {{output['text'][:100]}}...")
    else:
        # 其他节点等待主节点的指令
        print(f"Node {{node_rank}} waiting for tasks...")
        # 这里可以实现其他节点的逻辑
    
    print(f"Node {{node_rank}} finished.")

if __name__ == "__main__":
    main()
'''

    with open(output_file, "w") as f:
        f.write(script_content)

    # 设置执行权限
    os.chmod(output_file, 0o755)
    print(f"Launch script created: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Launch multi-machine TP parallel job")
    parser.add_argument("--master_addr", type=str, required=True, help="Master address")
    parser.add_argument("--master_port", type=int, default=2333, help="Master port")
    parser.add_argument("--nnodes", type=int, default=2, help="Number of nodes")
    parser.add_argument(
        "--nproc_per_node", type=int, default=8, help="Number of processes per node"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument("--script", type=str, help="Script to run")
    parser.add_argument(
        "--create_script",
        action="store_true",
        help="Create launch script for each node",
    )
    parser.add_argument(
        "--output_script",
        type=str,
        default="node_launch_script.py",
        help="Output script name",
    )
    parser.add_argument(
        "--node_rank", type=int, default=0, help="Node rank (for create_script)"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        help="Total GPU count across all nodes (required for --create_script)",
    )
    parser.add_argument(
        "--current_node_gpus",
        type=int,
        help="GPU count on current node (optional, for logging)",
    )

    args = parser.parse_args()

    if args.create_script:
        if not args.world_size:
            print("Error: --world_size is required when using --create_script")
            print("Example: --world_size 10 (for 8 GPUs on Node 1 + 2 GPUs on Node 2)")
            sys.exit(1)

        create_launch_script(
            master_addr=args.master_addr,
            master_port=args.master_port,
            node_rank=args.node_rank,
            world_size=args.world_size,
            model_path=args.model_path,
            output_file=args.output_script,
            nproc_per_node=args.current_node_gpus,
        )
    else:
        if not args.script:
            print("Error: --script is required when not using --create_script")
            sys.exit(1)

        launch_multi_machine_job(
            master_addr=args.master_addr,
            master_port=args.master_port,
            nnodes=args.nnodes,
            nproc_per_node=args.nproc_per_node,
            model_path=args.model_path,
            script=args.script,
        )


if __name__ == "__main__":
    main()
