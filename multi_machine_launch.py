#!/usr/bin/env python3
"""
多机多卡启动脚本
支持通过命令行参数或环境变量启动多机分布式训练
使用方式：
  python multi_machine_launch.py --nnodes=2 --nproc_per_node=8 --master_addr=115.190.188.193 --node_rank=0
"""

import os
import sys
import argparse
import subprocess
import time
from typing import List, Optional


def launch_torchrun(
    nnodes: int,
    nproc_per_node: int,
    master_addr: str,
    master_port: int,
    node_rank: int,
    script: str,
    script_args: Optional[List[str]] = None,
):
    """使用 torchrun 启动多机分布式训练"""
    
    print("=" * 80)
    print("多机分布式启动参数:")
    print("=" * 80)
    print(f"  nnodes:           {nnodes}")
    print(f"  nproc_per_node:   {nproc_per_node}")
    print(f"  master_addr:      {master_addr}")
    print(f"  master_port:      {master_port}")
    print(f"  node_rank:        {node_rank}")
    print(f"  script:           {script}")
    if script_args:
        print(f"  script_args:      {' '.join(script_args)}")
    print("=" * 80)
    print()
    
    # 构建 torchrun 命令
    cmd = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        f"--node_rank={node_rank}",
        script,
    ]
    
    if script_args:
        cmd.extend(script_args)
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"错误：启动失败 (exit code: {e.returncode})", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n中断启动", file=sys.stderr)
        sys.exit(1)


def launch_direct(
    nnodes: int,
    nproc_per_node: int,
    master_addr: str,
    master_port: int,
    node_rank: int,
    script: str,
):
    """直接启动脚本（用于调试，不推荐用于生产环境）"""
    
    print("=" * 80)
    print("警告：使用直接启动模式（非 torchrun）")
    print("=" * 80)
    print(f"  nnodes:           {nnodes}")
    print(f"  nproc_per_node:   {nproc_per_node}")
    print(f"  master_addr:      {master_addr}")
    print(f"  master_port:      {master_port}")
    print(f"  node_rank:        {node_rank}")
    print(f"  script:           {script}")
    print("=" * 80)
    print()
    
    # 计算全局 world_size 和该节点的起始 rank
    world_size = nnodes * nproc_per_node
    start_rank = node_rank * nproc_per_node
    
    print(f"计算得到:")
    print(f"  world_size:       {world_size}")
    print(f"  start_rank:       {start_rank}")
    print()
    
    # 为每个本地进程设置环境变量并启动
    for local_rank in range(nproc_per_node):
        global_rank = start_rank + local_rank
        
        print(f"启动进程 {local_rank} (全局 rank {global_rank})...")
        
        # 设置环境变量
        env = os.environ.copy()
        env["RANK"] = str(global_rank)
        env["LOCAL_RANK"] = str(local_rank)
        env["WORLD_SIZE"] = str(world_size)
        env["MASTER_ADDR"] = master_addr
        env["MASTER_PORT"] = str(master_port)
        env["PYTHONUNBUFFERED"] = "1"  # 立即输出日志
        
        # 后台启动进程
        proc = subprocess.Popen(
            [sys.executable, script],
            env=env,
            preexec_fn=os.setsid,  # 创建新的进程组
        )
        
        # 小延迟以避免启动冲突
        if local_rank < nproc_per_node - 1:
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(
        description="多机多卡分布式启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 使用 torchrun（推荐）:
   python multi_machine_launch.py \\
       --nnodes=2 \\
       --nproc_per_node=8 \\
       --master_addr=115.190.188.193 \\
       --master_port=2333 \\
       --node_rank=0 \\
       --use_torchrun

2. 直接启动（调试用）:
   python multi_machine_launch.py \\
       --nnodes=2 \\
       --nproc_per_node=8 \\
       --master_addr=115.190.188.193 \\
       --master_port=2333 \\
       --node_rank=0
        """
    )
    
    parser.add_argument(
        "--nnodes",
        type=int,
        required=True,
        help="节点总数",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        required=True,
        help="每个节点上的进程数（GPU 数量）",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        required=True,
        help="Master 节点 IP 地址",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=2333,
        help="Master 节点通信端口（默认：2333）",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        required=True,
        help="当前节点的 rank（0 为 Master，其他为 Worker）",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="node1_launch.py",
        help="要运行的启动脚本（默认：node1_launch.py）",
    )
    parser.add_argument(
        "--use_torchrun",
        action="store_true",
        help="使用 torchrun 启动（推荐）",
    )
    parser.add_argument(
        "--script_args",
        type=str,
        nargs="*",
        help="传递给启动脚本的额外参数",
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.node_rank >= args.nnodes:
        print(
            f"错误：node_rank ({args.node_rank}) 必须小于 nnodes ({args.nnodes})",
            file=sys.stderr,
        )
        sys.exit(1)
    
    if not os.path.exists(args.script):
        print(f"错误：脚本文件不存在: {args.script}", file=sys.stderr)
        sys.exit(1)
    
    # 确保脚本路径是绝对路径
    script_path = os.path.abspath(args.script)
    
    # 根据选项选择启动方式
    if args.use_torchrun:
        launch_torchrun(
            nnodes=args.nnodes,
            nproc_per_node=args.nproc_per_node,
            master_addr=args.master_addr,
            master_port=args.master_port,
            node_rank=args.node_rank,
            script=script_path,
            script_args=args.script_args,
        )
    else:
        print("⚠️  警告：不使用 torchrun 可能会导致进程管理问题")
        print("建议使用 --use_torchrun 参数\n")
        
        launch_direct(
            nnodes=args.nnodes,
            nproc_per_node=args.nproc_per_node,
            master_addr=args.master_addr,
            master_port=args.master_port,
            node_rank=args.node_rank,
            script=script_path,
        )


if __name__ == "__main__":
    main()
