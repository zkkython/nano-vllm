#!/usr/bin/env python3
"""
最简单的分布式测试程序
用于验证两个节点之间是否能正常通信
"""

import os
import sys
import time
import socket
import torch
import torch.distributed as dist


def get_local_ip():
    """获取本机 IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    print("\n" + "=" * 80)
    print("简单分布式通信测试")
    print("=" * 80 + "\n")
    
    # 设置 NCCL 环境变量
    # 这些设置可以在命令行前设置，或者在这里设置
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        # 尝试自动检测，默认 eth0
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    
    if "NCCL_IB_DISABLE" not in os.environ:
        os.environ["NCCL_IB_DISABLE"] = "1"  # 禁用 InfiniBand
    
    if "NCCL_DEBUG" not in os.environ:
        os.environ["NCCL_DEBUG"] = "WARN"  # 显示警告及以上
    
    if "PYTHONUNBUFFERED" not in os.environ:
        os.environ["PYTHONUNBUFFERED"] = "1"  # 无缓冲输出
    
    # 打印环境变量
    print(f"[INFO] NCCL 环境变量:")
    print(f"  NCCL_SOCKET_IFNAME:  {os.environ.get('NCCL_SOCKET_IFNAME', 'NOT SET')}")
    print(f"  NCCL_IB_DISABLE:     {os.environ.get('NCCL_IB_DISABLE', 'NOT SET')}")
    print(f"  NCCL_DEBUG:          {os.environ.get('NCCL_DEBUG', 'NOT SET')}")
    print()
    
    # 获取环境变量
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 2333))
    
    print(f"[INFO] 环境变量:")
    print(f"  WORLD_SIZE:     {world_size}")
    print(f"  MASTER_ADDR:    {master_addr}")
    print(f"  MASTER_PORT:    {master_port}")
    print(f"  本机 IP:        {get_local_ip()}")
    print()
    
    # 检查 GPU
    print(f"[INFO] GPU 信息:")
    print(f"  CUDA 可用:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 数量:       {torch.cuda.device_count()}")
        print(f"   当前 GPU:      {local_rank}")
        print(f"  GPU 名称:       {torch.cuda.get_device_name(local_rank)}")
    print()
    
    # 检查分布式环境
    print(f"[INFO] 分布式环境:")
    print(f"  NCCL 可用:      {dist.is_nccl_available()}")
    print(f"  GLOO 可用:      {dist.is_gloo_available()}")
    print(f"  分布式初始化:   {dist.is_available()}")
    print()
    
    # 如果还未初始化，则初始化分布式环境
    if not dist.is_initialized():
        print(f"[步骤 1] 初始化分布式环境...")
        try:
            # 设置 CUDA 设备
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                print(f"  ✓ 设置 CUDA 设备: {local_rank}")
            
            # 初始化进程组
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank,
                world_size=world_size,
                timeout=300,  # 5 分钟超时
            )
            print(f"  ✓ 分布式环境初始化成功 (backend: {backend})")
            
        except Exception as e:
            print(f"  ✗ 初始化失败: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[步骤 1] 分布式环境已初始化")
    
    print()
    print(f"[步骤 2] 测试通信...")
    
    # 测试 1: 广播
    try:
        print(f"  测试 1: 广播 (broadcast)")
        data = torch.tensor([rank], dtype=torch.int32)
        if torch.cuda.is_available():
            data = data.cuda()
        
        dist.broadcast(data, src=0)
        print(f"    ✓ Rank {rank} 收到广播数据: {data.item()}")
        
    except Exception as e:
        print(f"    ✗ 广播失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 测试 2: 同步屏障
    try:
        print(f"  测试 2: 屏障同步 (barrier)")
        dist.barrier()
        print(f"    ✓ Rank {rank} 通过屏障")
        
    except Exception as e:
        print(f"    ✗ 屏障同步失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 测试 3: AllGather
    try:
        print(f"  测试 3: 全部收集 (allgather)")
        data = torch.tensor([rank], dtype=torch.int32)
        if torch.cuda.is_available():
            data = data.cuda()
        
        gathered = [torch.zeros_like(data) for _ in range(world_size)]
        dist.all_gather(gathered, data)
        result = [t.item() for t in gathered]
        print(f"    ✓ Rank {rank} 收集结果: {result}")
        
    except Exception as e:
        print(f"    ✗ AllGather 失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 测试 4: AllReduce
    try:
        print(f"  测试 4: 全部化简 (allreduce)")
        data = torch.tensor([rank + 1], dtype=torch.int32)
        if torch.cuda.is_available():
            data = data.cuda()
        
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        print(f"    ✓ Rank {rank} AllReduce 结果 (求和): {data.item()}")
        
    except Exception as e:
        print(f"    ✗ AllReduce 失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    print()
    print(f"[步骤 3] 等待所有进程同步...")
    
    # 最后一个屏障确保所有进程都完成
    dist.barrier()
    
    print()
    print(f"[成功] Rank {rank} 所有测试通过! ✓")
    print()
    
    # 清理
    dist.destroy_process_group()
    print(f"[完成] 分布式环境已清理")
    print()


if __name__ == "__main__":
    main()
