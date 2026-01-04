#!/usr/bin/env python3
"""
多机分布式启动调试脚本
用于诊断 torchrun 和分布式通信问题
"""

import os
import sys
import socket
import torch
import torch.distributed as dist

def get_hostname():
    """获取当前机器的主机名"""
    return socket.gethostname()

def get_ip_address():
    """获取当前机器的 IP 地址"""
    try:
        # 这个方法获取本机 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def print_env_vars():
    """打印分布式相关的环境变量"""
    print("=" * 80)
    print("分布式环境变量:")
    print("=" * 80)
    
    vars_to_check = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
        "NODE_RANK", "NNODES", "NPROC_PER_NODE",
        "PYTHONUNBUFFERED", "CUDA_VISIBLE_DEVICES"
    ]
    
    for var in vars_to_check:
        value = os.environ.get(var, "NOT SET")
        print(f"  {var:25s} = {value}")
    
    print()

def print_network_info():
    """打印网络信息"""
    print("=" * 80)
    print("网络信息:")
    print("=" * 80)
    print(f"  主机名: {get_hostname()}")
    print(f"  本机IP: {get_ip_address()}")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")
    print()

def print_torch_info():
    """打印 PyTorch 分布式信息"""
    print("=" * 80)
    print("PyTorch 分布式信息:")
    print("=" * 80)
    print(f"  NCCL 可用: {torch.distributed.is_nccl_available()}")
    print(f"  GLOO 可用: {torch.distributed.is_gloo_available()}")
    print(f"  分布式初始化: {dist.is_available() and dist.is_initialized()}")
    
    if dist.is_available() and dist.is_initialized():
        print(f"  当前 Rank: {dist.get_rank()}")
        print(f"  World Size: {dist.get_world_size()}")
        print(f"  Backend: {dist.get_backend()}")
    print()

def test_connectivity():
    """测试与 Master 节点的网络连接"""
    print("=" * 80)
    print("网络连接测试:")
    print("=" * 80)
    
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "2333"))
    
    print(f"  尝试连接: {master_addr}:{master_port}")
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        result = s.connect_ex((master_addr, master_port))
        s.close()
        
        if result == 0:
            print(f"  ✓ 连接成功")
        else:
            print(f"  ✗ 连接失败 (错误代码: {result})")
    except Exception as e:
        print(f"  ✗ 连接异常: {e}")
    print()

def main():
    print("\n")
    print("*" * 80)
    print("多机分布式启动诊断工具")
    print("*" * 80)
    print()
    
    print_env_vars()
    print_network_info()
    print_torch_info()
    test_connectivity()
    
    print("=" * 80)
    print("诊断完成")
    print("=" * 80)
    print()
    
    # 打印建议
    print("故障排查建议:")
    print("-" * 80)
    
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    master_addr = os.environ.get("MASTER_ADDR")
    
    if rank == -1:
        print("⚠ 警告: RANK 未设置，请确保使用 torchrun 启动")
    if local_rank == -1:
        print("⚠ 警告: LOCAL_RANK 未设置，请确保使用 torchrun 启动")
    if master_addr is None:
        print("⚠ 警告: MASTER_ADDR 未设置，请检查 torchrun 参数")
    
    if master_addr:
        ip = get_ip_address()
        if master_addr != ip and rank != 0:
            print(f"✓ 正常: 这是 Worker 节点，指向 Master {master_addr}")
        elif master_addr == ip:
            print(f"✓ 正常: 这是 Master 节点")
    
    print()

if __name__ == "__main__":
    main()
