#!/usr/bin/env python3
"""
诊断网络接口和 NCCL 配置
"""

import os
import sys
import socket
import subprocess


def get_all_network_interfaces():
    """获取所有网络接口"""
    try:
        # 使用 ip addr 获取所有接口
        result = subprocess.run(
            ["ip", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout
    except Exception as e:
        print(f"获取网络接口失败: {e}")
        return None


def get_network_interfaces_ifconfig():
    """使用 ifconfig 获取网络接口"""
    try:
        result = subprocess.run(
            ["ifconfig"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout
    except Exception as e:
        print(f"ifconfig 不可用: {e}")
        return None


def test_socket_connection(host, port):
    """测试 socket 连接"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        result = s.connect_ex((host, port))
        s.close()
        return result == 0
    except Exception as e:
        return False


def main():
    print("\n" + "=" * 80)
    print("网络诊断工具")
    print("=" * 80 + "\n")
    
    # 获取本机 IP
    print("[步骤 1] 获取本机 IP 地址")
    print("-" * 80)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"本机 IP: {local_ip}")
    except Exception as e:
        print(f"无法获取本机 IP: {e}")
        local_ip = None
    
    print()
    
    # 获取所有网络接口
    print("[步骤 2] 网络接口信息")
    print("-" * 80)
    
    interfaces = get_all_network_interfaces()
    if interfaces:
        print(interfaces)
    else:
        interfaces = get_network_interfaces_ifconfig()
        if interfaces:
            print(interfaces)
        else:
            print("无法获取网络接口信息")
    
    print()
    
    # 检查 NCCL 环境变量
    print("[步骤 3] NCCL 环境变量")
    print("-" * 80)
    
    nccl_vars = [
        "NCCL_SOCKET_IFNAME",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG_FILE",
    ]
    
    for var in nccl_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"  {var:30s} = {value}")
    
    print()
    
    # 推荐的网络接口名称
    print("[步骤 4] 推荐的网络接口配置")
    print("-" * 80)
    
    print("根据常见的网络接口，建议按顺序尝试:")
    print()
    
    interface_priority = [
        ("eth0", "以太网（常见在云环境中）"),
        ("ens0", "以太网（常见在现代 Linux 中）"),
        ("ens3", "以太网虚拟化接口"),
        ("ens33", "以太网虚拟化接口（VMware）"),
        ("en0", "以太网（macOS/BSD）"),
        ("en1", "以太网（macOS/BSD）"),
        ("wlan0", "无线网络"),
    ]
    
    for i, (iface, description) in enumerate(interface_priority, 1):
        print(f"  {i}. {iface:10s} - {description}")
    
    print()
    print("设置方法:")
    print("  export NCCL_SOCKET_IFNAME=eth0  # 改成实际的接口名")
    print()
    
    # 诊断建议
    print("[步骤 5] 诊断建议")
    print("-" * 80)
    
    print("""
错误 'The hostname of the client socket cannot be retrieved. err=-3' 通常是由于:

1. ✗ NCCL_SOCKET_IFNAME 指定了不存在的网络接口
2. ✗ NCCL 尝试使用了不正确的网络接口
3. ✗ 网络接口没有分配 IP 地址

解决方案:

方案 1: 先找出正确的网络接口
  # 查看所有接口及其 IP
  ip addr show
  # 或者
  ifconfig

方案 2: 设置 NCCL_SOCKET_IFNAME 环境变量（在 torchrun 前执行）
  export NCCL_SOCKET_IFNAME=eth0
  torchrun ... test_distributed_simple.py

方案 3: 使用更详细的 NCCL 调试信息
  export NCCL_DEBUG=TRACE
  export NCCL_DEBUG_FILE=/tmp/nccl.log
  torchrun ... test_distributed_simple.py

方案 4: 禁用 InfiniBand（如果不需要）
  export NCCL_IB_DISABLE=1
  torchrun ... test_distributed_simple.py

方案 5: 同时尝试以上多个方案
  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_IB_DISABLE=1
  export NCCL_DEBUG=INFO
  torchrun ... test_distributed_simple.py
""")
    
    print()


if __name__ == "__main__":
    main()
