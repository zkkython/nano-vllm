#!/usr/bin/env python3
"""
最简单的网络连通性测试（不依赖 PyTorch）
用来诊断两个节点之间的 TCP 连接是否正常
"""

import socket
import sys
import time
import argparse


def is_master(node_rank):
    """判断是否为 Master 节点"""
    return node_rank == 0


def start_server(port, timeout=60):
    """Master 节点：启动 TCP 服务器，等待 Worker 连接"""
    print(f"\n[Master] 启动 TCP 服务器，监听端口 {port}...")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # 绑定到所有网络接口
        server_socket.bind(("0.0.0.0", port))
        server_socket.listen(5)
        print(f"[Master] ✓ 服务器已启动，等待 Worker 连接...")
        print(f"[Master] 监听地址: 0.0.0.0:{port}")
        
        # 等待连接，超时时间为 timeout 秒
        server_socket.settimeout(timeout)
        
        try:
            client_socket, client_address = server_socket.accept()
            print(f"[Master] ✓ 收到连接！来自: {client_address}")
            
            # 接收数据
            data = client_socket.recv(1024).decode()
            print(f"[Master] ✓ 收到数据: {data}")
            
            # 发送回应
            response = f"Master received: {data}"
            client_socket.send(response.encode())
            print(f"[Master] ✓ 发送回应: {response}")
            
            client_socket.close()
            print(f"[Master] ✓ 连接已关闭")
            print(f"[Master] ✓ 测试成功！")
            return True
            
        except socket.timeout:
            print(f"[Master] ✗ 等待连接超时（{timeout}秒）")
            print(f"[Master] 可能的原因:")
            print(f"  1. Worker 节点未启动")
            print(f"  2. Worker 无法访问 Master 的 IP")
            print(f"  3. 防火墙阻止了连接")
            print(f"  4. MASTER_ADDR 配置错误")
            return False
            
    finally:
        server_socket.close()


def connect_to_master(master_addr, master_port, timeout=60):
    """Worker 节点：连接到 Master TCP 服务器"""
    print(f"\n[Worker] 尝试连接到 Master: {master_addr}:{master_port}...")
    print(f"[Worker] 连接超时: {timeout} 秒")
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(timeout)
    
    try:
        # 尝试连接
        print(f"[Worker] 正在连接...")
        client_socket.connect((master_addr, master_port))
        print(f"[Worker] ✓ 连接成功！")
        
        # 发送数据
        message = f"Worker connected from {socket.gethostname()}"
        client_socket.send(message.encode())
        print(f"[Worker] ✓ 发送数据: {message}")
        
        # 接收回应
        response = client_socket.recv(1024).decode()
        print(f"[Worker] ✓ 收到回应: {response}")
        
        print(f"[Worker] ✓ 连接已关闭")
        print(f"[Worker] ✓ 测试成功！")
        return True
        
    except ConnectionRefusedError:
        print(f"[Worker] ✗ 连接被拒绝 (Connection refused)")
        print(f"[Worker] 可能的原因:")
        print(f"  1. Master 未启动或已停止")
        print(f"  2. Master 地址错误")
        print(f"  3. Master 端口错误")
        return False
        
    except socket.timeout:
        print(f"[Worker] ✗ 连接超时 ({timeout}秒)")
        print(f"[Worker] 可能的原因:")
        print(f"  1. Master 和 Worker 网络不通")
        print(f"  2. 防火墙阻止了连接")
        print(f"  3. Master 地址不可达")
        return False
        
    except socket.gaierror as e:
        print(f"[Worker] ✗ 无法解析 Master 地址: {e}")
        print(f"[Worker] 可能的原因:")
        print(f"  1. MASTER_ADDR 不是有效的 IP 地址或域名")
        return False
        
    except Exception as e:
        print(f"[Worker] ✗ 连接异常: {e}")
        return False
        
    finally:
        client_socket.close()


def main():
    parser = argparse.ArgumentParser(
        description="简单的 TCP 网络连通性测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

【Master（Node1）】
  python3 simple_network_test.py --role master --port 2333

【Worker（Node2）】
  python3 simple_network_test.py --role worker --master_addr 115.190.188.193 --master_port 2333
        """
    )
    
    parser.add_argument(
        "--role",
        type=str,
        choices=["master", "worker"],
        required=True,
        help="节点角色",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=2333,
        help="Master 服务器端口（仅在 role=master 时使用）",
    )
    
    parser.add_argument(
        "--master_addr",
        type=str,
        default="115.190.188.193",
        help="Master 地址（仅在 role=worker 时使用）",
    )
    
    parser.add_argument(
        "--master_port",
        type=int,
        default=2333,
        help="Master 端口（仅在 role=worker 时使用）",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="连接超时时间（秒），默认 60",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("TCP 网络连通性测试（不依赖 PyTorch）")
    print("=" * 80)
    
    if args.role == "master":
        print(f"\n[信息] 节点角色: Master")
        print(f"[信息] 监听端口: {args.port}")
        print(f"[信息] 超时时间: {args.timeout} 秒")
        
        success = start_server(args.port, timeout=args.timeout)
        
    else:  # worker
        print(f"\n[信息] 节点角色: Worker")
        print(f"[信息] Master 地址: {args.master_addr}")
        print(f"[信息] Master 端口: {args.master_port}")
        print(f"[信息] 超时时间: {args.timeout} 秒")
        
        # 等待一下，给 Master 时间启动
        print(f"[Worker] 等待 2 秒...")
        time.sleep(2)
        
        success = connect_to_master(
            args.master_addr,
            args.master_port,
            timeout=args.timeout
        )
    
    print()
    print("=" * 80)
    
    if success:
        print("[成功] 网络连通性测试通过！✓")
        print("\n现在可以尝试运行 PyTorch 分布式程序:")
        print("  bash run_distributed_test.sh ...")
        return 0
    else:
        print("[失败] 网络连通性测试失败！✗")
        print("\n故障排查:")
        print("  1. 检查两台机器是否能相互 ping 通")
        print("  2. 检查防火墙是否允许指定端口通信")
        print("  3. 检查 MASTER_ADDR 是否正确")
        print("  4. 确保 Master 在 Worker 之前启动")
        return 1
    
    print()


if __name__ == "__main__":
    sys.exit(main())
