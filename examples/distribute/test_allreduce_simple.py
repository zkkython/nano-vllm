#!/usr/bin/env python3
"""
双机PyTorch All-Reduce简化测试脚本
可以通过命令行参数指定角色

用法：
  主节点: python test_allreduce_simple.py --role master
  从节点: python test_allreduce_simple.py --role worker
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist


def test_allreduce(role):
    """执行All-Reduce测试"""
    # 配置
    MASTER_ADDR = '115.190.188.193'
    WORKER_ADDR = '115.190.188.194'
    MASTER_PORT = '29500'
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['WORLD_SIZE'] = '2'
    os.environ['RANK'] = '0' if role == 'master' else '1'
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 打印配置信息
    print("=" * 60)
    print(f"{'主节点' if role == 'master' else '从节点'} - All Reduce测试")
    print("=" * 60)
    print(f"主节点地址: {MASTER_ADDR}:{MASTER_PORT}")
    print(f"当前角色: {role}")
    print(f"Rank: {rank}, World Size: {world_size}")
    print("=" * 60)
    
    # 初始化分布式进程组
    print("\n[1] 初始化分布式进程组...")
    try:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        print("✓ 分布式进程组初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("\n提示：如果没有GPU，请修改backend为'gloo'")
        return
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ 使用CPU模式")
    
    # 创建测试张量
    print("\n[2] 创建测试张量...")
    if role == 'master':
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    else:
        tensor = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], device=device)
    print(f"原始张量: {tensor}")
    
    # 执行All-Reduce
    print("\n[3] 执行All-Reduce (SUM)...")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"✓ All-Reduce完成")
    print(f"结果张量: {tensor}")
    print(f"期望值: [11., 22., 33., 44., 55.]")
    
    # 验证结果
    expected = torch.tensor([11., 22., 33., 44., 55.], device=device)
    if torch.allclose(tensor, expected):
        print("✓ 结果正确！")
    else:
        print("❌ 结果不正确！")
    
    # 同步
    print("\n[4] 同步所有进程...")
    dist.barrier()
    print("✓ 同步完成")
    
    # 清理
    print("\n[5] 清理资源...")
    dist.destroy_process_group()
    print("✓ 分布式进程组已销毁")
    
    print("\n" + "=" * 60)
    print("✓ 测试完成！")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='双机PyTorch All-Reduce测试')
    parser.add_argument('--role', type=str, required=True, 
                       choices=['master', 'worker'],
                       help='节点角色: master(主节点) 或 worker(从节点)')
    
    args = parser.parse_args()
    
    try:
        test_allreduce(args.role)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
