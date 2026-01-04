#!/bin/bash

# 简单分布式测试启动脚本
# 用于快速验证两个节点之间的通信

set -e

MASTER_IP="${1:-115.190.188.193}"
MASTER_PORT="${2:-2333}"
NNODES="${3:-2}"
NODE_RANK="${4:-0}"
NPROC_PER_NODE="${5:-1}"

echo ""
echo "================================"
echo "分布式通信测试"
echo "================================"
echo ""
echo "参数:"
echo "  MASTER_IP:      $MASTER_IP"
echo "  MASTER_PORT:    $MASTER_PORT"
echo "  NNODES:         $NNODES"
echo "  NODE_RANK:      $NODE_RANK"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo ""

echo "设置 NCCL 环境变量..."
echo ""

# 设置 NCCL 环境变量
# 你可以为整个脚本修改这里的值

# eth0 是最常见的，但也可能是 ens0, ens3, ens33, en0 等
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

# 禁用 InfiniBand，仅使用 TCP
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# 打开 NCCL 调试信息
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# 无缓冲输出
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "环境变量:"
echo "  NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "  NCCL_IB_DISABLE:    $NCCL_IB_DISABLE"
echo "  NCCL_DEBUG:         $NCCL_DEBUG"
echo "  PYTHONUNBUFFERED:   $PYTHONUNBUFFERED"
echo ""
echo "启动命令:"
echo "  torchrun \\"
echo "      --nnodes=$NNODES \\"
echo "      --nproc_per_node=$NPROC_PER_NODE \\"
echo "      --master_addr=$MASTER_IP \\"
echo "      --master_port=$MASTER_PORT \\"
echo "      --node_rank=$NODE_RANK \\"
echo "      test_distributed_simple.py"
echo ""
echo "================================"
echo ""

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    test_distributed_simple.py

echo ""
echo "================================"
echo "测试完成"
echo "================================"
echo ""
