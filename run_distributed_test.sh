#!/bin/bash

# 简单分布式测试启动脚本
# 用于快速验证两个节点之间的通信

set -e

MASTER_IP="${1:-115.190.188.193}"
MASTER_PORT="${2:-2333}"
NNODES="${3:-2}"
NODE_RANK="${4:-0}"
NPROC_PER_NODE="${5:-1}"

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
