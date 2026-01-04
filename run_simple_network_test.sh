#!/bin/bash

# 最简单的网络测试启动脚本
# 这个脚本完全不依赖 PyTorch，只测试 TCP 连接

set -e

echo ""
echo "=========================================="
echo "TCP 网络连通性测试（不依赖 PyTorch）"
echo "=========================================="
echo ""

# 默认值
MASTER_ADDR="${1:-115.190.188.193}"
MASTER_PORT="${2:-2333}"
ROLE="${3:-master}"  # master 或 worker
TIMEOUT="${4:-60}"

echo "参数:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  ROLE:        $ROLE"
echo "  TIMEOUT:     $TIMEOUT 秒"
echo ""

if [ "$ROLE" = "master" ]; then
    echo "在 Master 节点上启动..."
    echo ""
    python3 simple_network_test.py \
        --role master \
        --port $MASTER_PORT \
        --timeout $TIMEOUT
        
elif [ "$ROLE" = "worker" ]; then
    echo "在 Worker 节点上启动..."
    echo ""
    python3 simple_network_test.py \
        --role worker \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --timeout $TIMEOUT
        
else
    echo "错误: ROLE 必须是 'master' 或 'worker'"
    echo ""
    echo "用法:"
    echo "  Master: bash simple_network_test.sh 115.190.188.193 2333 master"
    echo "  Worker: bash simple_network_test.sh 115.190.188.193 2333 worker"
    exit 1
fi

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo ""
