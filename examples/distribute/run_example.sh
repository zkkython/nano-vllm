#!/bin/bash
# PyTorch 分布式通信示例启动脚本
# 用法: ./run_example.sh <example_name> <rank> <world_size> <master_addr> [master_port] [backend]

EXAMPLE=$1
RANK=$2
WORLD_SIZE=$3
MASTER_ADDR=$4
MASTER_PORT=${5:-29500}
BACKEND=${6:-nccl}

if [ $# -lt 4 ]; then
    echo "用法: $0 <example_name> <rank> <world_size> <master_addr> [master_port] [backend]"
    echo ""
    echo "可用示例:"
    echo "  all_reduce     - 所有进程规约操作"
    echo "  broadcast      - 广播操作"
    echo "  gather         - 收集操作"
    echo "  scatter        - 分发操作"
    echo "  all_gather     - 全收集操作"
    echo "  reduce_scatter - 规约分发操作"
    echo "  send_recv      - 点对点通信"
    echo "  barrier        - 同步屏障"
    echo ""
    echo "示例:"
    echo "  # 在节点1 (master) 上运行 all_reduce"
    echo "  $0 all_reduce 0 2 192.168.1.100"
    echo ""
    echo "  # 在节点2 (worker) 上运行 all_reduce"
    echo "  $0 all_reduce 1 2 192.168.1.100"
    echo ""
    echo "  # 使用自定义端口和 gloo 后端"
    echo "  $0 broadcast 0 2 192.168.1.100 29501 gloo"
    exit 1
fi

# 检查示例文件是否存在
SCRIPT_DIR=$(dirname "$0")
EXAMPLE_FILE="${SCRIPT_DIR}/${EXAMPLE}.py"

if [ ! -f "$EXAMPLE_FILE" ]; then
    echo "错误: 找不到示例文件 ${EXAMPLE}.py"
    echo "可用示例: all_reduce, broadcast, gather, scatter, all_gather, reduce_scatter, send_recv, barrier"
    exit 1
fi

echo "========================================"
echo "启动 PyTorch 分布式示例: ${EXAMPLE}"
echo "========================================"
echo "Rank:        ${RANK}"
echo "World Size:  ${WORLD_SIZE}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "Backend:     ${BACKEND}"
echo "========================================"
echo ""

# 运行示例
python "${EXAMPLE_FILE}" \
    --rank "${RANK}" \
    --world-size "${WORLD_SIZE}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --backend "${BACKEND}"
