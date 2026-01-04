#!/bin/bash

# 多机多卡启动快速指南
# 这个脚本帮助用户快速启动多机分布式训练

set -e

# 配置参数
MASTER_NODE_IP="115.190.188.193"    # Node1 IP
MASTER_PORT="2333"
NNODES=2                             # 节点总数
NPROC_PER_NODE_NODE1=8               # Node1 GPU 数量
NPROC_PER_NODE_NODE2=2               # Node2 GPU 数量

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 检查环境
check_environment() {
    print_header "环境检查"
    
    # 检查 Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python3 已安装: $PYTHON_VERSION"
    else
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查 torchrun
    if python3 -m torch.distributed.launch --help &> /dev/null 2>&1 || \
       python3 -c "import torch; from torch.distributed import launch" &> /dev/null 2>&1; then
        print_success "PyTorch 分布式模块已安装"
    else
        print_warning "PyTorch 分布式模块未验证，继续..."
    fi
    
    # 检查 CUDA
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
        print_success "CUDA 已安装，检测到 $GPU_COUNT 个 GPU"
    else
        print_warning "未检测到 CUDA，继续..."
    fi
}

# 显示启动选项
show_options() {
    print_header "启动选项"
    
    echo "选择启动方式："
    echo ""
    echo "  1) 使用 torchrun 启动 Node1（推荐）"
    echo "  2) 使用 torchrun 启动 Node2（推荐）"
    echo "  3) 使用直接启动 Node1（调试用）"
    echo "  4) 使用直接启动 Node2（调试用）"
    echo "  5) 显示启动命令"
    echo "  6) 运行诊断工具"
    echo "  0) 退出"
    echo ""
    read -p "请选择 [0-6]: " choice
}

# 启动 Node1
launch_node1_torchrun() {
    print_header "启动 Node1（使用 torchrun）"
    
    CMD="torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE_NODE1 \
        --master_addr=$MASTER_NODE_IP \
        --master_port=$MASTER_PORT \
        --node_rank=0 \
        node1_launch.py"
    
    print_info "执行命令："
    echo "$CMD"
    echo ""
    
    eval $CMD
}

# 启动 Node2
launch_node2_torchrun() {
    print_header "启动 Node2（使用 torchrun）"
    
    CMD="torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE_NODE2 \
        --master_addr=$MASTER_NODE_IP \
        --master_port=$MASTER_PORT \
        --node_rank=1 \
        node2_launch.py"
    
    print_info "执行命令："
    echo "$CMD"
    echo ""
    
    eval $CMD
}

# 启动 Node1（直接）
launch_node1_direct() {
    print_header "启动 Node1（直接启动）"
    
    CMD="python3 multi_machine_launch.py \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE_NODE1 \
        --master_addr=$MASTER_NODE_IP \
        --master_port=$MASTER_PORT \
        --node_rank=0 \
        --script=node1_launch.py"
    
    print_info "执行命令："
    echo "$CMD"
    echo ""
    
    eval $CMD
}

# 启动 Node2（直接）
launch_node2_direct() {
    print_header "启动 Node2（直接启动）"
    
    CMD="python3 multi_machine_launch.py \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE_NODE2 \
        --master_addr=$MASTER_NODE_IP \
        --master_port=$MASTER_PORT \
        --node_rank=1 \
        --script=node2_launch.py"
    
    print_info "执行命令："
    echo "$CMD"
    echo ""
    
    eval $CMD
}

# 显示启动命令
show_commands() {
    print_header "启动命令参考"
    
    echo "Node1（Master）- 使用 torchrun:"
    echo "  torchrun \\"
    echo "      --nnodes=$NNODES \\"
    echo "      --nproc_per_node=$NPROC_PER_NODE_NODE1 \\"
    echo "      --master_addr=$MASTER_NODE_IP \\"
    echo "      --master_port=$MASTER_PORT \\"
    echo "      --node_rank=0 \\"
    echo "      node1_launch.py"
    echo ""
    
    echo "Node1（Master）- 直接启动:"
    echo "  python3 multi_machine_launch.py \\"
    echo "      --nnodes=$NNODES \\"
    echo "      --nproc_per_node=$NPROC_PER_NODE_NODE1 \\"
    echo "      --master_addr=$MASTER_NODE_IP \\"
    echo "      --master_port=$MASTER_PORT \\"
    echo "      --node_rank=0"
    echo ""
    
    echo "Node2（Worker）- 使用 torchrun:"
    echo "  torchrun \\"
    echo "      --nnodes=$NNODES \\"
    echo "      --nproc_per_node=$NPROC_PER_NODE_NODE2 \\"
    echo "      --master_addr=$MASTER_NODE_IP \\"
    echo "      --master_port=$MASTER_PORT \\"
    echo "      --node_rank=1 \\"
    echo "      node2_launch.py"
    echo ""
    
    echo "Node2（Worker）- 直接启动:"
    echo "  python3 multi_machine_launch.py \\"
    echo "      --nnodes=$NNODES \\"
    echo "      --nproc_per_node=$NPROC_PER_NODE_NODE2 \\"
    echo "      --master_addr=$MASTER_NODE_IP \\"
    echo "      --master_port=$MASTER_PORT \\"
    echo "      --node_rank=1"
    echo ""
}

# 运行诊断
run_diagnostics() {
    print_header "运行诊断工具"
    
    python3 debug_distributed.py
}

# 主循环
main() {
    print_header "多机多卡启动管理器"
    
    check_environment
    
    while true; do
        show_options
        
        case $choice in
            1)
                launch_node1_torchrun
                ;;
            2)
                launch_node2_torchrun
                ;;
            3)
                launch_node1_direct
                ;;
            4)
                launch_node2_direct
                ;;
            5)
                show_commands
                ;;
            6)
                run_diagnostics
                ;;
            0)
                print_info "退出"
                exit 0
                ;;
            *)
                print_error "无效选择"
                ;;
        esac
    done
}

# 如果提供了命令行参数，直接执行
if [ $# -eq 0 ]; then
    main
else
    case "$1" in
        node1-torchrun)
            launch_node1_torchrun
            ;;
        node2-torchrun)
            launch_node2_torchrun
            ;;
        node1-direct)
            launch_node1_direct
            ;;
        node2-direct)
            launch_node2_direct
            ;;
        commands)
            show_commands
            ;;
        diagnose)
            run_diagnostics
            ;;
        *)
            echo "用法: $0 [node1-torchrun|node2-torchrun|node1-direct|node2-direct|commands|diagnose]"
            exit 1
            ;;
    esac
fi
