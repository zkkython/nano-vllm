#!/bin/bash

# 快速检查网络接口的脚本

echo ""
echo "================================"
echo "网络接口检查"
echo "================================"
echo ""
echo "本机网络接口及 IP 地址:"
echo ""

# 尝试使用 ip addr show
if command -v ip &> /dev/null; then
    echo "【使用 ip addr show】"
    echo ""
    ip addr show | grep -E "^\d+:|inet "
else
    # 备选方案：使用 ifconfig
    echo "【使用 ifconfig】"
    echo ""
    ifconfig | grep -E "^[a-z]|inet " || echo "无法获取网络接口信息"
fi

echo ""
echo "================================"
echo ""
echo "常见网络接口名称:"
echo "  eth0    - 以太网（云环境）"
echo "  ens0    - 以太网（现代 Linux）"
echo "  ens3    - 虚拟化以太网"
echo "  ens33   - VMware 虚拟化以太网"
echo "  en0     - 以太网（macOS）"
echo "  wlan0   - 无线网络"
echo ""
echo "请从上面的接口列表中选择一个有 IP 地址的接口"
echo ""
echo "然后修改 run_distributed_test.sh 中的:"
echo "  export NCCL_SOCKET_IFNAME=\"eth0\""
echo ""
echo "改成你选择的接口，例如:"
echo "  export NCCL_SOCKET_IFNAME=\"ens0\""
echo ""
