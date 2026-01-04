# Socket 错误解决方案

## 问题

```
[W104 21:27:59.520842647 socket.cpp:209] [c10d] The hostname of the client socket cannot be retrieved. err=-3
```

虽然两台机器能 ping 通，但 NCCL 无法获取 socket 的主机名。

## 根本原因

NCCL（NVIDIA Collective Communications Library）默认尝试使用 `eth0` 网络接口，但在你的环境中：
- 可能不存在 `eth0` 接口
- 或者使用了不同名称的网络接口（如 `ens0`、`ens33`、`en0` 等）
- NCCL 无法正确识别和绑定到网络接口，导致 socket 操作失败

## 解决步骤

### 第一步：检查你的网络接口名称

```bash
# 在 Node1 上执行
bash CHECK_NETWORK.sh
```

或者手动检查：

```bash
# Linux
ip addr show

# 或者
ifconfig
```

**看起来像这样：**
```
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500
    inet 115.190.188.193/24 scope global eth0
3: eth1: <BROADCAST,MULTICAST> mtu 1500
```

**记住有 IP 地址的接口名称！** 比如 `eth0`、`ens0`、`ens33` 等

### 第二步：修改启动脚本

编辑 `run_distributed_test.sh`，修改第 28 行：

```bash
# 原来的（默认）
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

# 改成你的接口名称，比如：
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens0}"

# 或者
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens33}"

# 或者
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-en0}"
```

### 第三步：重新启动测试

**在 Node1 上：**
```bash
cd /root/mingtong/aiwork/nano-vllm
bash run_distributed_test.sh 115.190.188.193 2333 2 0 1
```

**在 Node2 上（等待 3 秒后）：**
```bash
cd /root/mingtong/aiwork/nano-vllm
bash run_distributed_test.sh 115.190.188.193 2333 2 1 1
```

## 如果还是不行

### 方案 A：禁用 InfiniBand，只用 TCP

```bash
# 在 Node1 和 Node2 上都设置
export NCCL_IB_DISABLE=1

bash run_distributed_test.sh ...
```

这告诉 NCCL 不要尝试使用 InfiniBand，只用基于 TCP 的通信。

### 方案 B：打开详细的 NCCL 调试信息

```bash
# 查看更详细的错误信息
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=/tmp/nccl_debug.log

bash run_distributed_test.sh ...

# 检查日志
cat /tmp/nccl_debug.log
```

### 方案 C：同时尝试多个选项

```bash
# Node1
export NCCL_SOCKET_IFNAME=eth0      # 改成你的接口
export NCCL_IB_DISABLE=1            # 禁用 InfiniBand
export NCCL_DEBUG=INFO              # 打开调试
bash run_distributed_test.sh 115.190.188.193 2333 2 0 1

# Node2（等待后）
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
bash run_distributed_test.sh 115.190.188.193 2333 2 1 1
```

## 所有可用的 NCCL 环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| NCCL_SOCKET_IFNAME | eth0, ens0, en0 等 | 网络接口名称 |
| NCCL_IB_DISABLE | 0 或 1 | 禁用 InfiniBand（1=禁用） |
| NCCL_DEBUG | WARN, INFO, TRACE | 调试级别 |
| NCCL_DEBUG_FILE | /path/to/file | 调试日志文件 |
| NCCL_P2P_DISABLE | 0 或 1 | 禁用点对点通信 |

## 常见网络接口

### 按环境类型

**云环境（AWS、Azure、GCP 等）**
- eth0 ✓

**本地 Linux 服务器**
- ens0
- ens1
- ens33

**VMware 虚拟机**
- eth0
- ens33

**macOS**
- en0
- en1
- en2

**Docker 容器**
- eth0

### 如何判断哪个是实际使用的接口

1. 查看哪个接口有 IP 地址（不是 127.0.0.1）
2. 查看哪个接口的 IP 与 `--master_addr` 对应
3. 确认该接口是 UP 状态（`<UP,RUNNING>`）

## 调试技巧

### 1. 如果显示 "Cannot get hostname"

说明 NCCL 找不到网络接口，尝试：
```bash
export NCCL_SOCKET_IFNAME=eth0  # 改成正确的接口
export NCCL_IB_DISABLE=1         # 禁用 InfiniBand
```

### 2. 如果显示 "Connection refused"

说明 Node1 还没启动或已退出，尝试：
1. 确认 Node1 正在运行
2. 使用正确的 MASTER_ADDR

### 3. 如果显示 "Address already in use"

端口被占用，尝试：
```bash
bash run_distributed_test.sh 115.190.188.193 2334 2 0 1  # 改用 2334 端口
```

### 4. 使用 tcpdump 查看网络流量

```bash
# 在 Node1 或 Node2 上监听
sudo tcpdump -i eth0 -n "tcp port 2333"

# 看是否有网络包通过
```

## 成功的标志

如果你看到类似这样的输出，说明修复成功：

```
[INFO] NCCL 环境变量:
  NCCL_SOCKET_IFNAME:  eth0
  NCCL_IB_DISABLE:     1
  NCCL_DEBUG:          WARN

[步骤 1] 初始化分布式环境...
  ✓ 设置 CUDA 设备: 0
  ✓ 分布式环境初始化成功 (backend: nccl)

[步骤 2] 测试通信...
  测试 1: 广播 (broadcast)
  ✓ Rank 0 收到广播数据: 0
```

## 下一步

一旦简单测试通过，你可以：

1. 应用同样的设置到 `node1_launch.py` 和 `node2_launch.py`
2. 测试更多 GPU 和进程
3. 运行完整的 LLM 推理

## 文件参考

- `CHECK_NETWORK.sh` - 检查网络接口
- `run_distributed_test.sh` - 启动分布式测试
- `test_distributed_simple.py` - 测试程序
- `diagnose_network.py` - 详细的网络诊断工具
