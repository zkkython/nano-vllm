# 基础连通性测试

## 问题

PyTorch 分布式测试卡住了，这可能是更深层的网络问题，甚至不是 NCCL 的问题。

## 诊断步骤

### 第一步：基础 Ping 测试

首先确认两台机器能基本通信：

**在 Node2 上执行：**
```bash
ping -c 4 115.190.188.193
```

**预期输出：**
```
PING 115.190.188.193 (115.190.188.193) 56(84) bytes of data.
64 bytes from 115.190.188.193: icmp_seq=1 ttl=64 time=10.2 ms
64 bytes from 115.190.188.193: icmp_seq=2 ttl=64 time=9.8 ms
...
--- 115.190.188.193 statistics ---
4 packets transmitted, 4 received, 0% packet loss
```

**如果 ping 失败：**
- 检查网络配置
- 检查防火墙规则
- 检查 IP 地址是否正确

### 第二步：使用 TCP 连接测试（不依赖 PyTorch）

这个测试完全使用 Python 标准库，不依赖 PyTorch，来诊断基础的 TCP 连接问题。

#### 在 Node1（Master）上启动：

```bash
cd /root/mingtong/aiwork/nano-vllm

python3 simple_network_test.py --role master --port 2333
```

**预期输出：**
```
[Master] 启动 TCP 服务器，监听端口 2333...
[Master] ✓ 服务器已启动，等待 Worker 连接...
[Master] 监听地址: 0.0.0.0:2333
<等待 Worker 连接...>
```

#### 在 Node2（Worker）上启动（等待 3 秒后）：

```bash
cd /root/mingtong/aiwork/nano-vllm

python3 simple_network_test.py --role worker --master_addr 115.190.188.193 --master_port 2333
```

**预期输出：**
```
[Worker] 尝试连接到 Master: 115.190.188.193:2333...
[Worker] 正在连接...
[Worker] ✓ 连接成功！
[Worker] ✓ 发送数据: Worker connected from ...
[Worker] ✓ 收到回应: Master received: ...
[Worker] ✓ 测试成功！
```

### 第三步：使用启动脚本（推荐）

**在 Node1 上：**
```bash
bash run_simple_network_test.sh 115.190.188.193 2333 master 60
```

**在 Node2 上：**
```bash
bash run_simple_network_test.sh 115.190.188.193 2333 worker 60
```

参数说明：
- `115.190.188.193` - Master IP
- `2333` - 端口
- `master` / `worker` - 角色
- `60` - 超时时间（秒）

## 可能的错误和解决方案

### 错误 1: Connection refused

```
[Worker] ✗ 连接被拒绝 (Connection refused)
```

**原因：** Master 未启动或已停止

**解决：**
1. 确认 Node1 上的 Master 已启动且没有退出
2. 检查 Master 是否在正确的端口监听
3. 使用 `netstat -tuln` 检查端口是否打开

### 错误 2: Connection timeout

```
[Master] ✗ 等待连接超时（60秒）
或
[Worker] ✗ 连接超时 (60秒)
```

**原因：** Master 和 Worker 网络不通

**解决：**
1. 确认两台机器能互相 ping 通
2. 检查防火墙配置
3. 尝试增加超时时间：`--timeout 120`

### 错误 3: Cannot resolve hostname

```
[Worker] ✗ 无法解析 Master 地址
```

**原因：** Master 地址格式错误

**解决：**
1. 使用 IP 地址而不是域名
2. 确保 IP 地址格式正确

## 故障排查清单

| 检查项 | 命令 | 预期结果 |
|-------|------|---------|
| Ping 通 | `ping -c 4 115.190.188.193` | 0% 丢包 |
| 端口开放 | `netstat -tuln \| grep 2333` | 看到 LISTEN |
| TCP 连接 | `python3 simple_network_test.py --role worker ...` | ✓ 连接成功 |
| 防火墙 | `sudo iptables -L -n` | 允许 2333 端口 |

## 如果 TCP 测试通过，但 PyTorch 测试卡住

说明问题出在 NCCL 或 PyTorch 分布式配置上：

1. 检查 `NCCL_SOCKET_IFNAME` 是否设置正确
2. 尝试禁用 InfiniBand：`NCCL_IB_DISABLE=1`
3. 检查 NCCL 版本是否兼容

## 快速诊断脚本

在 Node1 上运行（一条命令检查所有项）：

```bash
echo "=== Ping 测试 ===" && \
ping -c 1 115.190.188.193 && \
echo "=== 端口检查 ===" && \
netstat -tuln | grep LISTEN && \
echo "=== 网络接口 ===" && \
ip addr show | grep "inet " && \
echo "=== Python 版本 ===" && \
python3 --version && \
echo "=== PyTorch 版本 ===" && \
python3 -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null || echo "PyTorch 未安装"
```

## 建议的测试流程

1. ✅ **基础 Ping 测试** → 确认网络基础连通性
2. ✅ **TCP 连接测试** → 确认端口和防火墙配置
3. ✅ **PyTorch 分布式测试** → 测试 NCCL 通信

如果前两个测试通过但第三个卡住，说明是 NCCL 特定配置问题。

## 获取更多帮助

如果 TCP 测试失败，请提供以下信息：

1. TCP 测试的完整输出
2. Ping 测试结果
3. `ip addr show` 的输出
4. `netstat -tuln` 的输出
5. 防火墙配置（`sudo iptables -L -n`）
