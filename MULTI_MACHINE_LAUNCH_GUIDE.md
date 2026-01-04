# 多机多卡启动指南

## 问题说明

之前启动脚本中存在的问题：
1. **Node1 和 Node2 的 MASTER_ADDR 不同** - 导致分布式通信失败
2. **没有使用 torchrun** - torchrun 会自动设置 RANK、LOCAL_RANK、WORLD_SIZE 等环境变量
3. **错误的环境变量设置** - 每个节点上的进程应该有不同的 RANK，而不是固定的值

## 正确的启动方式

### 前置条件

- 两台机器的网络互通
- Node1 IP: `115.190.188.193`
- Node2 IP: `115.190.188.193`（注意：不是 115.190.188.194，必须指向 Master 节点）
- Master 节点：Node1

### 启动步骤

#### 1. 在 Node1 上启动（Master + Worker）

```bash
cd /root/mingtong/aiwork/nano-vllm

# 启动 Node1：8 个 GPU，使用 torchrun
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=0 \
    node1_launch.py
```

**环境变量映射：**
- 进程 0-7 的 RANK = 0-7
- 进程 0-7 的 LOCAL_RANK = 0-7
- 所有进程的 WORLD_SIZE = 10

#### 2. 在 Node2 上启动（Worker）

```bash
cd /root/mingtong/aiwork/nano-vllm

# 启动 Node2：2 个 GPU，同时连接到 Node1 的 Master
# 重要：MASTER_ADDR 必须是 Node1 的 IP（115.190.188.193），而不是本机 IP
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=1 \
    node2_launch.py
```

**环境变量映射：**
- 进程 0-1 的 RANK = 8-9（全局 rank）
- 进程 0-1 的 LOCAL_RANK = 0-1（本地 rank）
- 所有进程的 WORLD_SIZE = 10

### 关键区别：RANK vs LOCAL_RANK

| 参数 | 说明 | Node1 示例 | Node2 示例 |
|------|------|-----------|----------|
| **RANK** | 全局进程ID | 0-7 | 8-9 |
| **LOCAL_RANK** | 本机GPU索引 | 0-7 | 0-1 |
| **WORLD_SIZE** | 总进程数 | 10 | 10 |

### 为什么要这样设置？

1. **MASTER_ADDR 必须一致**
   - 所有节点的 `MASTER_ADDR` 必须指向 Master 节点的 IP
   - 不能各自指向自己的 IP，否则会导致：
     ```
     [W104 20:59:35.610101829 socket.cpp:209] [c10d] The hostname of the client socket cannot be retrieved. err=-3
     ```

2. **torchrun 自动管理分布式配置**
   - 自动设置环境变量
   - 自动处理进程启动和退出
   - 自动设置 LOCAL_RANK（本机 GPU 索引）

3. **local_rank 用于 CUDA 设备选择**
   - 每个进程使用 `cuda:{local_rank}` 来访问 GPU
   - 而不是 `cuda:{rank}`

## 故障排查

### 错误 1: socket hostname cannot be retrieved

**原因**：MASTER_ADDR 配置错误

**解决**：
```bash
# ❌ 错误做法
MASTER_ADDR=115.190.188.194  # Node2 自己的 IP

# ✅ 正确做法
MASTER_ADDR=115.190.188.193  # Master 节点（Node1）的 IP
```

### 错误 2: device_id cuda:8 is out of range

**原因**：使用全局 RANK 而不是 LOCAL_RANK

**解决**：已在 `model_runner.py` 中修复，使用 `local_rank` 代替 `rank`

### 错误 3: Process 卡死或无法连接

**检查清单**：
1. 确认 MASTER_ADDR 指向正确的 Master 节点 IP
2. 确认端口 2333 未被占用
3. 确认网络防火墙允许节点间通信
4. 检查 `torch.distributed` 是否正确初始化

## socket 超时问题排查

如果出现以下错误：
```
[W104 21:05:01.030155932 socket.cpp:209] [c10d] The hostname of the client socket cannot be retrieved. err=-3
[W104 21:07:12.188054319 socket.cpp:941] [c10d] The server socket on [...]:1815 has timed out
```

### 常见原因

1. **网络接口配置不正确**
   - NCCL 默认使用 eth0，但某些环境可能是 en0、ens33 等
   - 解决：设置 `NCCL_SOCKET_IFNAME` 环境变量

2. **网络不通**
   - Node1 和 Node2 之间无法通信
   - 解决：检查防火墙、网络连接

3. **Master 节点还未启动**
   - Worker 节点先启动，Master 还未初始化
   - 解决：先在 Node1 启动 torchrun，然后启动 Node2

### 环境变量设置

在启动脚本中设置以下环境变量：

```python
# 设置 NCCL 环境变量以改善跨节点通信
os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # 检查你的网络接口
os.environ.setdefault("NCCL_DEBUG", "INFO")  # 打开调试信息
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")  # 使用阻塞等待
```

### 诊断工具

运行诊断脚本检查配置：
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=<your_master_ip> \
    --master_port=2333 \
    debug_distributed.py
```

## 代码实现原理

### 1. 环境变量读取

启动脚本从环境变量读取配置：

```python
rank = int(os.environ.get("RANK", 0))          # 全局进程ID
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 本机GPU索引
world_size = int(os.environ.get("WORLD_SIZE", 1))  # 总进程数
master_addr = os.environ.get("MASTER_ADDR", "115.190.188.193")
master_port = int(os.environ.get("MASTER_PORT", 2333))
```

### 2. 只在 Rank 0 初始化 LLM

```python
if rank == 0:
    print("[Node1] Rank 0 initializing LLM...")
    llm = LLM(
        model="/data/Qwen3-8B/Qwen3-8B",
        tensor_parallel_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    )
    # 等待 Worker 初始化
    time.sleep(3)
    # 执行推理...
else:
    # 其他 rank 等待指令（在 ModelRunner.loop() 中）
    print(f"Rank {rank} waiting for tasks...")
```

### 3. ModelRunner 中使用 local_rank

```python
torch.cuda.set_device(local_rank)  # 使用本机GPU索引
if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
        device_id=local_rank,  # 关键：使用local_rank而不是rank
    )
```

### 4. 分布式通信流程

- **Rank 0**：初始化 LLM → 等待 Worker 初始化 → 发送广播信号和数据
- **其他 Rank**：在 loop() 中等待 → 接收广播信号 → 执行相应方法
