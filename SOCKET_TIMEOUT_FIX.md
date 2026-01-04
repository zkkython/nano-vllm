# Socket 超时问题修复说明

## 问题描述

多机多卡启动时出现以下错误：

```
[W104 21:05:01.030155932 socket.cpp:209] [c10d] The hostname of the client socket cannot be retrieved. err=-3
[W104 21:07:12.188054319 socket.cpp:941] [c10d] The server socket on [15b:1fdd:ff7e:0:80b5:79e5:ff7e:0]:1815 has timed out, will retry
```

这表明 Master 节点在等待 Worker 节点的连接时出现了网络超时。

## 根本原因分析

### 问题 1：网络接口配置
- NCCL 默认使用 `eth0` 网络接口进行通信
- 某些环境中网络接口可能是 `en0`、`ens33` 等
- 如果指定的网络接口不存在或不通，会导致 socket 错误

### 问题 2：Worker 节点初始化延迟
- 启动脚本中 Rank 0 立即开始初始化 LLM
- 其他 Rank（Worker 节点上的进程）的 ModelRunner 还未完成初始化
- 导致 Rank 0 的广播操作等不到 Worker 的响应

### 问题 3：LLM 在多个 Rank 上重复初始化
- 之前的代码逻辑不清晰，每个 Rank 都可能试图初始化 LLM
- 这会导致重复的网络连接和资源竞争

## 修复方案

### 修复 1：设置 NCCL 环境变量

在 `node1_launch.py` 和 `node2_launch.py` 中添加：

```python
# 设置 NCCL 环境变量以改善跨节点通信
os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # 根据实际情况修改
os.environ.setdefault("NCCL_DEBUG", "INFO")          # 打开调试信息查看通信过程
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")     # 使用阻塞等待避免竞态条件
```

**网络接口检查方法：**
```bash
# Linux 上查看网络接口
ip addr show

# 或者使用
ifconfig

# macOS 上查看
ifconfig

# 通常是 eth0、en0、ens0 等
```

### 修复 2：只在 Rank 0 初始化 LLM

```python
if rank == 0:
    print(f"[Node1] Rank 0 initializing LLM...")
    llm = LLM(
        model="/data/Qwen3-8B/Qwen3-8B",
        tensor_parallel_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    )
    
    # 等待所有 Worker rank 完成初始化
    # 这个延迟给 Worker 节点上的其他 rank 足够时间来启动 ModelRunner
    print(f"[Node1] Rank 0 waiting for all ranks to initialize...")
    time.sleep(3)  # 至少 2-3 秒以完成 Worker 初始化
    
    # 开始推理
    # ...
else:
    # 其他 rank 会自动进入 ModelRunner.loop() 等待主 rank 的指令
    print(f"[Node{node_id}] Rank {rank} waiting for tasks from rank 0...")
```

### 修复 3：LLMEngine 环境变量同步

在 `llm_engine.py` 中确保从环境变量读取 master_addr：

```python
if dist.is_available() and dist.is_initialized():
    config.tensor_parallel_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    
    # 从环境变量读取 master_addr（由 torchrun 设置）
    master_addr_env = os.environ.get("MASTER_ADDR")
    if master_addr_env:
        config.master_addr = master_addr_env  # 同步到配置
    
    self.model_runner = ModelRunner(config, rank, local_rank)
```

### 修复 4：ModelRunner 中的冗余初始化检查

```python
if dist.is_initialized():
    # 如果已经初始化（通过 torchrun），则使用现有环境
    print(
        f"[DEBUG] Rank {rank} (local_rank={local_rank}) "
        f"Distributed environment already initialized by torchrun",
        flush=True,
    )
    assert dist.get_world_size() == self.world_size
    assert dist.get_rank() == rank
else:
    # 否则初始化新的分布式环境（单机多卡模式）
    if self.world_size > 1:
        # ... 初始化代码 ...
```

## 启动步骤（修正版本）

### 1. 在 Node1（Master）上启动

```bash
cd /root/mingtong/aiwork/nano-vllm

torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=0 \
    node1_launch.py
```

**等待日志输出，确保看到：**
```
[Node1] Rank 0 initializing LLM...
[Node1] Rank 0 waiting for all ranks to initialize...
```

### 2. 在 Node2（Worker）上启动（在 Node1 启动后再启动）

```bash
cd /root/mingtong/aiwork/nano-vllm

torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=1 \
    node2_launch.py
```

**关键点：**
- ✅ Node1 要先启动
- ✅ Node1 上 Rank 0 会等待 3 秒给 Worker 初始化
- ✅ Node2 的 MASTER_ADDR 必须指向 Node1（115.190.188.193）

## 环境变量检查

### 查看当前网络接口

```bash
# Linux
ip addr show | grep "inet "

# 或者
hostname -I

# macOS
ifconfig | grep "inet "
```

### 如果 eth0 不存在

根据实际网络接口修改 `node1_launch.py` 和 `node2_launch.py`：

```python
# 如果是 en0
os.environ.setdefault("NCCL_SOCKET_IFNAME", "en0")

# 如果是 ens33
os.environ.setdefault("NCCL_SOCKET_IFNAME", "ens33")

# 如果是其他接口
os.environ.setdefault("NCCL_SOCKET_IFNAME", "<your_interface>")
```

## 调试建议

### 1. 打开 NCCL 调试信息

已在启动脚本中设置 `NCCL_DEBUG=INFO`，会输出详细的通信日志。

### 2. 运行诊断脚本

```bash
# 在 Node1 上
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    debug_distributed.py

# 在 Node2 上
torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=1 \
    debug_distributed.py
```

### 3. 检查网络连接

```bash
# 从 Node2 检查能否连接到 Node1
telnet 115.190.188.193 2333

# 或者使用 nc
nc -zv 115.190.188.193 2333
```

## 文件修改列表

- ✅ `nanovllm/engine/model_runner.py` - 改进初始化检查和调试信息
- ✅ `nanovllm/engine/llm_engine.py` - 从环境变量同步 master_addr
- ✅ `node1_launch.py` - 添加 NCCL 环境变量、等待延迟、清晰的初始化逻辑
- ✅ `node2_launch.py` - 添加 NCCL 环境变量、不初始化 LLM
- ✅ `debug_distributed.py` - 新增诊断工具
- ✅ `MULTI_MACHINE_LAUNCH_GUIDE.md` - 更新启动指南

## 如果问题仍未解决

1. **检查网络接口**
   - 运行 `ip addr show` 或 `ifconfig` 确认网络接口名称
   - 在启动脚本中修改 `NCCL_SOCKET_IFNAME` 为正确的接口

2. **检查防火墙**
   ```bash
   # 检查 2333 端口是否开放
   sudo iptables -L -n | grep 2333
   ```

3. **检查网络延迟**
   ```bash
   # 从 Node2 ping Node1
   ping 115.190.188.193
   ```

4. **增加超时时间**
   ```python
   # 在启动脚本中增加等待时间
   time.sleep(5)  # 改为 5 秒或更长
   ```

5. **查看详细日志**
   ```python
   # 设置更详细的 NCCL 日志
   os.environ.setdefault("NCCL_DEBUG", "TRACE")  # 改为 TRACE 获取更多细节
   ```
