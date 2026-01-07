# PyTorch 分布式集合通信示例

本目录包含 PyTorch 分布式训练中常用的集合通信操作示例，每个文件演示一种通信原语。所有示例都设计为在两个机器节点上运行。

## 📁 文件列表

| 文件 | 功能说明 | 通信模式 |
|------|---------|---------|
| `all_reduce.py` | 所有进程规约操作（求和、最大、最小） | 集合通信 |
| `broadcast.py` | 从一个进程广播数据到所有进程 | 一对多 |
| `gather.py` | 收集所有进程的数据到指定进程 | 多对一 |
| `scatter.py` | 从一个进程分发数据到所有进程 | 一对多 |
| `all_gather.py` | 收集所有进程的数据到所有进程 | 多对多 |
| `reduce_scatter.py` | 规约后分散到各进程 | 多对多 |
| `send_recv.py` | 点对点发送和接收（同步/异步） | 点对点 |
| `barrier.py` | 进程同步屏障 | 同步原语 |

## 🚀 快速开始

### 前置条件

- 两台机器，每台至少有一个 GPU
- PyTorch 已安装（支持分布式）
- 机器之间网络互通
- 已配置好 NCCL 环境（GPU通信）或使用 Gloo 后端（CPU通信）

### 基本使用方法

假设有两台机器：
- **节点1 (Master)**: IP 地址 `192.168.1.100`
- **节点2 (Worker)**: IP 地址 `192.168.1.101`

#### 示例 1: All-Reduce

在节点1上运行：
```bash
cd /root/mingtong/aiwork/nano-vllm/examples/dist
python all_reduce.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
```

在节点2上运行：
```bash
cd /root/mingtong/aiwork/nano-vllm/examples/dist
python all_reduce.py --rank 1 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
```

#### 示例 2: Broadcast

在节点1上运行：
```bash
python broadcast.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
```

在节点2上运行：
```bash
python broadcast.py --rank 1 --world-size 2 --master-addr 192.168.1.100 --master-port 29500
```

#### 其他示例

其他文件的使用方法相同，只需替换文件名：
```bash
# Gather
python gather.py --rank <RANK> --world-size 2 --master-addr <MASTER_IP> --master-port 29500

# Scatter
python scatter.py --rank <RANK> --world-size 2 --master-addr <MASTER_IP> --master-port 29500

# All-Gather
python all_gather.py --rank <RANK> --world-size 2 --master-addr <MASTER_IP> --master-port 29500

# Reduce-Scatter
python reduce_scatter.py --rank <RANK> --world-size 2 --master-addr <MASTER_IP> --master-port 29500

# Send-Recv
python send_recv.py --rank <RANK> --world-size 2 --master-addr <MASTER_IP> --master-port 29500

# Barrier
python barrier.py --rank <RANK> --world-size 2 --master-addr <MASTER_IP> --master-port 29500
```

## 🔧 参数说明

所有脚本都支持以下命令行参数：

| 参数 | 说明 | 必填 | 默认值 |
|------|------|------|--------|
| `--rank` | 当前进程的 rank（从0开始） | 是 | - |
| `--world-size` | 总进程数 | 是 | - |
| `--master-addr` | 主节点的 IP 地址 | 是 | - |
| `--master-port` | 主节点的端口号 | 否 | 29500 |
| `--backend` | 通信后端 (nccl/gloo) | 否 | nccl |

### Backend 选择

- **nccl**: 用于 GPU 通信，性能最优（推荐）
- **gloo**: 用于 CPU 通信，兼容性好

如果使用 CPU 或遇到 NCCL 问题，可以切换到 Gloo：
```bash
python all_reduce.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --backend gloo
```

## 📚 通信原语详解

### 1. All-Reduce
- **功能**: 对所有进程的数据进行规约操作（SUM/MAX/MIN），然后将结果分发给所有进程
- **使用场景**: 梯度聚合、全局统计信息计算
- **特点**: 所有进程都能获得相同的结果

### 2. Broadcast
- **功能**: 从一个进程（通常是 rank 0）广播数据到所有其他进程
- **使用场景**: 模型参数初始化、配置信息分发
- **特点**: 只有源进程需要准备数据

### 3. Gather
- **功能**: 将所有进程的数据收集到一个指定进程
- **使用场景**: 收集评估结果、日志收集
- **特点**: 只有目标进程能看到所有数据

### 4. Scatter
- **功能**: 将一个进程的数据分发到所有进程（每个进程接收不同部分）
- **使用场景**: 数据分片、任务分配
- **特点**: 与 Gather 相反的操作

### 5. All-Gather
- **功能**: 收集所有进程的数据，并分发给所有进程
- **使用场景**: 特征聚合、全局信息共享
- **特点**: 所有进程都能看到所有数据

### 6. Reduce-Scatter
- **功能**: 先规约再分散（All-Reduce + Scatter 的组合）
- **使用场景**: 分布式优化器、梯度切片
- **特点**: 每个进程只接收部分规约结果

### 7. Send-Recv
- **功能**: 点对点通信，支持同步和异步模式
- **使用场景**: 流水线并行、自定义通信模式
- **特点**: 灵活性高，可实现复杂通信拓扑

### 8. Barrier
- **功能**: 同步屏障，确保所有进程到达同一点后才继续
- **使用场景**: 阶段同步、调试、性能测量
- **特点**: 用于协调不同进程的执行进度

## 🔍 常见问题

### 1. NCCL 超时错误

如果遇到 NCCL 超时，可以尝试：
```bash
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口
export NCCL_DEBUG=INFO          # 开启调试信息
```

### 2. 网络连通性问题

确保两台机器之间可以互相访问：
```bash
# 在节点2上测试能否访问节点1
ping 192.168.1.100
telnet 192.168.1.100 29500
```

### 3. GPU 设备问题

如果只有一个 GPU，代码会自动处理。如果有多个 GPU，可以通过设置 `CUDA_VISIBLE_DEVICES` 指定：
```bash
CUDA_VISIBLE_DEVICES=0 python all_reduce.py --rank 0 --world-size 2 ...
```

### 4. 端口被占用

更改 `--master-port` 参数：
```bash
python all_reduce.py --rank 0 --world-size 2 --master-addr 192.168.1.100 --master-port 29501
```

## 📊 学习路径建议

1. **基础通信**: 从 `broadcast.py` 和 `barrier.py` 开始
2. **集合操作**: 学习 `all_reduce.py` 和 `all_gather.py`
3. **数据分发**: 理解 `scatter.py` 和 `gather.py`
4. **高级操作**: 掌握 `reduce_scatter.py`
5. **点对点**: 最后学习 `send_recv.py` 的灵活用法

## 🛠️ 调试技巧

1. **启用调试日志**:
   ```bash
   export NCCL_DEBUG=INFO
   export TORCH_DISTRIBUTED_DEBUG=INFO
   ```

2. **使用 Gloo 后端测试**:
   ```bash
   python xxx.py --backend gloo ...
   ```

3. **单机多进程测试**:
   ```bash
   # 节点1
   python xxx.py --rank 0 --world-size 2 --master-addr localhost --master-port 29500
   # 节点1 (另一个终端)
   python xxx.py --rank 1 --world-size 2 --master-addr localhost --master-port 29500
   ```

## 📖 参考资料

- [PyTorch 分布式官方文档](https://pytorch.org/docs/stable/distributed.html)
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [分布式训练最佳实践](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

## 💡 提示

- 确保两个节点的 PyTorch 版本一致
- 建议先在单机上测试，确认代码正确后再部署到多机
- 使用 `--backend gloo` 可以在 CPU 上快速验证逻辑
- 每个示例都有详细的输出，帮助理解数据流向

## 🎯 实际应用场景

| 通信操作 | 分布式训练场景 | 典型应用 |
|---------|---------------|---------|
| All-Reduce | 数据并行 | 梯度聚合 |
| Broadcast | 参数同步 | 模型权重分发 |
| All-Gather | 张量并行 | 收集分片输出 |
| Reduce-Scatter | 优化器并行 | ZeRO 优化器 |
| Send-Recv | 流水线并行 | GPipe、PipeDream |
| Barrier | 训练协调 | Epoch 同步 |
