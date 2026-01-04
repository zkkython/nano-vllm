# 分布式通信测试

## 说明

这是最简单的分布式测试程序，用来验证两个节点之间是否能正常通信。

## 运行步骤

### 1. 在 Node1（Master）上启动

```bash
cd /root/mingtong/aiwork/nano-vllm

# 方式 1：使用启动脚本（推荐）
bash run_distributed_test.sh 115.190.188.193 2333 2 0 1

# 方式 2：直接使用 torchrun
torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=0 \
    test_distributed_simple.py
```

**预期输出：**
```
================================================================================
简单分布式通信测试
================================================================================

[INFO] 环境变量:
  RANK:           0
  LOCAL_RANK:     0
  WORLD_SIZE:     2
  MASTER_ADDR:    115.190.188.193
  MASTER_PORT:    2333
  本机 IP:        115.190.188.193

[步骤 1] 初始化分布式环境...
  ✓ 设置 CUDA 设备: 0
  ✓ 分布式环境初始化成功 (backend: nccl)

[步骤 2] 测试通信...
  测试 1: 广播 (broadcast)
  <等待其他进程...>
```

### 2. 在 Node2（Worker）上启动（等待 Node1 启动后）

```bash
cd /root/mingtong/aiwork/nano-vllm

# 方式 1：使用启动脚本（推荐）
bash run_distributed_test.sh 115.190.188.193 2333 2 1 1

# 方式 2：直接使用 torchrun
torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=1 \
    test_distributed_simple.py
```

**预期输出：**
```
================================================================================
简单分布式通信测试
================================================================================

[INFO] 环境变量:
  RANK:           1
  LOCAL_RANK:     0
  WORLD_SIZE:     2
  MASTER_ADDR:    115.190.188.193
  MASTER_PORT:    2333
  本机 IP:        115.190.188.194

[步骤 1] 初始化分布式环境...
  ✓ 设置 CUDA 设备: 0
  ✓ 分布式环境初始化成功 (backend: nccl)

[步骤 2] 测试通信...
  测试 1: 广播 (broadcast)
  ✓ Rank 1 收到广播数据: 0
  
  测试 2: 屏障同步 (barrier)
  ✓ Rank 1 通过屏障
  
  测试 3: 全部收集 (allgather)
  ✓ Rank 1 收集结果: [0, 1]
  
  测试 4: 全部化简 (allreduce)
  ✓ Rank 1 AllReduce 结果 (求和): 3

[步骤 3] 等待所有进程同步...

[成功] Rank 1 所有测试通过! ✓

[完成] 分布式环境已清理
```

## 测试内容

程序会进行以下 4 个测试：

1. **Broadcast（广播）**
   - Rank 0 向所有 Rank 发送数据
   - 验证所有进程都能接收到相同的数据

2. **Barrier（屏障同步）**
   - 所有进程在屏障处同步
   - 验证所有进程能正确同步

3. **AllGather（全部收集）**
   - 每个进程发送自己的 Rank 编号
   - 所有进程收集所有数据
   - 结果应该是 `[0, 1]` 或类似（取决于进程数）

4. **AllReduce（全部化简）**
   - 每个进程发送 `rank + 1`
   - 所有进程将数据求和
   - 结果应该是所有 `rank + 1` 的总和

## 排查问题

### 如果 Node2 卡住，说明：

1. **网络不通**
   - 检查防火墙配置
   - 确认两台机器能互相 ping 通

2. **Master 地址错误**
   - 确保 `--master_addr` 是 Node1 的正确 IP
   - 不要使用本机 IP

3. **端口被占用**
   - 尝试使用其他端口：`--master_port=2334`

### 如果显示错误信息

**错误：`Connection refused`**
```
Node1 还未启动，先启动 Node1
```

**错误：`Address already in use`**
```
端口 2333 被占用，使用其他端口或杀死占用进程
```

**错误：`Hostname cannot be resolved`**
```
网络接口配置问题，检查 NCCL_SOCKET_IFNAME
```

## 成功标志

✅ 如果两边都输出类似以下内容，说明通信成功：

```
  ✓ 设置 CUDA 设备: 0
  ✓ 分布式环境初始化成功
  ✓ Rank X 收到广播数据: 0
  ✓ Rank X 通过屏障
  ✓ Rank X 收集结果: [0, 1]
  ✓ Rank X AllReduce 结果 (求和): 3
  [成功] Rank X 所有测试通过! ✓
```

## 下一步

如果测试通过，说明两个节点可以正常通信，然后可以：

1. 增加更多 GPU：修改 `--nproc_per_node` 参数
2. 测试 LLM 推理：使用 `node1_launch.py` 和 `node2_launch.py`
3. 查看 NCCL 日志：设置 `NCCL_DEBUG=INFO` 或 `NCCL_DEBUG=TRACE`

## 参数说明

### run_distributed_test.sh 参数

```bash
bash run_distributed_test.sh [MASTER_IP] [MASTER_PORT] [NNODES] [NODE_RANK] [NPROC_PER_NODE]
```

| 参数 | 默认值 | 说明 |
|------|-------|------|
| MASTER_IP | 115.190.188.193 | Master 节点 IP |
| MASTER_PORT | 2333 | Master 节点端口 |
| NNODES | 2 | 节点总数 |
| NODE_RANK | 0 | 当前节点编号（0 = Master） |
| NPROC_PER_NODE | 1 | 每个节点的进程数 |

### 示例

```bash
# Node1 (8 GPU)
bash run_distributed_test.sh 115.190.188.193 2333 2 0 8

# Node2 (2 GPU)
bash run_distributed_test.sh 115.190.188.193 2333 2 1 2
```

## 文件

- `test_distributed_simple.py` - 测试程序
- `run_distributed_test.sh` - 启动脚本
- `DISTRIBUTED_TEST_GUIDE.md` - 本文件
