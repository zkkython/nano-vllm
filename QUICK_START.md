# 快速启动指南

## 问题：环境变量未设置

如果你看到环境变量都显示 `NOT SET`，说明脚本没有通过 **torchrun** 启动。

```
RANK                      = NOT SET
LOCAL_RANK                = NOT SET
WORLD_SIZE                = NOT SET
```

## 正确的启动方式

### 方式 1：使用 torchrun（推荐）

#### 在 Node1（Master）上执行

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

#### 在 Node2（Worker）上执行

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

### 方式 2：使用启动脚本（更简单）

如果上面的命令太复杂，可以使用我们提供的启动脚本：

```bash
# 互动式启动
bash START_HERE.sh

# 或者直接启动 Node1
bash START_HERE.sh node1-torchrun

# 或者直接启动 Node2
bash START_HERE.sh node2-torchrun
```

### 方式 3：使用 Python 启动器（调试用）

```bash
# Node1
python3 multi_machine_launch.py \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=0 \
    --use_torchrun

# Node2
python3 multi_machine_launch.py \
    --nnodes=2 \
    --nproc_per_node=2 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=1 \
    --use_torchrun
```

## 关键点

### 1. 必须使用 torchrun

❌ **错误做法**：直接运行脚本
```bash
python node1_launch.py  # 这样不会设置环境变量！
```

✅ **正确做法**：使用 torchrun
```bash
torchrun ... node1_launch.py  # torchrun 会自动设置环境变量
```

### 2. 启动顺序

1. **先在 Node1 启动**（Master 节点）
2. **等待 3-5 秒**
3. **再在 Node2 启动**（Worker 节点）

> 不要同时启动，Worker 会找不到 Master

### 3. 参数检查

- `--nnodes=2` - 2 个节点
- `--nproc_per_node=8` - Node1 有 8 个 GPU
- `--nproc_per_node=2` - Node2 有 2 个 GPU
- `--master_addr=115.190.188.193` - Master 节点 IP
- `--node_rank=0` - Node1 是 rank 0（Master）
- `--node_rank=1` - Node2 是 rank 1（Worker）

### 4. 如果环境变量仍然未设置

检查以下几点：

```bash
# 检查 torchrun 是否安装
torchrun --help

# 检查 PyTorch 版本
python3 -c "import torch; print(torch.__version__)"

# 检查分布式模块
python3 -c "import torch.distributed as dist; print(dist.is_available())"
```

## 故障排查

### 错误：`torchrun command not found`

**解决**：安装 PyTorch 或使用 Python 模块方式

```bash
# 方式 1：通过 pip 安装
pip install torch torchvision torchaudio -U

# 方式 2：使用 Python 模块
python3 -m torch.distributed.launch \
    --nnodes=2 \
    --nproc_per_node=8 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    --node_rank=0 \
    node1_launch.py
```

### 错误：Socket 超时

**解决**：
1. 检查网络接口（修改 `NCCL_SOCKET_IFNAME`）
2. 检查防火墙设置
3. 运行诊断工具

```bash
# 运行诊断
torchrun --nnodes=1 --nproc_per_node=1 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    debug_distributed.py

# 或者用启动脚本
bash START_HERE.sh diagnose
```

### 错误：NCCL 连接失败

**解决**：检查网络接口配置

```python
# 在 node1_launch.py 或 node2_launch.py 中修改
os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # 改为 en0、ens33 等
```

## 环境变量说明

| 变量 | 说明 | 示例 |
|------|------|------|
| RANK | 全局进程 ID | 0-9（10 个进程） |
| LOCAL_RANK | 本机 GPU 索引 | 0-7（Node1），0-1（Node2） |
| WORLD_SIZE | 总进程数 | 10 |
| MASTER_ADDR | Master 节点 IP | 115.190.188.193 |
| MASTER_PORT | Master 节点端口 | 2333 |

## 验证环境变量已正确设置

运行诊断脚本查看环境变量：

```bash
# 使用 torchrun 启动诊断（会显示正确的环境变量）
torchrun --nnodes=1 --nproc_per_node=1 \
    --master_addr=115.190.188.193 \
    --master_port=2333 \
    debug_distributed.py

# 输出应该显示
# ================================================================================
# 分布式环境变量:
# ================================================================================
#   RANK                      = 0
#   LOCAL_RANK                = 0
#   WORLD_SIZE                = 1
#   MASTER_ADDR               = 115.190.188.193
#   MASTER_PORT               = 2333
```

## 文件参考

- `node1_launch.py` - Node1 启动脚本
- `node2_launch.py` - Node2 启动脚本
- `multi_machine_launch.py` - Python 启动器
- `START_HERE.sh` - Bash 启动脚本（互动式）
- `debug_distributed.py` - 诊断工具
- `MULTI_MACHINE_LAUNCH_GUIDE.md` - 详细指南
- `SOCKET_TIMEOUT_FIX.md` - Socket 超时问题排查
