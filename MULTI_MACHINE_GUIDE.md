# nano-vllm 多机TP并行使用指南

## 概述

nano-vllm 现在支持跨机器的张量并行（Tensor Parallelism, TP），允许在多台机器上分布模型权重以进行高效的推理。

## 主要修改

### 1. 配置文件更新
- 添加了 `master_addr`、`master_port`、`local_rank`、`node_rank` 参数
- 移除了 `tensor_parallel_size` 的8卡上限

### 2. 分布式初始化
- 修改了 `ModelRunner` 中的分布式初始化逻辑
- 支持通过 TCP 连接跨机器初始化

### 3. 通信机制
- 用分布式通信替代了共享内存机制
- 所有进程间通信现在通过 `torch.distributed` 进行

### 4. 模型层支持
- 现有的并行层（如 `QKVParallelLinear`、`RowParallelLinear` 等）已支持分布式

## 使用方法

### 单机多卡（向后兼容）

```python
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

llm = LLM(
    model="/path/to/your/model",
    tensor_parallel_size=8,  # 单机8张卡
    enforce_eager=False
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
prompts = ["Hello, how are you?", "What is machine learning?"]

outputs = llm.generate(prompts, sampling_params)
```

### 多机TP并行

#### 方法1：使用 torchrun (推荐)

使用torchrun启动分布式任务：
```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=MASTER_NODE_IP:2333 \
    examples/distributed_launch_example.py
```

#### 方法2：使用 launch_multi_machine.py 脚本

在主节点上运行：
```bash
python launch_multi_machine.py \
    --master_addr MASTER_IP \
    --master_port 2333 \
    --nnodes 2 \
    --nproc_per_node 8 \
    --model_path /path/to/your/model \
    --script examples/multi_machine_tp_example.py
```

#### 方法3：手动创建启动脚本

创建每个节点的启动脚本：
```bash
python launch_multi_machine.py \
    --master_addr MASTER_IP \
    --master_port 2333 \
    --nnodes 2 \
    --nproc_per_node 8 \
    --model_path /path/to/your/model \
    --create_script \
    --node_rank 0 \
    --output_script node0_script.py
```

然后在每台机器上运行相应的脚本。

## 配置参数说明

- `tensor_parallel_size`: 总的GPU数量（所有机器上的总和）
- `master_addr`: 主节点IP地址
- `master_port`: 主节点端口
- `node_rank`: 当前节点的rank（0表示主节点）

## 网络要求

- 所有机器必须能够通过TCP连接到主节点
- 指定的端口必须开放
- 建议使用高速网络（如InfiniBand或高速以太网）以获得最佳性能

## 注意事项

1. 所有机器必须安装相同的CUDA版本
2. 模型文件需要在所有机器上可用
3. 确保网络延迟较低以获得最佳性能
4. 目前支持NCCL后端进行GPU间通信

## 性能优化建议

1. 使用高速网络连接各机器
2. 确保网络带宽足够支持张量并行所需的通信
3. 根据模型大小和可用GPU数量调整TP大小
4. 考虑使用梯度累积等技术来优化内存使用