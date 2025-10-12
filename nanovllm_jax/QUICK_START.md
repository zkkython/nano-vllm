# Nano-vLLM JAX 快速开始指南

## 概述

这是nano-vllm的JAX实现版本，将原始的PyTorch实现迁移到了JAX/Flax框架，支持GPU加速的大语言模型推理。

## 环境要求

- Python 3.8+
- JAX with CUDA support
- Flax
- Transformers
- CUDA-capable GPU

## 安装

```bash
# 创建conda环境
conda create -n nvllm_jax python=3.12
conda activate nvllm_jax

# 安装依赖
cd nanovllm_jax
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用

```bash
# 运行Qwen3 demo
python qwen3_demo.py --model /path/to/qwen/model
```

### 2. 内存受限环境

如果遇到GPU内存不足的问题，可以使用以下参数：

```bash
python qwen3_demo.py \
  --model /path/to/qwen/model \
  --max_tokens 10 \
  --max_length 256 \
  --gpu_memory_utilization 0.3 \
  --max_num_batched_tokens 1024
```

### 3. Python API使用

```python
from nanovllm_jax import LLM, SamplingParams

# 创建LLM实例
llm = LLM(
    "/path/to/model",
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=256,
    gpu_memory_utilization=0.3
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=50
)

# 生成文本
prompts = ["Hello, how are you?"]
outputs = llm.generate(prompts, sampling_params)

# 查看结果
for output in outputs:
    print(f"Text: {output['text']}")
    print(f"Tokens: {output['token_ids']}")
```

## 配置参数说明

### LLM参数

- `model`: 模型路径
- `enforce_eager`: 是否使用eager模式（默认：True）
- `tensor_parallel_size`: 张量并行大小（默认：1）
- `max_model_len`: 最大模型长度（默认：2048）
- `max_num_batched_tokens`: 最大批处理token数（默认：2048）
- `gpu_memory_utilization`: GPU内存使用率（0.0-1.0，默认：0.9）
- `kvcache_block_size`: KV缓存块大小（必须是256的倍数，默认：256）

### SamplingParams参数

- `temperature`: 采样温度（默认：1.0）
- `max_tokens`: 最大生成token数（默认：256）

## 内存优化建议

### 低内存配置（< 8GB VRAM）

```python
llm = LLM(
    model_path,
    max_model_len=128,
    max_num_batched_tokens=512,
    gpu_memory_utilization=0.2,
    kvcache_block_size=256
)
```

### 中等内存配置（8-16GB VRAM）

```python
llm = LLM(
    model_path,
    max_model_len=512,
    max_num_batched_tokens=2048,
    gpu_memory_utilization=0.5,
    kvcache_block_size=256
)
```

### 高内存配置（> 16GB VRAM）

```python
llm = LLM(
    model_path,
    max_model_len=2048,
    max_num_batched_tokens=4096,
    gpu_memory_utilization=0.9,
    kvcache_block_size=256
)
```

## 性能调优

### 1. 使用JIT编译（未来版本）

```python
# 将在未来版本中支持
llm = LLM(model_path, enforce_eager=False)
```

### 2. 批处理推理

```python
# 一次处理多个提示
prompts = [
    "Prompt 1",
    "Prompt 2",
    "Prompt 3"
]
outputs = llm.generate(prompts, sampling_params)
```

### 3. 调整温度和采样参数

```python
# 更确定性的输出
sampling_params = SamplingParams(temperature=0.1, max_tokens=100)

# 更随机的输出
sampling_params = SamplingParams(temperature=1.5, max_tokens=100)
```

## 故障排除

### 问题1: GPU内存不足

**错误信息：**
```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate X bytes
```

**解决方案：**
- 减小 `max_model_len`
- 减小 `max_num_batched_tokens`
- 降低 `gpu_memory_utilization`

### 问题2: CUDA驱动版本不匹配

**警告信息：**
```
cudaErrorInsufficientDriver : CUDA driver version is insufficient
```

**解决方案：**
- 更新CUDA驱动
- 或者降级JAX/CUDA版本以匹配驱动

### 问题3: 生成文本质量差

**可能原因：**
- 模型权重未正确加载（当前版本限制）
- 温度设置不当

**临时解决方案：**
- 调整温度参数
- 等待权重加载功能完善

## 测试

```bash
# 运行测试
python nanovllm_jax/test_implementation.py

# 运行debug测试
python nanovllm_jax/debug_test.py
```

## 项目结构

```
nanovllm_jax/
├── __init__.py           # 包初始化
├── config.py             # 配置类
├── sampling_params.py    # 采样参数
├── llm.py               # 主LLM类
├── layers/              # 神经网络层
│   ├── linear.py        # 线性层
│   ├── attention.py     # 注意力机制
│   ├── layernorm.py     # LayerNorm
│   ├── activation.py    # 激活函数
│   ├── rotary_embedding.py  # 旋转位置编码
│   ├── embed_head.py    # 嵌入和LM头
│   └── sampler.py       # 采样器
├── models/              # 模型实现
│   └── qwen3.py         # Qwen3模型
├── engine/              # 推理引擎
│   ├── sequence.py      # 序列管理
│   ├── scheduler.py     # 调度器
│   └── model_runner.py  # 模型运行器
└── utils/               # 工具函数
    ├── context.py       # 上下文管理
    └── loader.py        # 模型加载
```

## 示例代码

查看 `qwen3_demo.py` 获取完整示例。

## 支持的模型

目前支持：
- Qwen3 系列模型

计划支持：
- Llama 系列
- Mistral 系列
- 其他主流LLM

## 贡献

欢迎提交问题和拉取请求！

## 许可证

与原始nano-vllm项目保持一致。

## 联系方式

如有问题，请提交GitHub issue。

