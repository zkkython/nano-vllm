# 日志配置指南 (Log Configuration Guide)

## 概述

`LogConfig` 提供了灵活的日志控制机制，允许你在**调试时启用详细日志**，在**生产环境中关闭日志**以提升性能。

## 日志级别

| 级别 | 说明 | 使用场景 |
|------|------|---------|
| `NONE` | 不输出任何日志 | 极致性能要求 |
| `ERROR` | 只输出错误 | 生产环境 |
| `WARNING` | 输出警告和错误 | 生产环境（推荐） |
| `INFO` | 输出信息、警告和错误 | 开发环境 |
| `DEBUG` | 输出所有日志 | 调试问题 |

## 支持的模块

- `chunked_prefill`: Chunked Prefill 功能的日志
- `scheduler`: Scheduler 调度器的日志
- `model_runner`: ModelRunner 模型执行的日志
- `block_manager`: BlockManager KV cache 管理的日志
- `warmup`: 模型预热的日志
- `attention`: Attention 计算的日志

## 使用方式

### 方式 1：通过代码配置（推荐）

```python
from nanovllm import LLM, SamplingParams
from nanovllm.log_config import LogConfig, LogLevel

# 生产环境：只输出错误
log_config = LogConfig(global_level=LogLevel.ERROR)

llm = LLM(
    model_path="/path/to/model",
    max_num_batched_tokens=256,
    enable_chunked_prefill=True,
    log_config=log_config,  # 传入日志配置
)
```

### 方式 2：通过环境变量（部署推荐）

```bash
# 设置全局日志级别
export NANOVLLM_LOG_LEVEL=ERROR

# 设置特定模块的日志级别
export NANOVLLM_LOG_CHUNKED_PREFILL=DEBUG
export NANOVLLM_LOG_WARMUP=INFO

python your_script.py
```

## 使用场景

### 场景 1：生产环境（性能优先）

```python
# 只输出错误，最大化性能
log_config = LogConfig(global_level=LogLevel.ERROR)

llm = LLM(model_path, log_config=log_config, ...)
```

或通过环境变量：

```bash
export NANOVLLM_LOG_LEVEL=ERROR
python production_server.py
```

### 场景 2：调试 Chunked Prefill 问题

```python
# 全局关闭日志，只开启 chunked_prefill 的调试日志
log_config = LogConfig(
    global_level=LogLevel.ERROR,
    chunked_prefill=LogLevel.DEBUG,  # 只调试这个模块
    warmup=LogLevel.INFO,            # 显示 warmup 基本信息
)

llm = LLM(model_path, log_config=log_config, ...)
```

或通过环境变量：

```bash
export NANOVLLM_LOG_LEVEL=ERROR
export NANOVLLM_LOG_CHUNKED_PREFILL=DEBUG
export NANOVLLM_LOG_WARMUP=INFO
python debug_script.py
```

### 场景 3：开发环境（详细日志）

```python
# 所有模块都输出详细日志
log_config = LogConfig(global_level=LogLevel.DEBUG)

llm = LLM(model_path, log_config=log_config, ...)
```

或通过环境变量：

```bash
export NANOVLLM_LOG_LEVEL=DEBUG
python dev_script.py
```

### 场景 4：默认配置

```python
# 不传 log_config，使用默认配置（WARNING 级别）
llm = LLM(model_path, ...)  # 默认 WARNING 级别
```

## 日志输出格式

日志格式：`[rank<N>][LEVEL][module] message`

示例：
```
[rank0][INFO][warmup] Warmup with Chunked Prefill: chunk_size=64
[rank0][DEBUG][chunked_prefill] Processing 1 sequences
[rank0][DEBUG][chunked_prefill] Seq 0: start=0, end=64, chunk=64, total=64, blocks=1
[rank0][INFO][warmup] Warmup completed: 4 seqs x 64 tokens
```

## 性能影响

| 配置 | 性能影响 | 适用场景 |
|------|---------|---------|
| `NONE` / `ERROR` | 几乎无影响 | 生产环境 |
| `WARNING` | 轻微影响 | 生产环境（推荐） |
| `INFO` | 中等影响 | 开发环境 |
| `DEBUG` | 明显影响 | 调试问题 |

**建议**：
- ✅ 生产环境使用 `ERROR` 或 `WARNING`
- ✅ 调试时只开启需要的模块的 `DEBUG` 日志
- ❌ 避免在生产环境使用 `DEBUG` 级别

## 完整示例

查看 `test_log_config.py` 获取完整的使用示例：

```bash
# 运行默认示例（调试 Chunked Prefill）
python test_log_config.py

# 运行生产环境示例
python test_log_config.py 2

# 运行调试模式示例
python test_log_config.py 4
```

## 最佳实践

1. **开发时**：使用代码配置，方便快速调整
   ```python
   log_config = LogConfig(
       global_level=LogLevel.ERROR,
       chunked_prefill=LogLevel.DEBUG,  # 只调试需要的模块
   )
   ```

2. **部署时**：使用环境变量，无需修改代码
   ```bash
   export NANOVLLM_LOG_LEVEL=WARNING
   python server.py
   ```

3. **调试线上问题**：临时修改环境变量，重启服务
   ```bash
   export NANOVLLM_LOG_LEVEL=ERROR
   export NANOVLLM_LOG_CHUNKED_PREFILL=DEBUG  # 只开启问题模块
   python server.py
   ```

4. **性能测试**：完全关闭日志
   ```bash
   export NANOVLLM_LOG_LEVEL=NONE
   python benchmark.py
   ```

## 扩展日志模块

如果需要为新模块添加日志支持，只需：

1. 在代码中导入日志函数：
   ```python
   from nanovllm.log_config import log_debug, log_info
   ```

2. 使用日志函数输出：
   ```python
   log_debug("my_module", f"Debug message: {value}", rank=self.rank)
   log_info("my_module", "Info message")
   ```

3. 在 `LogConfig` 中添加该模块的配置字段（可选）

## 常见问题

### Q: 如何完全关闭日志？
A: 设置 `global_level=LogLevel.NONE` 或环境变量 `NANOVLLM_LOG_LEVEL=NONE`

### Q: 如何只看某个模块的日志？
A: 设置全局为 `ERROR`，单独设置该模块为 `DEBUG`

### Q: 环境变量和代码配置哪个优先？
A: 代码配置优先。如果代码中指定了 `log_config`，环境变量只会填充未设置的模块

### Q: 如何在多机环境中使用？
A: 日志会自动包含 `rank` 信息，可以通过 `rank` 区分不同节点的日志

### Q: 日志会影响性能吗？
A: 只有在日志级别允许时才会执行字符串格式化和输出，`ERROR` 级别几乎无性能影响
