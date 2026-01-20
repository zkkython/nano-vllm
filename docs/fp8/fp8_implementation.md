# DeepSeek-V3 FP8 推理实现方案文档

本文档详细介绍了在 `nano-vllm` 框架下为 DeepSeek-V3 模型实现 FP8 量化推理的方案、过程、代码改动及使用说明。

## 1. 技术背景与方案概述

DeepSeek-V3 是一个拥有 671B 参数的巨型模型，为了在有限的显存环境下实现高效推理，我们引入了 FP8（e4m3fn）量化支持。

### 核心方案：
*   **权重格式**：使用 `torch.float8_e4m3fn` 存储线性层权重。
*   **量化粒度**：采用 Block-wise 量化（128x128 块大小），每个块对应一个缩放系数（Scale）。
*   **算子实现**：基于 Triton 开发高性能 FP8 GEMM、激活值量化（ActQuant）和权重反量化（Dequant）算子。
*   **解耦设计**：将 FP8 核心逻辑封装在 `utils/fp8.py` 中，通过通用线性层（`linear.py`）透明支持多种并行策略。

## 2. 代码改造详述

### 2.1 核心工具库 (`nanovllm/utils/fp8.py`)
创建了独立的工具库，包含以下核心功能：
*   **`act_quant`**：将输入张量实时量化为 FP8，并计算缩放系数。增加了 `.contiguous()` 处理以确保内存布局兼容。
*   **`fp8_gemm`**：执行 FP8 矩阵乘法，支持指定输出 DType（默认随输入精度，如 BF16）。
*   **`linear_fp8`**：高层抽象接口，自动处理量化、计算及 Bias 加法。

### 2.2 通用线性层 (`nanovllm/layers/linear.py`)
对 `ReplicatedLinear`、`ColumnParallelLinear` 和 `RowParallelLinear` 进行了深度改造：
*   **按需初始化**：根据 `quantization="fp8"` 配置，将 `weight` 初始化为 `torch.float8_e4m3fn` 类型。
*   **Scale 参数管理**：通过 `register_parameter` 管理 `weight_scale`。针对 MLA 中维度不为 128 倍数的情况（如 576），使用了向上取整的 Block 分配逻辑：`(size + 127) // 128`。
*   **Forward 路由**：在 `forward` 方法中检测 `weight_scale`。如果存在，则自动切换到 `linear_fp8` 执行路径。

### 2.3 权重加载器 (`nanovllm/utils/weight_loader.py` & `models/deepseek_v3_weight_mapping.py`)
*   **系数转换**：DeepSeek-V3 权重文件中存储的是 `scale_inv`。在 `weight_scale_loader` 中直接加载这些系数作为乘法系数使用（注意：DeepSeek-V3 的 `scale_inv` 在推理时通常就是作为 dequantization 的乘数）。
*   **并行切分**：确保 `weight_scale` 在 Tensor Parallel 场景下能像 `weight` 一样正确切分（Shard）。

### 2.4 模型配置与结构 (`nanovllm/models/deepseek_v3.py`)
*   **配置扩展**：在 `DeepSeekV3Config` 中增加了 `quantization` 字段。
*   **透传机制**：将量化配置从模型构造函数透传至每一个线性层实例。
*   **健壮性优化**：将关键路径的 `.view()` 替换为 `.reshape()`，增强对非连续内存布局的容错性。

## 3. 使用方式

### 3.1 开启 FP8 推理
在初始化模型配置或 `LLM` 引擎时，指定 `quantization` 参数：

```python
from transformers import AutoConfig
from nanovllm.models.deepseek_v3 import DeepSeekV3ForCausalLM

config = AutoConfig.from_pretrained(model_path)
config.quantization = "fp8"  # 开启 FP8 开关

model = DeepSeekV3ForCausalLM(config).cuda()
```

### 3.2 精度建议
由于 FlashAttention 算子不支持 FP32，建议在脚本开头设置默认精度为 `bfloat16`：

```python
import torch
torch.set_default_dtype(torch.bfloat16)
```

## 4. 注意事项与常见陷阱

### 4.1 张量连续性 (Contiguous)
Triton 算子要求输入张量在内存中连续。虽然 `act_quant` 内部已添加处理，但在进行复杂的 `split`、`slice` 或 `view` 操作后，显式调用 `.contiguous()` 有助于提升性能并避免潜在的断言错误。

### 4.2 数据类型对齐
*   **FlashAttention 限制**：FlashAttention 仅支持 `fp16` 和 `bf16`。FP8 线性层的输出必须转换为其中一种精度（通常为 `bf16`）后再进入 Attention 层。
*   **Scale 加载**：确保原始权重中的 `scale_inv` 正确加载为乘数。

### 4.3 显存占用
在 FP8 模式下，参数部分的显存占用将下降至原来的 1/4（相比 FP32）或 1/2（相比 BF16）。但在初始化阶段，PyTorch 可能会因临时张量产生峰值占用，建议配合 `load_partial_layers` 进行调试。

## 5. 验证方式
*   **DType 检查**：打印 `linear.weight.dtype` 应为 `torch.float8_e4m3fn`。
*   **Scale 检查**：打印 `linear.weight_scale` 不应为 `None` 且值应为合理的缩放系数。
*   **算子验证**：开启代码中的 `[FP8_EXEC]` 日志，观察推理时是否触发 Triton 内核。
