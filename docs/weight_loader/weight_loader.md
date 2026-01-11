## 通用 WeightLoader 设计与使用说明

本文档介绍 nanovllm 中通用权重加载器 **WeightLoader** 的设计背景、实现原理以及使用方法，帮助你为不同模型实现统一、可扩展的权重加载能力，兼容并行 Linear 等自定义切分策略。

---

### 1. 背景与目标

在实际项目中，存在如下需求：

- **源权重文件的命名（通常是 HF 风格）与本地模型结构命名不一致**，需要通过 mapping 才能正确加载；
- 模型内部大量使用 **列并行 / 行并行 / QKV 合并等自定义 Linear 实现**，它们通过参数上的 `weight_loader` 完成“全局权重 → 当前 rank shard”的切分逻辑；
- 希望有一个 **通用、解耦的权重加载器**，做到：
  - 不理解具体并行策略细节；
  - 只负责“找到正确的参数 + 把 tensor 交给它”；
  - 支持不同模型通过自定义 mapping 来适配 HF 权重命名差异。

为此，引入了 [`WeightLoader`](file:///root/mingtong/aiwork/nano-vllm/nanovllm/utils/weight_loader.py) 和映射配置 [`WeightMapping`](file:///root/mingtong/aiwork/nano-vllm/nanovllm/utils/weight_loader.py)，并在 Qwen3 模型上完成了示例实现和单测验证。

---

### 2. 核心设计

#### 2.1 WeightMapping 结构

定义位置：[`nanovllm/utils/weight_loader.py`](file:///root/mingtong/aiwork/nano-vllm/nanovllm/utils/weight_loader.py)

```python
@dataclass
class WeightMapping:

    target_path: str | list[str]
    transpose: bool = False
    dist_strategy: str | None = None
    loader_arg: object | None = None
```

含义：

- **target_path**: 模型参数的路径（或路径列表），用于 `named_parameters()` 查找；
- **transpose**: 是否在调用参数的 `weight_loader` 或默认 copy 前执行一次转置 `tensor.T`；
- **dist_strategy**: 可选字段，用于标记列并行 / 行并行策略（当前实现中主要是信息性标记）；
- **loader_arg**: 传给自定义 `weight_loader` 的额外参数，用于区分 packed 权重的不同 shard，例如：
  - QKV 合并权重中 `"q" / "k" / "v"`；
  - MLP 合并权重中 gate / up 的 `0 / 1` 索引。

权重映射总表 `weight_mappings` 的类型为：

```python
weight_mappings: dict[str, str | list[str] | WeightMapping]
```

- key 为 **源权重名**（通常是 safetensors 中的 key，支持模板形式，见下文）；
- value 为目标路径或 `WeightMapping`。

#### 2.2 WeightLoader 行为

定义位置：[`nanovllm/utils/weight_loader.py`](file:///root/mingtong/aiwork/nano-vllm/nanovllm/utils/weight_loader.py)

构造函数：

```python
class WeightLoader:
    def __init__(self, model: nn.Module, config, model_path: str):
        self.model_path = model_path
        self.model_config = config
        self.model = model
```

核心方法：

```python
def load_weights_from_safetensors(
    self, weight_mappings: dict[str, str | list[str] | WeightMapping]
):
    ...
```

**主要逻辑步骤**：

1. **标准化映射**：所有 value 统一转为 `WeightMapping` 实例；
2. **记录模型参数**：`param_dict = dict(self.model.named_parameters())`，方便通过字符串路径查找；
3. **匹配映射**：
   - 直接以源 key 精确查找；
   - 支持 `{layer}` 模板：
     - 例如 key 为 `"model.layers.{layer}.input_layernorm.weight"`；
     - 对于实际权重名 `"model.layers.3.input_layernorm.weight"`，自动解析出 `layer=3`；
     - 将 `{"layer": 3}` 作为格式化参数传入 `target_path.format(**fmt_kwargs)`。
4. **遍历权重文件**：通过 `_iterate_weights()` 按文件、按 key 依次取出 `weight_tensor`；
5. **根据映射决定目标参数名**：
   - 若无显式 mapping，则默认目标名与源 key 相同；
   - 若有 mapping，展开 `target_path`（支持列表）并处理格式化占位符；
6. **处理 transpose**：若 `mapping.transpose=True`，在传递给参数前先执行一次 `tensor = tensor.T`；
7. **优先调用参数自己的 `weight_loader`**：

   ```python
   weight_loader = getattr(param, "weight_loader", None)
   if weight_loader is not None:
       if mapping is not None and mapping.loader_arg is not None:
           weight_loader(param, tensor, mapping.loader_arg)
       else:
           weight_loader(param, tensor)
   else:
       # 默认行为：shape 校验 + 直接 copy
       ...
   ```

   - 如果参数上挂了 `weight_loader`（例如并行 Linear 中的权重和 bias），则由该函数完成真正的 shard 切分与赋值；
   - 否则，执行 shape 校验并直接 `param.data.copy_(tensor.to(...))`。

通过这种设计，**并行策略完全封装在各个层的 `weight_loader` 中**，`WeightLoader` 只承担“把对的 tensor 送到对的 param 上，并附带必要的 `loader_arg`”的职责，实现了解耦。

#### 2.3 与并行 Linear 的配合

相关代码：[`nanovllm/layers/linear.py`](file:///root/mingtong/aiwork/nano-vllm/nanovllm/layers/linear.py)

几个重要类：

- `ColumnParallelLinear`
- `RowParallelLinear`
- `MergedColumnParallelLinear`
- `QKVParallelLinear`

它们在初始化时会将自定义 `weight_loader` 挂在权重 / bias 参数上，例如：

```python
self.weight = nn.Parameter(...)
self.weight.weight_loader = self.weight_loader
```

典型示例：

- **列并行 Linear**：

  ```python
  def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
      param_data = param.data
      shard_size = param_data.size(self.tp_dim)
      start_idx = self.tp_rank * shard_size
      loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
      param_data.copy_(loaded_weight)
  ```

- **QKV 合并 Linear**：通过 `loader_arg in {"q", "k", "v"}` 决定在大矩阵中切哪一段；

- **Merged MLP Linear**：通过 `loader_arg in {0, 1}` 区分 gate / up 的位置后再切 shard。

得益于 `WeightLoader` 的调用约定：

- 对于 packed 权重，`WeightMapping.loader_arg` 会作为第三个参数传入；
- 对于普通并行 Linear，则只传两个参数 `(param, loaded_weight)`；
- 单机 world_size=1 的情况下，这些并行 `weight_loader` 的行为会退化到简单的转置 + 复制，便于在单测中进行精确比对。

---

### 3. 代码改造点

#### 3.1 新增 / 扩展的核心组件

- **通用加载器与映射结构**：
  - [`nanovllm/utils/weight_loader.py`](file:///root/mingtong/aiwork/nano-vllm/nanovllm/utils/weight_loader.py)
    - 新增 `WeightMapping` dataclass；
    - 实现 `WeightLoader.load_weights_from_safetensors`；
    - `_iterate_weights` 支持按层数过滤（基于 `config.num_hidden_layers`）。

- **Qwen3 模型权重加载接口**：
  - [`nanovllm/models/qwen3_2.py`](file:///root/mingtong/aiwork/nano-vllm/nanovllm/models/qwen3_2.py)
    - 在 `Qwen3ForCausalLM.__init__` 中保存 `self.config`，修正 `tie_word_embeddings` 逻辑；
    - 实现 `load_weights(self, config, model_path)`，内部调用 `WeightLoader`；
    - 实现 `_build_weight_mappings`，按 HF 命名展开各层的权重映射关系。

- **单测验证**：
  - [`nanovllm/tests/test_qwen3_2_weight_loading.py`](file:///root/mingtong/aiwork/nano-vllm/nanovllm/tests/test_qwen3_2_weight_loading.py)
    - 使用“小模型 + 假 HF 权重”验证映射逻辑与并行切分逻辑；
    - 可选地，使用真实 Qwen3 模型目录验证加载后参数与原始 safetensors 匹配。

---

### 4. 使用方法

#### 4.1 为模型定义映射规则

以 Qwen3 为例，在模型类中实现一个 `_build_weight_mappings` 方法（或单独放在对应的 mapping 模块中）：

```python
from nanovllm.utils.weight_loader import WeightMapping

class Qwen3ForCausalLM(nn.Module):
    ...

    def _build_weight_mappings(self) -> dict[str, WeightMapping]:
        weight_mappings: dict[str, WeightMapping] = {}

        # 全局权重
        weight_mappings["model.embed_tokens.weight"] = WeightMapping(
            target_path="transformers.embed_tokens.weight"
        )
        weight_mappings["lm_head.weight"] = WeightMapping(target_path="lm_head.weight")
        weight_mappings["model.norm.weight"] = WeightMapping(
            target_path="transformers.norm.weight"
        )

        # 按层展开
        for layer_id in range(self.config.num_hidden_layers):
            hf_prefix = f"model.layers.{layer_id}"
            current_prefix = f"transformers.layers.{layer_id}"

            weight_mappings[f"{hf_prefix}.input_layernorm.weight"] = WeightMapping(
                target_path=f"{current_prefix}.input_layernorm.weight"
            )
            weight_mappings[f"{hf_prefix}.post_attention_layernorm.weight"] = (
                WeightMapping(
                    target_path=f"{current_prefix}.post_attention_layernorm.weight"
                )
            )

            # attention q/k/v（列并行 + 转置）
            for proj in ["q_proj", "k_proj", "v_proj"]:
                hf_key = f"{hf_prefix}.self_attn.{proj}.weight"
                target = f"{current_prefix}.self_attn.{proj}.weight"
                weight_mappings[hf_key] = WeightMapping(
                    target_path=target,
                    transpose=True,
                    dist_strategy="col",
                )

            # q_norm / k_norm
            for qknorm in ["q_norm", "k_norm"]:
                hf_key = f"{hf_prefix}.self_attn.{qknorm}.weight"
                target = f"{current_prefix}.self_attn.{qknorm}.weight"
                weight_mappings[hf_key] = WeightMapping(target_path=target)

            # o_proj（行并行 + 转置）
            hf_key = f"{hf_prefix}.self_attn.o_proj.weight"
            target = f"{current_prefix}.self_attn.o_proj.weight"
            weight_mappings[hf_key] = WeightMapping(
                target_path=target,
                transpose=True,
                dist_strategy="row",
            )

            # MLP，gate/up（列并行 + 转置），down（行并行 + 转置）
            for proj in ["gate_proj", "up_proj"]:
                hf_key = f"{hf_prefix}.mlp.{proj}.weight"
                target = f"{current_prefix}.mlp.{proj}.weight"
                weight_mappings[hf_key] = WeightMapping(
                    target_path=target,
                    transpose=True,
                    dist_strategy="col",
                )

            hf_key = f"{hf_prefix}.mlp.down_proj.weight"
            target = f"{current_prefix}.mlp.down_proj.weight"
            weight_mappings[hf_key] = WeightMapping(
                target_path=target,
                transpose=True,
                dist_strategy="row",
            )

        return weight_mappings
```

对于需要额外 shard 标记的 packed 权重（如使用 `QKVParallelLinear` 或 `MergedColumnParallelLinear` 的模型），可以通过 `loader_arg` 传入标识，例如：

```python
# QKV packed 示例
weight_mappings[
    "model.layers.{layer}.self_attn.q_proj.weight"
] = WeightMapping(
    target_path="model.layers.{layer}.self_attn.qkv_proj.weight",
    loader_arg="q",
)

# MLP packed 示例
weight_mappings[
    "model.layers.{layer}.mlp.gate_proj.weight"
] = WeightMapping(
    target_path="model.layers.{layer}.mlp.gate_up_proj.weight",
    loader_arg=0,
)
```

#### 4.2 在模型中调用 WeightLoader

以 Qwen3 为例：

```python
from nanovllm.utils.weight_loader import WeightLoader

class Qwen3ForCausalLM(nn.Module):
    ...

    def load_weights(self, config, model_path):
        weight_mappings = self._build_weight_mappings()
        loader = WeightLoader(config=config, model_path=model_path, model=self)
        loader.load_weights_from_safetensors(weight_mappings)
```

使用方式：

```python
from transformers import Qwen3Config

config = Qwen3Config.from_pretrained(model_path)
model = Qwen3ForCausalLM(config)
model.load_weights(config=config, model_path=model_path)
```

这样：

- `WeightLoader` 会遍历指定路径下的所有 `*.safetensors` 文件；
- 按照 `_build_weight_mappings` 定义的规则，将 HF 权重 key 映射到模型参数；
- 自动处理 `transpose` 和 `loader_arg`，并调用各参数的自定义 `weight_loader` 完成切分；
- 最终模型即可在 nanovllm 的并行推理框架中使用。

---

### 5. 注意事项与常见坑

- **transpose 语义**：
  - 约定 HF 权重为 `(out_features, in_features)`；
  - 若 `mapping.transpose=True`，则在传给参数 / 自定义 `weight_loader` 前会执行 `tensor.T`，得到 `(in, out)` 形状，符合本地并行 Linear 的参数定义。

- **优先使用参数的 weight_loader**：
  - 对于并行 Linear，请务必在初始化时为参数挂上自定义 `weight_loader`；
  - `WeightLoader` 会优先调用该函数，只有在不存在 `weight_loader` 时才走默认 copy 逻辑。

- **loader_arg 的使用**：
  - 仅在 packed 权重（如 QKV 合并、MLP 合并）时需要；
  - 其含义完全由对应层的 `weight_loader` 自己解释，`WeightLoader` 不做语义假设。

- **单测构造“假 HF 权重”**：
  - 对于 `transpose=False` 的权重，可以直接用参数形状构造；
  - 对于 `transpose=True` 且为 2D 矩阵的权重，应使用 `hf_shape = (param.shape[1], param.shape[0])` 模拟 HF 的 `(out, in)` 形状，否则在并行 Linear 的 `weight_loader` 中可能出现 narrow 越界错误。

- **分布式环境**：
  - 在真实多卡/多机 TP 环境下，`RowParallelLinear` / `ColumnParallelLinear` 等层会根据 `tp_rank` 和 `tp_size` 自动切分权重；
  - 在单测中，可以通过打补丁的方式将 `world_size=1` / `rank=0` / `all_reduce` 变为 no-op，以便在 CPU 单进程环境下验证权重加载逻辑。

通过以上设计与实现，nanovllm 的权重加载能力实现了：

- **命名解耦**：通过映射表适配不同模型 / 不同权重命名；
- **并行策略解耦**：具体切分逻辑完全由各层的 `weight_loader` 决定；
- **可测试性**：既能用“假 HF 权重”做快速单测，又能对真实模型权重进行精确校验。
