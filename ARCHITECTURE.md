## Nano-vLLM 设计与架构

本文基于当前代码实现，系统性阐述推理架构、关键模块设计、与源码对应关系；并提供如何扩展增量接入新模型（以 LLaMA 为例）的方案与端到端验证步骤。

### 总览

- 数据流：请求 -> 调度器批处理与KV缓存分配 -> 多进程/多卡 ModelRunner 执行 -> 采样 -> 回传输出。
- 关键优化：前缀缓存、块化KV缓存、Tensor Parallel、FlashAttention、CUDA Graph 捕获与复放。

### 配置与入口

配置集中在 `nanovllm/config.py`，由 `LLMEngine` 初始化并在全局使用。

```1:26:/home/kason/python_workspace/nano-vllm/nanovllm/config.py
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1 

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
```

`LLMEngine` 负责：组建进程、创建调度器、初始化 `ModelRunner`、管理生成主循环。

```15:35:/home/kason/python_workspace/nano-vllm/nanovllm/engine/llm_engine.py
class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
```

### 调度与KV缓存块管理

调度器将等待队列中的 `Sequence` 合并成批，先 prefill 再 decode，期间与 `BlockManager` 合作进行KV缓存块分配、复用与逐步扩展。

```8:34:/home/kason/python_workspace/nano-vllm/nanovllm/engine/scheduler.py
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```

块化KV缓存与前缀复用逻辑（哈希+引用计数）在 `BlockManager`：

```26:43:/home/kason/python_workspace/nano-vllm/nanovllm/engine/block_manager.py
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

### 执行器 ModelRunner 与 CUDA Graph

`ModelRunner` 管理模型构建、权重加载、CUDA 图捕获、KVCache 分配，以及一次批处理的前后处理。

```17:45:/home/kason/python_workspace/nano-vllm/nanovllm/engine/model_runner.py
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        # self.model = Qwen3ForCausalLM(hf_config)
        self.model = MODELS_MAPPING[hf_config.model_type](hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
```

CUDA Graph 捕获沿用上下文中的 KV 布局，提升 decode 小批次吞吐：

```296:336:/home/kason/python_workspace/nano-vllm/nanovllm/engine/model_runner.py
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None
        ...
```

### 注意力与RoPE、FlashAttention、KV写回

注意力核心封装于 `nanovllm/layers/attention.py`。prefill 阶段支持前缀缓存直接从 cache 读；decode 阶段使用带 KVCache 的 FlashAttention。KV 写回通过 Triton kernel 实现：

```10:29:/home/kason/python_workspace/nano-vllm/nanovllm/layers/attention.py
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

```68:79:/home/kason/python_workspace/nano-vllm/nanovllm/layers/attention.py
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
```

RoPE 缓存与按位旋转在 `rotary_embedding.py`：

```39:55:/home/kason/python_workspace/nano-vllm/nanovllm/layers/rotary_embedding.py
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key
```

### Tensor Parallel 线性算子与词表并行

张量并行线性层（列并行、行并行、QKV合并）定义在 `nanovllm/layers/linear.py`，词表并行 `Embedding/LMHead` 在 `nanovllm/layers/embed_head.py`。权重加载时通过参数挂载的 `weight_loader` 拆分装载。

```115:164:/home/kason/python_workspace/nano-vllm/nanovllm/layers/linear.py
class QKVParallelLinear(ColumnParallelLinear):
    ...
    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        ...
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[
            self.tp_rank
        ]
        param_data.copy_(loaded_weight)
```

```68:89:/home/kason/python_workspace/nano-vllm/nanovllm/layers/embed_head.py
    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

### 模型抽象与权重加载

模型类型通过 `models_mapping.py` 选择，实际权重加载在 `utils/loader.py`，支持 packed 权重名称到模块参数的映射与切片载入。

```1:9:/home/kason/python_workspace/nano-vllm/nanovllm/models/models_mapping.py
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.llama import LLamaForCausalLM

MODELS_MAPPING = {
    "qwen3": Qwen3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "llama": LLamaForCausalLM,
}
```

```22:53:/home/kason/python_workspace/nano-vllm/nanovllm/utils/loader.py
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    with open("model_structure_write.txt", "w") as model_f:
        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    model_f.write(
                        f"{weight_name}：shape = {f.get_tensor(weight_name).shape}\n"
                    )
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            try:
                                param = model.get_parameter(param_name)
                                weight_loader = getattr(param, "weight_loader")
                                weight_loader(
                                    param, f.get_tensor(weight_name), shard_id
                                )
                                break
                            except AttributeError:
                                print(
                                    f"Warning: {param_name} not found in the model, skip loading {weight_name}"
                                )
                    else:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, f.get_tensor(weight_name))
```

### 采样与上下文

`Sampler` 在 rank0 上进行贪心/温度采样；`Context` 承载 prefill/decode 的动态信息供注意力与 LM Head 读取。

```251:293:/home/kason/python_workspace/nano-vllm/nanovllm/engine/model_runner.py
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids
```

### 扩展新模型：设计准则与步骤

目标：新增模型需适配统一调用接口 `forward(input_ids, positions)` 与 `compute_logits(hidden_states)`，并暴露 `packed_modules_mapping` 以支持权重文件命名与模块参数名称的映射。

步骤：
1) 在 `nanovllm/models/` 新增模型实现（参考 Qwen3/LLaMA 拆分 Attention/MLP/DecoderLayer/Model/ForCausalLM）。
2) 在 `nanovllm/models/models_mapping.py` 登记 `model_type -> Class`。要求 HuggingFace `config.json` 中 `model_type` 与此键一致。
3) 保持张量并行线性层、Embedding/LM Head 的用法一致；如权重切片策略不同，覆写对应参数的 `weight_loader`。
4) 在 `utils/loader.load_model` 的 packed 映射中通过 `packed_modules_mapping` 解决权重名差异（如 q_proj/k_proj/v_proj -> qkv_proj）。

### LLaMA 端到端实现（本仓库新增）

新增 `nanovllm/models/llama.py`，实现与 Qwen3 对齐的注意力/MLP/Decoder/Model/LMHead 结构：

```1:120:/home/kason/python_workspace/nano-vllm/nanovllm/models/llama.py
import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
...
class LlamaAttention(nn.Module):
    ...
```

权重名映射保持与 Qwen3 相同风格，便于 `utils.loader` 复用：

```120:170:/home/kason/python_workspace/nano-vllm/nanovllm/models/llama.py
class LLamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
```

### 运行与验证

示例：`llama_infer.py`（新增），指定本地 HuggingFace LLaMA 权重目录：

```1:999:/home/kason/python_workspace/nano-vllm/llama_infer.py
import argparse
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
...
```

用法：

```bash
python /home/kason/python_workspace/nano-vllm/llama_infer.py --model /PATH/TO/LLAMA-MODEL
```

预期：程序完成两条提示的生成输出；若 `config.json` 中包含 `"model_type": "llama"`，则会自动选择 `LLamaForCausalLM` 实现。

### 增量优化与扩展建议

- 增量图捕获：对常见 batch size 维护 CUDA Graph 池，已在 `capture_cudagraph` 实现，可按业务分布调整 `graph_bs` 列表。
- KV 压缩与混合精度：通过 `hf_config.torch_dtype` 统一控制；可增加 FP8/压缩KV策略。
- Prefix Cache 策略：可接入跨请求共享（当前块级哈希已具备基础设施）。
- 多机并行：当前使用 NCCL 初始化单机多卡；可扩展为多节点地址簇及参数切片初始化流程。



