# LLM 并行示例（两节点 PyTorch 分布式）

本目录 `examples/llm_dist` 基于 PyTorch 提供了 5 种典型并行方式的**最小可运行示例**，
假设有 **2 个 GPU 节点**（也可以在单机上用 2 张卡模拟），每个示例只跑一次前向/反向，方便理解数据流。

示例基于 `torch.distributed`，启动方式与 `examples/distribute` 下保持类似：
- 显式传入 `--rank / --world-size / --master-addr / --master-port`
- 每个 Rank 启动一个独立 Python 进程

---

## 目录结构

- **`dist_utils.py`**: 通用分布式初始化/参数解析工具
- **`toy_models.py`**: 简单 Toy MLP / Pipeline Stage / Expert 模型
- **`dp_example.py`**: 数据并行 (Data Parallel, DP)
- **`tp_example.py`**: 张量并行 (Tensor Parallel, TP)
- **`pp_example.py`**: 流水线并行 (Pipeline Parallel, PP)
- **`ep_example.py`**: Expert Parallel (EP, 类 MoE)
- **`sp_example.py`**: 序列并行 (Sequence Parallel, SP)

所有示例都使用非常小的网络和数据规模，方便在两张 GPU 上快速验证。

---

## 通用启动方式（两节点）

假设：
- 节点 1 IP: `NODE1_IP`
- 节点 2 IP: `NODE2_IP`
- 使用 NCCL 后端；每个节点只启动一个进程（共 2 个进程，`world_size=2`）

### 节点 1（Rank 0）

```bash
cd examples/llm_dist

python dp_example.py \
  --rank 0 \
  --world-size 2 \
  --master-addr NODE1_IP \
  --master-port 29500 \
  --backend nccl
```

### 节点 2（Rank 1）

```bash
cd examples/llm_dist

python dp_example.py \
  --rank 1 \
  --world-size 2 \
  --master-addr NODE1_IP \
  --master-port 29500 \
  --backend nccl
```

> **注意**：
> - `--master-addr` 统一指向 Rank 0 所在节点 IP；
> - 如果在单机多卡上模拟，只需把 `NODE1_IP` 替换为 `127.0.0.1` 即可；
> - 使用 NCCL 时，确保每个进程绑定到不同 GPU（`dist_utils.py` 中已根据 `rank % num_gpus` 自动设置）。

其他示例（`tp_example.py / pp_example.py / ep_example.py / sp_example.py`）的启动方式相同，只需要把脚本名替换掉。

---

## 1. 数据并行 DP (`dp_example.py`)

- **核心思想**: 每个 Rank 拥有一份完整模型，各自处理不同数据分片，通过 `all_reduce` 聚合梯度，使得结果等价于单卡更大 batch 训练。
- **本示例**：
  - 模型：`ToyMLP`（两层小 MLP）；
  - 每个 Rank 上随机生成不同数据；
  - 前向 -> MSE Loss -> 反向；
  - 使用 `dist.all_reduce` 对每个 `param.grad` 做 `SUM` 后除以 `world_size`；
  - 调用一次 `optimizer.step()`。

运行完成后，你会看到每个 Rank 各自的 local loss，以及一次同步梯度更新的日志。

---

## 2. 张量并行 TP (`tp_example.py`)

- **核心思想**: 将单个算子（如 Linear）的权重/输出切分到多个 Rank 上，每个 Rank 只负责一部分计算，最后通过通信拼接结果。
- **本示例**：
  - 实现了一个简化的 `ColumnParallelLinear`：
    - `out_features` 沿列方向划分到各个 Rank；
    - Rank i 上只持有 `in_features x (out_features / world_size)` 的子矩阵；
  - 前向：
    - 每个 Rank 局部计算 `local_out = x @ W_i`；
    - 使用 `dist.all_gather` 收集所有 `local_out`，在特征维 `dim=-1` 上拼接为完整输出；
  - 反向：
    - 普通 `loss.backward()`；
    - 对每个参数梯度执行 `all_reduce` + 取平均，保证所有 Rank 参数一致。

> 通过这个示例可以直观理解 TP 的数据流：**运算分摊在多卡，结果通过通信合并**。

---

## 3. 流水线并行 PP (`pp_example.py`)

- **核心思想**: 将模型按层拆分成多个 Stage，每个 Stage 放在不同 Rank 上，数据像流水线一样从前向后流动；反向时梯度反向流动。
- **本示例**（2 Stage）：
  - Rank 0: `ToyPipelineStage0`（第一层 Linear + ReLU）；
  - Rank 1: `ToyPipelineStage1`（第二层 Linear 输出层）；
  - 前向：
    - Rank 0: `h = stage0(x)`，使用 `dist.send(h, dst=1)` 发送给 Rank 1；
    - Rank 1: `h = recv()` 后，做 `out = stage1(h)`，计算 loss；
  - 反向：
    - Rank 1: `loss.backward()`，得到 `h.grad`，再通过 `dist.send(h.grad, dst=0)` 发送梯度给 Rank 0；
    - Rank 0: 收到 `grad_h` 后，对 `h.backward(grad_h)`；
  - 两边各自调用一次 `optimizer.step()`。

> 这个示例只跑了单个 micro-batch，没有做流水线填充，但能清楚展示 **激活从前往后传，梯度从后往前传** 的过程。

---

## 4. Expert Parallel EP (`ep_example.py`)

- **核心思想**: MoE 中的 Expert Parallel 将不同 Expert 分布到不同 Rank 上，Token 通过门控被路由到对应 Expert 进行计算，然后再聚回。
- **本示例**（极简化版 MoE）：
  - 有 2 个 Expert：
    - Expert0 放在 Rank 0；
    - Expert1 放在 Rank 1；
  - 所有 Rank 上构造相同的输入 `x`，并用简单规则划分 Token：
    - 前半部分 Token 走 Expert0；
    - 后半部分 Token 走 Expert1；
  - 通信流程：
    - 每个 Rank 上按本地负责的 Token 生成 `local_input`；
    - 使用 `all_gather` 汇集所有 `local_input`，得到自己负责的 Expert 的真正输入 `my_tokens`；
    - 通过 `ToyExpert` 计算 `local_output`；
    - 再次 `all_gather` 将各 Expert 的输出收集回来，根据掩码 `mask_expert0/mask_expert1` 还原到全局输出 `out`；
  - 反向同样用 `all_reduce` 聚合梯度，保证各 Rank 上 Expert 参数一致。

> 真实 MoE 中会使用 `all_to_all` 和更复杂的路由策略，这里只保留最核心的数据路由与 Expert 并行思想。

---

## 5. 序列并行 SP (`sp_example.py`)

- **核心思想**: 在 Transformer 等模型中沿 **序列维 (sequence length)** 切分激活，每个 Rank 只负责一部分序列片段，以减小单卡激活开销。
- **本示例**：
  - 输入张量形状为 `(batch_size, seq_len, dim)`，其中：
    - `batch_size=2`，`seq_len=8`，`world_size=2` 时，每个 Rank 负责 4 个 token；
  - Rank i 负责序列切片：`x_local = x[:, i*4:(i+1)*4, :]`；
  - 每个 Rank 上各自跑一份 `ToyMLP`：
    - 将 `x_local` reshape 到 `(batch_size * local_seq_len, dim)`，做逐 token 的 MLP；
  - 使用 `dist.all_gather` 在序列维 (`dim=1`) 上拼回完整输出 `out`；
  - 用简单的 `out.pow(2).mean()` 构造 loss，反向后对参数梯度做 `all_reduce` + 平均。

> 这个示例展示了 **SP 的核心模式：沿序列切分 + 本地计算 + 通信聚合**，
> 实际大模型中会与 TP/DP 等组合使用，并在注意力/归一化等模块中引入更多通信。

---

## 在两节点上快速验证的建议

- **建议 1**：先在单机多卡上把示例跑通（`MASTER_ADDR=127.0.0.1`），确认 PyTorch/NCCL 环境 OK；
- **建议 2**：再切换到两机，按 `examples/distribute/README.md` 中的网络排查方法，确认端口互通；
- **建议 3**：从最简单的 `dp_example.py` 开始，再依次尝试 `tp/pp/ep/sp`，观察每种并行方式下打印的 loss 与日志，理解梯度/激活的流向。

如果你希望把这些示例进一步改造成 **真正的 LLM Block（带 Attention 等）**，我们也可以在此基础上继续扩展。
