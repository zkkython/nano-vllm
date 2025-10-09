import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 生成当前rank[vocab_start_idx, vocab_end_idx)范围内的boolean mask，x的值范围应该在当前rank的vocab范围内
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # (x - self.vocab_start_idx) 将x映射到当前rank的本地的vocab范围(shift操作)
            # mask * (x - self.vocab_start_idx) 将x映射到当前rank的位置索引，把不在范围内的置0
            x = mask * (x - self.vocab_start_idx)
        # x shape:  (seq1_len+seq2_len+..+seq_bs_len,), weights shape: (num_embeddings_per_partition, embedding_dim)
        # y shape: (seq1_len+seq2_len+..+seq_bs_len, embedding_dim)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        # 如果是prefill，则只取最后一个token的logits，所以x shape: (bs, hidden_size)，weights shape: (num_embeddings_per_partition, embedding_dim)
        # 如果是decode，则取所有token的logits，所以x shape: (bs, hidden_size)，weights shape: (num_embeddings_per_partition, embedding_dim)
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # x shape: (bs, hidden_size), weights shape: (num_embeddings_per_partition, embedding_dim)，其中embedding_dim=hidden_size
        #  logits shape: (bs, num_embeddings_per_partition)
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            # 在rank 0上收集所有rank的logits
            dist.gather(logits, all_logits, 0)
            # rank 0上拼接所有rank的logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
