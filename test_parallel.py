

import torch

import torch.nn as nn
import torch.nn.functional as F
class ParallelLMHead(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        logits = F.linear(x, self.weight, self.bias)



p = ParallelLMHead(10000, 768)
print(p.named_parameters())
print(p.get_parameter("weight").shape)
for k, v in p.named_parameters():
    print(k, v.shape)