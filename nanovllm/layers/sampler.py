import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 每次decode都是吐出一个token，所以logits shape: (batch_size, vocab_size)
        # temperatures shape: (batch_size,)，每个batch的每个token的采样温度
        logits = logits.to(torch.float)
        # greedy_tokens shape: (batch_size,)，每次decode都吐出概率最大的token
        greedy_tokens = logits.argmax(dim=-1)
        # 温度缩放
        logits.div_(temperatures.unsqueeze(dim=1))
        # probs shape: (batch_size, vocab_size)，每个token的概率
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10
        # sample_tokens shape: (batch_size,)，每次decode都吐出一个采样得到的token
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1) + epsilon
        ).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
