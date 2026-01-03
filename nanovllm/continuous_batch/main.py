from dataclasses import dataclass
import random
import torch
from torch import nn

from nanovllm.continuous_batch.request import listen_request


class ToyModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config


class ContinueBatcingEngine: ...


@dataclass
class Config:
    max_batch_size = 4  # 最大批次
    max_seq_len = 32  # 最长文本长度
    max_prompt_len = 16  # 最大prompt长度

    # model 与kv_cache
    num_layers = 3  # 模型最大3层
    dim = 16
    num_kv_heads = 2
    head_dim = 8
    vocab_size = 20  # 词汇表20个词
    EOS_TOKEN = 0


config = Config()
toymodel = ToyModel(config)
worker = ContinueBatcingEngine()

# 核心主流程
N = 100
count = 0  # 当前计数

while count < N:
    prompt, prompt_len = listen_request(config, p=0.5)
    if prompt_len > 0:
        count += 1  # 是一个合理的请求
        if count % (N // 10) == 0:
            print(f"Processed {count} requests")
        generate_len = random.randint(prompt_len, config.max_seq_len)
        worker.add_request(prompt, generate_len)
        pending, total = worker.get_requests_info()
        print(f"pending requests = {pending}, total: {total}, N:{N}")
    # 执行推理
    worker.step()
    if worker.is_finished() and count == N:
        print("All requests are finished")
        break
