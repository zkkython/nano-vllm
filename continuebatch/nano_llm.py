from dataclasses import dataclass
import random
from typing import List
from collections import deque
import torch
from torch import nn
from itertools import count


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


def listen_request(config, p=0.5):
    if random.random() < p:
        prompt = []
        prompt_len = random.randint(config.max_prompt_len // 4, config.max_prompt_len)

        prompt = torch.randint(1, config.vocab_size, (1, prompt_len))
        return prompt[0].tolist(), prompt_len

    else:
        return [], 0


request_counter = count()


class Request:

    def __init__(self, request_id: int, prompt: list[int], max_len: int) -> None:
        self.request_id = request_id
        self.prompt = prompt  # prompt对应的token int 数组
        self.max_len = max_len  # 最大生成长度，总长度如果超过这个值就得停止了
        self.generated_tokens = []  # 生成的token id
        self.cur_length = len(
            prompt
        )  # 当前位置，随着decode解码，当前位置会不断地按1累计递增
        self.request_state = "WAITING"  # 请求的状态WAITING, RUNNING, FINISHED


class ToyModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config


class ModelWrapper:
    def __init__(self, model, kv_cache_manager) -> None:
        self.model = model
        self.kv_cache_manager = kv_cache_manager


class RequestManager:

    def __init__(self, max_batch_size: int) -> None:
        self.max_batch_size = max_batch_size
        self.requests = {}  # request id - > Req
        self.wait_queue = deque()
        self.running_queue = set()

    def add_request(self, prompt: List[int], max_seq_len: int):
        request_id = next(request_counter)
        req = Request(request_id, prompt, max_seq_len)
        self.requests[request_id] = req
        self.wait_queue.append(req)
        return 0


# 核心是step方法
class ContinueBatchingEngine:

    def __init__(self, model, config) -> None:
        self.kv_cache_manager = KVCacheManager(config)
        self.model = ModelWrapper(model, self.kv_cache_manager)
        self.request_manager = RequestManager(
            config.max_batch_size,
        )

    def add_request(self, prompt: List[int], max_seq_len: int) -> int:
        return self.request_manager.add_request(prompt, max_seq_len)
    
  


    # 优先decode, 其次prefill
    # 这里体现continue batching， 先decoding, 在prefill
    # 这里假设每次step 增加一个请求
    # step1 prefill req1
    # step2 decoding req1 prefill req2
    # step3 decoding req1,req2, prefill req3
    def step(self):
        # decoding阶段(已经有请求)
        if has_activate_requests():
            # 获取kvcache的槽和运行中的requests_ids
            activate_slots, request_ids = get_activate_slots_info()
            # 获取最新token
            input_tokens = get_input_ids(self.request_manager, request_ids)
            # 进行decode, 同时更新kvcache也在decode的程序内部
            decoding_logits = self.model.decode(
                input_tokens,
                activate_slots,
            )

            next_tokens = generate_next_tokens(decoding_logits)

            # 关联request_id 和 token，然后更新状态
            for i, request_id in enumerate(request_ids):
                # 更新状态
                update_request(request_id, next_tokens[i])
                update_slots(
                    request_id
                )  # 如果decode到eos 或者 达到max length，则结束释放slot

        # prefill 新请求
        if has_available_slots():
            pending_requests = get_pending_requests()
            if pending_requests:
                prefill_logits = self.model.prefill(pending_requests)
                prefill_tokens = generate_next_tokens(prefill_logits)
                # 更新状态
                for i, (request_id, _) in enumerate(pending_requests):
                    update_request(request_id, prefill_tokens[i].item())


import torch


class KVCacheManager:
    # 当前的kvcache 的问题是： 固定了max_batch_size 和 max_seq_len， 所以有可能造成浪费。因为请求长度不一，有长有短，短请求较多就浪费了。
    def __init__(self, config):
        self.k_cache = torch.zeros(
            config.num_layers,
            config.max_batch_size,
            config.max_seq_len,
            config.num_kv_heads,
            config.head_dim,
        )
        self.v_cache = torch.zeros(
            config.num_layers,
            config.max_batch_size,
            config.max_seq_len,
            config.num_kv_heads,
            config.head_dim,
        )
        # 记录每个请求推理到的len，这个是动态变化的，因为随着decode，这个长度不断的在加
        self.sequence_lengths = torch.zeros(config.max_batch_size, dtype=torch.long)
        self.request_to_slot = (
            {}
        )  # request id --> slot_index, 当前没有PagedKVCache，所以是一对一的
        self.slot_to_request = {}  # slot_index -> request id
        self.free_slots = set(
            range(config.max_batch_size)
        )  # kvcache 所有的槽位，因为有max_batch_size请求上线，所以这里也是max_batch_size个

    def update_slots(self, slot_ids, new_kv_cache):
        print(f"update slots: {slot_ids}")
        # 按照层更新kvcache
        for i, layer_kv_cache in enumerate(new_kv_cache):
            # 获取这一批请求的batcj_size，对齐的sequence长度，kv头, 头维度的大小
            batch_size, seq_len, num_kv_heads, head_dim = layer_kv_cache[0].shape
            if seq_len == 1:  # 代表是Decoding
                # 每个序列request的当前的长度，decode会不断的加， slots_ids 一一对应requests
                cur_len = self.sequence_lengths[slot_ids]
                self.k_cache[i, slot_ids, cur_len, :, :] = layer_kv_cache[0][:, 0, :, :]
                self.v_cache[i, slot_ids, cur_len, :, :] = layer_kv_cache[1][:, 0, :, :]
            else:  # prefill
                self.k_cache[i, slot_ids, :seq_len, :, :] = layer_kv_cache[0]
                self.v_cache[i, slot_ids, :seq_len, :, :] = layer_kv_cache[1]

    def free_slot(self, request_id: int):
        print(f"free request_id {request_id}, free it's kvcache")
        if request_id in self.request_to_slot:
            slot_id = self.request_to_slot[request_id]
            del self.request_to_slot[request_id]
            del self.slot_to_request[slot_id]

            # 释放槽位，更新k_cahce, v_cache
            self.free_slots.add(slot_id)
            self.k_cache[:, slot_id, :, :, :] = 0
            self.v_cache[:, slot_id, :, :, :] = 0


config = Config()
toymodel = ToyModel(config)
worker = ContinueBatchingEngine(toymodel, config)

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
