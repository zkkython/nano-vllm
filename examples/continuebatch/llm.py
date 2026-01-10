import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class ContinueBatchingEngineConfig:
    max_batch_size = 4  # 64
    max_seq_len = 32  # 512
    max_prompt_len: int = 16  # 128

    # model & kv cache
    num_layers = 3
    dim = 16
    num_heads = 2
    head_dim = 8
    vocab_size = 20


config = ContinueBatchingEngineConfig()
EOS_TOKEN = 0


class Request:
    def __init__(self, request_id: int, prompt: List[int], max_len: int = 2048):
        self.request_id = request_id
        self.prompt = prompt
        self.generated_tokens = []
        self.status = "REQUEST_WAITING"  # waiting, running, completed
        self.current_length = len(prompt)
        self.max_length = max_len

    def add_token(self, token: int):
        """添加生成的token到请求中"""
        self.generated_tokens.append(token)
        self.current_length += 1
        if self.is_finished():
            self.status = "REQUEST_COMPLETED"
            print(
                f"finished: ID.{self.request_id}, new_len:{len(self.generated_tokens)}"
            )

    def is_finished(self) -> bool:
        """检查请求是否完成（达到最大长度或生成了EOS）"""
        return self.current_length >= self.max_length or (
            self.generated_tokens and self.generated_tokens[-1] == EOS_TOKEN
        )

    def get_full_sequence(self) -> List[int]:
        """获取完整的序列（prompt + 生成的tokens）"""
        return self.prompt + self.generated_tokens


from queue import deque


class RequestManager:
    """管理所有请求的调度和状态"""

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.requests = {}  # request_id -> Request
        self.waiting_queue = deque()
        self.running_requests = set()
        self.next_request_id = 0

    def add_request(self, prompt: List[int], max_seq_len: int) -> int:
        """添加新请求，返回请求ID"""
        request_id = self.next_request_id
        self.next_request_id += 1
        request = Request(request_id, prompt, max_seq_len)
        self.requests[request_id] = request
        self.waiting_queue.append(request_id)
        return request_id

    def get_available_slots(self) -> int:
        """获取可用的批次空位数量"""
        return self.max_batch_size - len(self.running_requests)

    def get_pending_requests(self, max_count: int) -> List[Tuple[int, List[int]]]:
        """获取等待处理的请求"""
        available_slots = self.get_available_slots()
        count = min(max_count, available_slots, len(self.waiting_queue))

        requests_to_process = []
        for _ in range(count):
            if not self.waiting_queue:
                break
            request_id = self.waiting_queue.popleft()
            request = self.requests[request_id]
            request.status = "REQUEST_RUNNING"
            self.running_requests.add(request_id)
            requests_to_process.append((request_id, request.prompt))
        return requests_to_process

    def update_request(self, request_id: int, next_token: int):
        """更新请求状态"""
        if request_id in self.requests:
            request = self.requests[request_id]
            request.add_token(next_token)
            if request.is_finished():
                self.running_requests.discard(request_id)

    def has_pending_requests(self) -> bool:
        """检查是否有未完成的请求"""
        return len(self.waiting_queue) > 0 or len(self.running_requests) > 0

    def get_num_pending_requests(self) -> bool:
        return len(self.waiting_queue)

    def get_num_running_requests(self) -> bool:
        return len(self.running_requests)

    def get_running_request_ids(self) -> List[int]:
        """获取当前正在运行的请求ID"""
        return list(self.running_requests)


class KVCacheManager:
    """管理Transformer的KV缓存"""

    def __init__(self, config):
        # 初始化KV缓存 [layer, batch, seq, head, dim]

        self.k_cache = torch.zeros(
            config.num_layers,
            config.max_batch_size,
            config.max_seq_len,
            config.num_heads,
            config.head_dim,
        )
        self.v_cache = torch.zeros(
            config.num_layers,
            config.max_batch_size,
            config.max_seq_len,
            config.num_heads,
            config.head_dim,
        )

        self.sequence_lengths = torch.zeros(config.max_batch_size, dtype=torch.long)
        self.request_to_slot = {}  # request_id -> slot_index
        self.slot_to_request = {}  # slot_index -> request_id
        self.free_slots = set(range(config.max_batch_size))

    def has_active_requests(self) -> bool:
        """检查是否有活跃的请求"""
        return len(self.slot_to_request) > 0

    def has_available_slots(self) -> bool:
        """检查是否有可用的槽位"""
        return len(self.free_slots) > 0

    def get_available_slots(self) -> bool:
        """检查是否有可用的槽位"""
        return len(self.free_slots)

    def allocate_slots(self, request_ids: List[int]) -> List[int]:
        """为请求分配槽位"""
        allocated_slots = []
        for request_id in request_ids:
            if not self.free_slots:
                break
            slot_id = self.free_slots.pop()
            self.request_to_slot[request_id] = slot_id
            self.slot_to_request[slot_id] = request_id
            self.sequence_lengths[slot_id] = 0
            allocated_slots.append(slot_id)
        return allocated_slots

    def free_slot(self, request_id: int):
        """释放请求占用的槽位"""
        if request_id in self.request_to_slot:
            slot_id = self.request_to_slot[request_id]
            del self.request_to_slot[request_id]
            del self.slot_to_request[slot_id]
            self.free_slots.add(slot_id)
            # 清空该槽位的缓存
            self.k_cache[:, slot_id, :, :, :] = 0
            self.v_cache[:, slot_id, :, :, :] = 0

    def update_slots(self, slot_ids, new_kv_cache):
        for i, layer_kv_cache in enumerate(new_kv_cache):
            bsz, seq_len, num_heads, head_dim = layer_kv_cache[0].shape
            if seq_len == 1:  # decoding mode
                cur_len = self.sequence_lengths[slot_ids]
                self.k_cache[i, slot_ids, cur_len, :, :] = layer_kv_cache[0][:, 0, :, :]
                self.v_cache[i, slot_ids, cur_len, :, :] = layer_kv_cache[1][:, 0, :, :]
            else:  # Prefill mode
                self.k_cache[i, slot_ids, :seq_len, :, :] = layer_kv_cache[0]
                self.v_cache[i, slot_ids, :seq_len, :, :] = layer_kv_cache[1]

    def update_after_step(self, new_tokens: torch.Tensor):
        """更新步进后的序列长度"""
        active_slots = list(self.slot_to_request.keys())
        self.sequence_lengths[active_slots] += 1

    def get_active_slots_info(self) -> Tuple[List[int], List[int]]:
        """获取活跃槽位的信息"""
        active_slots = list(self.slot_to_request.keys())
        request_ids = [self.slot_to_request[slot] for slot in active_slots]
        return active_slots, request_ids

    def get_kv_cache_for_slots(
        self, slot_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定槽位的KV缓存"""
        return self.k_cache[:, slot_ids], self.v_cache[:, slot_ids]


import math


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.WQ = nn.Linear(config.dim, config.dim, bias=False)
        self.WK = nn.Linear(config.dim, config.dim, bias=False)
        self.WV = nn.Linear(config.dim, config.dim, bias=False)
        self.WO = nn.Linear(config.dim, config.dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, X, kvcache=None, current_length=None):
        bsz, seq_len, _ = X.shape
        Q, K, V = self.WQ(X), self.WK(X), self.WV(X)
        Q = Q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(bsz, seq_len, self.num_heads, self.head_dim)

        if kvcache is None:
            K_, V_ = K, V
        else:
            K_cache = kvcache[0]
            V_cache = kvcache[1]

            # # left padding 拼接方式
            # K_ = torch.cat((K_cache, K), dim = 1)
            # V_ = torch.cat((V_cache, V), dim = 1)

            # right padding 填充方式
            cache_col = torch.zeros(bsz, 1, self.num_heads, self.head_dim)
            K_ = torch.cat((K_cache, cache_col.clone()), dim=1)
            V_ = torch.cat((V_cache, cache_col.clone()), dim=1)
            K_[:, current_length, :, :] = K
            V_[:, current_length, :, :] = V

        K_ = K_.transpose(1, 2)
        V_ = V_.transpose(1, 2)

        S = Q @ K_.transpose(2, 3) // math.sqrt(self.head_dim)
        P = F.softmax(S, dim=-1)
        Z = P @ V_
        Z = Z.transpose(1, 2).reshape(bsz, seq_len, self.dim)
        O = self.WO(Z)

        # activate & shorcut
        O_ = X + self.act(O)

        return O_, [K, V]


class ToyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size)
        self.decoder = nn.ModuleList(
            [DecoderBlock(config) for i in range(config.num_layers)]
        )

    def forward(self, x, kvcaches=None, current_length=None):
        layer_kvcaches = []
        X = self.embd(x)

        for i, block in enumerate(self.decoder):
            if kvcaches == None:
                X, layer_kvcache = block(X, None, None)
            else:
                X, layer_kvcache = block(
                    X,
                    kvcache=[kvcaches[0][i], kvcaches[1][i]],
                    current_length=current_length,
                )

            layer_kvcaches.append(layer_kvcache)
        logits = self.lm_head(X)
        return logits, layer_kvcaches


class ModelWrapper:
    """封装模型的前向传播"""

    def __init__(self, model, kv_cache_manager: KVCacheManager):
        self.model = model
        self.kv_cache_manager = kv_cache_manager

    def prefill_requests(self, requests: List[Tuple[int, List[int]]]) -> torch.Tensor:
        """预填充新请求"""
        if not requests:
            return torch.tensor([])

        # 分配槽位
        request_ids = [req_id for req_id, _ in requests]
        slot_ids = self.kv_cache_manager.allocate_slots(request_ids)

        # 准备输入 - 将不同长度的 prompt 填充到相同长度
        prompts = [prompt for _, prompt in requests]
        max_len = max(len(prompt) for prompt in prompts)

        input_ids = torch.zeros(len(prompts), max_len, dtype=torch.long)
        for i, prompt in enumerate(prompts):
            input_ids[i, : len(prompt)] = torch.tensor(prompt)

        # 执行预填充
        with torch.no_grad():
            logits, layer_kvcaches = self.model(
                input_ids,
            )

        # 更新KV缓存
        self._update_kv_cache(slot_ids, layer_kvcaches)

        # 返回最后一个token的logits
        return logits[:, -1, :].unsqueeze(1)  # bsz, seq_len, vocab_size

    def decode_next_tokens(
        self, next_tokens: torch.Tensor, slot_ids: List[int], current_length
    ) -> torch.Tensor:
        """解码下一个token"""
        if len(slot_ids) == 0:
            return torch.tensor([])

        with torch.no_grad():
            logits, layer_kvcaches = self.model(
                next_tokens,
                kvcaches=self.kv_cache_manager.get_kv_cache_for_slots(slot_ids),
                current_length=current_length,
            )

        # 更新KV缓存
        self._update_kv_cache(slot_ids, layer_kvcaches)

        return logits

    def _update_kv_cache(self, slot_ids: List[int], new_kv_cache):
        """更新KV缓存"""
        self.kv_cache_manager.update_slots(slot_ids, new_kv_cache)
        return

    def generate_next_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """从logits生成下一个token（贪婪采样）"""
        if len(logits) == 0:
            return torch.tensor([])
        return torch.argmax(logits, dim=-1)


class ContinueBatchingEngine:
    """连续批处理主引擎"""

    def __init__(self, model, config):
        self.kv_cache_manager = KVCacheManager(config)
        self.model_wrapper = ModelWrapper(model, self.kv_cache_manager)
        self.request_manager = RequestManager(
            config.max_batch_size,
        )

    def add_request(self, prompt: List[int], max_seq_len) -> int:
        """添加新请求"""
        return self.request_manager.add_request(prompt, max_seq_len)

    def step(self):
        """
        step 函数先 decoding 再 prefill, 模拟多个 step, 假设每一步都新增一个请求
        t1: prefill req1
        t2: decoding req1, prefill req2
        t3: decoding req1,2 prefill req3
        """
        # 阶段1: 处理解码（已有请求）
        if self.kv_cache_manager.has_active_requests():
            active_slots, request_ids = self.kv_cache_manager.get_active_slots_info()

            # 准备输入token (上一个 step 生成的token)
            input_tokens = torch.tensor(
                [
                    self.request_manager.requests[req_id].generated_tokens[-1]
                    for req_id in request_ids
                ],
                dtype=torch.long,
            )
            current_length = torch.tensor(
                [
                    self.request_manager.requests[req_id].current_length
                    for req_id in request_ids
                ],
                dtype=torch.long,
            )
            input_tokens = input_tokens.unsqueeze(dim=1)

            # 解码
            decoding_logits = self.model_wrapper.decode_next_tokens(
                input_tokens, active_slots, current_length
            )
            next_tokens = self.model_wrapper.generate_next_tokens(decoding_logits)

            # 更新状态
            self.kv_cache_manager.update_after_step(next_tokens)
            for i, request_id in enumerate(request_ids):
                self.request_manager.update_request(request_id, next_tokens[i].item())
                if self.request_manager.requests[request_id].is_finished():
                    self.kv_cache_manager.free_slot(request_id)

        # 阶段2: 处理预填充（新请求）
        if self.kv_cache_manager.has_available_slots():
            pending_requests = self.request_manager.get_pending_requests(
                self.kv_cache_manager.get_available_slots()
            )
            if pending_requests:
                prefill_logits = self.model_wrapper.prefill_requests(pending_requests)
                prefill_tokens = self.model_wrapper.generate_next_tokens(prefill_logits)
                for i, (request_id, _) in enumerate(pending_requests):
                    self.request_manager.update_request(
                        request_id, prefill_tokens[i].item()
                    )

    def has_pending_work(self) -> bool:
        """检查是否还有未完成的工作"""
        return self.request_manager.has_pending_requests()

    def get_requests_info(self):
        pending = self.request_manager.get_num_pending_requests()
        running = self.request_manager.get_num_running_requests()
        total_request = len(self.request_manager.requests)
        return pending, running, total_request


from random import randint


def listen_request(config, p=0.01):
    prompt = []
    prompt_len = 0
    num = randint(1, 100)
    if num / 100.0 < p:
        prompt_len = randint(config.max_prompt_len // 4, config.max_prompt_len)
        prompt = torch.randint(low=1, high=config.vocab_size, size=(1, prompt_len))
        prompt = prompt[0].tolist()
    return prompt, prompt_len


N = 32
count = 0

model = ToyModel(config)
engine = ContinueBatchingEngine(model, config)

# main
while 1:
    # 监听进程
    if count != N:
        prompt, prompt_len = listen_request(config, p=0.5)
        if prompt_len != 0:
            count += 1
            if count % (N // 10) == 0:
                per = count / (N // 10)
                print("Running...:", "*" * int(per), "-" * (10 - int(per)))
            generate_len = randint(prompt_len, config.max_seq_len)
            engine.add_request(prompt, generate_len)
            pending, running, total = engine.get_requests_info()
            print(f"[Request Info] pending:{pending}/running:{running}/total:{total}")

    # 处理进程
    engine.step()

    if not engine.has_pending_work() and count == N:
        print("process done")
        break
