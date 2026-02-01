from collections import deque

from nanovllm.config import Config, EngineRole
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.config = config
        # 支持的prefill最大语句数量
        self.max_num_seqs_prefill = config.max_num_seqs_prefill
        self.max_num_seqs_decode = config.max_num_seqs_decode
        # 支持的最大prefill tokens数量
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos

        # Chunked Prefill 配置
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.chunked_prefill_size = config.chunked_prefill_size

        # Paged KVCache
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        # 存放待执行prefill的req
        self.waiting: deque[Sequence] = deque()
        # 存放待执行decode model runner 的req
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # continuous batch
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # 如果是 DECODE 节点，跳过 prefill 逻辑（除非我们允许 local prefill）
        if self.config.engine_role == EngineRole.DECODE:
            goto_decode = True
        else:
            goto_decode = False

        if not goto_decode:
            # 拿到prefiling的requests 给model_runner执行
            while self.waiting and num_seqs < self.max_num_seqs_prefill:
                seq = self.waiting[0]

                # 判断是否启用 Chunked Prefill
                if not self.enable_chunked_prefill:
                    # 不启用 Chunked Prefill，使用原有逻辑
                    if num_batched_tokens + len(
                        seq
                    ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(
                        seq
                    ):
                        break
                    num_seqs += 1
                    self.block_manager.allocate(seq)
                    num_batched_tokens += len(seq) - seq.num_cached_tokens
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                    scheduled_seqs.append(seq)
                else:
                    # 启用 Chunked Prefill
                    # 既然要做prefill首先申请paged kvcache来承载kv cache
                    if not seq.block_table:
                        if not self.block_manager.can_allocate(seq):
                            break
                        self.block_manager.allocate(seq)
                        seq.num_prefilled_tokens = seq.num_cached_tokens

                    remaining_tokens = seq.num_prompt_tokens - seq.num_prefilled_tokens
                    if remaining_tokens <= 0:
                        self.waiting.popleft()
                        self.running.append(seq)
                        continue

                    # 使用配置的 chunk size
                    max_chunk_size = self.chunked_prefill_size
                    chunk_size = min(
                        remaining_tokens,
                        max_chunk_size,
                        self.max_num_batched_tokens - num_batched_tokens,
                    )
                    if chunk_size <= 0:
                        break

                    seq.current_chunk_size = chunk_size
                    num_batched_tokens += chunk_size
                    num_seqs += 1
                    seq.status = SequenceStatus.RUNNING

                    if seq.num_prefilled_tokens + chunk_size >= seq.num_prompt_tokens:
                        self.waiting.popleft()
                        self.running.append(seq)

                    scheduled_seqs.append(seq)

                    if num_batched_tokens >= self.max_num_batched_tokens:
                        break

        if scheduled_seqs:
            return scheduled_seqs, True

        # 如果是 PREFILL 节点且没有 prefill 请求，返回空等待
        if self.config.engine_role == EngineRole.PREFILL:
            return [], False

        # decode, 拿到decoding的requests 给model runner执行, 能进到decode 说明一定没有prefill的请求
        while self.running and num_seqs < self.max_num_seqs_decode:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        if not scheduled_seqs:
             return [], False
             
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def add_running_sequence(self, seq: Sequence):
        """用于从外部（如 KV Transfer）直接添加已完成 Prefill 的序列"""
        seq.status = SequenceStatus.RUNNING
        self.running.append(seq)

    def remove_sequence(self, seq: Sequence):
        """从调度器中彻底移除序列（通常在 Prefill 节点发送完 KV 后调用）"""
        if seq in self.waiting:
            self.waiting.remove(seq)
        if seq in self.running:
            self.running.remove(seq)
        # 注意：这里可能需要释放 block_manager 中的 blocks，
        # 但如果是 PD 分离，这些 blocks 实际上已经完成使命并被传输了。
        # 如果不释放，可能会导致内存泄漏。
        self.block_manager.deallocate(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            if self.enable_chunked_prefill and token_id is None:
                # Chunked prefill 中间块，没有输出token
                seq.num_prefilled_tokens += seq.current_chunk_size
                continue

            # 更新 prefilled tokens（如果是最后一块 prefill 或者 decode）
            if (
                self.enable_chunked_prefill
                and seq.num_prefilled_tokens < seq.num_prompt_tokens
            ):
                seq.num_prefilled_tokens += getattr(seq, "current_chunk_size", 0)

            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
