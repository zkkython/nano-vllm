import os
from dataclasses import dataclass
from transformers import AutoConfig
from typing import Optional

from nanovllm.log_config import LogConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = (
        4096  # max_model_len确实代表了单个请求（sequence）的最大长度限制，这个长度包括了prefill阶段的输入token数量加上后续decode阶段生成的token数量的总和
    )
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False

    # Chunked Prefill 配置
    enable_chunked_prefill: bool = False  # 是否启用 Chunked Prefill
    chunked_prefill_size: int | None = (
        None  # Chunk 大小，None 表示使用 max_num_batched_tokens
    )

    # 权重加载配置
    load_partial_layers: int | None = None  # 只加载前 N 层，None 表示加载所有层

    # 日志配置
    log_config: Optional[LogConfig] = None

    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    master_addr: str = "localhost"
    master_port: int = 2333
    local_rank: int = 0
    node_rank: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )

        # 初始化日志配置
        if self.log_config is None:
            from nanovllm.log_config import get_log_config, set_log_config

            self.log_config = get_log_config()
        else:
            from nanovllm.log_config import set_log_config

            set_log_config(self.log_config)

        # Chunked Prefill 配置验证
        if self.enable_chunked_prefill:
            # 如果启用 Chunked Prefill，允许 max_num_batched_tokens < max_model_len
            if self.chunked_prefill_size is None:
                # 默认使用 max_num_batched_tokens 作为 chunk size
                self.chunked_prefill_size = self.max_num_batched_tokens
            else:
                # 验证 chunk size 合理性
                assert (
                    self.chunked_prefill_size > 0
                ), "chunked_prefill_size must be positive"
                assert (
                    self.chunked_prefill_size <= self.max_num_batched_tokens
                ), "chunked_prefill_size should not exceed max_num_batched_tokens"
        else:
            # 如果不启用 Chunked Prefill，保持原有限制
            from nanovllm.log_config import log_info

            log_info(
                "config",
                f"max num batch tokens {self.max_num_batched_tokens}, max model len {self.max_model_len}",
            )
            assert (
                self.max_num_batched_tokens >= self.max_model_len
            ), "max_num_batched_tokens must be >= max_model_len when chunked_prefill is disabled"
