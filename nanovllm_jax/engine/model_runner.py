import jax
import jax.numpy as jnp
from typing import List, Tuple
from nanovllm_jax.config import Config
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.layers.sampler import Sampler
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM, Qwen3ForCausalLMVarlen
from nanovllm_jax.utils.context import set_context, get_context, reset_context
from nanovllm_jax.configs.model_config import ModelConfig
from nanovllm_jax.utils.weight_utils import WeightLoader
from nanovllm_jax.utils.test_weight_emd_lm import TestQwen3NNXForward
import logging

logger = logging.getLogger(__name__)


class ModelRunner:
    """JAX model runner for inference."""

    def __init__(self, config: Config, rank: int = 0):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        # Initialize JAX devices
        self.devices = jax.devices()
        if len(self.devices) > 1:
            self.device = self.devices[rank % len(self.devices)]
        else:
            self.device = self.devices[0]

        # Create a simple 1D mesh over available devices
        self.mesh = jax.sharding.Mesh(self.devices[: self.world_size], ("tensor",))

        # Set default dtype
        self.default_dtype = jnp.bfloat16

        # Initialize NNX Qwen3 model
        self.model = Qwen3ForCausalLM(
            config=hf_config,
            dtype=self.default_dtype,
            rngs=None,
            mesh=self.mesh,
        )

        # Load pretrained weights using existing NNX weight loader utilities
        model_config = ModelConfig(model_path=self.config.model, trust_remote_code=True)
        helper = TestQwen3NNXForward()
        helper.hf_config = hf_config
        weight_mappings = helper._build_weight_mappings()
        loader = WeightLoader(
            model=self.model,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.default_dtype,
        )
        loader.load_weights_from_safetensors(weight_mappings)

        # Initialize sampler
        self.sampler = Sampler()

        # Warmup and allocate KV cache
        self.warmup_model()
        self.allocate_kv_cache()

    def warmup_model(self):
        """Warmup model to measure peak memory."""
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)

    def allocate_kv_cache(self):
        """Allocate KV cache based on available memory."""
        config = self.config
        hf_config = config.hf_config

        # Calculate KV cache dimensions
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = (
            hf_config.head_dim
            if hasattr(hf_config, "head_dim")
            else hf_config.hidden_size // hf_config.num_attention_heads
        )

        # Get dtype from hf_config
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is None:
            dtype = jnp.float32
            itemsize = 4
        else:
            dt_str = str(torch_dtype)
            if "bfloat16" in dt_str:
                dtype = jnp.bfloat16
                itemsize = 2
            elif "float16" in dt_str or "half" in dt_str:
                dtype = jnp.float16
                itemsize = 2
            else:
                dtype = jnp.float32
                itemsize = 4

        # Calculate block size in bytes
        block_bytes = (
            2  # k and v cache
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * itemsize
        )

        # Estimate available memory (JAX doesn't have direct memory query like CUDA)
        # Use a conservative estimate based on gpu_memory_utilization
        # For simplicity, use a default number of blocks if not specified
        if config.num_kvcache_blocks <= 0:
            # Conservative default: ~100 blocks
            config.num_kvcache_blocks = 100

        assert config.num_kvcache_blocks > 0

        # Allocate KV cache
        self.kv_cache = jnp.zeros(
            (
                2,  # k and v
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
            ),
            dtype=dtype,
        )

        # Note: In JAX version, we don't directly assign k_cache and v_cache to modules
        # as the model manages its own internal cache. This is kept for potential
        # future external cache implementation.

    def prepare_block_tables(self, seqs: List[Sequence]) -> jnp.ndarray:
        """Prepare block tables from sequences."""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = jnp.array(block_tables, dtype=jnp.int32)
        return block_tables

    def prepare_prefill(self, seqs: List[Sequence]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare inputs for prefill phase."""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        # Check for prefix cache
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        input_ids = jnp.array(input_ids, dtype=jnp.int64)
        positions = jnp.array(positions, dtype=jnp.int64)
        cu_seqlens_q = jnp.array(cu_seqlens_q, dtype=jnp.int32)
        cu_seqlens_k = jnp.array(cu_seqlens_k, dtype=jnp.int32)
        slot_mapping = (
            jnp.array(slot_mapping, dtype=jnp.int32) if slot_mapping else None
        )

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: List[Sequence]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare inputs for decode phase."""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        input_ids = jnp.array(input_ids, dtype=jnp.int64)
        positions = jnp.array(positions, dtype=jnp.int64)
        slot_mapping = jnp.array(slot_mapping, dtype=jnp.int32)
        context_lens = jnp.array(context_lens, dtype=jnp.int32)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: List[Sequence]) -> jnp.ndarray:
        """Prepare sampling parameters."""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = jnp.array(temperatures, dtype=jnp.float32)
        return temperatures

    def run_model(
        self, input_ids: jnp.ndarray, positions: jnp.ndarray, is_prefill: bool
    ) -> jnp.ndarray:
        """Run the model forward pass with external KV cache."""
        # Context is already set by prepare_prefill or prepare_decode
        # JAX version now supports varlen attention like PyTorch

        if is_prefill:
            # For prefill, input_ids is flattened [total_tokens]
            # Pass to model which will use varlen attention internally
            # The model should process flattened sequences using cu_seqlens from context
            logits = self.run_model_with_cache(input_ids, positions, is_prefill)
        else:
            # For decode, input_ids is [batch_size]
            # Use paged attention with block tables
            logits = self.run_model_with_cache(input_ids, positions, is_prefill)

        return logits

    def run_model_with_cache(
        self, input_ids: jnp.ndarray, positions: jnp.ndarray, is_prefill: bool
    ) -> jnp.ndarray:
        """Run model with external KV cache support."""
        # For now, we use the model's internal KV cache
        # TODO: Integrate external KV cache with varlen attention

        if is_prefill:
            # Prefill: Clear cache and process flattened input
            self.model.clear_kv_cache()
            # Reshape to [1, seq_len] for model input
            input_ids_batched = input_ids[None, :]  # [1, total_tokens]
            logits = self.model(input_ids_batched, use_cache=True, is_decode=False)
            # Output shape is [1, seq_len, vocab_size]
            logits = logits[0, :, :]  # [seq_len, vocab_size]

            # Extract logits for the last token of each sequence using cu_seqlens_q
            context = get_context()
            if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
                # Get the indices of the last token for each sequence
                # cu_seqlens_q: [0, len1, len1+len2, ...]
                # Last token indices: [len1-1, len1+len2-1, ...]
                last_indices = []
                for i in range(1, len(context.cu_seqlens_q)):
                    last_idx = int(context.cu_seqlens_q[i]) - 1
                    last_indices.append(last_idx)
                # Extract logits for these positions
                logits = logits[jnp.array(last_indices), :]  # [num_seqs, vocab_size]
            else:
                # Single sequence, take last token
                logits = logits[-1:, :]  # [1, vocab_size]
        else:
            # Decode: Process each token with KV cache
            # input_ids is [batch_size], reshape to [batch_size, 1]
            input_ids_batched = input_ids[:, None]  # [batch_size, 1]
            logits = self.model(input_ids_batched, use_cache=True, is_decode=True)
            # Output shape is [batch_size, 1, vocab_size]
            logits = logits[:, 0, :]  # [batch_size, vocab_size]

        return logits

    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        """Run inference on sequences."""
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)

        # Sample tokens
        if self.rank == 0:
            import time

            rng = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
            token_ids = self.sampler(logits, temperatures, rng).tolist()
        else:
            token_ids = None

        reset_context()
        return token_ids


class ModelRunnerVarlen:
    """JAX model runner with varlen attention support.

    This version uses Qwen3ForCausalLMVarlen with:
    - External KV cache management per layer
    - Varlen attention for flattened sequences
    - Paged attention for decode phase
    """

    def __init__(self, config: Config, rank: int = 0, use_varlen: bool = True):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.use_varlen = use_varlen

        # Initialize JAX devices
        self.devices = jax.devices()
        if len(self.devices) > 1:
            self.device = self.devices[rank % len(self.devices)]
        else:
            self.device = self.devices[0]

        # Create a simple 1D mesh over available devices
        self.mesh = jax.sharding.Mesh(self.devices[: self.world_size], ("tensor",))

        # Set default dtype
        self.default_dtype = jnp.bfloat16

        # Initialize model with varlen support
        if use_varlen:
            self.model = Qwen3ForCausalLMVarlen(
                config=hf_config,
                dtype=self.default_dtype,
                block_size=self.block_size,
                rngs=None,
                mesh=self.mesh,
            )
        else:
            self.model = Qwen3ForCausalLM(
                config=hf_config,
                dtype=self.default_dtype,
                rngs=None,
                mesh=self.mesh,
            )

        # Load pretrained weights
        self.model.load_weights(self.config)
        logger.info(f"{self.config.model} Weights loaded finished.")
        # model_config = ModelConfig(model_path=self.config.model, trust_remote_code=True)
        # helper = TestQwen3NNXForward()
        # helper.hf_config = hf_config
        # weight_mappings = helper._build_weight_mappings()
        # loader = WeightLoader(
        #     model=self.model,
        #     model_config=model_config,
        #     mesh=self.mesh,
        #     dtype=self.default_dtype,
        # )
        # loader.load_weights_from_safetensors(weight_mappings)

        # Initialize sampler
        self.sampler = Sampler()

        # Warmup and allocate KV cache
        self.warmup_model()
        self.allocate_kv_cache()

    def warmup_model(self):
        """Warmup model to measure peak memory."""
        if not self.use_varlen:
            # Use standard warmup for non-varlen model
            max_num_batched_tokens = self.config.max_num_batched_tokens
            max_model_len = self.config.max_model_len
            num_seqs = min(
                max_num_batched_tokens // max_model_len, self.config.max_num_seqs
            )
            seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
            self.run(seqs, True)
        # For varlen, skip warmup or implement a simple one

    def allocate_kv_cache(self):
        """Allocate external KV cache for each layer."""
        config = self.config
        hf_config = config.hf_config

        # Calculate KV cache dimensions
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = (
            hf_config.head_dim
            if hasattr(hf_config, "head_dim")
            else hf_config.hidden_size // hf_config.num_attention_heads
        )

        # Get dtype from hf_config
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is None:
            dtype = jnp.float32
        else:
            dt_str = str(torch_dtype)
            if "bfloat16" in dt_str:
                dtype = jnp.bfloat16
            elif "float16" in dt_str or "half" in dt_str:
                dtype = jnp.float16
            else:
                dtype = jnp.float32

        # Use conservative default if not specified
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = 100

        assert config.num_kvcache_blocks > 0

        # Allocate KV cache for each layer
        num_layers = hf_config.num_hidden_layers
        self.kv_caches = []

        for layer_idx in range(num_layers):
            k_cache = jnp.zeros(
                (
                    config.num_kvcache_blocks,
                    self.block_size,
                    num_kv_heads,
                    head_dim,
                ),
                dtype=dtype,
            )
            v_cache = jnp.zeros(
                (
                    config.num_kvcache_blocks,
                    self.block_size,
                    num_kv_heads,
                    head_dim,
                ),
                dtype=dtype,
            )
            self.kv_caches.append((k_cache, v_cache))

    def prepare_block_tables(self, seqs: List[Sequence]) -> jnp.ndarray:
        """Prepare block tables from sequences."""
        if not seqs:
            return jnp.array([], dtype=jnp.int32)
        max_len = (
            max(len(seq.block_table) for seq in seqs)
            if any(seq.block_table for seq in seqs)
            else 0
        )
        if max_len == 0:
            return None
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = jnp.array(block_tables, dtype=jnp.int32)
        return block_tables

    def prepare_prefill(self, seqs: List[Sequence]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare inputs for prefill phase."""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        # Check for prefix cache
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        input_ids = jnp.array(input_ids, dtype=jnp.int64)
        positions = jnp.array(positions, dtype=jnp.int64)
        cu_seqlens_q = jnp.array(cu_seqlens_q, dtype=jnp.int32)
        cu_seqlens_k = jnp.array(cu_seqlens_k, dtype=jnp.int32)
        slot_mapping = (
            jnp.array(slot_mapping, dtype=jnp.int32) if slot_mapping else None
        )

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: List[Sequence]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare inputs for decode phase."""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        input_ids = jnp.array(input_ids, dtype=jnp.int64)
        positions = jnp.array(positions, dtype=jnp.int64)
        slot_mapping = jnp.array(slot_mapping, dtype=jnp.int32)
        context_lens = jnp.array(context_lens, dtype=jnp.int32)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: List[Sequence]) -> jnp.ndarray:
        """Prepare sampling parameters."""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = jnp.array(temperatures, dtype=jnp.float32)
        return temperatures

    def run_model(
        self, input_ids: jnp.ndarray, positions: jnp.ndarray, is_prefill: bool
    ) -> jnp.ndarray:
        """Run the model forward pass with varlen attention."""
        if not self.use_varlen:
            # Fallback to standard model
            if is_prefill:
                self.model.clear_kv_cache()
                input_ids_batched = input_ids[None, :]
                logits = self.model(input_ids_batched, use_cache=True, is_decode=False)
                logits = logits[0, :, :]

                context = get_context()
                if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
                    last_indices = []
                    for i in range(1, len(context.cu_seqlens_q)):
                        last_idx = int(context.cu_seqlens_q[i]) - 1
                        last_indices.append(last_idx)
                    logits = logits[jnp.array(last_indices), :]
                else:
                    logits = logits[-1:, :]
            else:
                input_ids_batched = input_ids[:, None]
                logits = self.model(input_ids_batched, use_cache=True, is_decode=True)
                logits = logits[:, 0, :]
            return logits

        # Use varlen model with external KV cache
        logits, self.kv_caches = self.model(
            input_ids,
            positions=positions,
            kv_caches=self.kv_caches,
        )

        # Extract logits for sampling
        if is_prefill:
            context = get_context()
            if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
                # Get the indices of the last token for each sequence
                last_indices = []
                for i in range(1, len(context.cu_seqlens_q)):
                    last_idx = int(context.cu_seqlens_q[i]) - 1
                    last_indices.append(last_idx)
                logits = logits[jnp.array(last_indices), :]
            else:
                # Single sequence, take last token
                logits = logits[-1:, :]
        # For decode, logits are already [batch_size, vocab_size]

        return logits

    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        """Run inference on sequences."""
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)

        # Sample tokens
        if self.rank == 0:
            import time

            rng = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
            token_ids = self.sampler(logits, temperatures, rng).tolist()
        else:
            token_ids = None

        reset_context()
        return token_ids
