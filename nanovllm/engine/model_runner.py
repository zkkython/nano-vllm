import torch
import torch.distributed as dist
from nanovllm.utils.distributed import broadcast_object

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence

from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.models.models_mapping import MODELS_MAPPING
from nanovllm.log_config import (
    log_debug,
    log_info,
    log_warning,
)


class ModelRunner:

    def __init__(self, config: Config, rank: int, local_rank: int):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        # 如果没有指定local_rank，则使用rank作为local_rank（适用于单机情况）
        # 在多机环境中，应该通过环境变量或参数传入正确的local_rank
        if local_rank is None:
            local_rank = rank
        self.local_rank = local_rank

        # 添加调试信息，看看进入了哪个分支
        log_debug(
            "model_runner",
            f"ModelRunner.__init__: rank={rank}, local_rank={local_rank}, "
            f"world_size={self.world_size}, dist.is_initialized()={dist.is_initialized()}, "
            f"available_gpus={torch.cuda.device_count()}",
            rank=rank,
        )

        # 初始化分布式通信组
        if dist.is_initialized():
            # 如果已经初始化（通过torchrun），则使用现有的分布式环境
            log_debug(
                "model_runner",
                f"Rank {rank} (local_rank={local_rank}) Distributed environment already initialized by torchrun",
                rank=rank,
            )
            log_debug(
                "model_runner",
                f"Rank {rank}: Available GPUs = {torch.cuda.device_count()}, local_rank = {local_rank}",
                rank=rank,
            )
            assert (
                dist.get_world_size() == self.world_size
            ), f"World size mismatch: expected {self.world_size}, got {dist.get_world_size()}"
            assert (
                dist.get_rank() == rank
            ), f"Rank mismatch: expected {rank}, got {dist.get_rank()}"

            # 重要：即使分布式环境已初始化，我们仍需要设置正确的CUDA设备
            # 确保local_rank在可用GPU范围内
            num_gpus = torch.cuda.device_count()
            if local_rank >= num_gpus:
                # 如果local_rank超出了本地GPU数量，使用rank对本地GPU数取模
                actual_local_rank = rank % num_gpus
                log_warning(
                    "model_runner",
                    f"Rank {rank}: local_rank {local_rank} exceeds available GPUs ({num_gpus}), using device {actual_local_rank} based on rank % num_gpus",
                    rank=rank,
                )
            else:
                actual_local_rank = local_rank

            # 立即设置CUDA设备，确保后续操作使用正确的GPU
            torch.cuda.set_device(actual_local_rank)
            # 保存实际使用的设备ID，供后续使用
            self.actual_device_id = actual_local_rank
            log_debug(
                "model_runner",
                f"Rank {rank}: Set CUDA device to {actual_local_rank} (torch.cuda.current_device() = {torch.cuda.current_device()})",
                rank=rank,
            )
        else:
            if self.world_size > 1:
                log_debug(
                    "model_runner",
                    f"Rank {rank} (local_rank={local_rank}) Initializing new distributed environment",
                    rank=rank,
                )
                # 否则初始化新的分布式环境（单机多卡情况）
                init_method = (
                    f"tcp://{self.config.master_addr}:{self.config.master_port}"
                )
                # 确保device_id在可用GPU范围内
                num_gpus = torch.cuda.device_count()
                log_debug(
                    "model_runner",
                    f"Rank {rank}: Before device_id calculation - local_rank={local_rank}, num_gpus={num_gpus}",
                    rank=rank,
                )
                if local_rank >= num_gpus:
                    # 如果local_rank超出了本地GPU数量，使用rank对本地GPU数取模
                    actual_device_id = rank % num_gpus
                    log_warning(
                        "model_runner",
                        f"local_rank {local_rank} exceeds available GPUs ({num_gpus}), using device {actual_device_id} based on rank % num_gpus",
                        rank=rank,
                    )
                else:
                    actual_device_id = local_rank
                log_debug(
                    "model_runner",
                    f"Rank {rank}: Calculated actual_device_id={actual_device_id}",
                    rank=rank,
                )

                # 在初始化分布式环境之前先设置CUDA设备
                torch.cuda.set_device(actual_device_id)
                log_debug(
                    "model_runner",
                    f"Rank {rank}: Set CUDA device to {actual_device_id} before init_process_group",
                    rank=rank,
                )

                # 不传递device_id参数，让NCCL使用当前设置的CUDA设备
                dist.init_process_group(
                    backend="nccl",
                    init_method=init_method,
                    world_size=self.world_size,
                    rank=rank,
                )
                log_debug(
                    "model_runner",
                    f"Rank {rank} (local_rank={local_rank}) Distributed environment initialized",
                    rank=rank,
                )

                # CUDA设备已经在init_process_group之前设置了，保存实际使用的设备ID
                self.actual_device_id = actual_device_id
                log_debug(
                    "model_runner",
                    f"Rank {rank}: Using CUDA device {actual_device_id}",
                    rank=rank,
                )
            else:
                # 单GPU情况，直接设置设备
                num_gpus = torch.cuda.device_count()
                if local_rank >= num_gpus:
                    actual_local_rank = 0  # 单GPU默认使用设备 0
                else:
                    actual_local_rank = local_rank
                torch.cuda.set_device(actual_local_rank)
                self.actual_device_id = actual_local_rank
                log_debug(
                    "model_runner",
                    f"Rank {rank}: Single GPU mode, set CUDA device to {actual_local_rank}",
                    rank=rank,
                )
                dist.init_process_group(
                    "nccl",
                    "tcp://localhost:2333",
                    world_size=self.world_size,
                    rank=rank,
                )

        # CUDA设备已经在上面设置好了，这里不需要重复设置
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        # 为 hf_config 注入 load_partial_layers，方便模型初始化时按需分配内存
        setattr(hf_config, "load_partial_layers", config.load_partial_layers)
        setattr(hf_config, "quantization", config.quantization)
        self.model = MODELS_MAPPING[hf_config.model_type](hf_config)
        # load_model(self.model, config.model)
        self.model.load_weights(
            config=hf_config,
            model_path=config.model,
            load_partial_layers=config.load_partial_layers,
        )
        self.sampler = Sampler()
        self.allocate_kv_cache()

        # Warmup 模型（已适配 chunked prefill）
        if self.rank == 0:
            log_info(
                "model_runner",
                "Warming up model...",
                rank=self.rank,
            )
        self.warmup_model()
        if self.rank == 0:
            log_info(
                "model_runner",
                "Warmup completed",
                rank=self.rank,
            )

        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass  # barrier可能失败，继续执行

            if rank != 0:
                # 非主rank进入循环处理来自主rank的请求
                self.loop()

    def exit(self):
        """退出并清理资源"""
        log_debug(
            "model_runner",
            f"ModelRunner exit started",
            rank=self.rank,
        )
        try:
            if not self.enforce_eager:
                if hasattr(self, "graphs"):
                    del self.graphs
                if hasattr(self, "graph_pool"):
                    del self.graph_pool
        except Exception:
            pass

        try:
            torch.cuda.synchronize()
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            log_debug(
                "model_runner",
                f"synchronize Exception in exit try block: {e}",
                rank=self.rank,
            )
            pass

        log_debug(
            "model_runner",
            f"Rank {self.rank} ModelRunner exit completed",
            rank=self.rank,
        )

    def loop(self):
        # 在分布式环境中，非主rank等待主rank的指令
        # 这里使用分布式通信来协调工作
        while True:
            # 从主rank接收指令 - 使用一个信号来标记是否有真实数据
            # 使用初始化时保存的实际设备ID
            actual_device_id = self.actual_device_id
            signal = torch.zeros(1, dtype=torch.long, device=f"cuda:{actual_device_id}")
            dist.broadcast(signal, src=0)

            if signal.item() == 0:
                # 信号为0表示没有新指令
                continue

            # 信号非零表示有新指令，接收指令数据
            call_data = broadcast_object(None, src=0)
            if call_data is not None:
                method_name, args = call_data
                # 执行方法（非rank 0只执行方法）
                method = getattr(self, method_name, None)
                if method:
                    method(*args)
                else:
                    raise AttributeError(f"Method {method_name} not found")
                # 如果子进程接受到的是exit方法，那么在该方法执行完之后，就需要退出循环了
                if method_name == "exit":
                    break

    def call(self, method_name, *args):
        # rank 0主动发送指令给其他rank
        if self.world_size > 1:
            if self.rank != 0:
                raise RuntimeError(
                    f"call() should only be invoked on rank 0, got rank {self.rank}"
                )

            # 发送信号：1表示有指令
            # 使用初始化时保存的实际设备ID
            actual_device_id = self.actual_device_id
            signal = torch.ones(1, dtype=torch.long, device=f"cuda:{actual_device_id}")
            dist.broadcast(signal, src=0)

            # 广播指令
            call_data = (method_name, args)
            broadcast_object(call_data, src=0)

        # rank 0直接执行方法
        method = getattr(self, method_name, None)
        if method:
            result = method(*args)
            return result
        else:
            raise AttributeError(f"Method {method_name} not found")

    def warmup_model(self):
        """模型预热，已适配 Chunked Prefill"""
        from nanovllm.log_config import log_info, log_debug

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        config = self.config
        max_num_batched_tokens = config.max_num_batched_tokens
        max_model_len = config.max_model_len

        # 根据是否启用 chunked prefill 决定预热策略
        if config.enable_chunked_prefill:
            # Chunked Prefill 模式：使用 chunked_prefill_size
            warmup_len = min(config.chunked_prefill_size, max_model_len)
            log_info(
                "warmup",
                f"Warmup with Chunked Prefill: chunk_size={warmup_len}",
                rank=self.rank,
            )
        else:
            # 传统模式：使用 max_num_batched_tokens
            warmup_len = min(max_num_batched_tokens, max_model_len)
            log_info(
                "warmup",
                f"Warmup without Chunked Prefill: warmup_len={warmup_len}",
                rank=self.rank,
            )

        num_seqs = min(max_num_batched_tokens // warmup_len, config.max_num_seqs)
        if num_seqs == 0:
            num_seqs = 1

        # 创建预热序列
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]

        # 重要：为 warmup 序列分配 blocks，这样才能生成 slot_mapping
        # 因为 KV cache 已经分配了，store_kvcache 需要 slot_mapping
        from nanovllm.engine.block_manager import BlockManager

        temp_block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )

        for seq in seqs:
            seq.num_prefilled_tokens = 0
            seq.current_chunk_size = warmup_len
            # 分配 blocks 以便生成 slot_mapping
            if temp_block_manager.can_allocate(seq):
                temp_block_manager.allocate(seq)

        log_debug(
            "warmup",
            f"Starting warmup: {num_seqs} seqs x {warmup_len} tokens",
            rank=self.rank,
        )

        # 执行预热
        self.run(seqs, True)

        # 清理 warmup 分配的 blocks
        for seq in seqs:
            if seq.block_table:
                temp_block_manager.deallocate(seq)

        torch.cuda.empty_cache()

        log_info(
            "warmup",
            f"Warmup completed: {num_seqs} seqs x {warmup_len} tokens",
            rank=self.rank,
        )

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = (
            getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
            // self.world_size
        )
        # handling the kv heads < attention heads
        num_kv_heads = max(1, num_kv_heads)
        if hf_config.model_type == "deepseek_v3":
            # DeepSeek-V3 (MLA) uses different head dims for QK and V
            qk_head_dim = getattr(hf_config, "qk_nope_head_dim", 0) + getattr(
                hf_config, "qk_rope_head_dim", 0
            )
            v_head_dim = getattr(hf_config, "v_head_dim", 0)
            head_dim = max(qk_head_dim, v_head_dim)
        else:
            head_dim = getattr(hf_config, "head_dim", None) or (
                hf_config.hidden_size // hf_config.num_attention_heads
            )
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.torch_dtype.itemsize
        )
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        # KV 补全，补齐到最大的KV Cache table 个数
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        # (bs, max_len)
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        from nanovllm.log_config import log_debug

        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        log_debug(
            "chunked_prefill", f"Processing {len(seqs)} sequences", rank=self.rank
        )

        for seq_idx, seq in enumerate(seqs):
            chunk_size = getattr(
                seq,
                "current_chunk_size",
                seq.num_prompt_tokens - seq.num_prefilled_tokens,
            )
            start_pos = seq.num_prefilled_tokens
            end_pos = start_pos + chunk_size

            log_debug(
                "chunked_prefill",
                f"Seq {seq_idx}: start={start_pos}, end={end_pos}, chunk={chunk_size}, "
                f"total={seq.num_prompt_tokens}, blocks={len(seq.block_table) if seq.block_table else 0}",
                rank=self.rank,
            )

            input_ids.extend(
                seq.token_ids[start_pos:end_pos]
            )  # 真正执行prefill的input ids
            positions.extend(list(range(start_pos, end_pos)))  # 位置

            cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end_pos)

            max_seqlen_q = max(chunk_size, max_seqlen_q)
            max_seqlen_k = max(end_pos, max_seqlen_k)

            # slot_mapping 告诉 kernel「新 token → KV Cache 位置」
            # 只为当前 chunk 中实际需要写入的 tokens 生成 slot_mapping
            if seq.block_table:
                # 注意：只处理当前 chunk 的 tokens [start_pos:end_pos]
                for i in range(start_pos, end_pos):
                    block_idx = i // self.block_size
                    block_offset = i % self.block_size

                    # 确保 block_idx 在有效范围内
                    if block_idx >= len(seq.block_table):
                        from nanovllm.log_config import log_error

                        log_error(
                            "chunked_prefill",
                            f"Block index out of range: block_idx={block_idx} >= len(block_table)={len(seq.block_table)}, "
                            f"token_pos={i}, block_size={self.block_size}, num_blocks={seq.num_blocks}",
                            rank=self.rank,
                        )
                        raise IndexError(
                            f"Block index {block_idx} out of range for block_table of length {len(seq.block_table)}"
                        )

                    slot_mapping.append(
                        seq.block_table[block_idx] * self.block_size + block_offset
                    )
            else:
                # 如果没有 block_table，slot_mapping 应该为空（首次 prefill 且没有 KV cache）
                # Flash attention 会直接计算，不需要写入 KV cache
                pass

        log_debug(
            "chunked_prefill",
            f"Total input_ids={len(input_ids)}, slot_mapping={len(slot_mapping)}",
            rank=self.rank,
        )

        # 决定是否需要使用 block_tables
        # 1. 存在 prefix cache (num_cached_tokens > 0)
        # 2. 是 Chunked Prefill 的非第一块 (start_pos > 0)
        use_block_tables = False
        for seq in seqs:
            if seq.num_prefilled_tokens > 0 or seq.num_cached_tokens > 0:
                use_block_tables = True
                break

        if use_block_tables:
            block_tables = self.prepare_block_tables(seqs)  # (bs, max_seq_len)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )  # (len1+len22+len3...,)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
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

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(
                seq.last_token
            )  # 每次添加单个token，上一次推理出的词作为输入
            positions.append(len(seq))  # 位置是动态那个动态的值
            context_lens.append(len(seq))  # context_lens 记录每一个seq 的动态长度
            # 告诉 kernel「新 token → KV Cache 位置」，decode做完生成新的token时，只是生成了token，但是还没写入k,vcache，所以需要
            # 在下一次forward的时候，写入kvcache，所以就需要给出的kvcache的位置-1
            """
            看一下 decode 的时序（结合 Scheduler 和 ModelRunner.prepare_decode）：
            上一步已经做完一次 forward + sampling：
            新采样出来的 token 已经通过 seq.append_token(token_id) 加到了 Sequence 里
            所以现在的 len(seq)、seq.last_token 都已经包含了这个新 token
            但是：这个新 token 的 KV 还没写进 KV cache（因为 KV 只有在 forward 里面 store_kvcache 时才会写入）
            本次 prepare_decode 做的事是：
            用 seq.last_token 作为 input_ids，再 forward 一次；
            在 Attention.forward 里，会先调用
            python
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            把“这一步的 token”（也就是当前 `last_token`）的 KV 写到 KV cache 里对应的 slot。
            所以，slot_mapping 现在应该指向的就是：
            “当前这条序列里最后一个 token（last_token）在全局 KV cache 中的下标”
            而不是“下一个还没生成的 token 的位置”。
            """
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        # 补齐
        block_tables = self.prepare_block_tables(seqs)
        # 历史 token 的 K/V 已经在 paged KV cache (k_cache / v_cache) 里，block_tables 告诉 kernel 用的是哪些块
        # context_lens 告诉 kernel：这些块里有多少真实的 token 需要参与 attention
        """
        context_lens 是一个 shape 为 [batch_size] 的一维 tensor
        第 i 个元素就是第 i 条 seq 的当前长度，也就是 len(seq)，等于 Sequence.num_tokens
        在 decode 阶段，每个 step 只喂进去 last_token，但注意力需要对“之前所有 token + 当前 token”做自注意力
        历史 token 的 K/V 已经在 paged KV cache (k_cache / v_cache) 里
        block_tables 告诉 kernel 用的是哪些块
        context_lens 告诉 kernel：这些块里有多少真实的 token 需要参与 attention
        flash_attn_with_kvcache 的 cache_seqlens 参数就是干这个用的：变长 batch + 正确的 causal mask + 不用去看 cache 中还没写入的空位。

        """
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions)

            if is_prefill:
                # prefill 阶段，只对每个序列的最后一个 token 计算 logits
                context = get_context()
                last_token_indices = (context.cu_seqlens_q[1:] - 1).long()
                hidden_states = hidden_states[last_token_indices]

                # 重要：标记我们已经提取了最后一个 token，告诉 lm_head 不要再次提取
                context.is_prefill = False

            logits = self.model.compute_logits(hidden_states)
            return logits
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )

        # 只在rank 0上准备temperatures
        temperatures = None
        if self.rank == 0:
            temperatures = self.prepare_sample(seqs)

        logits = self.run_model(input_ids, positions, is_prefill)

        # 只在rank 0上进行采样
        token_ids = None
        if self.rank == 0:
            full_token_ids = self.sampler(logits, temperatures).tolist()
            if is_prefill:
                # 判断是否启用 chunked prefill
                if self.config.enable_chunked_prefill:
                    # Chunked Prefill 模式：中间 chunk 返回 None
                    token_ids = []
                    for i, seq in enumerate(seqs):
                        chunk_size = getattr(seq, "current_chunk_size", 0)
                        if (
                            seq.num_prefilled_tokens + chunk_size
                            < seq.num_prompt_tokens
                        ):
                            token_ids.append(None)
                        else:
                            token_ids.append(full_token_ids[i])
                else:
                    # 传统模式：直接返回所有 tokens
                    token_ids = full_token_ids
            else:
                token_ids = full_token_ids

        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        device = f"cuda:{self.actual_device_id}"
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=device)
        positions = torch.zeros(max_bs, dtype=torch.int64, device=device)
        # 初始化为有效值，避免 kernel 报错
        slot_mapping = torch.arange(max_bs, dtype=torch.int32, device=device)
        context_lens = torch.ones(max_bs, dtype=torch.int32, device=device)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=device)
        outputs = torch.zeros(max_bs, hf_config.hidden_size, device=device)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            # Warmup: 确保所有 kernel 都已编译（包括 Triton kernels）
            hidden_states = self.model(input_ids[:bs], positions[:bs])
            outputs[:bs].copy_(hidden_states)
            
            with torch.cuda.graph(graph, self.graph_pool):
                hidden_states = self.model(input_ids[:bs], positions[:bs])
                outputs[:bs].copy_(hidden_states)
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
