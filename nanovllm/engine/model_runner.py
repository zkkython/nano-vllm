import torch
import torch.distributed as dist
from nanovllm.utils.distributed import broadcast_object

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence

from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.models.models_mapping import MODELS_MAPPING


class ModelRunner:

    def __init__(self, config: Config, rank: int, local_rank: int = None):
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

        # 初始化分布式通信组
        if dist.is_initialized():
            # 如果已经初始化（通过torchrun），则使用现有的分布式环境
            print(
                f"[DEBUG] Rank {rank} (local_rank={local_rank}) Distributed environment already initialized by torchrun",
                flush=True,
            )
            print(
                f"[DEBUG] Rank {rank}: Available GPUs = {torch.cuda.device_count()}, local_rank = {local_rank}",
                flush=True,
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
                print(
                    f"[WARNING] Rank {rank}: local_rank {local_rank} exceeds available GPUs ({num_gpus}), using device {actual_local_rank} based on rank % num_gpus",
                    flush=True,
                )
            else:
                actual_local_rank = local_rank

            # 立即设置CUDA设备，确保后续操作使用正确的GPU
            torch.cuda.set_device(actual_local_rank)
            # 保存实际使用的设备ID，供后续使用
            self.actual_device_id = actual_local_rank
            print(
                f"[DEBUG] Rank {rank}: Set CUDA device to {actual_local_rank} (torch.cuda.current_device() = {torch.cuda.current_device()})",
                flush=True,
            )
        else:
            if self.world_size > 1:
                print(
                    f"[DEBUG] Rank {rank} (local_rank={local_rank}) Initializing new distributed environment",
                    flush=True,
                )
                # 否则初始化新的分布式环境（单机多卡情况）
                init_method = (
                    f"tcp://{self.config.master_addr}:{self.config.master_port}"
                )
                # 确保device_id在可用GPU范围内
                num_gpus = torch.cuda.device_count()
                if local_rank >= num_gpus:
                    # 如果local_rank超出了本地GPU数量，使用rank对本地GPU数取模
                    actual_device_id = rank % num_gpus
                    print(
                        f"[WARNING] local_rank {local_rank} exceeds available GPUs ({num_gpus}), using device {actual_device_id} based on rank % num_gpus",
                        flush=True,
                    )
                else:
                    actual_device_id = local_rank
                dist.init_process_group(
                    backend="nccl",
                    init_method=init_method,
                    world_size=self.world_size,
                    rank=rank,
                    device_id=actual_device_id,
                )
                print(
                    f"[DEBUG] Rank {rank} (local_rank={local_rank}) Distributed environment initialized",
                    flush=True,
                )

                # 立即设置CUDA设备
                num_gpus = torch.cuda.device_count()
                if local_rank >= num_gpus:
                    actual_local_rank = rank % num_gpus
                    print(
                        f"[WARNING] Rank {rank}: local_rank {local_rank} exceeds available GPUs ({num_gpus}), using device {actual_local_rank}",
                        flush=True,
                    )
                else:
                    actual_local_rank = local_rank
                torch.cuda.set_device(actual_local_rank)
                # 保存实际使用的设备ID，供后续使用
                self.actual_device_id = actual_local_rank
                print(
                    f"[DEBUG] Rank {rank}: Set CUDA device to {actual_local_rank}",
                    flush=True,
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
                print(
                    f"[DEBUG] Rank {rank}: Single GPU mode, set CUDA device to {actual_local_rank}",
                    flush=True,
                )

        # CUDA设备已经在上面设置好了，这里不需要重复设置
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        self.model = MODELS_MAPPING[hf_config.model_type](hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
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
        print(f"[DEBUG] {self.rank} ModelRunner exit started", flush=True)
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
            print(f"[DEBUG] synchronize Exception in exit try block: {e}", flush=True)
            pass

        print(f"[DEBUG] Rank {self.rank} ModelRunner exit completed", flush=True)

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
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = (
            hf_config.head_dim
            if hasattr(hf_config, "head_dim")
            else hf_config.hidden_size // hf_config.num_attention_heads
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
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
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
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
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
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
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
        block_tables = self.prepare_block_tables(seqs)
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
            return self.model.compute_logits(self.model(input_ids, positions))
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

        # 将temperatures广播给所有rank
        temperatures = broadcast_object(temperatures, src=0)

        logits = self.run_model(input_ids, positions, is_prefill)

        # 只在rank 0上进行采样
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, temperatures).tolist()

        # 将结果广播给所有rank
        token_ids = broadcast_object(token_ids, src=0)

        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
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
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
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
