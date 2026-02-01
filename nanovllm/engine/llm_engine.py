import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp
import os
import torch.distributed as dist
from nanovllm.config import Config, EngineRole
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.log_config import log_info


class LLMEngine:

    def __init__(self, model, master_addr="localhost", master_port=2333, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 添加分布式相关的配置参数
        if "master_addr" not in config_kwargs:
            config_kwargs["master_addr"] = master_addr
        if "master_port" not in config_kwargs:
            config_kwargs["master_port"] = master_port
        config = Config(model, **config_kwargs)

        # 1. 优先初始化 Scheduler 和 KVTransferAgent，确保端口尽早开启
        self.scheduler = Scheduler(config)
        self.kv_transfer_agent = None
        if config.engine_role != EngineRole.SINGLE:
            # 使用 Mooncake 传输
            from nanovllm.engine.kv_transfer_mooncake import MooncakeTransferAgent

            print(
                f"[DEBUG] Initializing MooncakeTransferAgent for role: {config.engine_role}",
                flush=True,
            )
            self.kv_transfer_agent = MooncakeTransferAgent(config, self.scheduler)
            print(f"[DEBUG] MooncakeTransferAgent initialized", flush=True)

        # 2. 检查并初始化分布式环境/ModelRunner
        print(
            f"[DEBUG] About to initialize ModelRunner, tensor_parallel_size={config.tensor_parallel_size}",
            flush=True,
        )
        if dist.is_available() and dist.is_initialized():
            print("[DEBUG] Distributed environment detected", flush=True)
            # 如果已经初始化了分布式环境，使用现有的配置
            config.tensor_parallel_size = dist.get_world_size()
            rank = dist.get_rank()
            # 从环境变量中获取local_rank（由torchrun设置）
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            # 从环境变量中获取master_addr（由torchrun设置）
            # 重要：必须指向Master节点，所有worker节点都用同一个MASTER_ADDR
            master_addr_env = os.environ.get("MASTER_ADDR")
            if master_addr_env:
                config.master_addr = master_addr_env

            print(
                f"[DEBUG] Distributed config: rank={rank}, local_rank={local_rank}, "
                f"world_size={config.tensor_parallel_size}, master_addr={config.master_addr}",
                flush=True,
            )

            # 只在rank 0上启动其他rank的进程（单机多卡情况）
            # 或者在分布式环境中每个rank都运行自己的ModelRunner
            self.model_runner = ModelRunner(config, rank, local_rank)
            print(f"[DEBUG] ModelRunner initialized for rank {rank}", flush=True)
        else:
            # 否则使用原来的多进程方式（主要用于单机多卡）
            if config.tensor_parallel_size > 1:
                print(
                    f"[DEBUG] Starting subprocesses for TP={config.tensor_parallel_size}",
                    flush=True,
                )
                # 设置环境变量，以便子进程可以访问
                os.environ["MASTER_ADDR"] = config.master_addr
                os.environ["MASTER_PORT"] = str(config.master_port)
                os.environ["WORLD_SIZE"] = str(config.tensor_parallel_size)

                # 获取主进程绑定的设备索引作为偏移量
                num_local_gpus = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 1
                )
                base_device_id = config.local_rank

                self.ps = []

                ctx = mp.get_context("spawn")
                for i in range(1, config.tensor_parallel_size):
                    # 子进程的设备索引应基于 base_device_id 递增
                    local_rank = (base_device_id + i) % num_local_gpus
                    print(
                        f"[DEBUG] Starting subprocess for rank {i} with device {local_rank}",
                        flush=True,
                    )
                    process = ctx.Process(
                        target=ModelRunner, args=(config, i, local_rank)
                    )
                    process.start()
                    self.ps.append(process)
                    print(
                        f"[DEBUG] Subprocess started: rank={i}, pid={process.pid}",
                        flush=True,
                    )

                # 主进程也应使用正确的 local_rank
                main_local_rank = base_device_id % num_local_gpus
                print(
                    f"[DEBUG] Main process rank 0 using device {main_local_rank}",
                    flush=True,
                )
                self.model_runner = ModelRunner(config, 0, main_local_rank)
                print(f"[DEBUG] Main process ModelRunner initialized", flush=True)
            else:
                # 单 GPU 情况也要使用正确的local_rank
                num_local_gpus = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 1
                )
                main_local_rank = config.local_rank % num_local_gpus
                print(f"[DEBUG] Single GPU mode, device={main_local_rank}", flush=True)
                self.model_runner = ModelRunner(config, 0, main_local_rank)
                print(f"[DEBUG] Single GPU ModelRunner initialized", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.config = config
        self.world_size = config.tensor_parallel_size
        self._exited = False

        # 3. 同步 ModelRunner 计算出的 num_kvcache_blocks 到 Scheduler/BlockManager
        actual_num_blocks = self.model_runner.call(
            "get_config_attr", "num_kvcache_blocks"
        )
        self.scheduler.block_manager.update_num_blocks(actual_num_blocks)

        # 注册自动退出处理 - 使用实例编号确保只注册一次
        import sys

        atexit_key = f"_llm_atexit_registered_{id(self)}"
        if not getattr(sys, atexit_key, False):
            atexit.register(self.exit)
            setattr(sys, atexit_key, True)

    def exit(self):
        print(f"[DEBUG] LLMEngine.exit() called, _exited={self._exited}", flush=True)
        """优雅退出，清理所有资源"""
        # 防止重复调用
        if self._exited:
            print("[DEBUG] Already exited, returning", flush=True)
            return
        self._exited = True

        try:
            # 子进程是守护进程，会自动清理
            if dist.is_initialized():
                print(f"[DEBUG] world_size={self.world_size}", flush=True)
                self.model_runner.call("exit")
        except Exception as e:
            print(f"[DEBUG] Exception in exit try block: {e}", flush=True)
        print("[DEBUG] LLMEngine.exit() completed", flush=True)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # Decode 节点可能需要从 KV Transfer Agent 接收新的 Sequence
        if self.config.engine_role == EngineRole.DECODE and self.kv_transfer_agent:
            new_seq_items = self.kv_transfer_agent.recv_sequences()
            for seq, kv_data in new_seq_items:
                # 关键：接收到的 seq.block_table 指向的是 Prefill 节点的物理块 ID，在当前节点无效
                seq.block_table = []
                # 在 Decode 节点重新分配本地物理块
                self.scheduler.block_manager.allocate(seq)
                # 将数据导入本地分配的物理块
                self.model_runner.call("import_kv_cache", seq, kv_data)
                self.scheduler.add_running_sequence(seq)

        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0

        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        # 校验产出的kvcache是否存在问题
        # finished_prefill_seqs = [s for s in seqs if s.status == SequenceStatus.RUNNING]
        # for seq in finished_prefill_seqs:
        #     log_info("llm_engine", f"seq_id = {seq.seq_id}, kv data={seq.token_ids}")

        # Prefill 节点完成后需要发送 KV Cache
        if self.config.engine_role == EngineRole.PREFILL and is_prefill:
            finished_prefill_seqs = [
                s for s in seqs if s.status == SequenceStatus.RUNNING
            ]
            if finished_prefill_seqs and self.kv_transfer_agent:
                # 导出 KV blocks 数据
                kv_data = self.model_runner.call(
                    "export_kv_cache", finished_prefill_seqs
                )
                self.kv_transfer_agent.send_sequences(finished_prefill_seqs, kv_data)
                # Prefill 节点在发送完后可以清理掉这些 sequence
                for seq in finished_prefill_seqs:
                    self.scheduler.remove_sequence(seq)

        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        last_update_time = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                now = perf_counter()
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (now - t)
                elif num_tokens < 0:
                    decode_throughput = -num_tokens / (now - t)

                # 限制刷新频率，避免在非 TTY 环境下产生大量日志
                if now - last_update_time >= 0.1:
                    pbar.set_postfix(
                        {
                            "Prefill": f"{int(prefill_throughput)}tok/s",
                            "Decode": f"{int(decode_throughput)}tok/s",
                        }
                    )
                    last_update_time = now
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
