import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp
import os
import torch.distributed as dist
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
import logging

log = logging.getLogger(__name__)


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

        # 检查是否已经通过torchrun等方式启动了分布式环境
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
        else:
            # 否则使用原来的多进程方式（主要用于单机多卡）
            if config.tensor_parallel_size > 1:
                # 设置环境变量，以便子进程可以访问
                os.environ["MASTER_ADDR"] = config.master_addr
                os.environ["MASTER_PORT"] = str(config.master_port)
                os.environ["WORLD_SIZE"] = str(config.tensor_parallel_size)

                self.ps = []

                ctx = mp.get_context("spawn")
                for i in range(1, config.tensor_parallel_size):
                    # 单机多卡时，local_rank应该限制在本地GPU范围内
                    # 如果i超出了本地GPU数量，则使用i % 本地GPU数
                    num_local_gpus = (
                        torch.cuda.device_count() if torch.cuda.is_available() else 1
                    )
                    local_rank = i % num_local_gpus
                    process = ctx.Process(
                        target=ModelRunner, args=(config, i, local_rank)
                    )
                    process.start()
                    self.ps.append(process)

                # 主进程也应使用正确的local_rank
                num_local_gpus = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 1
                )
                main_local_rank = 0 % num_local_gpus  # 对于rank 0，local_rank总是0
                self.model_runner = ModelRunner(config, 0, main_local_rank)
            else:
                # 单GPU情况也要使用正确的local_rank
                num_local_gpus = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 1
                )
                main_local_rank = 0 % num_local_gpus  # 对于rank 0，local_rank总是0
                self.model_runner = ModelRunner(config, 0, main_local_rank)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.world_size = config.tensor_parallel_size
        self._exited = False

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
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
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
