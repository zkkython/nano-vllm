import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.log_config import LogConfig, LogLevel
import argparse


def main(args):
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # path = os.path.expanduser("/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B")
    log_config = LogConfig(
        global_level=LogLevel.ERROR,
        chunked_prefill=LogLevel.INFO,  # 只调试 chunked prefill
        warmup=LogLevel.INFO,  # warmup 显示基本信息
        model_runner=LogLevel.INFO,
    )
    llm = LLM(
        args.model,
        log_config=log_config,
        enforce_eager=False,
        max_model_len=2050,
        tensor_parallel_size=args.tp,
        use_fused_moe=args.use_fused_moe,
        use_triton_moe=args.use_triton_moe,
        gpu_memory_utilization=args.gpu_memory_utilization,
        ep_size=args.ep_size,
        enable_epmoe=args.enable_epmoe,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]
    # warm up
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=True)
    print(f"warm up finished")
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    )
    args.add_argument(
        "--tp",
        type=int,
        default=8,
    )
    args.add_argument("--use_fused_moe", action="store_true")
    args.add_argument("--use_triton_moe", action="store_true")
    args.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    args.add_argument("--ep_size", type=int, default=1)
    args.add_argument("--enable_epmoe", action="store_true")

    main(args=args.parse_args())
