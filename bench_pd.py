import time
import multiprocessing as mp
import os
import argparse
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.config import EngineRole
from nanovllm.log_config import LogConfig, LogLevel


def run_decode_node(args, result_queue, ready_event):
    os.environ["NANOVLLM_LOG_LEVEL"] = "INFO"
    log_config = LogConfig(global_level=LogLevel.INFO)

    llm = LLM(
        args.model,
        engine_role=EngineRole.DECODE,
        kv_transfer_port=args.port,
        kv_transfer_address="127.0.0.1",
        enforce_eager=False,
        tensor_parallel_size=args.tp_decode,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs_decode=args.max_num_seqs_decode,
        master_port=args.master_port_decode,
        local_rank=args.device_decode,
    )

    print(f"Decode Node warmed up and ready (ZMQ port: {args.port}).", flush=True)
    ready_event.set()

    total_output_tokens = 0
    finished_count = 0
    start_time = None

    while finished_count < args.num_seqs:
        outputs, num_tokens = llm.step()

        if outputs:
            if start_time is None:
                start_time = time.time()

            for seq_id, token_ids in outputs:
                finished_count += 1
                total_output_tokens += len(token_ids)

        if num_tokens == 0:
            time.sleep(0.001)

    end_time = time.time()
    duration = end_time - start_time if start_time else 0
    result_queue.put((total_output_tokens, duration))


def run_prefill_node(args, prompt_token_ids, sampling_params):
    os.environ["NANOVLLM_LOG_LEVEL"] = "INFO"
    log_config = LogConfig(global_level=LogLevel.INFO)

    llm = LLM(
        args.model,
        engine_role=EngineRole.PREFILL,
        kv_transfer_port=args.port_prefill,
        kv_transfer_address="127.0.0.1",
        enforce_eager=False,
        tensor_parallel_size=args.tp_prefill,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs_prefill=args.max_num_seqs_prefill,
        master_port=args.master_port_prefill,
        local_rank=args.device_prefill,
    )

    print(f"Prefill Node LLM initialized successfully", flush=True)
    print(f"Prefill Node starting batch of {len(prompt_token_ids)} requests")

    for p, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(p, sp)

    print(f"All {len(prompt_token_ids)} requests added to Prefill Node")

    start_time = time.time()
    while not llm.is_finished():
        llm.step()

    end_time = time.time()
    print(f"Prefill Node finished all transfers in {end_time - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Single-machine PD Separation Benchmark"
    )
    parser.add_argument(
        "--model", type=str, default="/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    )
    parser.add_argument("--num_seqs", type=int, default=128)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--port", type=int, default=2345, help="Decode 节点 ZMQ 控制端口"
    )
    parser.add_argument(
        "--port_prefill", type=int, default=2346, help="Prefill 节点 ZMQ 控制端口"
    )
    parser.add_argument("--max_num_seqs_prefill", type=int, default=16)
    parser.add_argument("--max_num_seqs_decode", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument(
        "--device_prefill", type=int, default=0, help="Starting device ID for prefill"
    )
    parser.add_argument(
        "--device_decode", type=int, default=4, help="Starting device ID for decode"
    )
    parser.add_argument(
        "--tp_prefill", type=int, default=4, help="TP size for prefill node"
    )
    parser.add_argument(
        "--tp_decode", type=int, default=4, help="TP size for decode node"
    )
    parser.add_argument("--master_port_prefill", type=int, default=2337)
    parser.add_argument("--master_port_decode", type=int, default=2338)

    args = parser.parse_args()
    seed(0)

    max_input_len = 512
    max_output_len = 512

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.0, ignore_eos=True, max_tokens=randint(100, max_output_len)
        )
        for _ in range(args.num_seqs)
    ]

    total_expected_tokens = sum(sp.max_tokens for sp in sampling_params)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    ready_event = ctx.Event()

    d_proc = ctx.Process(target=run_decode_node, args=(args, result_queue, ready_event))
    p_proc = ctx.Process(
        target=run_prefill_node, args=(args, prompt_token_ids, sampling_params)
    )

    print("Starting Benchmark Processes...")
    d_proc.start()

    print("Waiting for Decode Node to be ready (this may take a minute for TP=4)...")
    if not ready_event.wait(timeout=300):  # 5 minutes max
        print("Error: Decode Node timed out during initialization")
        d_proc.terminate()
        return

    p_proc.start()

    p_proc.join()

    # Get results from decode node
    total_tokens, duration = result_queue.get()

    d_proc.terminate()

    print("\n" + "=" * 40)
    print(f"PD Separation Benchmark Results")
    print(f"Model: {args.model}")
    print(f"Num Seqs: {args.num_seqs}")
    print(f"Prefill TP: {args.tp_prefill}, Decode TP: {args.tp_decode}")
    print(
        f"Prefill Batch: {args.max_num_seqs_prefill}, Decode Batch: {args.max_num_seqs_decode}"
    )
    print(f"Total Output Tokens: {total_tokens}")
    print(f"Total Time: {duration:.2f}s")
    print(f"Throughput: {total_tokens / duration:.2f} tokens/s")
    print("=" * 40)


if __name__ == "__main__":
    main()
