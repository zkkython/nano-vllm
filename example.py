import argparse
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)


def main(args):
    path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    print(f"eager force: {args.enforce_eager}")
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp,
        use_fused_moe=args.use_fused_moe,
        use_triton_moe=args.use_triton_moe,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "请用中文介绍你自己",
        "列举出100以内的质数",
        "解析下量子力学",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    # llm.exit()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model", type=str, default="/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    )
    args.add_argument("--tp", type=int, default=8)
    args.add_argument("--use_fused_moe", action="store_true")
    args.add_argument("--use_triton_moe", action="store_true")
    args.add_argument("--enforce_eager", action="store_false")
    args = args.parse_args()
    main(args)
