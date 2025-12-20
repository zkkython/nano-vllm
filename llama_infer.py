import argparse
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main(args):
    path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    prompts = [
        "Hello, introduce yourself in one sentence.",
        "Explain the difference between CPU and GPU in 2 lines.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print("\n")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, required=True, help="Path to a HuggingFace LLaMA model dir")
    args = args.parse_args()
    main(args)



