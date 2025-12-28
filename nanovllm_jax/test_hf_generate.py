"""测试 HF 官方模型的生成结果"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

print("加载 HF 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model = model.to("cpu")
model.eval()

prompt = "中国的首都"
print(f"\nPrompt: {prompt}")

input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(f"Input IDs: {input_ids.tolist()}")

print("\n" + "=" * 60)
print("Greedy Decoding (temperature=0) - 生成6个token")
print("=" * 60)

with torch.no_grad():
    # 逐步 greedy decode
    tokens = []
    cache = None

    # Prefill
    outputs = model(input_ids, use_cache=True, return_dict=True)
    next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
    tokens.append(next_token_id)
    cache = outputs.past_key_values
    print(f"[Prefill] Token {next_token_id}: '{tokenizer.decode([next_token_id])}'")

    # Decode 4 步
    for step in range(5):
        outputs = model(
            torch.tensor([[next_token_id]]),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        next_token_id = torch.argmax(outputs.logits[0, 0, :]).item()
        tokens.append(next_token_id)
        cache = outputs.past_key_values
        print(
            f"[Decode {step+1}] Token {next_token_id}: '{tokenizer.decode([next_token_id])}'"
        )

    full_text = prompt + tokenizer.decode(tokens)
    print(f"\n完整生成:\n{full_text}\n")
