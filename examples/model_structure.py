from transformers import Qwen3ForCausalLM, Qwen2ForCausalLM

model_name = "/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
model = Qwen3ForCausalLM.from_pretrained(
    model_name, dtype="auto", device_map="auto"
)
print(f"model config = \n {model.config}")
print("=" * 20)
print(model)

qwen2_06b = "/home/kason/models/qwen206b"
model = Qwen2ForCausalLM.from_pretrained(
    qwen2_06b, dtype="auto", device_map="auto"
)
print(f"model config = \n {model.config}")
print("=" * 20)
print(model)
