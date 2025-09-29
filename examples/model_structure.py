from transformers import Qwen3ForCausalLM

model_name = "/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
model = Qwen3ForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
print(f"model config = \n {model.config}")
print("=" * 20)
print(model)
