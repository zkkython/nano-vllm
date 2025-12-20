from transformers import Qwen2VLForConditionalGeneration
from transformers.utils import logging

logging.set_verbosity_info()

# model_name = "/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
# model = Qwen3ForCausalLM.from_pretrained(
#     model_name, dtype="auto", device_map="auto"
# )
# print(f"model config = \n {model.config}")
# print("=" * 20)
# print(model)

qwen_cl = "/home/kason/models/qwen2_vl_2b"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    qwen_cl, dtype="auto", device_map="auto"
)

out_path = "/home/kason/python_workspace/nano-vllm/examples/model_structure.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("model config = \n" + str(model.config) + "\n")
    f.write("=" * 80 + "\n")
    f.write("[Qwen2-VL 模型整体结构]\n")
    f.write(str(model) + "\n\n")

    f.write("=" * 80 + "\n")
    f.write("[按层打印 - 文本解码器 layers]\n")
    text_layers = getattr(model.model.language_model, "layers", [])
    for idx, layer in enumerate(text_layers):
        f.write(f"Text Layer {idx}: {layer.__class__.__name__}\n")
        f.write(str(layer) + "\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("[按层打印 - 视觉编码器 blocks（如存在）]\n")
    visual_blocks = getattr(model.model.visual, "blocks", [])
    for idx, block in enumerate(visual_blocks):
        f.write(f"Vision Block {idx}: {block.__class__.__name__}\n")
        f.write(str(block) + "\n")

print(f"已写入: {out_path}")