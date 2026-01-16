from nanovllm.models.models_mapping import MODELS_MAPPING

print("Registered models:", list(MODELS_MAPPING.keys()))

if "qwen2" in MODELS_MAPPING:
    print("qwen2 found!")
    cls = MODELS_MAPPING["qwen2"]
    print(f"qwen2 class: {cls}")
else:
    print("qwen2 NOT found!")
