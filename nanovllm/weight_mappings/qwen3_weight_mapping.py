"""Qwen3 模型权重映射配置

Qwen3ForCausalLM 使用 QKVParallelLinear 和 MergedColumnParallelLinear 实现权重打包，
本映射文件将 HF safetensors 中的 q/k/v/gate/up 独立权重映射到打包后的参数上。
"""

from nanovllm.utils.weight_loader import WeightMapping


def build_qwen3_weight_mappings(num_hidden_layers: int) -> dict[str, WeightMapping]:
    """构建 Qwen3 模型的权重映射规则.

    Args:
        num_hidden_layers: 模型层数，从 config.num_hidden_layers 获取

    Returns:
        权重映射字典，key 为 HF safetensors 中的权重名，value 为 WeightMapping 实例
    """
    weight_mappings: dict[str, WeightMapping] = {}

    # ===== 全局权重 =====
    # embed_tokens
    weight_mappings["model.embed_tokens.weight"] = WeightMapping(
        target_path="model.embed_tokens.weight"
    )

    # lm_head
    weight_mappings["lm_head.weight"] = WeightMapping(target_path="lm_head.weight")

    # 最后的 RMSNorm
    weight_mappings["model.norm.weight"] = WeightMapping(
        target_path="model.norm.weight"
    )

    # ===== 按层映射 =====
    for layer_id in range(num_hidden_layers):
        prefix = f"model.layers.{layer_id}"

        # LayerNorm 权重
        weight_mappings[f"{prefix}.input_layernorm.weight"] = WeightMapping(
            target_path=f"{prefix}.input_layernorm.weight"
        )
        weight_mappings[f"{prefix}.post_attention_layernorm.weight"] = WeightMapping(
            target_path=f"{prefix}.post_attention_layernorm.weight"
        )

        # Self-Attention: q/k/v → qkv_proj（QKVParallelLinear）
        # 通过 loader_arg 传入 "q" / "k" / "v"
        weight_mappings[f"{prefix}.self_attn.q_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.qkv_proj.weight",
            loader_arg="q",
        )
        weight_mappings[f"{prefix}.self_attn.k_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.qkv_proj.weight",
            loader_arg="k",
        )
        weight_mappings[f"{prefix}.self_attn.v_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.qkv_proj.weight",
            loader_arg="v",
        )

        # Qwen3 的 qkv_bias 默认为 False，如果你的配置里是 True，需要取消下面注释
        # weight_mappings[f"{prefix}.self_attn.q_proj.bias"] = WeightMapping(
        #     target_path=f"{prefix}.self_attn.qkv_proj.bias",
        #     loader_arg="q",
        # )
        # weight_mappings[f"{prefix}.self_attn.k_proj.bias"] = WeightMapping(
        #     target_path=f"{prefix}.self_attn.qkv_proj.bias",
        #     loader_arg="k",
        # )
        # weight_mappings[f"{prefix}.self_attn.v_proj.bias"] = WeightMapping(
        #     target_path=f"{prefix}.self_attn.qkv_proj.bias",
        #     loader_arg="v",
        # )

        # q_norm / k_norm
        weight_mappings[f"{prefix}.self_attn.q_norm.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.q_norm.weight"
        )
        weight_mappings[f"{prefix}.self_attn.k_norm.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.k_norm.weight"
        )

        # o_proj（RowParallelLinear，bias=False）
        weight_mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.o_proj.weight"
        )

        # MLP: gate/up → gate_up_proj（MergedColumnParallelLinear）
        # 通过 loader_arg 传入 0（gate）/ 1（up）
        weight_mappings[f"{prefix}.mlp.gate_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.mlp.gate_up_proj.weight",
            loader_arg=0,
        )
        weight_mappings[f"{prefix}.mlp.up_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.mlp.gate_up_proj.weight",
            loader_arg=1,
        )

        # down_proj（RowParallelLinear，bias=False）
        weight_mappings[f"{prefix}.mlp.down_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.mlp.down_proj.weight"
        )

    return weight_mappings
