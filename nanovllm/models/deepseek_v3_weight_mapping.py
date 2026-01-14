"""DeepSeek-V3 模型权重映射配置

DeepSeek-V3 使用 MLA (Multi-head Latent Attention) 和 MoE (Mixture of Experts) 架构。
本映射文件将 HF safetensors 中的权重映射到模型参数上。

特殊结构：
1. MLA Attention: 使用 LoRA 风格的 Q/KV 投影
   - Q: wq_a -> q_norm -> wq_b (如果 q_lora_rank > 0)
   - KV: wkv_a -> kv_norm -> wkv_b
2. MoE: 第一层使用标准 MLP，其余层使用 MoE
   - Routed experts: gate/up/down_proj
   - Shared experts: 使用标准 MLP 结构
"""

from nanovllm.utils.weight_loader import WeightMapping


def build_deepseek_v3_weight_mappings(
    num_hidden_layers: int,
    q_lora_rank: int,
    n_shared_experts: int,
) -> dict[str, WeightMapping]:
    """构建 DeepSeek-V3 模型的权重映射规则.

    Args:
        num_hidden_layers: 模型层数，从 config.num_hidden_layers 获取
        q_lora_rank: Q 投影的 LoRA rank，从 config.q_lora_rank 获取
        n_shared_experts: 共享专家数量，从 config.n_shared_experts 获取

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

        # ===== MLA Attention =====
        # Q 投影: 根据 q_lora_rank 决定是否使用 LoRA
        if q_lora_rank == 0:
            # 直接投影
            # TODO Need to check when the feature is turning on
            weight_mappings[f"{prefix}.self_attn.wq.weight"] = WeightMapping(
                target_path=f"{prefix}.self_attn.wq.weight"
            )
        else:
            # LoRA 投影
            weight_mappings[f"{prefix}.self_attn.q_a_proj.weight"] = WeightMapping(
                target_path=f"{prefix}.self_attn.wq_a.weight"
            )
            weight_mappings[f"{prefix}.self_attn.q_a_layernorm.weight"] = WeightMapping(
                target_path=f"{prefix}.self_attn.q_norm.weight"
            )
            weight_mappings[f"{prefix}.self_attn.q_b_proj.weight"] = WeightMapping(
                target_path=f"{prefix}.self_attn.wq_b.weight"
            )

        # KV 投影: 使用 LoRA
        weight_mappings[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.wkv_a.weight"
        )
        weight_mappings[f"{prefix}.self_attn.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.kv_norm.weight"
        )
        weight_mappings[f"{prefix}.self_attn.kv_b_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.wkv_b.weight"
        )
        

        # Output projection
        weight_mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.wo.weight"
        )

        # ===== MLP / MoE =====
        """
        前三层
        model.layers.0.input_layernorm.weight：shape = torch.Size([7168])
        model.layers.0.mlp.down_proj.weight：shape = torch.Size([7168, 18432])
        model.layers.0.mlp.down_proj.weight_scale_inv：shape = torch.Size([56, 144])
        model.layers.0.mlp.gate_proj.weight：shape = torch.Size([18432, 7168])
        model.layers.0.mlp.gate_proj.weight_scale_inv：shape = torch.Size([144, 56])
        model.layers.0.mlp.up_proj.weight：shape = torch.Size([18432, 7168])
        model.layers.0.mlp.up_proj.weight_scale_inv：shape = torch.Size([144, 56])
        model.layers.0.post_attention_layernorm.weight：shape = torch.Size([7168])
        model.layers.0.self_attn.kv_a_layernorm.weight：shape = torch.Size([512])
        model.layers.0.self_attn.kv_a_proj_with_mqa.weight：shape = torch.Size([576, 7168])
        model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv：shape = torch.Size([5, 56])
        model.layers.0.self_attn.kv_b_proj.weight：shape = torch.Size([32768, 512])
        model.layers.0.self_attn.kv_b_proj.weight_scale_inv：shape = torch.Size([256, 4])
        model.layers.0.self_attn.o_proj.weight：shape = torch.Size([7168, 16384])
        model.layers.0.self_attn.o_proj.weight_scale_inv：shape = torch.Size([56, 128])
        model.layers.0.self_attn.q_a_layernorm.weight：shape = torch.Size([1536])
        model.layers.0.self_attn.q_a_proj.weight：shape = torch.Size([1536, 7168])
        model.layers.0.self_attn.q_a_proj.weight_scale_inv：shape = torch.Size([12, 56])
        model.layers.0.self_attn.q_b_proj.weight：shape = torch.Size([24576, 1536])
        model.layers.0.self_attn.q_b_proj.weight_scale_inv：shape = torch.Size([192, 12])

        """
        if layer_id >= 0 and layer_id < 3:
            # 第一层使用标准 MLP
            weight_mappings[f"{prefix}.mlp.gate_proj.weight"] = WeightMapping(
                target_path=f"{prefix}.mlp.gate_proj.weight"
            )
            weight_mappings[f"{prefix}.mlp.up_proj.weight"] = WeightMapping(
                target_path=f"{prefix}.mlp.up_proj.weight"
            )
            weight_mappings[f"{prefix}.mlp.down_proj.weight"] = WeightMapping(
                target_path=f"{prefix}.mlp.down_proj.weight"
            )
        else:
            # 其余层使用 MoE
            # Gate 权重
            weight_mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{prefix}.mlp.gate.weight"
            )
            # Gate bias (可选，某些配置有)
            # 在实际加载时会自动处理不存在的权重

            # Routed experts
            # 注意：专家权重按 experts.{expert_id}.{proj} 命名
            # 这里不预先生成所有专家的映射，而是使用模板匹配
            # 如果需要显式映射，可以遍历 n_routed_experts

            # Shared experts (如果存在)
            if n_shared_experts > 0:
                weight_mappings[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = (
                    WeightMapping(
                        target_path=f"{prefix}.mlp.shared_experts.gate_proj.weight"
                    )
                )
                weight_mappings[f"{prefix}.mlp.shared_experts.up_proj.weight"] = (
                    WeightMapping(
                        target_path=f"{prefix}.mlp.shared_experts.up_proj.weight"
                    )
                )
                weight_mappings[f"{prefix}.mlp.shared_experts.down_proj.weight"] = (
                    WeightMapping(
                        target_path=f"{prefix}.mlp.shared_experts.down_proj.weight"
                    )
                )

    return weight_mappings


def build_deepseek_v3_expert_mappings(
    num_hidden_layers: int, n_routed_experts: int
) -> dict[str, WeightMapping]:
    """构建 DeepSeek-V3 MoE 专家的权重映射规则.

    这是一个辅助函数，用于生成所有专家的显式映射。
    可以与 build_deepseek_v3_weight_mappings 的结果合并。

    Args:
        num_hidden_layers: 模型层数
        n_routed_experts: 路由专家数量，从 config.n_routed_experts 获取

    Returns:
        专家权重映射字典
    """
    expert_mappings: dict[str, WeightMapping] = {}

    for layer_id in range(1, num_hidden_layers):  # 从第 1 层开始（第 0 层是 MLP）
        prefix = f"model.layers.{layer_id}"

        for expert_id in range(n_routed_experts):
            expert_prefix = f"{prefix}.mlp.experts.{expert_id}"

            expert_mappings[f"{expert_prefix}.gate_proj.weight"] = WeightMapping(
                target_path=f"{expert_prefix}.gate_proj.weight"
            )
            expert_mappings[f"{expert_prefix}.up_proj.weight"] = WeightMapping(
                target_path=f"{expert_prefix}.up_proj.weight"
            )
            expert_mappings[f"{expert_prefix}.down_proj.weight"] = WeightMapping(
                target_path=f"{expert_prefix}.down_proj.weight"
            )

    return expert_mappings
