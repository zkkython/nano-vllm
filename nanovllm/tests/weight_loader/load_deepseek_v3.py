"""DeepSeek-V3 模型权重加载示例."""

import torch.distributed as dist

from nanovllm.models.deepseek_v3_origin import DeepSeekV3Config, DeepSeekV3ForCausalLM


def example_load_deepseek_v3():
    """示例：加载 DeepSeek-V3 模型权重."""
    print("=" * 70)
    print("DeepSeek-V3 权重加载示例")
    print("=" * 70)

    # 单进程环境下的分布式初始化
    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    # 模型路径（需要根据实际路径调整）
    model_path = "/data/ds-671"

    # 加载配置（可以从 HF 加载或手动创建）
    # config = DeepSeekV3Config.from_pretrained(model_path)

    # 或者手动创建一个小配置用于测试
    config = DeepSeekV3Config(
        vocab_size=102400,
        hidden_size=2048,
        intermediate_size=10944,
        moe_intermediate_size=1408,
        num_hidden_layers=4,  # 测试用小模型
        num_attention_heads=16,
        num_key_value_heads=16,
        # MoE 配置
        n_routed_experts=8,  # 测试用少量专家
        n_shared_experts=2,
        num_experts_per_tok=2,
        # MLA 配置
        q_lora_rank=512,  # 或 0（直接投影）
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )

    print(f"\n模型配置:")
    print(f"  层数: {config.num_hidden_layers}")
    print(f"  隐藏维度: {config.hidden_size}")
    print(f"  路由专家数: {config.n_routed_experts}")
    print(f"  共享专家数: {config.n_shared_experts}")
    print(f"  Q LoRA rank: {config.q_lora_rank}")

    # 创建模型
    print("\n创建模型...")
    model = DeepSeekV3ForCausalLM(config)

    # 加载权重（完整加载）
    print("\n加载权重（完整）...")
    stats = model.load_weights(config=config, model_path=model_path)

    print("\n权重加载统计:")
    print(f"  总权重数量: {stats['total_weights']}")
    print(f"  成功加载: {stats['loaded_weights']}")
    print(f"  跳过数量: {stats['skipped_weights']}")
    print(f"  已加载层: {stats['loaded_layers']}")
    if stats["skipped_layers"]:
        print(f"  跳过的层: {stats['skipped_layers']}")

    print(f"\n已加载的权重示例（前 10 个）:")
    for name in stats["loaded_weight_names"][:10]:
        print(f"  - {name}")

    print("\n✅ 权重加载完成！")
    print("=" * 70)


def example_partial_load_deepseek_v3():
    """示例：部分层加载 DeepSeek-V3."""
    print("\n" + "=" * 70)
    print("DeepSeek-V3 部分层加载示例")
    print("=" * 70)

    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    model_path = "/data/ds-671"
    # 使用真实配置
    config = DeepSeekV3Config.from_pretrained(model_path)
    # config = DeepSeekV3Config(
    #     vocab_size=102400,
    #     hidden_size=2048,
    #     num_hidden_layers=60,  # 假设是完整模型
    #     n_routed_experts=64,
    #     n_shared_experts=2,
    #     q_lora_rank=512,
    #     kv_lora_rank=512,
    # )
    config.load_partial_layers = 2
    model = DeepSeekV3ForCausalLM(config)

    # 只加载前 2 层进行快速测试
    print(f"\n只加载前 2 层（共 {config.num_hidden_layers} 层）...")
    stats = model.load_weights(
        config=config,
        model_path=model_path,
        load_partial_layers=config.load_partial_layers,
    )

    print("\n权重加载统计:")
    print(f"  总权重数量: {stats['total_weights']}")
    print(f"  成功加载: {stats['loaded_weights']}")
    print(f"  跳过数量: {stats['skipped_weights']}")
    print(f"  已加载层: {stats['loaded_layers']}")
    print(f"  跳过的层: {stats['skipped_layers']}")

    print(
        f"\n跳过的专家权重数量: {len([n for n in stats['skipped_weight_names'] if 'experts' in n])}"
    )

    print("\n✅ 部分层加载完成！节省内存和时间")
    print("=" * 70)


def example_load_with_verification():
    """示例：加载权重并验证."""
    print("\n" + "=" * 70)
    print("DeepSeek-V3 权重加载验证")
    print("=" * 70)

    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    model_path = "/data/ds-671"

    # 使用真实配置
    config = DeepSeekV3Config.from_pretrained(model_path)

    model = DeepSeekV3ForCausalLM(config)

    # 加载权重
    stats = model.load_weights(config=config, model_path=model_path)

    # 验证关键权重
    print("\n验证关键权重:")

    # 检查 embedding
    embed_weight = model.model.embed_tokens.weight
    print(f"  embed_tokens: shape={embed_weight.shape}, mean={embed_weight.mean():.6f}")

    # 检查第一层的 MLA attention
    layer0 = model.model.layers[0]
    if config.q_lora_rank > 0:
        wq_a = layer0.self_attn.wq_a.weight
        print(f"  layer0.wq_a: shape={wq_a.shape}, mean={wq_a.mean():.6f}")
    else:
        wq = layer0.self_attn.wq.weight
        print(f"  layer0.wq: shape={wq.shape}, mean={wq.mean():.6f}")

    # 检查第一层的 MLP
    gate_proj = layer0.mlp.gate_proj.weight
    print(
        f"  layer0.mlp.gate_proj: shape={gate_proj.shape}, mean={gate_proj.mean():.6f}"
    )

    # 检查 MoE 层（如果有）
    if config.num_hidden_layers > 1:
        layer1 = model.model.layers[1]
        if hasattr(layer1.mlp, "gate"):
            gate_weight = layer1.mlp.gate.weight
            print(
                f"  layer1.moe.gate: shape={gate_weight.shape}, mean={gate_weight.mean():.6f}"
            )

            # 检查第一个专家
            if layer1.mlp.experts[0] is not None:
                expert0_gate = layer1.mlp.experts[0].gate_proj.weight
                print(
                    f"  layer1.expert0.gate: shape={expert0_gate.shape}, mean={expert0_gate.mean():.6f}"
                )

    print("\n✅ 权重验证完成！")
    print("=" * 70)


if __name__ == "__main__":
    # 选择运行的示例
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "full"

    if mode == "full":
        example_load_deepseek_v3()
    elif mode == "partial":
        example_partial_load_deepseek_v3()
    elif mode == "verify":
        example_load_with_verification()
    else:
        print("Usage: python load_deepseek_v3.py [full|partial|verify]")
        print("\nAvailable modes:")
        print("  full    - 完整加载所有层")
        print("  partial - 部分层加载（快速测试）")
        print("  verify  - 加载并验证权重")
