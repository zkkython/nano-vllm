from transformers import AutoConfig
from nanovllm_jax.models.qwen3 import SelfAttentionVarlen
from nanovllm_jax.utils.context import set_context
import jax.numpy as jnp

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
print(f"Loading config from: {MODEL_PATH}")
hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
num_kv_heads = getattr(hf_config, "num_key_value_heads")


hidden_size = hf_config.hidden_size
num_heads = hf_config.num_attention_heads
head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)

print(f"\nModel config:")
print(f"  hidden_size: {hidden_size}")
print(f"  num_heads: {num_heads}")
print(f"  num_kv_heads: {num_kv_heads}")
print(f"  head_dim: {head_dim}")

# 1. 初始化 attention layer
print(f"\nInitializing SelfAttentionVarlen...")
attn = SelfAttentionVarlen(hf_config, dtype=jnp.bfloat16, block_size=256)

# 2. 准备 flattened sequences (例如：2个序列，长度分别为10和15)
hidden_states = jnp.zeros((25, hidden_size))  # [total_tokens, hidden_size]
positions = jnp.arange(25)  # position indices

print(f"\nInput shapes:")
print(f"  hidden_states: {hidden_states.shape}")
print(f"  positions: {positions.shape}")

# 3. 设置 context (varlen attention 需要)
print(f"\nSetting context for varlen attention...")
set_context(
    is_prefill=True,
    cu_seqlens_q=jnp.array([0, 10, 25]),  # 累积序列长度
    cu_seqlens_k=jnp.array([0, 10, 25]),
    max_seqlen_q=15,
    max_seqlen_k=15,
    slot_mapping=jnp.arange(25),  # KV cache 的 slot indices
    context_lens=None,
    block_tables=None,
)

# 4. 分配外部 KV cache
print(f"\nAllocating KV cache...")
num_blocks = 100
k_cache = jnp.zeros((num_blocks, 256, num_kv_heads, head_dim))
v_cache = jnp.zeros((num_blocks, 256, num_kv_heads, head_dim))
print(f"  k_cache shape: {k_cache.shape}")
print(f"  v_cache shape: {v_cache.shape}")

# 5. Forward pass
print(f"\nRunning forward pass...")
output, k_cache_updated, v_cache_updated = attn(
    hidden_states, positions=positions, k_cache=k_cache, v_cache=v_cache
)

print(f"\nOutput shapes:")
print(f"  output: {output.shape}")
print(f"  k_cache_updated: {k_cache_updated.shape}")
print(f"  v_cache_updated: {v_cache_updated.shape}")

print(f"\n✅ Test passed successfully!")
