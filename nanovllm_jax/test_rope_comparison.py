"""
Unit test to compare two RoPE implementations:
1. qwen3.py: _build_rope_cos_sin + _apply_rope (inline implementation)
2. rotary_embedding.py: RotaryEmbedding class (reusable module)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from transformers import PretrainedConfig

# Import the new implementation from qwen3.py
from nanovllm_jax.models.qwen3 import _build_rope_cos_sin, _apply_rope

# Import the old implementation from rotary_embedding.py
from nanovllm_jax.layers.rotary_embedding import RotaryEmbedding


def test_rope_comparison():
    """Test if both RoPE implementations produce the same results."""

    # Setup test parameters
    batch_size = 2
    seq_len = 8
    num_heads = 4
    head_dim = 64
    rope_theta = 10000.0
    max_position = 128
    dtype = jnp.float32

    print("=" * 80)
    print("RoPE Comparison Test")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  RoPE theta (base): {rope_theta}")
    print(f"  Max position: {max_position}")
    print(f"  Data type: {dtype}")
    print()

    # Create random Q and K tensors
    key = jax.random.PRNGKey(42)
    key_q, key_k = jax.random.split(key)

    # Shape for qwen3.py: [batch, seq, heads, head_dim]
    q_new = jax.random.normal(
        key_q, (batch_size, seq_len, num_heads, head_dim), dtype=dtype
    )
    k_new = jax.random.normal(
        key_k, (batch_size, seq_len, num_heads, head_dim), dtype=dtype
    )

    # Shape for rotary_embedding.py: [num_tokens, num_heads * head_dim]
    # num_tokens = batch_size * seq_len
    num_tokens = batch_size * seq_len
    q_old = q_new.reshape(num_tokens, num_heads * head_dim)
    k_old = k_new.reshape(num_tokens, num_heads * head_dim)

    print(f"Input shapes:")
    print(f"  qwen3.py format - q: {q_new.shape}, k: {k_new.shape}")
    print(f"  rotary_embedding.py format - q: {q_old.shape}, k: {k_old.shape}")
    print()

    # ========== New implementation (qwen3.py) ==========
    print("-" * 80)
    print("Testing NEW implementation (qwen3.py)")
    print("-" * 80)

    # Build cos/sin tables
    cos_new, sin_new = _build_rope_cos_sin(seq_len, head_dim, rope_theta, dtype=dtype)
    print(f"cos shape: {cos_new.shape}, sin shape: {sin_new.shape}")

    # Apply RoPE
    q_rope_new = _apply_rope(q_new, cos_new, sin_new)
    k_rope_new = _apply_rope(k_new, cos_new, sin_new)
    print(f"After RoPE - q: {q_rope_new.shape}, k: {k_rope_new.shape}")
    print()

    # ========== Old implementation (rotary_embedding.py) ==========
    print("-" * 80)
    print("Testing OLD implementation (rotary_embedding.py with NNX)")
    print("-" * 80)

    # Initialize RotaryEmbedding using NNX (no need for init/bind)
    rotary_emb = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=max_position,
        base=rope_theta,
        dtype=dtype,
    )

    # Create positions array [0, 1, 2, ..., seq_len-1] repeated for each batch
    positions = jnp.tile(jnp.arange(seq_len), batch_size)
    print(f"positions: {positions}")

    # Apply RoPE directly (NNX modules are callable immediately)
    q_rope_old, k_rope_old = rotary_emb(positions, q_old, k_old)
    print(f"After RoPE - q: {q_rope_old.shape}, k: {k_rope_old.shape}")

    # Reshape old results back to [batch, seq, heads, head_dim] for comparison
    q_rope_old_reshaped = q_rope_old.reshape(batch_size, seq_len, num_heads, head_dim)
    k_rope_old_reshaped = k_rope_old.reshape(batch_size, seq_len, num_heads, head_dim)
    print(
        f"After reshape - q: {q_rope_old_reshaped.shape}, k: {k_rope_old_reshaped.shape}"
    )
    print()

    # ========== Comparison ==========
    print("=" * 80)
    print("Comparison Results")
    print("=" * 80)

    # Check if results are close
    q_match = jnp.allclose(q_rope_new, q_rope_old_reshaped, rtol=1e-5, atol=1e-5)
    k_match = jnp.allclose(k_rope_new, k_rope_old_reshaped, rtol=1e-5, atol=1e-5)

    print(f"Q tensors match: {q_match}")
    print(f"K tensors match: {k_match}")

    # Compute differences
    q_max_diff = jnp.max(jnp.abs(q_rope_new - q_rope_old_reshaped))
    k_max_diff = jnp.max(jnp.abs(k_rope_new - k_rope_old_reshaped))
    q_mean_diff = jnp.mean(jnp.abs(q_rope_new - q_rope_old_reshaped))
    k_mean_diff = jnp.mean(jnp.abs(k_rope_new - k_rope_old_reshaped))

    print(f"\nQ tensor differences:")
    print(f"  Max absolute difference: {q_max_diff:.2e}")
    print(f"  Mean absolute difference: {q_mean_diff:.2e}")

    print(f"\nK tensor differences:")
    print(f"  Max absolute difference: {k_max_diff:.2e}")
    print(f"  Mean absolute difference: {k_mean_diff:.2e}")

    # Show sample values for inspection
    print(f"\nSample values comparison (first element of first batch, first head):")
    print(f"  New Q[0,0,0,:5]: {q_rope_new[0,0,0,:5]}")
    print(f"  Old Q[0,0,0,:5]: {q_rope_old_reshaped[0,0,0,:5]}")
    print(f"  Difference:      {q_rope_new[0,0,0,:5] - q_rope_old_reshaped[0,0,0,:5]}")

    print(f"\n  New K[0,0,0,:5]: {k_rope_new[0,0,0,:5]}")
    print(f"  Old K[0,0,0,:5]: {k_rope_old_reshaped[0,0,0,:5]}")
    print(f"  Difference:      {k_rope_new[0,0,0,:5] - k_rope_old_reshaped[0,0,0,:5]}")

    print()
    print("=" * 80)

    # Assert that both implementations produce the same results
    assert q_match, f"Q tensors do not match! Max diff: {q_max_diff}"
    assert k_match, f"K tensors do not match! Max diff: {k_max_diff}"

    print("✅ TEST PASSED: Both RoPE implementations produce identical results!")
    print("=" * 80)


def test_rope_decode_scenario():
    """Test RoPE in decode scenario (with position offset)."""

    print("\n\n")
    print("=" * 80)
    print("RoPE Decode Scenario Test (with position offset)")
    print("=" * 80)

    # Setup test parameters
    batch_size = 1
    current_seq_len = 1  # Decode processes 1 token at a time
    past_len = 10  # Already processed 10 tokens
    num_heads = 4
    head_dim = 64
    rope_theta = 10000.0
    max_position = 128
    dtype = jnp.float32

    print(f"Configuration:")
    print(f"  Past length: {past_len}")
    print(f"  Current sequence length: {current_seq_len}")
    print(f"  Head dimension: {head_dim}")
    print()

    # Create random Q and K tensors for current token
    key = jax.random.PRNGKey(123)
    key_q, key_k = jax.random.split(key)

    q_new = jax.random.normal(
        key_q, (batch_size, current_seq_len, num_heads, head_dim), dtype=dtype
    )
    k_new = jax.random.normal(
        key_k, (batch_size, current_seq_len, num_heads, head_dim), dtype=dtype
    )

    # ========== New implementation (qwen3.py) - decode mode ==========
    print("-" * 80)
    print("NEW implementation (decode mode with position offset)")
    print("-" * 80)

    # Build full cos/sin table up to past_len + current_seq_len
    cos_new, sin_new = _build_rope_cos_sin(
        past_len + current_seq_len, head_dim, rope_theta, dtype=dtype
    )
    # Take only the positions for current tokens
    cos_new_slice = cos_new[past_len : past_len + current_seq_len, :]
    sin_new_slice = sin_new[past_len : past_len + current_seq_len, :]

    print(
        f"cos slice shape: {cos_new_slice.shape}, sin slice shape: {sin_new_slice.shape}"
    )

    q_rope_new = _apply_rope(q_new, cos_new_slice, sin_new_slice)
    k_rope_new = _apply_rope(k_new, cos_new_slice, sin_new_slice)

    print(f"After RoPE - q: {q_rope_new.shape}, k: {k_rope_new.shape}")
    print()

    # ========== Old implementation (rotary_embedding.py) ==========
    print("-" * 80)
    print("OLD implementation (rotary_embedding.py with NNX)")
    print("-" * 80)

    # Initialize RotaryEmbedding using NNX (no need for init/bind)
    rotary_emb = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=max_position,
        base=rope_theta,
        dtype=dtype,
    )

    # Positions array should be [past_len] for the current token
    positions = jnp.array([past_len])
    print(f"positions: {positions}")

    num_tokens = batch_size * current_seq_len
    q_old = q_new.reshape(num_tokens, num_heads * head_dim)
    k_old = k_new.reshape(num_tokens, num_heads * head_dim)

    # Apply RoPE directly (NNX modules are callable immediately)
    q_rope_old, k_rope_old = rotary_emb(positions, q_old, k_old)

    q_rope_old_reshaped = q_rope_old.reshape(
        batch_size, current_seq_len, num_heads, head_dim
    )
    k_rope_old_reshaped = k_rope_old.reshape(
        batch_size, current_seq_len, num_heads, head_dim
    )

    print(
        f"After RoPE - q: {q_rope_old_reshaped.shape}, k: {k_rope_old_reshaped.shape}"
    )
    print()

    # ========== Comparison ==========
    print("=" * 80)
    print("Decode Scenario Comparison Results")
    print("=" * 80)

    q_match = jnp.allclose(q_rope_new, q_rope_old_reshaped, rtol=1e-5, atol=1e-5)
    k_match = jnp.allclose(k_rope_new, k_rope_old_reshaped, rtol=1e-5, atol=1e-5)

    print(f"Q tensors match: {q_match}")
    print(f"K tensors match: {k_match}")

    q_max_diff = jnp.max(jnp.abs(q_rope_new - q_rope_old_reshaped))
    k_max_diff = jnp.max(jnp.abs(k_rope_new - k_rope_old_reshaped))

    print(f"\nQ max difference: {q_max_diff:.2e}")
    print(f"K max difference: {k_max_diff:.2e}")

    print(f"\nSample values (position {past_len}):")
    print(f"  New Q[0,0,0,:5]: {q_rope_new[0,0,0,:5]}")
    print(f"  Old Q[0,0,0,:5]: {q_rope_old_reshaped[0,0,0,:5]}")

    assert q_match, f"Q tensors do not match in decode mode! Max diff: {q_max_diff}"
    assert k_match, f"K tensors do not match in decode mode! Max diff: {k_max_diff}"

    print(
        "\n✅ DECODE TEST PASSED: Both RoPE implementations produce identical results!"
    )
    print("=" * 80)


if __name__ == "__main__":
    # Run both tests
    test_rope_comparison()
    test_rope_decode_scenario()

    print("\n\n")
    print("🎉 All tests passed! Both RoPE implementations are equivalent.")
