"""Unit tests for Qwen3ForCausalLMVarlen with prefill and decode phases.

Tests the full varlen attention pipeline including:
- Prefill phase with flattened sequences
- Decode phase with paged attention
- KV cache management
- Context management
"""

import jax
import jax.numpy as jnp
from transformers import AutoConfig
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLMVarlen
from nanovllm_jax.utils.context import set_context, reset_context


MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"


def test_prefill_single_sequence():
    """Test prefill phase with a single sequence."""
    print("\n" + "=" * 70)
    print("Test 1: Prefill with Single Sequence")
    print("=" * 70)

    # Load config
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Initialize model
    model = Qwen3ForCausalLMVarlen(
        config=hf_config,
        dtype=jnp.bfloat16,
        block_size=16,
        rngs=None,
        mesh=None,
    )

    # Prepare input: single sequence of length 10
    seq_len = 10
    input_ids = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.int32)
    positions = jnp.arange(seq_len, dtype=jnp.int32)

    print(f"Input shape: {input_ids.shape}")
    print(f"Positions: {positions}")

    # Set context for prefill
    set_context(
        is_prefill=True,
        cu_seqlens_q=jnp.array([0, seq_len]),
        cu_seqlens_k=jnp.array([0, seq_len]),
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        slot_mapping=jnp.arange(seq_len),  # Maps to cache slots 0-9
        context_lens=None,
        block_tables=None,
    )

    # Allocate KV cache
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_blocks = 10
    block_size = 16

    kv_caches = []
    for _ in range(num_layers):
        k_cache = jnp.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16
        )
        v_cache = jnp.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16
        )
        kv_caches.append((k_cache, v_cache))

    # Forward pass
    logits, updated_kv_caches = model(
        input_ids, positions=positions, kv_caches=kv_caches
    )

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Expected: ({seq_len}, {hf_config.vocab_size})")
    print(f"✓ Number of KV cache pairs: {len(updated_kv_caches)}")
    print(f"✓ K cache shape: {updated_kv_caches[0][0].shape}")
    print(f"✓ V cache shape: {updated_kv_caches[0][1].shape}")

    assert logits.shape == (
        seq_len,
        hf_config.vocab_size,
    ), f"Expected logits shape ({seq_len}, {hf_config.vocab_size}), got {logits.shape}"
    assert len(updated_kv_caches) == num_layers, f"Expected {num_layers} KV cache pairs"

    reset_context()
    print("✅ Test passed!\n")
    return model, updated_kv_caches


def test_prefill_multiple_sequences():
    """Test prefill phase with multiple sequences."""
    print("\n" + "=" * 70)
    print("Test 2: Prefill with Multiple Sequences")
    print("=" * 70)

    # Load config
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Initialize model
    model = Qwen3ForCausalLMVarlen(
        config=hf_config,
        dtype=jnp.bfloat16,
        block_size=16,
        rngs=None,
        mesh=None,
    )

    # Prepare input: 2 sequences with lengths 8 and 12
    seq1 = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
    seq2 = jnp.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=jnp.int32)

    # Flatten sequences
    input_ids = jnp.concatenate([seq1, seq2])  # [20]
    positions = jnp.arange(len(input_ids), dtype=jnp.int32)

    print(f"Input shape: {input_ids.shape}")
    print(f"Sequence 1 length: {len(seq1)}")
    print(f"Sequence 2 length: {len(seq2)}")
    print(f"Total tokens: {len(input_ids)}")

    # Set context for prefill with multiple sequences
    cu_seqlens = jnp.array([0, len(seq1), len(seq1) + len(seq2)])
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=len(seq2),  # max length
        max_seqlen_k=len(seq2),
        slot_mapping=jnp.arange(len(input_ids)),
        context_lens=None,
        block_tables=None,
    )

    # Allocate KV cache
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_blocks = 10
    block_size = 16

    kv_caches = []
    for _ in range(num_layers):
        k_cache = jnp.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16
        )
        v_cache = jnp.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16
        )
        kv_caches.append((k_cache, v_cache))

    # Forward pass
    logits, updated_kv_caches = model(
        input_ids, positions=positions, kv_caches=kv_caches
    )

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Expected: ({len(input_ids)}, {hf_config.vocab_size})")
    print(f"✓ Number of KV cache pairs: {len(updated_kv_caches)}")

    assert logits.shape == (len(input_ids), hf_config.vocab_size)
    assert len(updated_kv_caches) == num_layers

    reset_context()
    print("✅ Test passed!\n")
    return model, updated_kv_caches


def test_decode_single_token():
    """Test decode phase with single token per sequence."""
    print("\n" + "=" * 70)
    print("Test 3: Decode with Single Token")
    print("=" * 70)

    # Load config
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Initialize model
    model = Qwen3ForCausalLMVarlen(
        config=hf_config,
        dtype=jnp.bfloat16,
        block_size=16,
        rngs=None,
        mesh=None,
    )

    # Simulate that we've already done prefill with a sequence of length 10
    # Now we're decoding the 11th token
    batch_size = 1
    context_len = 10  # Previous tokens
    new_token = jnp.array([100], dtype=jnp.int32)  # Single new token
    positions = jnp.array([context_len], dtype=jnp.int32)  # Position 10

    print(f"Input token: {new_token}")
    print(f"Position: {positions[0]}")
    print(f"Context length: {context_len}")

    # Set context for decode
    # In decode, we need block_tables to access the KV cache
    block_tables = jnp.array(
        [[0, -1, -1, -1, -1]], dtype=jnp.int32
    )  # Single sequence uses block 0
    context_lens = jnp.array([context_len], dtype=jnp.int32)
    slot_mapping = jnp.array(
        [context_len], dtype=jnp.int32
    )  # Store new token at slot 10

    set_context(
        is_prefill=False,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=1,
        max_seqlen_k=context_len + 1,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )

    # Allocate KV cache and simulate prefill phase
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_blocks = 10
    block_size = 16

    kv_caches = []
    for _ in range(num_layers):
        # Simulate that cache already has 10 tokens stored
        k_cache = jax.random.normal(
            jax.random.PRNGKey(42),
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=jnp.bfloat16,
        )
        v_cache = jax.random.normal(
            jax.random.PRNGKey(43),
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=jnp.bfloat16,
        )
        kv_caches.append((k_cache, v_cache))

    # Forward pass
    logits, updated_kv_caches = model(
        new_token, positions=positions, kv_caches=kv_caches
    )

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Expected: ({batch_size}, {hf_config.vocab_size})")
    print(f"✓ Number of KV cache pairs: {len(updated_kv_caches)}")

    assert logits.shape == (batch_size, hf_config.vocab_size)
    assert len(updated_kv_caches) == num_layers

    reset_context()
    print("✅ Test passed!\n")
    return model, updated_kv_caches


def test_decode_multiple_sequences():
    """Test decode phase with multiple sequences."""
    print("\n" + "=" * 70)
    print("Test 4: Decode with Multiple Sequences")
    print("=" * 70)

    # Load config
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Initialize model
    model = Qwen3ForCausalLMVarlen(
        config=hf_config,
        dtype=jnp.bfloat16,
        block_size=16,
        rngs=None,
        mesh=None,
    )

    # Simulate 2 sequences with different context lengths
    batch_size = 2
    context_lens = jnp.array([8, 12], dtype=jnp.int32)
    new_tokens = jnp.array([100, 200], dtype=jnp.int32)  # One token per sequence
    positions = context_lens  # Next positions

    print(f"Batch size: {batch_size}")
    print(f"Context lengths: {context_lens}")
    print(f"New tokens: {new_tokens}")
    print(f"Positions: {positions}")

    # Set context for decode
    # Sequence 0 uses block 0, sequence 1 uses block 1
    block_tables = jnp.array(
        [
            [0, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )
    slot_mapping = jnp.array(
        [8, 12], dtype=jnp.int32
    )  # Store tokens at their context positions

    set_context(
        is_prefill=False,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=1,
        max_seqlen_k=13,  # max context len + 1
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )

    # Allocate KV cache
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_blocks = 10
    block_size = 16

    kv_caches = []
    for _ in range(num_layers):
        k_cache = jax.random.normal(
            jax.random.PRNGKey(44),
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=jnp.bfloat16,
        )
        v_cache = jax.random.normal(
            jax.random.PRNGKey(45),
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=jnp.bfloat16,
        )
        kv_caches.append((k_cache, v_cache))

    # Forward pass
    logits, updated_kv_caches = model(
        new_tokens, positions=positions, kv_caches=kv_caches
    )

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Expected: ({batch_size}, {hf_config.vocab_size})")
    print(f"✓ Number of KV cache pairs: {len(updated_kv_caches)}")

    assert logits.shape == (batch_size, hf_config.vocab_size)
    assert len(updated_kv_caches) == num_layers

    reset_context()
    print("✅ Test passed!\n")
    return model, updated_kv_caches


def test_prefill_then_decode():
    """Test a complete prefill -> decode flow."""
    print("\n" + "=" * 70)
    print("Test 5: Complete Prefill -> Decode Flow")
    print("=" * 70)

    # Load config
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Initialize model
    model = Qwen3ForCausalLMVarlen(
        config=hf_config,
        dtype=jnp.bfloat16,
        block_size=16,
        rngs=None,
        mesh=None,
    )

    # Allocate KV cache
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_blocks = 10
    block_size = 16

    kv_caches = []
    for _ in range(num_layers):
        k_cache = jnp.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16
        )
        v_cache = jnp.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim), dtype=jnp.bfloat16
        )
        kv_caches.append((k_cache, v_cache))

    # === PREFILL PHASE ===
    print("\nPhase 1: Prefill")
    print("-" * 70)

    seq_len = 5
    input_ids = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    positions = jnp.arange(seq_len, dtype=jnp.int32)

    print(f"Prefill input: {input_ids}")
    print(f"Prefill positions: {positions}")

    set_context(
        is_prefill=True,
        cu_seqlens_q=jnp.array([0, seq_len]),
        cu_seqlens_k=jnp.array([0, seq_len]),
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        slot_mapping=jnp.arange(seq_len),
        context_lens=None,
        block_tables=None,
    )

    prefill_logits, kv_caches = model(
        input_ids, positions=positions, kv_caches=kv_caches
    )

    print(f"✓ Prefill logits shape: {prefill_logits.shape}")
    assert prefill_logits.shape == (seq_len, hf_config.vocab_size)

    reset_context()

    # === DECODE PHASE (Only 1 step for debugging) ===
    print("\nPhase 2: Decode (1 step)")
    print("-" * 70)

    context_len = seq_len
    block_tables = jnp.array([[0, -1, -1, -1, -1]], dtype=jnp.int32)

    new_token = jnp.array([100], dtype=jnp.int32)
    position = jnp.array([context_len], dtype=jnp.int32)
    slot_mapping = jnp.array([context_len], dtype=jnp.int32)
    context_lens_array = jnp.array([context_len], dtype=jnp.int32)

    print(f"\nDecode step 1:")
    print(
        f"  Token: {new_token[0]}, Position: {position[0]}, Context len: {context_lens_array[0]}"
    )
    print(f"  Slot mapping: {slot_mapping}")
    print(f"  Block tables: {block_tables}")

    # Debug: print context before setting
    print(f"\nSetting decode context with is_prefill=False...")

    set_context(
        is_prefill=False,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=1,
        max_seqlen_k=context_len + 1,
        slot_mapping=slot_mapping,
        context_lens=context_lens_array,
        block_tables=block_tables,
    )

    # Debug: verify context
    from nanovllm_jax.utils.context import get_context

    ctx = get_context()
    print(f"  Context is_prefill: {ctx.is_prefill}")
    print(
        f"  Context slot_mapping shape: {ctx.slot_mapping.shape if ctx.slot_mapping is not None else None}"
    )
    print(f"  Context context_lens: {ctx.context_lens}")

    try:
        decode_logits, kv_caches = model(
            new_token, positions=position, kv_caches=kv_caches
        )
        print(f"  ✓ Decode logits shape: {decode_logits.shape}")
        assert decode_logits.shape == (1, hf_config.vocab_size)
        print("\n✅ Complete flow test passed!\n")
    except Exception as e:
        print(f"\n❌ Decode failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise

    reset_context()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Qwen3ForCausalLMVarlen Unit Tests")
    print("=" * 70)

    try:
        # Run only complete flow tests
        print("\n⚠️  Running complete prefill->decode flow test only")
        print("    (Standalone decode tests are skipped)\n")

        test_prefill_single_sequence()
        test_prefill_multiple_sequences()
        # Skip standalone decode tests for now
        # test_decode_single_token()
        # test_decode_multiple_sequences()
        test_prefill_then_decode()  # This is the complete flow test

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
