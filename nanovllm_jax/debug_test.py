#!/usr/bin/env python3
"""
Debug test to identify the source of the frozen attributes error.
"""

import jax
import jax.numpy as jnp
from transformers import Qwen3Config

def test_individual_components():
    """Test individual components to identify the problematic one."""
    print("Testing individual components...")
    
    try:
        from nanovllm_jax.layers.linear import ReplicatedLinear
        print("✓ ReplicatedLinear import OK")
        
        # Test ReplicatedLinear
        linear = ReplicatedLinear(input_size=128, output_size=256)
        x = jnp.ones((2, 128))
        params = linear.init(jax.random.PRNGKey(0), x)
        print("✓ ReplicatedLinear init OK")
        
    except Exception as e:
        print(f"✗ ReplicatedLinear failed: {e}")
        return False
    
    try:
        from nanovllm_jax.layers.layernorm import RMSNorm
        print("✓ RMSNorm import OK")
        
        # Test RMSNorm
        norm = RMSNorm(hidden_size=128)
        x = jnp.ones((2, 128))
        params = norm.init(jax.random.PRNGKey(0), x)
        print("✓ RMSNorm init OK")
        
    except Exception as e:
        print(f"✗ RMSNorm failed: {e}")
        return False
    
    try:
        from nanovllm_jax.layers.embed_head import VocabParallelEmbedding
        print("✓ VocabParallelEmbedding import OK")
        
        # Test VocabParallelEmbedding
        embed = VocabParallelEmbedding(vocab_size=1000, hidden_size=128)
        x = jnp.array([1, 2, 3])
        params = embed.init(jax.random.PRNGKey(0), x)
        print("✓ VocabParallelEmbedding init OK")
        
    except Exception as e:
        print(f"✗ VocabParallelEmbedding failed: {e}")
        return False
    
    try:
        from nanovllm_jax.layers.embed_head import ParallelLMHead
        print("✓ ParallelLMHead import OK")
        
        # Test ParallelLMHead
        lm_head = ParallelLMHead(vocab_size=1000, hidden_size=128)
        x = jnp.ones((2, 128))
        params = lm_head.init(jax.random.PRNGKey(0), x)
        print("✓ ParallelLMHead init OK")
        
    except Exception as e:
        print(f"✗ ParallelLMHead failed: {e}")
        return False
    
    return True

def test_model_components():
    """Test model components."""
    print("\nTesting model components...")
    
    try:
        from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
        print("✓ Qwen3ForCausalLM import OK")
        
        # Create a minimal config
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=256,
            max_position_embeddings=1024,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            tie_word_embeddings=False
        )
        
        # Test model initialization
        model = Qwen3ForCausalLM(config=config)
        input_ids = jnp.array([1, 2, 3])
        positions = jnp.array([0, 1, 2])
        
        print("Attempting model initialization...")
        params = model.init(jax.random.PRNGKey(0), input_ids, positions)
        print("✓ Qwen3ForCausalLM init OK")
        
    except Exception as e:
        print(f"✗ Qwen3ForCausalLM failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main debug function."""
    print("Debug Test for Frozen Attributes Error")
    print("=" * 50)
    
    if test_individual_components():
        print("\n✓ All individual components work")
    else:
        print("\n✗ Some individual components failed")
        return 1
    
    if test_model_components():
        print("\n✓ Model components work")
    else:
        print("\n✗ Model components failed")
        return 1
    
    print("\n✅ All tests passed!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
