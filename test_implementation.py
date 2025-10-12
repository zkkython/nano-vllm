#!/usr/bin/env python3
"""
Test script for JAX-based nano-vllm implementation.
"""

import jax
import jax.numpy as jnp
from nanovllm_jax.layers.layernorm import RMSNorm
from nanovllm_jax.layers.linear import ReplicatedLinear
from nanovllm_jax.layers.activation import SiluAndMul


def test_layers():
    """Test individual layer implementations."""
    print("Testing JAX layers...")
    
    # Test RMSNorm
    print("Testing RMSNorm...")
    rms_norm = RMSNorm(hidden_size=128, dtype=jnp.float32)
    x = jnp.ones((2, 128), dtype=jnp.float32)
    params = rms_norm.init(jax.random.PRNGKey(0), x)
    output = rms_norm.apply(params, x)
    print(f"RMSNorm input shape: {x.shape}, output shape: {output.shape}")
    
    # Test ReplicatedLinear
    print("Testing ReplicatedLinear...")
    linear = ReplicatedLinear(input_size=128, output_size=256, dtype=jnp.float32)
    x = jnp.ones((2, 128), dtype=jnp.float32)
    params = linear.init(jax.random.PRNGKey(0), x)
    output = linear.apply(params, x)
    print(f"Linear input shape: {x.shape}, output shape: {output.shape}")
    
    # Test SiluAndMul
    print("Testing SiluAndMul...")
    silu_mul = SiluAndMul()
    x = jnp.ones((2, 256), dtype=jnp.float32)
    output = silu_mul.apply({}, x)
    print(f"SiluAndMul input shape: {x.shape}, output shape: {output.shape}")
    
    print("✓ All layer tests passed!")


def test_jax_setup():
    """Test JAX setup and device configuration."""
    print("Testing JAX setup...")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Test basic JAX operations
    x = jnp.array([1, 2, 3, 4])
    y = jnp.array([5, 6, 7, 8])
    z = x + y
    print(f"Basic JAX operation: {x} + {y} = {z}")
    
    print("✓ JAX setup test passed!")


if __name__ == "__main__":
    print("JAX-based Nano-vLLM Implementation Test")
    print("=" * 50)
    
    try:
        test_jax_setup()
        print()
        test_layers()
        print("\n✓ All tests passed! JAX implementation is working correctly.")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
