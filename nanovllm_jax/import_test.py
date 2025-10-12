#!/usr/bin/env python3
"""
Import test for JAX implementation.
This test checks if all modules can be imported without errors.
"""

import sys
import importlib

def test_import(module_name):
    """Test importing a module."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name}: Import OK")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: Import Error - {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name}: Error - {e}")
        return False

def main():
    """Test all module imports."""
    print("Testing JAX implementation imports...")
    print("=" * 50)
    
    modules_to_test = [
        "nanovllm_jax",
        "nanovllm_jax.config",
        "nanovllm_jax.sampling_params",
        "nanovllm_jax.layers.linear",
        "nanovllm_jax.layers.layernorm",
        "nanovllm_jax.layers.activation",
        "nanovllm_jax.layers.rotary_embedding",
        "nanovllm_jax.layers.attention",
        "nanovllm_jax.layers.embed_head",
        "nanovllm_jax.layers.sampler",
        "nanovllm_jax.engine.sequence",
        "nanovllm_jax.engine.scheduler",
        "nanovllm_jax.utils.context",
        "nanovllm_jax.utils.loader",
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module_name in modules_to_test:
        if test_import(module_name):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {success_count}/{total_count} modules imported successfully")
    
    if success_count == total_count:
        print("✓ All modules can be imported!")
        return 0
    else:
        print("✗ Some modules failed to import")
        return 1

if __name__ == "__main__":
    sys.exit(main())
