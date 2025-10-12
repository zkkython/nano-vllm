#!/usr/bin/env python3
"""
Syntax test for JAX implementation without requiring JAX installation.
This test only checks for syntax errors and import issues.
"""

import sys
import ast

def test_syntax(file_path):
    """Test syntax of a Python file."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source)
        print(f"✓ {file_path}: Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {file_path}: Syntax Error - {e}")
        return False
    except Exception as e:
        print(f"✗ {file_path}: Error - {e}")
        return False

def main():
    """Test all Python files in the JAX implementation."""
    print("Testing JAX implementation syntax...")
    print("=" * 50)
    
    files_to_test = [
        "nanovllm_jax/__init__.py",
        "nanovllm_jax/config.py",
        "nanovllm_jax/sampling_params.py",
        "nanovllm_jax/llm.py",
        "nanovllm_jax/layers/__init__.py",
        "nanovllm_jax/layers/linear.py",
        "nanovllm_jax/layers/layernorm.py",
        "nanovllm_jax/layers/activation.py",
        "nanovllm_jax/layers/rotary_embedding.py",
        "nanovllm_jax/layers/attention.py",
        "nanovllm_jax/layers/embed_head.py",
        "nanovllm_jax/layers/sampler.py",
        "nanovllm_jax/engine/__init__.py",
        "nanovllm_jax/engine/sequence.py",
        "nanovllm_jax/engine/scheduler.py",
        "nanovllm_jax/engine/model_runner.py",
        "nanovllm_jax/models/__init__.py",
        "nanovllm_jax/models/qwen3.py",
        "nanovllm_jax/utils/__init__.py",
        "nanovllm_jax/utils/context.py",
        "nanovllm_jax/utils/loader.py",
    ]
    
    success_count = 0
    total_count = len(files_to_test)
    
    for file_path in files_to_test:
        if test_syntax(file_path):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {success_count}/{total_count} files passed syntax check")
    
    if success_count == total_count:
        print("✓ All files have correct syntax!")
        return 0
    else:
        print("✗ Some files have syntax errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
