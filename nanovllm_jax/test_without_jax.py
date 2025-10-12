#!/usr/bin/env python3
"""
Test JAX implementation without requiring JAX installation.
This test only checks for syntax and import structure issues.
"""

import sys
import ast
import importlib.util

def test_syntax_without_import(file_path):
    """Test syntax of a Python file without importing it."""
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

def test_import_structure(module_path):
    """Test if a module can be loaded without executing it."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec is None:
            print(f"✗ {module_path}: Cannot create spec")
            return False
        
        # Load the module without executing it
        module = importlib.util.module_from_spec(spec)
        print(f"✓ {module_path}: Structure OK")
        return True
    except Exception as e:
        print(f"✗ {module_path}: Structure Error - {e}")
        return False

def main():
    """Test all files in the JAX implementation."""
    print("Testing JAX implementation (without JAX dependency)...")
    print("=" * 60)
    
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
    
    syntax_success = 0
    structure_success = 0
    total_count = len(files_to_test)
    
    print("Testing syntax...")
    for file_path in files_to_test:
        if test_syntax_without_import(file_path):
            syntax_success += 1
    
    print("\nTesting import structure...")
    for file_path in files_to_test:
        if test_import_structure(file_path):
            structure_success += 1
    
    print("\n" + "=" * 60)
    print(f"Syntax Results: {syntax_success}/{total_count} files passed")
    print(f"Structure Results: {structure_success}/{total_count} files passed")
    
    if syntax_success == total_count and structure_success == total_count:
        print("✓ All files are syntactically correct and have proper structure!")
        print("✓ Ready for JAX installation and testing!")
        return 0
    else:
        print("✗ Some files have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
