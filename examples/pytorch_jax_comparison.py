"""
PyTorch vs JAX Comparison for VocabParallelEmbedding

This script compares the PyTorch and JAX implementations of VocabParallelEmbedding,
showing the key differences in API, performance, and distributed execution patterns.

Run: python examples/pytorch_jax_comparison.py
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.embed_head import VocabParallelEmbedding as TorchVocabParallelEmbedding
from nanovllm.layers.jax_embed_head import VocabParallelEmbedding as JaxVocabParallelEmbedding


def pytorch_implementation_demo():
    """Demonstrate PyTorch VocabParallelEmbedding implementation."""
    print("="*60)
    print("PYTORCH IMPLEMENTATION")
    print("="*60)
    
    # Configuration
    num_embeddings = 8
    embedding_dim = 4
    batch_size = 2
    seq_len = 3
    
    # Create test data
    x = torch.randint(0, num_embeddings, (batch_size, seq_len))
    print(f"Input tokens shape: {x.shape}")
    print(f"Input tokens:\n{x}")
    
    # Simulate distributed environment (single process for demo)
    # In real usage, this would be initialized with dist.init_process_group()
    class MockDist:
        @staticmethod
        def get_rank():
            return 0
        
        @staticmethod
        def get_world_size():
            return 2
    
    # Monkey patch dist for demo
    original_dist = dist
    dist.get_rank = MockDist.get_rank
    dist.get_world_size = MockDist.get_world_size
    
    try:
        # Create PyTorch embedding
        torch_embedding = TorchVocabParallelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        
        print(f"PyTorch embedding weight shape: {torch_embedding.weight.shape}")
        print(f"Vocabulary range: {torch_embedding.vocab_start_idx}-{torch_embedding.vocab_end_idx-1}")
        
        # Forward pass
        with torch.no_grad():
            output = torch_embedding(x)
        
        print(f"PyTorch output shape: {output.shape}")
        print(f"PyTorch output:\n{output}")
        
    finally:
        # Restore original dist
        dist.get_rank = original_dist.get_rank
        dist.get_world_size = original_dist.get_world_size


def jax_implementation_demo():
    """Demonstrate JAX VocabParallelEmbedding implementation."""
    print("\n" + "="*60)
    print("JAX IMPLEMENTATION")
    print("="*60)
    
    # Configuration
    num_embeddings = 8
    embedding_dim = 4
    batch_size = 2
    seq_len = 3
    tp_size = 2
    
    # Create test data
    rng = random.PRNGKey(42)
    x = random.randint(rng, (batch_size, seq_len), 0, num_embeddings)
    print(f"Input tokens shape: {x.shape}")
    print(f"Input tokens:\n{x}")
    
    # Create JAX embedding
    jax_embedding = JaxVocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=0
    )
    
    # Initialize parameters
    rng, init_rng = random.split(rng)
    params = jax_embedding.init(init_rng, x)
    
    print(f"JAX embedding weight shape: {params['params']['embedding'].shape}")
    print(f"Vocabulary range: {jax_embedding.vocab_start_idx}-{jax_embedding.vocab_end_idx-1}")
    
    # Forward pass
    output = jax_embedding.apply(params, x)
    
    print(f"JAX output shape: {output.shape}")
    print(f"JAX output:\n{output}")


def compare_key_differences():
    """Compare key differences between PyTorch and JAX implementations."""
    print("\n" + "="*60)
    print("KEY DIFFERENCES COMPARISON")
    print("="*60)
    
    differences = [
        {
            "Aspect": "Framework",
            "PyTorch": "PyTorch with torch.distributed",
            "JAX": "JAX with Flax and JAX primitives"
        },
        {
            "Aspect": "Distributed Communication",
            "PyTorch": "torch.distributed.all_reduce()",
            "JAX": "lax.psum() and collective operations"
        },
        {
            "Aspect": "Device Management",
            "PyTorch": "Manual device placement with .cuda()",
            "JAX": "Automatic device placement with mesh and sharding"
        },
        {
            "Aspect": "Compilation",
            "PyTorch": "torch.compile() or TorchScript",
            "JAX": "jax.jit() with XLA compilation"
        },
        {
            "Aspect": "Parameter Initialization",
            "PyTorch": "nn.Parameter with manual initialization",
            "JAX": "Flax parameter system with init functions"
        },
        {
            "Aspect": "Sharding Strategy",
            "PyTorch": "Manual tensor slicing and masking",
            "JAX": "NamedSharding with PartitionSpec"
        },
        {
            "Aspect": "Memory Management",
            "PyTorch": "Manual memory management",
            "JAX": "Functional programming with immutable arrays"
        },
        {
            "Aspect": "Gradient Computation",
            "PyTorch": "Automatic differentiation with autograd",
            "JAX": "Functional differentiation with jax.grad"
        }
    ]
    
    print(f"{'Aspect':<25} {'PyTorch':<35} {'JAX':<35}")
    print("-" * 95)
    for diff in differences:
        print(f"{diff['Aspect']:<25} {diff['PyTorch']:<35} {diff['JAX']:<35}")


def compare_performance():
    """Compare performance characteristics."""
    print("\n" + "="*60)
    print("PERFORMANCE CHARACTERISTICS")
    print("="*60)
    
    characteristics = [
        {
            "Characteristic": "Compilation Time",
            "PyTorch": "Faster initial compilation",
            "JAX": "Slower initial compilation, faster execution"
        },
        {
            "Characteristic": "Memory Usage",
            "PyTorch": "Higher memory usage due to mutable state",
            "JAX": "Lower memory usage with functional approach"
        },
        {
            "Characteristic": "Distributed Efficiency",
            "PyTorch": "Good with NCCL backend",
            "JAX": "Excellent with XLA collective operations"
        },
        {
            "Characteristic": "GPU Utilization",
            "PyTorch": "Good with proper device placement",
            "JAX": "Excellent with automatic optimization"
        },
        {
            "Characteristic": "Debugging",
            "PyTorch": "Easier debugging with imperative style",
            "JAX": "More complex due to functional nature"
        },
        {
            "Characteristic": "Ecosystem",
            "PyTorch": "Mature ecosystem with many libraries",
            "JAX": "Growing ecosystem, excellent for research"
        }
    ]
    
    print(f"{'Characteristic':<20} {'PyTorch':<40} {'JAX':<40}")
    print("-" * 100)
    for char in characteristics:
        print(f"{char['Characteristic']:<20} {char['PyTorch']:<40} {char['JAX']:<40}")


def migration_guide():
    """Provide migration guide from PyTorch to JAX."""
    print("\n" + "="*60)
    print("MIGRATION GUIDE: PyTorch → JAX")
    print("="*60)
    
    migration_steps = [
        "1. Replace torch.nn.Module with flax.linen.Module",
        "2. Convert torch.distributed calls to JAX collective operations",
        "3. Replace nn.Parameter with Flax parameter system",
        "4. Use JAX sharding instead of manual tensor slicing",
        "5. Implement functional forward pass (no mutable state)",
        "6. Use jax.jit() for compilation instead of torch.compile()",
        "7. Replace torch tensors with jax.numpy arrays",
        "8. Use JAX mesh for device placement instead of .cuda()",
        "9. Implement proper weight loading with shard_map",
        "10. Test with JAX's functional differentiation"
    ]
    
    for step in migration_steps:
        print(step)
    
    print("\nKey Benefits of JAX Version:")
    benefits = [
        "• Better GPU utilization with XLA compilation",
        "• More efficient distributed communication",
        "• Functional programming reduces bugs",
        "• Better memory efficiency",
        "• Seamless integration with JAX ecosystem",
        "• Excellent performance on TPUs",
        "• More composable and testable code"
    ]
    
    for benefit in benefits:
        print(benefit)


def main():
    """Run the comparison demo."""
    print("PyTorch vs JAX VocabParallelEmbedding Comparison")
    print("="*60)
    
    # Run demonstrations
    pytorch_implementation_demo()
    jax_implementation_demo()
    compare_key_differences()
    compare_performance()
    migration_guide()
    
    print("\n" + "="*60)
    print("Comparison completed!")
    print("="*60)


if __name__ == "__main__":
    main()
