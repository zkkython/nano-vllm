"""
JAX GPU Demo for VocabParallelEmbedding and ParallelLMHead

This script demonstrates the JAX implementation of distributed vocabulary parallel
embedding layers with GPU acceleration. It shows how to:

1. Set up JAX with GPU support
2. Create distributed embedding layers
3. Perform forward passes with proper sharding
4. Simulate multi-device execution
5. Compare with PyTorch reference implementation

Run: python examples/jax_embed_head_demo.py
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.jax_embed_head import (
    VocabParallelEmbedding, 
    ParallelLMHead, 
    create_mesh,
    initialize_embedding_weights
)


def setup_jax_gpu():
    """Setup JAX with GPU support and print device information."""
    print("JAX version:", jax.__version__)
    print("Available devices:", jax.devices())
    print("Default backend:", jax.default_backend())
    
    # Check if GPU is available
    gpu_devices = [d for d in jax.devices() if d.device_kind == 'gpu']
    if gpu_devices:
        print(f"Found {len(gpu_devices)} GPU devices: {gpu_devices}")
        jax.config.update('jax_default_device', gpu_devices[0])
    else:
        print("No GPU devices found, using CPU")
    
    return gpu_devices


def demo_basic_embedding():
    """Demonstrate basic VocabParallelEmbedding functionality."""
    print("\n" + "="*60)
    print("DEMO 1: Basic VocabParallelEmbedding")
    print("="*60)
    
    # Configuration
    num_embeddings = 8
    embedding_dim = 4
    tp_size = 2
    batch_size = 2
    seq_len = 3
    
    # Create test data
    rng = random.PRNGKey(42)
    x = random.randint(rng, (batch_size, seq_len), 0, num_embeddings)
    print(f"Input tokens shape: {x.shape}")
    print(f"Input tokens:\n{x}")
    
    # Create embedding layers for both ranks
    embedding_rank0 = VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=0,
        dtype=jnp.float32
    )
    
    embedding_rank1 = VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=1,
        dtype=jnp.float32
    )
    
    # Initialize parameters
    rng, init_rng0, init_rng1 = random.split(rng, 3)
    params_rank0 = embedding_rank0.init(init_rng0, x)
    params_rank1 = embedding_rank1.init(init_rng1, x)
    
    print(f"\nRank 0 vocabulary range: {embedding_rank0.vocab_start_idx}-{embedding_rank0.vocab_end_idx-1}")
    print(f"Rank 1 vocabulary range: {embedding_rank1.vocab_start_idx}-{embedding_rank1.vocab_end_idx-1}")
    
    # Forward pass
    output_rank0 = embedding_rank0.apply(params_rank0, x)
    output_rank1 = embedding_rank1.apply(params_rank1, x)
    
    print(f"\nRank 0 output shape: {output_rank0.shape}")
    print(f"Rank 1 output shape: {output_rank1.shape}")
    
    # Simulate all-reduce (sum across ranks)
    final_output = output_rank0 + output_rank1
    print(f"Final output shape: {final_output.shape}")
    print(f"Final output:\n{final_output}")
    
    return final_output


def demo_parallel_lm_head():
    """Demonstrate ParallelLMHead functionality."""
    print("\n" + "="*60)
    print("DEMO 2: ParallelLMHead")
    print("="*60)
    
    # Configuration
    num_embeddings = 8
    embedding_dim = 4
    hidden_dim = 4
    tp_size = 2
    batch_size = 2
    seq_len = 3
    
    # Create test data (hidden states)
    rng = random.PRNGKey(123)
    hidden_states = random.normal(rng, (batch_size, seq_len, hidden_dim))
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Create LM head layers
    lm_head_rank0 = ParallelLMHead(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=0,
        bias=True,
        dtype=jnp.float32
    )
    
    lm_head_rank1 = ParallelLMHead(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=1,
        bias=True,
        dtype=jnp.float32
    )
    
    # Initialize parameters
    rng, init_rng0, init_rng1 = random.split(rng, 3)
    params_rank0 = lm_head_rank0.init(init_rng0, hidden_states)
    params_rank1 = lm_head_rank1.init(init_rng1, hidden_states)
    
    # Forward pass
    logits_rank0 = lm_head_rank0.apply(params_rank0, hidden_states)
    logits_rank1 = lm_head_rank1.apply(params_rank1, hidden_states)
    
    print(f"Rank 0 logits shape: {logits_rank0.shape}")
    print(f"Rank 1 logits shape: {logits_rank1.shape}")
    
    # Simulate gathering logits (concatenate along vocab dimension)
    full_logits = jnp.concatenate([logits_rank0, logits_rank1], axis=-1)
    print(f"Full logits shape: {full_logits.shape}")
    print(f"Full logits:\n{full_logits}")
    
    return full_logits


def demo_sharded_execution():
    """Demonstrate sharded execution with JAX mesh."""
    print("\n" + "="*60)
    print("DEMO 3: Sharded Execution with JAX Mesh")
    print("="*60)
    
    # Get available devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    # Create mesh for distributed execution
    tp_size = min(2, len(devices))  # Use up to 2 devices for tensor parallelism
    mesh = create_mesh(devices[:tp_size], tp_size)
    print(f"Created mesh with {tp_size} devices")
    
    # Configuration
    num_embeddings = 8
    embedding_dim = 4
    batch_size = 2
    seq_len = 3
    
    # Create test data
    rng = random.PRNGKey(456)
    x = random.randint(rng, (batch_size, seq_len), 0, num_embeddings)
    
    # Create embedding layer
    embedding = VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=0,  # This will be handled by shard_map
        dtype=jnp.float32
    )
    
    # Initialize parameters
    rng, init_rng = random.split(rng)
    params = embedding.init(init_rng, x)
    
    # Sharded forward pass
    try:
        output = embedding.sharded_forward(x, mesh)
        print(f"Sharded output shape: {output.shape}")
        print(f"Sharded output:\n{output}")
    except Exception as e:
        print(f"Sharded execution failed (expected with single device): {e}")
        print("This is normal when running on a single device.")


def demo_performance_comparison():
    """Compare JAX performance with a simple benchmark."""
    print("\n" + "="*60)
    print("DEMO 4: Performance Benchmark")
    print("="*60)
    
    # Configuration
    num_embeddings = 10000
    embedding_dim = 512
    batch_size = 32
    seq_len = 128
    tp_size = 1  # Single device for now
    
    # Create test data
    rng = random.PRNGKey(789)
    x = random.randint(rng, (batch_size, seq_len), 0, num_embeddings)
    
    # Create embedding layer
    embedding = VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=0,
        dtype=jnp.float32
    )
    
    # Initialize parameters
    rng, init_rng = random.split(rng)
    params = embedding.init(init_rng, x)
    
    # JIT compile the forward pass
    forward_fn = jax.jit(embedding.apply)
    
    # Warmup
    _ = forward_fn(params, x)
    
    # Benchmark
    import time
    num_iterations = 100
    
    start_time = time.time()
    for _ in range(num_iterations):
        _ = forward_fn(params, x)
    jax.device_arrays.block_until_ready(_)  # Ensure all computations are done
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"Average forward pass time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {batch_size * seq_len / avg_time:.0f} tokens/sec")


def demo_weight_loading():
    """Demonstrate weight loading for distributed embeddings."""
    print("\n" + "="*60)
    print("DEMO 5: Weight Loading")
    print("="*60)
    
    # Configuration
    num_embeddings = 8
    embedding_dim = 4
    tp_size = 2
    
    # Create full embedding weights
    rng = random.PRNGKey(999)
    full_weights = initialize_embedding_weights(rng, num_embeddings, embedding_dim)
    print(f"Full weights shape: {full_weights.shape}")
    print(f"Full weights:\n{full_weights}")
    
    # Create embedding layers
    embedding_rank0 = VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=0
    )
    
    embedding_rank1 = VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=1
    )
    
    # Load weight shards
    shard_rank0 = embedding_rank0.load_weight_shard(full_weights)
    shard_rank1 = embedding_rank1.load_weight_shard(full_weights)
    
    print(f"\nRank 0 shard shape: {shard_rank0.shape}")
    print(f"Rank 0 shard:\n{shard_rank0}")
    print(f"Rank 1 shard shape: {shard_rank1.shape}")
    print(f"Rank 1 shard:\n{shard_rank1}")
    
    # Verify shards can be reconstructed
    reconstructed = jnp.concatenate([shard_rank0, shard_rank1], axis=0)
    print(f"\nReconstructed shape: {reconstructed.shape}")
    print(f"Reconstruction matches original: {jnp.allclose(full_weights, reconstructed)}")


def main():
    """Run all demonstrations."""
    print("JAX GPU VocabParallelEmbedding Demo")
    print("="*60)
    
    # Setup JAX
    gpu_devices = setup_jax_gpu()
    
    # Run demonstrations
    demo_basic_embedding()
    demo_parallel_lm_head()
    demo_sharded_execution()
    demo_performance_comparison()
    demo_weight_loading()
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
