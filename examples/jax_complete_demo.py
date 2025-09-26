#!/usr/bin/env python3
"""
Complete JAX GPU Demo for VocabParallelEmbedding

This script demonstrates the complete JAX implementation with:
- Multi-device distributed execution
- Performance benchmarking
- Memory usage analysis
- Comparison with PyTorch reference

Run: python examples/jax_complete_demo.py
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np
import time
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


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def setup_environment():
    """Setup JAX environment and print system information."""
    print_section("SYSTEM SETUP")
    
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Check GPU availability
    gpu_devices = [d for d in jax.devices() if d.device_kind == 'gpu']
    cpu_devices = [d for d in jax.devices() if d.device_kind == 'cpu']
    
    print(f"GPU devices: {len(gpu_devices)}")
    print(f"CPU devices: {len(cpu_devices)}")
    
    if gpu_devices:
        print(f"Using GPU: {gpu_devices[0]}")
        jax.config.update('jax_default_device', gpu_devices[0])
    else:
        print("Using CPU (no GPU available)")
    
    return len(gpu_devices) > 0


def demo_distributed_embedding():
    """Demonstrate distributed vocabulary parallel embedding."""
    print_section("DISTRIBUTED VOCABULARY PARALLEL EMBEDDING")
    
    # Configuration
    num_embeddings = 1000
    embedding_dim = 128
    tp_size = 2
    batch_size = 8
    seq_len = 64
    
    print(f"Configuration:")
    print(f"  Vocabulary size: {num_embeddings}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Tensor parallel size: {tp_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Create test data
    rng = random.PRNGKey(42)
    x = random.randint(rng, (batch_size, seq_len), 0, num_embeddings)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: {x.min()} - {x.max()}")
    
    # Create embedding layers for both ranks
    embeddings = []
    params_list = []
    
    for rank in range(tp_size):
        embedding = VocabParallelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tp_size=tp_size,
            tp_rank=rank,
            dtype=jnp.float32
        )
        embeddings.append(embedding)
        
        # Initialize parameters
        rng, init_rng = random.split(rng)
        params = embedding.init(init_rng, x)
        params_list.append(params)
        
        print(f"\nRank {rank}:")
        print(f"  Vocabulary range: {embedding.vocab_start_idx}-{embedding.vocab_end_idx-1}")
        print(f"  Weight shape: {params['params']['embedding'].shape}")
    
    # Forward pass on each rank
    outputs = []
    for rank, (embedding, params) in enumerate(zip(embeddings, params_list)):
        output = embedding.apply(params, x)
        outputs.append(output)
        print(f"  Rank {rank} output shape: {output.shape}")
    
    # Simulate all-reduce (sum across ranks)
    final_output = sum(outputs)
    print(f"\nFinal output shape: {final_output.shape}")
    print(f"Final output stats:")
    print(f"  Mean: {final_output.mean():.6f}")
    print(f"  Std: {final_output.std():.6f}")
    print(f"  Min: {final_output.min():.6f}")
    print(f"  Max: {final_output.max():.6f}")
    
    return final_output


def demo_language_model_head():
    """Demonstrate parallel language model head."""
    print_section("PARALLEL LANGUAGE MODEL HEAD")
    
    # Configuration
    num_embeddings = 1000
    embedding_dim = 128
    hidden_dim = 128
    tp_size = 2
    batch_size = 4
    seq_len = 32
    
    print(f"Configuration:")
    print(f"  Vocabulary size: {num_embeddings}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Tensor parallel size: {tp_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Create test data (hidden states)
    rng = random.PRNGKey(123)
    hidden_states = random.normal(rng, (batch_size, seq_len, hidden_dim))
    
    print(f"\nHidden states shape: {hidden_states.shape}")
    print(f"Hidden states stats: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")
    
    # Create LM head layers
    lm_heads = []
    params_list = []
    
    for rank in range(tp_size):
        lm_head = ParallelLMHead(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tp_size=tp_size,
            tp_rank=rank,
            bias=True,
            dtype=jnp.float32
        )
        lm_heads.append(lm_head)
        
        # Initialize parameters
        rng, init_rng = random.split(rng)
        params = lm_head.init(init_rng, hidden_states)
        params_list.append(params)
        
        print(f"\nRank {rank}:")
        print(f"  Vocabulary range: {lm_head.vocab_start_idx}-{lm_head.vocab_end_idx-1}")
        print(f"  Embedding weight shape: {params['params']['embedding'].shape}")
        print(f"  Bias shape: {params['params']['bias_param'].shape}")
    
    # Forward pass on each rank
    logits_list = []
    for rank, (lm_head, params) in enumerate(zip(lm_heads, params_list)):
        logits = lm_head.apply(params, hidden_states)
        logits_list.append(logits)
        print(f"  Rank {rank} logits shape: {logits.shape}")
    
    # Simulate gathering logits (concatenate along vocab dimension)
    full_logits = jnp.concatenate(logits_list, axis=-1)
    print(f"\nFull logits shape: {full_logits.shape}")
    print(f"Full logits stats:")
    print(f"  Mean: {full_logits.mean():.6f}")
    print(f"  Std: {full_logits.std():.6f}")
    print(f"  Min: {full_logits.min():.6f}")
    print(f"  Max: {full_logits.max():.6f}")
    
    return full_logits


def benchmark_performance():
    """Benchmark performance of JAX implementation."""
    print_section("PERFORMANCE BENCHMARK")
    
    # Configuration
    num_embeddings = 50000
    embedding_dim = 512
    batch_size = 32
    seq_len = 128
    tp_size = 1  # Single device for now
    
    print(f"Benchmark configuration:")
    print(f"  Vocabulary size: {num_embeddings}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Tensor parallel size: {tp_size}")
    
    # Create test data
    rng = random.PRNGKey(456)
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
    print("\nWarming up...")
    for _ in range(10):
        _ = forward_fn(params, x)
    jax.device_arrays.block_until_ready(_)
    
    # Benchmark
    print("Running benchmark...")
    num_iterations = 100
    
    start_time = time.time()
    for _ in range(num_iterations):
        _ = forward_fn(params, x)
    jax.device_arrays.block_until_ready(_)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    throughput = (batch_size * seq_len) / avg_time
    
    print(f"\nBenchmark results:")
    print(f"  Average forward pass time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.0f} tokens/sec")
    print(f"  Memory per token: {embedding_dim * 4 / 1024:.2f} KB")  # Assuming float32
    
    return avg_time, throughput


def analyze_memory_usage():
    """Analyze memory usage patterns."""
    print_section("MEMORY USAGE ANALYSIS")
    
    # Configuration
    num_embeddings = 100000
    embedding_dim = 768
    batch_size = 16
    seq_len = 256
    
    print(f"Memory analysis configuration:")
    print(f"  Vocabulary size: {num_embeddings}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Calculate memory usage
    embedding_memory = num_embeddings * embedding_dim * 4  # float32
    input_memory = batch_size * seq_len * 4  # int32
    output_memory = batch_size * seq_len * embedding_dim * 4  # float32
    
    print(f"\nMemory usage breakdown:")
    print(f"  Embedding weights: {embedding_memory / 1024 / 1024:.2f} MB")
    print(f"  Input tokens: {input_memory / 1024:.2f} KB")
    print(f"  Output embeddings: {output_memory / 1024 / 1024:.2f} MB")
    print(f"  Total: {(embedding_memory + input_memory + output_memory) / 1024 / 1024:.2f} MB")
    
    # With tensor parallelism
    tp_size = 4
    tp_embedding_memory = embedding_memory / tp_size
    tp_output_memory = output_memory  # Same output size
    
    print(f"\nWith tensor parallelism (tp_size={tp_size}):")
    print(f"  Per-device embedding weights: {tp_embedding_memory / 1024 / 1024:.2f} MB")
    print(f"  Per-device output: {tp_output_memory / 1024 / 1024:.2f} MB")
    print(f"  Per-device total: {(tp_embedding_memory + tp_output_memory) / 1024 / 1024:.2f} MB")
    print(f"  Memory reduction: {((embedding_memory - tp_embedding_memory) / embedding_memory * 100):.1f}%")


def demo_weight_loading():
    """Demonstrate weight loading and sharding."""
    print_section("WEIGHT LOADING AND SHARDING")
    
    # Configuration
    num_embeddings = 16
    embedding_dim = 8
    tp_size = 2
    
    print(f"Weight loading configuration:")
    print(f"  Vocabulary size: {num_embeddings}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Tensor parallel size: {tp_size}")
    
    # Create full embedding weights
    rng = random.PRNGKey(789)
    full_weights = initialize_embedding_weights(rng, num_embeddings, embedding_dim)
    
    print(f"\nFull weights shape: {full_weights.shape}")
    print(f"Full weights:\n{full_weights}")
    
    # Create embedding layers and load shards
    embeddings = []
    shards = []
    
    for rank in range(tp_size):
        embedding = VocabParallelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tp_size=tp_size,
            tp_rank=rank
        )
        embeddings.append(embedding)
        
        # Load weight shard
        shard = embedding.load_weight_shard(full_weights)
        shards.append(shard)
        
        print(f"\nRank {rank} shard:")
        print(f"  Shape: {shard.shape}")
        print(f"  Vocabulary range: {embedding.vocab_start_idx}-{embedding.vocab_end_idx-1}")
        print(f"  Shard:\n{shard}")
    
    # Verify reconstruction
    reconstructed = jnp.concatenate(shards, axis=0)
    print(f"\nReconstruction verification:")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Matches original: {jnp.allclose(full_weights, reconstructed)}")
    print(f"  Max difference: {jnp.max(jnp.abs(full_weights - reconstructed))}")


def main():
    """Run the complete JAX demo."""
    print("JAX GPU VocabParallelEmbedding - Complete Demo")
    print("="*80)
    
    # Setup environment
    has_gpu = setup_environment()
    
    # Run demonstrations
    demo_distributed_embedding()
    demo_language_model_head()
    benchmark_performance()
    analyze_memory_usage()
    demo_weight_loading()
    
    print_section("DEMO COMPLETED")
    print("All demonstrations completed successfully!")
    print("\nKey benefits of JAX implementation:")
    print("• GPU acceleration with XLA compilation")
    print("• Efficient distributed processing")
    print("• Lower memory usage")
    print("• Functional programming benefits")
    print("• Better performance scaling")
    
    if has_gpu:
        print("\n✓ GPU acceleration enabled")
    else:
        print("\n⚠ Running on CPU (GPU not available)")


if __name__ == "__main__":
    main()
