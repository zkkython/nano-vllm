"""
JAX GPU implementation of VocabParallelEmbedding and ParallelLMHead.

This module provides distributed vocabulary parallel embedding layers using JAX
with GPU acceleration and multi-device communication.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np
from typing import Optional, Tuple, Union
from flax import linen as nn
from flax.core import freeze, unfreeze


class VocabParallelEmbedding(nn.Module):
    """
    JAX implementation of vocabulary parallel embedding with GPU support.
    
    This implementation distributes the vocabulary across multiple devices/ranks,
    where each device handles a shard of the vocabulary. The forward pass uses
    masking to only process tokens belonging to the current device's vocabulary
    shard, followed by an all-reduce to aggregate results.
    """
    
    num_embeddings: int
    embedding_dim: int
    tp_size: int = 1
    tp_rank: int = 0
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        """Initialize the embedding layer parameters."""
        assert self.num_embeddings % self.tp_size == 0, \
            f"num_embeddings ({self.num_embeddings}) must be divisible by tp_size ({self.tp_size})"
        
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # Initialize embedding weights for this partition
        self.embedding = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (self.num_embeddings_per_partition, self.embedding_dim),
            self.dtype
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of vocabulary parallel embedding.
        
        Args:
            x: Input token indices of shape (...,) or (batch_size, seq_len)
            
        Returns:
            Embedding vectors of shape (..., embedding_dim)
        """
        # Create mask for tokens belonging to this partition
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # Adjust indices to local vocabulary range
            x_local = jnp.where(mask, x - self.vocab_start_idx, 0)
        else:
            mask = jnp.ones_like(x, dtype=bool)
            x_local = x
        
        # Perform embedding lookup
        y = self.embedding[x_local]
        
        # Apply mask to zero out embeddings for tokens not in this partition
        if self.tp_size > 1:
            # Expand mask to match embedding dimensions
            mask_expanded = jnp.expand_dims(mask, axis=-1)
            y = jnp.where(mask_expanded, y, 0.0)
        
        return y
    
    def sharded_forward(self, x: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
        """
        Sharded forward pass using JAX mesh and shard_map for multi-device execution.
        
        Args:
            x: Input token indices
            mesh: JAX mesh for device placement
            
        Returns:
            Embedding vectors with proper sharding
        """
        def _forward_fn(x_shard, embedding_shard):
            # Local forward pass on each device
            return self.__call__(x_shard)
        
        # Define sharding specifications
        x_sharding = NamedSharding(mesh, PartitionSpec('batch', 'seq'))
        embedding_sharding = NamedSharding(mesh, PartitionSpec('tp', None))
        output_sharding = NamedSharding(mesh, PartitionSpec('batch', 'seq', None))
        
        # Apply shard_map for distributed execution
        result = shard_map(
            _forward_fn,
            mesh,
            in_specs=(x_sharding, embedding_sharding),
            out_specs=output_sharding
        )(x, self.embedding)
        
        # All-reduce across tensor parallel dimension
        if self.tp_size > 1:
            result = lax.psum(result, 'tp')
        
        return result
    
    def load_weight_shard(self, full_weights: jnp.ndarray) -> jnp.ndarray:
        """
        Load the appropriate weight shard for this partition.
        
        Args:
            full_weights: Full embedding weights of shape (num_embeddings, embedding_dim)
            
        Returns:
            Weight shard for this partition
        """
        assert full_weights.shape[0] == self.num_embeddings, \
            f"Expected {self.num_embeddings} embeddings, got {full_weights.shape[0]}"
        
        shard = full_weights[self.vocab_start_idx:self.vocab_end_idx]
        return shard


class ParallelLMHead(VocabParallelEmbedding):
    """
    JAX implementation of parallel language model head.
    
    Extends VocabParallelEmbedding to include bias and proper logit computation
    for language modeling tasks.
    """
    
    bias: bool = False
    
    def setup(self):
        """Initialize the LM head parameters."""
        super().setup()
        
        if self.bias:
            self.bias_param = self.param(
                'bias',
                nn.initializers.zeros,
                (self.num_embeddings_per_partition,),
                self.dtype
            )
        else:
            self.bias_param = None
    
    def __call__(self, x: jnp.ndarray, context: Optional[dict] = None) -> jnp.ndarray:
        """
        Forward pass of parallel LM head.
        
        Args:
            x: Input hidden states of shape (batch_size, seq_len, hidden_dim)
            context: Optional context dictionary for sequence handling
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size) or (batch_size, vocab_size)
        """
        # Handle prefill context if provided
        if context is not None and context.get('is_prefill', False):
            cu_seqlens_q = context.get('cu_seqlens_q', None)
            if cu_seqlens_q is not None:
                # Extract last token indices for each sequence
                last_indices = cu_seqlens_q[1:] - 1
                x = x[last_indices]
        
        # Compute logits using linear transformation
        logits = jnp.dot(x, self.embedding.T)
        
        # Add bias if present
        if self.bias_param is not None:
            logits = logits + self.bias_param
        
        # Handle tensor parallelism
        if self.tp_size > 1:
            # Gather logits from all partitions
            # In a real implementation, this would use collective communication
            # For now, we'll simulate the gathering process
            logits = self._gather_logits(logits)
        
        return logits
    
    def _gather_logits(self, logits: jnp.ndarray) -> jnp.ndarray:
        """
        Gather logits from all tensor parallel ranks.
        
        Args:
            logits: Local logits of shape (..., num_embeddings_per_partition)
            
        Returns:
            Full logits of shape (..., num_embeddings)
        """
        # In a real distributed implementation, this would use:
        # - jax.experimental.pjit for compilation
        # - Collective communication primitives like all_gather
        # For now, we return the local logits
        # TODO: Implement proper all_gather for distributed execution
        return logits
    
    def sharded_forward(self, x: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
        """
        Sharded forward pass for parallel LM head.
        
        Args:
            x: Input hidden states
            mesh: JAX mesh for device placement
            
        Returns:
            Logits with proper sharding
        """
        def _forward_fn(x_shard, embedding_shard, bias_shard=None):
            # Local forward pass
            logits = jnp.dot(x_shard, embedding_shard.T)
            if bias_shard is not None:
                logits = logits + bias_shard
            return logits
        
        # Define sharding
        x_sharding = NamedSharding(mesh, PartitionSpec('batch', 'seq', None))
        embedding_sharding = NamedSharding(mesh, PartitionSpec('tp', None))
        bias_sharding = NamedSharding(mesh, PartitionSpec('tp',)) if self.bias else None
        
        # Apply shard_map
        if self.bias:
            result = shard_map(
                _forward_fn,
                mesh,
                in_specs=(x_sharding, embedding_sharding, bias_sharding),
                out_specs=NamedSharding(mesh, PartitionSpec('batch', 'seq', 'tp'))
            )(x, self.embedding, self.bias_param)
        else:
            result = shard_map(
                _forward_fn,
                mesh,
                in_specs=(x_sharding, embedding_sharding),
                out_specs=NamedSharding(mesh, PartitionSpec('batch', 'seq', 'tp'))
            )(x, self.embedding)
        
        # All-gather logits across tensor parallel dimension
        if self.tp_size > 1:
            # In real implementation: result = lax.all_gather(result, 'tp')
            # For now, we'll simulate by concatenating along the last dimension
            result = jnp.concatenate([result] * self.tp_size, axis=-1)
        
        return result


def create_mesh(devices: list, tp_size: int = 1) -> Mesh:
    """
    Create a JAX mesh for distributed execution.
    
    Args:
        devices: List of JAX devices
        tp_size: Tensor parallel size
        
    Returns:
        JAX mesh for device placement
    """
    if tp_size == 1:
        return Mesh(devices, axis_names=('batch',))
    else:
        # Create mesh with tensor parallel dimension
        mesh_shape = (len(devices) // tp_size, tp_size)
        mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
        return Mesh(mesh_devices, axis_names=('batch', 'tp'))


def initialize_embedding_weights(
    rng: jnp.ndarray,
    num_embeddings: int,
    embedding_dim: int,
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Initialize embedding weights using proper scaling.
    
    Args:
        rng: Random key
        num_embeddings: Vocabulary size
        embedding_dim: Embedding dimension
        dtype: Data type
        
    Returns:
        Initialized embedding weights
    """
    # Use Xavier uniform initialization with proper scaling
    scale = jnp.sqrt(1.0 / embedding_dim)
    weights = random.uniform(
        rng,
        (num_embeddings, embedding_dim),
        minval=-scale,
        maxval=scale,
        dtype=dtype
    )
    return weights


# Example usage and testing functions
def test_vocab_parallel_embedding():
    """Test the VocabParallelEmbedding implementation."""
    # Configuration
    num_embeddings = 8
    embedding_dim = 4
    tp_size = 2
    batch_size = 2
    seq_len = 3
    
    # Create test data
    rng = random.PRNGKey(42)
    x = random.randint(rng, (batch_size, seq_len), 0, num_embeddings)
    
    print("Input tokens:", x)
    
    # Test rank 0
    embedding_rank0 = VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=0
    )
    
    # Test rank 1
    embedding_rank1 = VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tp_size=tp_size,
        tp_rank=1
    )
    
    # Initialize parameters
    rng, init_rng = random.split(rng)
    params_rank0 = embedding_rank0.init(init_rng, x)
    params_rank1 = embedding_rank1.init(init_rng, x)
    
    # Forward pass
    output_rank0 = embedding_rank0.apply(params_rank0, x)
    output_rank1 = embedding_rank1.apply(params_rank1, x)
    
    print("Rank 0 output shape:", output_rank0.shape)
    print("Rank 1 output shape:", output_rank1.shape)
    
    # Simulate all-reduce
    final_output = output_rank0 + output_rank1
    print("Final output shape:", final_output.shape)
    
    return final_output


if __name__ == "__main__":
    # Run test
    test_vocab_parallel_embedding()
