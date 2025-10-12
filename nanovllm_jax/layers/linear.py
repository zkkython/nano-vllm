import jax
import jax.numpy as jnp
from typing import Optional, List
from flax import linen as nn


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    input_size: int
    output_size: int
    bias: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.weight = self.param(
            'weight',
            nn.initializers.normal(stddev=0.02),
            (self.output_size, self.input_size),
            self.dtype
        )
        if self.bias:
            self.bias_param = self.param(
                'bias',
                nn.initializers.zeros,
                (self.output_size,),
                self.dtype
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.dot(x, self.weight.T)
        if self.bias:
            y = y + self.bias_param
        return y


class ReplicatedLinear(LinearBase):
    """Linear layer with replicated weights across devices."""
    pass


class ColumnParallelLinear(LinearBase):
    """Column parallel linear layer for tensor parallelism."""
    tp_size: int = 1
    tp_rank: int = 0

    def setup(self):
        self.input_size_per_partition = self.input_size
        self.output_size_per_partition = divide(self.output_size, self.tp_size)
        
        self.weight = self.param(
            'weight',
            nn.initializers.normal(stddev=0.02),
            (self.output_size_per_partition, self.input_size),
            self.dtype
        )
        if self.bias:
            self.bias_param = self.param(
                'bias',
                nn.initializers.zeros,
                (self.output_size_per_partition,),
                self.dtype
            )


class MergedColumnParallelLinear(LinearBase):
    """Merged column parallel linear layer for gate and up projections."""
    output_sizes: List[int] = None
    tp_size: int = 1
    tp_rank: int = 0

    def setup(self):
        if self.output_sizes is None:
            raise ValueError("output_sizes must be provided")
        total_output_size = sum(self.output_sizes)
        self.output_size_per_partition = divide(total_output_size, self.tp_size)
        
        # Initialize weight
        self.weight = self.param(
            'weight',
            nn.initializers.normal(stddev=0.02),
            (self.output_size_per_partition, self.input_size),
            self.dtype
        )
        
        if self.bias:
            self.bias_param = self.param(
                'bias',
                nn.initializers.zeros,
                (self.output_size_per_partition,),
                self.dtype
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.dot(x, self.weight.T)
        if self.bias:
            y = y + self.bias_param
        return y


class QKVParallelLinear(LinearBase):
    """QKV parallel linear layer for attention."""
    head_size: int = 64
    total_num_heads: int = 8
    total_num_kv_heads: int = 8
    tp_size: int = 1
    tp_rank: int = 0

    def setup(self):
        self.num_heads = divide(self.total_num_heads, self.tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, self.tp_size)
        
        # Calculate output size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        
        # Initialize weight
        self.weight = self.param(
            'weight',
            nn.initializers.normal(stddev=0.02),
            (self.output_size_per_partition, self.input_size),
            self.dtype
        )
        
        if self.bias:
            self.bias_param = self.param(
                'bias',
                nn.initializers.zeros,
                (self.output_size_per_partition,),
                self.dtype
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.dot(x, self.weight.T)
        if self.bias:
            y = y + self.bias_param
        return y


class RowParallelLinear(LinearBase):
    """Row parallel linear layer for tensor parallelism."""
    tp_size: int = 1
    tp_rank: int = 0

    def setup(self):
        self.input_size_per_partition = divide(self.input_size, self.tp_size)
        self.output_size_per_partition = self.output_size
        
        self.weight = self.param(
            'weight',
            nn.initializers.normal(stddev=0.02),
            (self.output_size, self.input_size_per_partition),
            self.dtype
        )
        if self.bias:
            self.bias_param = self.param(
                'bias',
                nn.initializers.zeros,
                (self.output_size,),
                self.dtype
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.dot(x, self.weight.T)
        if self.bias and self.tp_rank == 0:
            y = y + self.bias_param
        
        # All-reduce across tensor parallel ranks
        if self.tp_size > 1:
            y = jax.lax.psum(y, 'tp')
        
        return y
