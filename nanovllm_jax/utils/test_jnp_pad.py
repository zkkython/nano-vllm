import jax
from jax import numpy as jnp
arr = jnp.arange(0, 10)
print(arr)

arr = jnp.reshape(arr, (2, 5))
print(arr)

arr2 = jnp.pad(arr, ((0, 0), (0, 5)))
print(arr2)

arr3 = jnp.pad(arr, ((0, 1), (0,5)))
print(arr3)