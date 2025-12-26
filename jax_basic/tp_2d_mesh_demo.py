import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils


def create_2d_mesh() -> Mesh:
    """Create a 2D mesh with axes ('dp', 'tp').

    - 'dp': data parallel axis (typically shards the batch dimension)
    - 'tp': tensor parallel axis (typically shards model dimensions like hidden/intermediate)

    The function tries to factor the available devices into (dp, tp) such that
    dp * tp == num_devices, preferring larger tp when possible, which is common
    for large-model tensor parallel inference.
    """
    devices = jax.devices()
    num_devices = len(devices)

    if num_devices < 4:
        mesh_shape = (num_devices, 1)
    else:
        tp = 2
        while tp * 2 <= num_devices and num_devices % (tp * 2) == 0:
            tp *= 2
        dp = num_devices // tp
        mesh_shape = (dp, tp)

    mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(mesh_devices, axis_names=("dp", "tp"))


def demo_llm_inference_sharding():
    """Demonstrate typical 2D mesh sharding patterns for LLM inference.

    This covers a simplified block with:
    - Input activations X [batch, hidden]
    - Column-parallel projections (e.g. q_proj, k_proj, v_proj, gate_proj, up_proj)
    - Row-parallel projections (e.g. o_proj, down_proj)

    Sharding patterns on a 2D mesh ('dp', 'tp'):
    - Activations: PartitionSpec('dp', None)
      batch dimension sharded over data-parallel axis, hidden replicated
    - Column-parallel weights: PartitionSpec(None, 'tp')
      output/features sharded over tensor-parallel axis, input replicated
    - Row-parallel weights: PartitionSpec('tp', None)
      input/features sharded over tensor-parallel axis, output replicated
    """
    mesh = create_2d_mesh()

    batch = 8
    hidden = 16
    intermediate = 32

    x = jnp.arange(batch * hidden, dtype=jnp.float32).reshape(batch, hidden)
    w_col = jnp.arange(hidden * intermediate, dtype=jnp.float32).reshape(
        hidden, intermediate
    )
    w_row = jnp.arange(intermediate * hidden, dtype=jnp.float32).reshape(
        intermediate, hidden
    )

    print("\n========================================")
    print("2D Mesh configuration")
    print("========================================")
    print(f"Mesh axis names: {mesh.axis_names}")
    print(f"Mesh shape: dp={mesh.shape['dp']}, tp={mesh.shape['tp']}")

    print("\n========================================")
    print("Activations: data-parallel sharding (PartitionSpec('dp', None))")
    print("========================================")
    act_sharding = NamedSharding(mesh, PartitionSpec("dp", None))
    x_sharded = jax.device_put(x, act_sharding)
    print(f"X shape: {x.shape}")
    print(f"X sharding spec: {x_sharded.sharding.spec}")
    print(f"Number of shards: {len(x_sharded.addressable_shards)}")
    for i, shard in enumerate(x_sharded.addressable_shards):
        print(f"Shard {i}: index={shard.index}, shape={shard.data.shape}")

    print("\n========================================")
    print("Column-parallel weight: PartitionSpec(None, 'tp')")
    print("========================================")
    w_col_sharding = NamedSharding(mesh, PartitionSpec(None, "tp"))
    w_col_sharded = jax.device_put(w_col, w_col_sharding)
    print(f"W_col shape: {w_col.shape}")
    print(f"W_col sharding spec: {w_col_sharded.sharding.spec}")
    print(f"Number of shards: {len(w_col_sharded.addressable_shards)}")
    for i, shard in enumerate(w_col_sharded.addressable_shards):
        print(f"Shard {i}: index={shard.index}, shape={shard.data.shape}")

    print("\n========================================")
    print("Row-parallel weight: PartitionSpec('tp', None)")
    print("========================================")
    w_row_sharding = NamedSharding(mesh, PartitionSpec("tp", None))
    w_row_sharded = jax.device_put(w_row, w_row_sharding)
    print(f"W_row shape: {w_row.shape}")
    print(f"W_row sharding spec: {w_row_sharded.sharding.spec}")
    print(f"Number of shards: {len(w_row_sharded.addressable_shards)}")
    for i, shard in enumerate(w_row_sharded.addressable_shards):
        print(f"Shard {i}: index={shard.index}, shape={shard.data.shape}")

    print("\n========================================")
    print("Combined pattern: data-parallel activations + tensor-parallel weights")
    print("========================================")
    col_out = jnp.matmul(x_sharded, w_col_sharded)
    print(f"Column-parallel matmul output shape: {col_out.shape}")
    print(f"Column-parallel output sharding: {col_out.sharding.spec}")

    row_in = col_out
    w_row_in_sharding = w_row_sharding
    w_row_in = w_row_sharded
    row_out = jnp.matmul(row_in, w_row_in)
    print(f"Row-parallel matmul partial output shape: {row_out.shape}")
    print(f"Row-parallel partial sharding: {row_out.sharding.spec}")


def create_2d_mesh_dp_tp_fixed() -> Mesh:
    """Create a 2D mesh that uses dp > 1 when possible.

    For example, on 8 devices this will create a mesh with shape (dp=2, tp=4),
    so that:
    - 'dp' axis shards the batch dimension into 2 parts
    - 'tp' axis provides 4-way tensor parallelism per data-parallel shard
    """
    devices = jax.devices()
    num_devices = len(devices)

    if num_devices >= 4 and num_devices % 2 == 0:
        dp = 2
        tp = num_devices // dp
        mesh_shape = (dp, tp)
    else:
        mesh_shape = (num_devices, 1)

    mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(mesh_devices, axis_names=("dp", "tp"))


def demo_partition_none_dp():
    mesh = create_2d_mesh_dp_tp_fixed()
    arr = jnp.arange(4 * 8, dtype=jnp.bfloat16).reshape(4, 8)

    arr_sharding = NamedSharding(mesh, PartitionSpec(0, "dp"))
    arr_sharded = jax.device_put(arr, arr_sharding)

    for i, shard in enumerate(arr_sharded.addressable_shards):
        print(f"Device {i}, index: {shard.index},  data shape = {shard.data.shape}")
        print(f"\n data = {shard.data}")


def demo_llm_inference_sharding_dp_tp():
    """Demonstrate 2D mesh sharding with dp > 1 explicitly.

    This shows how:
    - batch is split across 'dp'
    - model dimensions are split across 'tp'
    """
    mesh = create_2d_mesh_dp_tp_fixed()

    batch = 8
    hidden = 16
    intermediate = 32

    x = jnp.arange(batch * hidden, dtype=jnp.float32).reshape(batch, hidden)
    w_col = jnp.arange(hidden * intermediate, dtype=jnp.float32).reshape(
        hidden, intermediate
    )
    w_row = jnp.arange(intermediate * hidden, dtype=jnp.float32).reshape(
        intermediate, hidden
    )

    print("\n========================================")
    print("2D Mesh configuration (dp > 1)")
    print("========================================")
    print(f"Mesh axis names: {mesh.axis_names}")
    print(f"Mesh shape: dp={mesh.shape['dp']}, tp={mesh.shape['tp']}")

    print("\n========================================")
    print("Activations: data-parallel sharding (PartitionSpec('dp', None))")
    print("========================================")
    act_sharding = NamedSharding(mesh, PartitionSpec("dp", None))
    x_sharded = jax.device_put(x, act_sharding)
    print(f"X shape: {x.shape}")
    print(f"X sharding spec: {x_sharded.sharding.spec}")
    print(f"Number of shards (devices): {len(x_sharded.addressable_shards)}")

    shard_indices = [shard.index for shard in x_sharded.addressable_shards]
    print("Distinct batch index slices across shards:")
    distinct_batch_slices = sorted({idx[0] for idx in shard_indices})
    for s in distinct_batch_slices:
        print(f"  batch slice: {s}")

    for i, shard in enumerate(x_sharded.addressable_shards):
        print(f"Shard {i}: index={shard.index}, shape={shard.data.shape}")

    print("\n========================================")
    print("Column-parallel weight: PartitionSpec(None, 'tp')")
    print("========================================")
    w_col_sharding = NamedSharding(mesh, PartitionSpec(None, "tp"))
    w_col_sharded = jax.device_put(w_col, w_col_sharding)
    print(f"W_col shape: {w_col.shape}")
    print(f"W_col sharding spec: {w_col_sharded.sharding.spec}")
    print(f"Number of shards (devices): {len(w_col_sharded.addressable_shards)}")
    for i, shard in enumerate(w_col_sharded.addressable_shards):
        print(f"Shard {i}: index={shard.index}, shape={shard.data.shape}")

    print("\n========================================")
    print("Row-parallel weight: PartitionSpec('tp', None)")
    print("========================================")
    w_row_sharding = NamedSharding(mesh, PartitionSpec("tp", None))
    w_row_sharded = jax.device_put(w_row, w_row_sharding)
    print(f"W_row shape: {w_row.shape}")
    print(f"W_row sharding spec: {w_row_sharded.sharding.spec}")
    print(f"Number of shards (devices): {len(w_row_sharded.addressable_shards)}")
    for i, shard in enumerate(w_row_sharded.addressable_shards):
        print(f"Shard {i}: index={shard.index}, shape={shard.data.shape}")

    print("\n========================================")
    print("Combined pattern: dp-sharded activations + tp weights")
    print("========================================")
    col_out = jnp.matmul(x_sharded, w_col_sharded)
    print(f"Column-parallel matmul output shape: {col_out.shape}")
    print(f"Column-parallel output sharding: {col_out.sharding.spec}")

    row_in = col_out
    row_out = jnp.matmul(row_in, w_row_sharded)
    print(f"Row-parallel matmul partial output shape: {row_out.shape}")
    print(f"Row-parallel partial sharding: {row_out.sharding.spec}")


if __name__ == "__main__":
    print("\n=== demo_llm_inference_sharding (auto factorization) ===")
    demo_llm_inference_sharding()
    print("\n=== demo_llm_inference_sharding_dp_tp (dp > 1) ===")
    demo_llm_inference_sharding_dp_tp()
    print("\n special demo of partition spec: (None, 'dp')")
    demo_partition_none_dp()
