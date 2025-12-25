import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import numpy as np


def create_3d_mesh_dp_tp_ep() -> Mesh:
    """Create a 3D mesh with axes ('dp', 'tp', 'ep') for MoE expert parallel.

    Target shape on 8 devices: (dp=2, tp=2, ep=2).
    - 'dp': data parallel axis (shards batch dimension)
    - 'tp': tensor parallel axis (shards model dimensions like hidden/ff)
    - 'ep': expert parallel axis (shards experts in MoE)

    If the device count does not match 2*2*2, we fall back to a best-effort
    factorization where ep=2 if possible, otherwise ep=1.
    """
    devices = np.array(jax.devices())
    num_devices = devices.size

    if num_devices >= 8 and num_devices % 8 == 0:
        dp = 2
        tp = 2
        ep = num_devices // (dp * tp)
    elif num_devices % 4 == 0:
        # Try dp=2, tp=1, ep=num_devices/2  or similar
        dp = 2
        tp = 1
        ep = num_devices // (dp * tp)
    else:
        # Fallback: no dp/tp, only ep or single axis
        dp = 1
        tp = 1
        ep = num_devices

    mesh_shape = (dp, tp, ep)
    mesh_devices = devices.reshape(mesh_shape)
    return Mesh(mesh_devices, axis_names=("dp", "tp", "ep"))


def demo_moe_expert_parallel():
    """Demonstrate MoE expert-parallel sharding on a (dp, tp, ep) mesh.

    We simulate a simple MoE layer with:
    - Input activations X: [batch, hidden]
    - Experts weights W_expert: [num_experts, hidden, ff_dim]

    Sharding on mesh ('dp', 'tp', 'ep'):
    - X: PartitionSpec('dp', None)
        → batch is sharded over data-parallel axis, hidden replicated
    - W_expert: PartitionSpec('ep', None, 'tp')
        → experts dimension sharded over 'ep' (expert parallel)
        → ff_dim sharded over 'tp' (tensor parallel inside each expert)
    """
    mesh = create_3d_mesh_dp_tp_ep()

    batch = 8
    hidden = 16
    ff_dim = 32

    # num_experts must be divisible by ep to shard cleanly
    num_experts_total = 4
    assert (
        num_experts_total % mesh.shape["ep"] == 0
    ), f"num_experts ({num_experts_total}) must be divisible by ep ({mesh.shape['ep']})"

    x = jnp.arange(batch * hidden, dtype=jnp.float32).reshape(batch, hidden)
    w_expert = jnp.arange(
        num_experts_total * hidden * ff_dim, dtype=jnp.float32
    ).reshape(num_experts_total, hidden, ff_dim)

    print("\n========================================")
    print("3D Mesh configuration (dp, tp, ep)")
    print("========================================")
    print(f"Mesh axis names: {mesh.axis_names}")
    print(
        f"Mesh shape: dp={mesh.shape['dp']}, tp={mesh.shape['tp']}, ep={mesh.shape['ep']}"
    )

    # 1) Shard activations over 'dp'
    print("\n========================================")
    print("Activations: data-parallel over 'dp' (PartitionSpec('dp', None))")
    print("========================================")
    act_sharding = NamedSharding(mesh, PartitionSpec("dp", None))
    x_sharded = jax.device_put(x, act_sharding)
    print(f"X shape: {x.shape}")
    print(f"X sharding spec: {x_sharded.sharding.spec}")
    print(f"Number of shards (devices): {len(x_sharded.addressable_shards)}")

    shard_indices = [shard.index for shard in x_sharded.addressable_shards]
    distinct_batch_slices = sorted({idx[0] for idx in shard_indices})
    print("Distinct batch index slices across shards:")
    for s in distinct_batch_slices:
        print(f"  batch slice: {s}")

    # 2) Shard experts over 'ep' and ff_dim over 'tp'
    print("\n========================================")
    print("Experts weights: expert-parallel + tensor-parallel")
    print("  PartitionSpec('ep', None, 'tp')")
    print("========================================")
    expert_sharding = NamedSharding(mesh, PartitionSpec("ep", None, "tp"))
    w_expert_sharded = jax.device_put(w_expert, expert_sharding)
    print(f"W_expert shape: {w_expert.shape}")
    print(f"W_expert sharding spec: {w_expert_sharded.sharding.spec}")
    print(f"Number of shards (devices): {len(w_expert_sharded.addressable_shards)}")
    for i, shard in enumerate(w_expert_sharded.addressable_shards):
        print(f"Shard {i}: index={shard.index}, shape={shard.data.shape}")

    # 3) Simulate a router selecting experts (top-1) for each token
    print("\n========================================")
    print("Router example: tokens choose experts (not fully distributed logic)")
    print("========================================")
    # Simple deterministic router: token i -> expert (i % num_experts_total)
    token_ids = jnp.arange(batch)
    expert_ids = token_ids % num_experts_total
    print(f"Token ids:   {token_ids}")
    print(f"Expert ids:  {expert_ids} (token i -> expert i % {num_experts_total})")

    # 4) Expert-parallel compute: each expert processes all tokens locally
    #    This is not sparse routing yet, but shows EP sharding in compute.
    print("\n========================================")
    print("Expert-parallel compute: vmap over experts")
    print("========================================")

    def expert_apply(w_e: jnp.ndarray, x_in: jnp.ndarray) -> jnp.ndarray:
        """Apply one expert: [hidden, ff_dim] with input [batch, hidden]."""
        return x_in @ w_e  # [B, H] @ [H, F] -> [B, F]

    # vmap over expert dimension 0; w_expert_sharded is sharded over 'ep' and 'tp'
    # x_sharded is sharded over 'dp'
    moe_outputs = jax.vmap(expert_apply, in_axes=(0, None))(w_expert_sharded, x_sharded)
    print(f"MoE outputs shape (local shard): {moe_outputs.shape}")
    print(f"MoE outputs sharding: {moe_outputs.sharding.spec}")

    print(
        '\nNote: In a real MoE, you\'d use the router to "scatter" tokens to experts\n'
        'and then "gather" outputs back, often with all-to-all communication.\n'
        "Here we focus on showing how experts weights are sharded over the 'ep' axis\n"
        "and combined with 'dp' and 'tp' in a (dp, tp, ep) mesh."
    )


if __name__ == "__main__":
    demo_moe_expert_parallel()
