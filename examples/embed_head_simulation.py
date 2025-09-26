"""
Simulation of VocabParallelEmbedding forward when tp_size=2.
This script fakes two tensor-parallel ranks locally to demonstrate the mask,
index adjustment, local embedding lookup, and the final all-reduce.

Run: python examples/embed_head_simulation.py
"""

import torch
import torch.nn.functional as F

# Config: vocab size and embedding dim
vocab_size = 8
emb_dim = 3
tp_size = 2

# Example input token ids (batch of 4)
# include ids from 0..7
x = torch.tensor([0, 1, 4, 6], dtype=torch.long)
print("input x:", x.tolist())

# Create a full embedding weight for reference (what a non-sharded embedding would use)
full_weight = torch.arange(vocab_size * emb_dim, dtype=torch.float32).reshape(
    vocab_size, emb_dim
)
print("full embedding weights:\n", full_weight)

# Reference: embedding result with full weight
y_ref = F.embedding(x, full_weight)
print("reference embedding (no sharding):\n", y_ref)

# Now simulate two tensor-parallel ranks with equal partition
shard_size = vocab_size // tp_size
print("shard_size:", shard_size)

# We'll simulate rank0 and rank1
results = []
for tp_rank in range(tp_size):
    vocab_start_idx = tp_rank * shard_size
    vocab_end_idx = vocab_start_idx + shard_size
    # local weight is the corresponding shard from full_weight
    local_weight = full_weight[vocab_start_idx:vocab_end_idx].clone()
    print(
        f"\n--- simulating tp_rank={tp_rank} shard {vocab_start_idx}-{vocab_end_idx - 1} ---"
    )
    print("local_weight:\n", local_weight)

    # Step 1: mask = (x >= start) & (x < end)
    mask = (x >= vocab_start_idx) & (x < vocab_end_idx)
    print("mask:", mask.tolist())

    # Step 2: x = mask * (x - vocab_start_idx)
    # Note: multiplication with mask turns out-of-range indices into 0.
    x_local = mask * (x - vocab_start_idx)
    print("x_local (after mask and shift):", x_local.tolist())

    # Step 3: local embedding lookup using local_weight
    y_local = F.embedding(x_local, local_weight)
    print("y_local (before zeroing out with mask unsqueeze):\n", y_local)

    # Step 4: zero out embeddings for tokens not belonging to this shard
    y_local = mask.unsqueeze(1) * y_local
    print("y_local (after applying mask unsqueeze):\n", y_local)

    results.append(y_local)

# Simulate all_reduce by summing results across ranks
y_sum = torch.stack(results, dim=0).sum(dim=0)
print("\nResult after simulated all_reduce (sum of per-rank outputs):\n", y_sum)

# Verify equality with reference
print(
    "\nAre summed sharded embeddings equal to reference?", torch.allclose(y_sum, y_ref)
)

# For clarity, print side-by-side
print("\nreference vs summed sharded:")
for i in range(x.size(0)):
    print(
        i, x[i].item(), "-> ref:", y_ref[i].tolist(), "sharded_sum:", y_sum[i].tolist()
    )
