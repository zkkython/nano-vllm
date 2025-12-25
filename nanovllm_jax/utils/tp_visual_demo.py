"""
Visual demonstration of tensor parallelism sharding patterns.
Run this to see ASCII diagrams of how tensors are split across devices.
"""


def print_sharding_visualization():
    """Print visual diagrams of different sharding patterns."""

    print("=" * 80)
    print(" 📊 TENSOR PARALLELISM SHARDING VISUALIZATION")
    print("=" * 80)

    print(
        """
SCENARIO: 2 devices (Device 0, Device 1), Tensor Parallel axis = 'tensor'

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. REPLICATED SHARDING: (None, None)                                        │
│    Used for: Embeddings, LM Head                                            │
└─────────────────────────────────────────────────────────────────────────────┘

Original Tensor [8 x 4]:
┌─────────────────┐
│ 0  1  2  3      │
│ 4  5  6  7      │
│ 8  9  10 11     │
│ 12 13 14 15     │
│ 16 17 18 19     │
│ 20 21 22 23     │
│ 24 25 26 27     │
│ 28 29 30 31     │
└─────────────────┘

After Sharding (Both devices get full copy):

Device 0:                Device 1:
┌─────────────────┐      ┌─────────────────┐
│ 0  1  2  3      │      │ 0  1  2  3      │
│ 4  5  6  7      │      │ 4  5  6  7      │
│ 8  9  10 11     │      │ 8  9  10 11     │
│ 12 13 14 15     │      │ 12 13 14 15     │
│ 16 17 18 19     │      │ 16 17 18 19     │
│ 20 21 22 23     │      │ 20 21 22 23     │
│ 24 25 26 27     │      │ 24 25 26 27     │
│ 28 29 30 31     │      │ 28 29 30 31     │
└─────────────────┘      └─────────────────┘

Memory per device: 8x4 = 32 elements (100% of original)
"""
    )

    print(
        """
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. COLUMN-PARALLEL SHARDING: (None, 'tensor')                               │
│    Used for: Q/K/V projections, Gate/Up projections                         │
└─────────────────────────────────────────────────────────────────────────────┘

Original Weight [4 x 8]:
┌───────────────────────────────┐
│ 0  1  2  3  │ 4  5  6  7      │
│ 8  9  10 11 │ 12 13 14 15     │
│ 16 17 18 19 │ 20 21 22 23     │
│ 24 25 26 27 │ 28 29 30 31     │
└───────────────────────────────┘
           ↓ Split along column (dimension 1)

Device 0 [4 x 4]:        Device 1 [4 x 4]:
┌───────────────┐        ┌───────────────┐
│ 0  1  2  3    │        │ 4  5  6  7    │
│ 8  9  10 11   │        │ 12 13 14 15   │
│ 16 17 18 19   │        │ 20 21 22 23   │
│ 24 25 26 27   │        │ 28 29 30 31   │
└───────────────┘        └───────────────┘

Matrix Multiplication Example: X @ W
Input X [2 x 4] (replicated):     Weight W [4 x 8] (column-parallel):

Device 0:                Device 1:
X @ W_0                  X @ W_1
[2x4] @ [4x4]           [2x4] @ [4x4]
    ↓                        ↓
  [2x4]                    [2x4]

Final Output [2 x 8] = concat([2x4], [2x4]) along dim 1
                       ↑ Sharded in output dimension

Memory per device: 4x4 = 16 elements (50% of original) ✅ Memory saved!
"""
    )

    print(
        """
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. ROW-PARALLEL SHARDING: ('tensor', None)                                  │
│    Used for: O projection, Down projection                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Original Weight [8 x 4]:
┌─────────────────┐
│ 0  1  2  3      │
│ 4  5  6  7      │
│ 8  9  10 11     │
│ 12 13 14 15     │
├─────────────────┤ ← Split along row (dimension 0)
│ 16 17 18 19     │
│ 20 21 22 23     │
│ 24 25 26 27     │
│ 28 29 30 31     │
└─────────────────┘

Device 0 [4 x 4]:        Device 1 [4 x 4]:
┌─────────────────┐      ┌─────────────────┐
│ 0  1  2  3      │      │ 16 17 18 19     │
│ 4  5  6  7      │      │ 20 21 22 23     │
│ 8  9  10 11     │      │ 24 25 26 27     │
│ 12 13 14 15     │      │ 28 29 30 31     │
└─────────────────┘      └─────────────────┘

Matrix Multiplication Example: X @ W
Input X [2 x 8] (column-parallel):    Weight W [8 x 4] (row-parallel):
                                      PartitionSpec(None, 'tensor')    PartitionSpec('tensor', None)

Device 0:                Device 1:
X_0 @ W_0               X_1 @ W_1
[2x4] @ [4x4]           [2x4] @ [4x4]
    ↓                        ↓
  [2x4]                    [2x4]
   partial                  partial
   result                   result
       ↓                       ↓
       └───── All-Reduce ──────┘
                  ↓
            [2 x 4] Final (replicated)

💡 Key Point: Input X is column-parallel (split along dim 1)
             Weight W is row-parallel (split along dim 0)
             This ensures dimension alignment for matmul!

⚠️  Requires All-Reduce communication to sum partial results!

Memory per device: 4x4 = 16 elements (50% of original) ✅ Memory saved!
"""
    )

    print(
        """
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. COMPLETE TP PIPELINE: Column-Parallel → Row-Parallel                     │
│    Example: MLP layer (Gate → Activation → Down)                            │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Input (replicated)
┌───────┐        ┌───────┐
│   X   │   →    │   X   │
│ [2x4] │        │ [2x4] │
└───────┘        └───────┘
Device 0         Device 1

Step 2: Column-Parallel Matmul (W_gate)
┌───────┐        ┌───────┐
│ [2x4] │        │ [2x4] │
│   ↓   │        │   ↓   │
│  @W_0 │        │  @W_1 │
│ [4x4] │        │ [4x4] │
│   ↓   │        │   ↓   │
│ [2x4] │        │ [2x4] │
└───────┘        └───────┘
 Hidden_0         Hidden_1
(sharded output, no communication needed)

Step 3: Activation (element-wise, no communication)
┌───────┐        ┌───────┐
│ GELU  │        │ GELU  │
│ [2x4] │        │ [2x4] │
└───────┘        └───────┘

Step 4: Row-Parallel Matmul (W_down)
┌───────┐        ┌───────┐
│ [2x4] │        │ [2x4] │
│   ↓   │        │   ↓   │
│  @W_0 │        │  @W_1 │
│ [4x4] │        │ [4x4] │
│   ↓   │        │   ↓   │
│ [2x4] │        │ [2x4] │
└───────┘        └───────┘
Partial_0        Partial_1

Step 5: All-Reduce
┌───────────────────┐
│   Partial_0 +     │
│   Partial_1       │
│        ↓          │
│   Output [2x4]    │
│   (replicated)    │
└───────────────────┘

Communication: Only 1 All-Reduce operation per layer!
Memory Savings: Weights split across devices (50% per device)
"""
    )

    print("=" * 80)
    print(" 🎯 KEY TAKEAWAYS")
    print("=" * 80)
    print(
        """
1️⃣  Column-Parallel (None, 'tensor'):
   - Splits OUTPUT dimension
   - No communication during forward pass
   - Each device computes different features

2️⃣  Row-Parallel ('tensor', None):
   - Splits INPUT dimension
   - Requires All-Reduce to combine results
   - Follows column-parallel to minimize communication

3️⃣  Why this pattern works:
   - Column → Row sequence minimizes communication
   - Only one All-Reduce per layer pair
   - Weight memory distributed across devices
   - Computation fully parallel

4️⃣  Memory Savings:
   - Without TP: Each device holds 100% of weights
   - With 2-way TP: Each device holds 50% of weights
   - With 4-way TP: Each device holds 25% of weights
"""
    )


if __name__ == "__main__":
    print_sharding_visualization()
