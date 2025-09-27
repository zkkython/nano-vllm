import torch
import torch.nn.functional as F
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.utils.context import set_context, reset_context


def test_parallel_lmhead_forward_single_rank():
    """Test ParallelLMHead forward with tp_size=1 (no parallelism)."""
    # Mock dist to simulate single rank
    with patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.get_world_size', return_value=1):

        num_embeddings = 100
        embedding_dim = 64
        lm_head = ParallelLMHead(num_embeddings, embedding_dim, bias=True)

        # Initialize weights
        lm_head.weight.data = torch.randn_like(lm_head.weight)
        lm_head.bias.data = torch.randn_like(lm_head.bias)

        # Test input: batch_size=2, seq_len=5, embedding_dim=64
        x = torch.randn(2, 5, embedding_dim)

        # Test non-prefill case
        reset_context()
        set_context(is_prefill=False)
        logits = lm_head(x)
        expected_logits = F.linear(x, lm_head.weight, lm_head.bias)
        assert torch.allclose(logits, expected_logits), "Non-prefill logits mismatch"

        # Test prefill case
        reset_context()
        # cu_seqlens_q: cumulative sequence lengths, e.g., [0, 3, 5] for two sequences of len 3 and 2
        cu_seqlens_q = torch.tensor([0, 3, 5])
        set_context(is_prefill=True, cu_seqlens_q=cu_seqlens_q)
        logits = lm_head(x)
        # Should extract last tokens: indices 2 and 4 (0-based)
        last_indices = cu_seqlens_q[1:] - 1  # [2, 4]
        expected_x = x[last_indices]  # shape (2, 64)
        expected_logits = F.linear(expected_x, lm_head.weight, lm_head.bias)
        assert torch.allclose(logits, expected_logits), "Prefill logits mismatch"


def test_parallel_lmhead_forward_multi_rank():
    """Test ParallelLMHead forward with tp_size=2 (tensor parallelism)."""
    tp_size = 2
    num_embeddings = 100  # must be divisible by tp_size
    embedding_dim = 64

    # Mock dist functions
    with patch('torch.distributed.get_rank', side_effect=[0, 1]), \
         patch('torch.distributed.get_world_size', return_value=tp_size), \
         patch('torch.distributed.gather') as mock_gather:

        # Create two instances for rank 0 and 1
        lm_heads = []
        for rank in range(tp_size):
            lm_head = ParallelLMHead(num_embeddings, embedding_dim, bias=False)
            lm_head.weight.data = torch.randn_like(lm_head.weight)
            lm_heads.append(lm_head)

        # Test input: batch_size=2, embedding_dim=64 (after prefill extraction)
        x = torch.randn(2, embedding_dim)

        # Simulate gather: collect logits from both ranks
        all_logits = [torch.empty(2, num_embeddings // tp_size) for _ in range(tp_size)]
        mock_gather.side_effect = lambda tensor, gather_list, dst: gather_list.__setitem__(0, tensor.clone()) if dst == 0 else None

        # For rank 0
        reset_context()
        set_context(is_prefill=False)
        logits_0 = lm_heads[0](x)
        # Check that gather was called
        assert mock_gather.called, "gather should be called for multi-rank"

        # For rank 1
        reset_context()
        set_context(is_prefill=False)
        logits_1 = lm_heads[1](x)

        # Manually simulate the full logits concatenation
        full_logits = torch.cat([logits_0, logits_1], dim=-1)
        assert full_logits.shape == (2, num_embeddings), f"Expected shape (2, {num_embeddings}), got {full_logits.shape}"


if __name__ == "__main__":
    test_parallel_lmhead_forward_single_rank()
    test_parallel_lmhead_forward_multi_rank()
    print("All tests passed!")
