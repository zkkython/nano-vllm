"""Sequence Parallel (SP) 示例。

我们在序列维度上切分输入张量：
- world_size=2 时，Rank 0 负责序列前半部分，Rank 1 负责后半部分；
- 每个 Rank 上各自执行一份 MLP，仅作用于自己的序列切片；
- 通过 all_gather 在序列维上拼接回完整输出。

这个示例演示 SP 的核心思想：沿着序列维度切分计算，减小单卡激活占用。
"""

from typing import Tuple

import torch
import torch.distributed as dist

from dist_utils import cleanup, init_distributed, parse_args
from toy_models import ToyMLP


def get_data(batch_size: int = 2, seq_len: int = 8, dim: int = 16) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch_size, seq_len, dim).cuda()


def run_sp(rank: int, world_size: int, master_addr: str, master_port: str, backend: str = "nccl") -> None:
    assert world_size == 2, "这个简化 SP 示例假设 world_size=2"

    init_distributed(rank, world_size, master_addr, master_port, backend)

    batch_size, seq_len, dim = 2, 8, 16
    x = get_data(batch_size=batch_size, seq_len=seq_len, dim=dim)

    assert seq_len % world_size == 0
    local_seq_len = seq_len // world_size

    start = rank * local_seq_len
    end = (rank + 1) * local_seq_len
    x_local = x[:, start:end, :]

    model = ToyMLP(dim=dim).cuda()

    x_local_flat = x_local.reshape(batch_size * local_seq_len, dim)
    out_local_flat = model(x_local_flat)
    out_local = out_local_flat.reshape(batch_size, local_seq_len, dim)

    outputs = [torch.zeros_like(out_local) for _ in range(world_size)]
    dist.all_gather(outputs, out_local)

    out = torch.cat(outputs, dim=1)

    loss = out.pow(2).mean()
    print(f"[Rank {rank}] SP loss: {loss.item():.4f}")

    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size

    print(f"[Rank {rank}] SP 步骤完成")

    cleanup()


if __name__ == "__main__":
    args = parse_args(default_world_size=2)
    run_sp(args.rank, args.world_size, args.master_addr, args.master_port, args.backend)
