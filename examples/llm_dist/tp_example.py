"""张量并行 (Tensor Parallel, TP) 示例。

这里我们实现一个非常简化的列并行线性层：权重按列切分到两个 rank 上，
每个 rank 只计算自己那份输出，最后通过 all_gather 将分片拼接成完整输出。

假设 world_size=2，对应 2 个 GPU 节点，每个节点一个进程（也可以在单机上模拟）。
"""

from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from dist_utils import cleanup, init_distributed, parse_args


class ColumnParallelLinear(nn.Module):
    """一个简化版列并行 Linear: out_features 按列在多个 rank 间划分。"""

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        assert out_features % world_size == 0, "out_features 必须能被 world_size 整除"
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        per_rank_out = out_features // world_size
        self.local_linear = nn.Linear(in_features, per_rank_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = self.local_linear(x)

        outputs = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(outputs, local_out)
        full_out = torch.cat(outputs, dim=-1)
        return full_out


def get_data(batch_size: int = 4, dim: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch_size, dim).cuda()
    y = torch.randn(batch_size, dim).cuda()
    return x, y


def run_tp(rank: int, world_size: int, master_addr: str, master_port: str, backend: str = "nccl") -> None:
    init_distributed(rank, world_size, master_addr, master_port, backend)

    dim = 16
    model = ColumnParallelLinear(dim, dim, world_size=world_size, rank=rank).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x, y = get_data(dim=dim)

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    print(f"[Rank {rank}] TP loss: {loss.item():.4f}")

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

    optimizer.step()
    print(f"[Rank {rank}] 完成一次 TP 步骤")

    cleanup()


if __name__ == "__main__":
    args = parse_args(default_world_size=2)
    run_tp(args.rank, args.world_size, args.master_addr, args.master_port, args.backend)
