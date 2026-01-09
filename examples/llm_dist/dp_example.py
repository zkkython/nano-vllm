"""数据并行 (Data Parallel, DP) 示例。

假设有 2 个 GPU 节点，总共 2 个进程 (world_size=2)，每个进程持有一份完整模型，
但只看到自己那部分数据，通过 all_reduce 聚合梯度，实现等价于单卡更大 batch 的训练。

示例中我们仅做一次前向 + 反向 + 梯度 all_reduce，不做真正训练循环，便于快速验证。
"""

from typing import Tuple

import torch
import torch.distributed as dist

from dist_utils import cleanup, init_distributed, parse_args
from toy_models import ToyMLP


def get_data(rank: int, batch_size: int = 4, dim: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """每个 rank 拿到不同的数据分片。"""

    torch.manual_seed(42 + rank)
    x = torch.randn(batch_size, dim).cuda()
    y = torch.randn(batch_size, dim).cuda()
    return x, y


def run_dp(rank: int, world_size: int, master_addr: str, master_port: str, backend: str = "nccl") -> None:
    init_distributed(rank, world_size, master_addr, master_port, backend)

    model = ToyMLP().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x, y = get_data(rank)

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    print(f"[Rank {rank}] local loss: {loss.item():.4f}")

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

    optimizer.step()

    print(f"[Rank {rank}] 完成一次 DP 步骤")

    cleanup()


if __name__ == "__main__":
    args = parse_args(default_world_size=2)
    run_dp(args.rank, args.world_size, args.master_addr, args.master_port, args.backend)
