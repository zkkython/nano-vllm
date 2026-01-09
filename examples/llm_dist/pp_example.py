"""流水线并行 (Pipeline Parallel, PP) 示例。

我们将一个简单的两层 MLP 拆成两个 Stage：
- Rank 0: ToyPipelineStage0
- Rank 1: ToyPipelineStage1

两个 Rank 之间使用 send/recv 传递中间激活，仅展示单 batch 前向/反向的基本数据流。
"""

from typing import Tuple

import torch
import torch.distributed as dist

from dist_utils import cleanup, init_distributed, parse_args
from toy_models import ToyPipelineStage0, ToyPipelineStage1


def get_data(batch_size: int = 4, dim: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch_size, dim).cuda()
    y = torch.randn(batch_size, dim).cuda()
    return x, y


def run_pp(rank: int, world_size: int, master_addr: str, master_port: str, backend: str = "nccl") -> None:
    assert world_size == 2, "这个简化示例假设 world_size=2"

    init_distributed(rank, world_size, master_addr, master_port, backend)

    dim = 16
    hidden_dim = 32

    if rank == 0:
        model = ToyPipelineStage0(dim=dim, hidden_dim=hidden_dim).cuda()
    else:
        model = ToyPipelineStage1(dim=dim, hidden_dim=hidden_dim).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    if rank == 0:
        x, y = get_data(dim=dim)
        x.requires_grad_()

        h = model(x)

        dist.send(h.detach(), dst=1)

        grad_h = torch.zeros_like(h)
        dist.recv(grad_h, src=1)

        h.backward(grad_h)

        optimizer.step()

        print("[Rank 0] PP 前后向完成")

    else:
        _, y = get_data(dim=dim)

        h = torch.zeros((4, hidden_dim), device=torch.cuda.current_device())
        dist.recv(h, src=0)
        h.requires_grad_()

        out = model(h)
        loss = torch.nn.functional.mse_loss(out, y)
        print(f"[Rank 1] PP loss: {loss.item():.4f}")

        loss.backward()

        dist.send(h.grad.detach(), dst=0)

        optimizer.step()

        print("[Rank 1] PP 前后向完成")

    cleanup()


if __name__ == "__main__":
    args = parse_args(default_world_size=2)
    run_pp(args.rank, args.world_size, args.master_addr, args.master_port, args.backend)
