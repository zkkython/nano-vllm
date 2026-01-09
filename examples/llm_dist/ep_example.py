"""Expert Parallel (EP) 示例。

我们构造一个非常简化的 MoE：
- 有 2 个 Expert，分别放在 Rank 0 和 Rank 1 上；
- 所有 Rank 上有相同的门控 (gating) 逻辑，决定一半 Token 走 Expert0，一半 Token 走 Expert1；
- 使用 all_to_all 进行 Token 路由与结果回传。

这里只是演示通信与分组方式，未做负载均衡等细节。
"""

from typing import Tuple

import torch
import torch.distributed as dist

from dist_utils import cleanup, init_distributed, parse_args
from toy_models import ToyExpert


def get_data(batch_size: int = 4, dim: int = 16) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch_size, dim).cuda()


def run_ep(rank: int, world_size: int, master_addr: str, master_port: str, backend: str = "nccl") -> None:
    assert world_size == 2, "这个简化 EP 示例假设 world_size=2 (2 个 Expert)"

    init_distributed(rank, world_size, master_addr, master_port, backend)

    dim = 16
    model = ToyExpert(dim=dim).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = get_data(dim=dim)
    batch_size = x.size(0)

    indices = torch.arange(batch_size, device=x.device)
    mask_expert0 = indices < batch_size // 2
    mask_expert1 = ~mask_expert0

    if rank == 0:
        local_input = x[mask_expert0]
    else:
        local_input = x[mask_expert1]

    inputs = [torch.zeros_like(local_input) for _ in range(world_size)]
    dist.all_gather(inputs, local_input)

    if rank == 0:
        my_tokens = inputs[0]
    else:
        my_tokens = inputs[1]

    local_output = model(my_tokens)

    outputs = [torch.zeros_like(local_output) for _ in range(world_size)]
    dist.all_gather(outputs, local_output)

    out = torch.zeros_like(x)
    out[mask_expert0] = outputs[0]
    out[mask_expert1] = outputs[1]

    loss = (out.pow(2).mean())
    print(f"[Rank {rank}] EP loss: {loss.item():.4f}")

    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size

    optimizer.step()

    print(f"[Rank {rank}] EP 步骤完成")

    cleanup()


if __name__ == "__main__":
    args = parse_args(default_world_size=2)
    run_ep(args.rank, args.world_size, args.master_addr, args.master_port, args.backend)
