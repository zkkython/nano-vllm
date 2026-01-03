"""分布式通信工具，支持跨机器TP并行"""
import torch
import torch.distributed as dist
from typing import Any, Optional
import pickle
import socket
import struct


def init_distributed_environment(
    rank: int, 
    world_size: int, 
    master_addr: str, 
    master_port: int,
    backend: str = "nccl"
):
    """初始化分布式环境"""
    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())


def get_world_size() -> int:
    """获取world size"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """获取当前rank"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_distributed() -> bool:
    """检查是否在分布式环境中"""
    return dist.is_available() and dist.is_initialized()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """广播对象到所有rank"""
    if not is_distributed():
        return obj
    
    rank = get_rank()
    if rank == src:
        obj_bytes = pickle.dumps(obj)
        obj_size = len(obj_bytes)
        size_tensor = torch.tensor([obj_size], dtype=torch.long, device=f"cuda:{get_rank()}")
    else:
        size_tensor = torch.zeros(1, dtype=torch.long, device=f"cuda:{get_rank()}")
    
    dist.broadcast(size_tensor, src=src)
    
    if rank != src:
        obj_size = size_tensor.item()
        obj_bytes_tensor = torch.zeros(obj_size, dtype=torch.uint8, device=f"cuda:{get_rank()}")
    else:
        obj_bytes_tensor = torch.tensor(list(obj_bytes), dtype=torch.uint8, device=f"cuda:{get_rank()}")
    
    dist.broadcast(obj_bytes_tensor, src=src)
    
    if rank != src:
        obj_bytes = bytes(obj_bytes_tensor.cpu().numpy())
        obj = pickle.loads(obj_bytes)
    
    return obj


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """执行all reduce操作"""
    if not is_distributed():
        return tensor
    
    dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """执行all gather操作"""
    if not is_distributed():
        return tensor
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=dim)


def barrier():
    """同步所有进程"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()