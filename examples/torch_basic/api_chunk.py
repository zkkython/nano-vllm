import torch

# 创建张量
x = torch.arange(10).reshape(5, 2)
print(x.shape)  # torch.Size([5, 2])

# 沿着第一个维度（dim=0）分割成3个子张量
chunks = x.chunk(3, dim=0)
print(len(chunks))  # 3
for i, c in enumerate(chunks):
    print(f"chunk {i}: {c.shape} -> {c}")
# 输出：
# chunk 0: torch.Size([1, 2]) -> tensor([[0, 1]])
# chunk 1: torch.Size([1, 2]) -> tensor([[2, 3]])
# chunk 2: torch.Size([3, 2]) -> tensor([[4, 5], [6, 7], [8, 9]])  # 最后一块较大
print("----" * 10)

x = torch.arange(24).reshape(3, 8)  # shape: (3, 8)

# 沿着dim=1（列维度）分割成4个子张量，每块2列
chunks = x.chunk(4, dim=1)
for i, c in enumerate(chunks):
    print(f"chunk {i}: {c.shape}")
# chunk 0: torch.Size([3, 2])
# chunk 1: torch.Size([3, 2])
# chunk 2: torch.Size([3, 2])
# chunk 3: torch.Size([3, 2])  # 8列能被4整除，每块2列

print("----" * 10)

# 模拟4-GPU张量并行，权重矩阵(shape: class_num, feature_dim)
weight = torch.randn(1000, 512)  # vocab_size=1000, hidden_size=512

# 按GPU数量分割
num_gpus = 4
weight_chunks = weight.chunk(num_gpus, dim=0)
for rank in range(num_gpus):
    local_weight = weight_chunks[rank]
    print(f"GPU {rank}: {local_weight.shape}")  # 每个GPU: (250, 512)


print("----" * 10)
# chunk size > tensor.size(dim)
x = torch.arange(10).reshape(5, 2)
print(x.size(0))  # torch.Size([5, 2])
chunks = x.chunk(10, dim=0)
for i, c in enumerate(chunks):
    print(f"chunk {i}: {c.shape} -> {c}")
