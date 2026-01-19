import torch

# topk
origin_tensor = torch.rand(10, 5)
print(f"origin tensor = {origin_tensor}")
topk_tensor = torch.topk(origin_tensor, k=3, dim=-1)
print(f"top k {topk_tensor}")


a = torch.rand(10, 5)
print(a)
b = a[:, :, None]
print(b)
c = a.unsqueeze(2)
print(c)

print(torch.allclose(b, c))
