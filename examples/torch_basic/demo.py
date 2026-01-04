import torch


torch.manual_seed(42)
a = torch.full(size=(1,), fill_value=-1)
print(a.item())

b = torch.ones(1)
print(b)
