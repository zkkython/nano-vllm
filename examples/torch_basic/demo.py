import torch


torch.manual_seed(42)
a = torch.full(size=(1,), fill_value=-1)
print(a.item())

b = torch.ones(1)
print(b)


d=[]
d.extend([1,23])
d.extend([4,5])
print(d)


print([-1] * 5)

print('/root/model/deepseek'.split('/')[-1])