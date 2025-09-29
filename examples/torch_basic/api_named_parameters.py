import torch
from torch import nn


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.rand(3, 3))
        self.param2 = torch.nn.Parameter(torch.rand(3, 3))

    def forward(self, x):
        return self.param1 + self.param2


class MyModule2(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.randn(10, 5)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(10)))


class MyModule0(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 10)
        self.param = nn.Parameter(torch.randn(20, 30))
        self.module1 = MyModule()


class MyModule3(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = MyModule0()
        self.m2 = MyModule2()


if __name__ == "__main__":
    model = MyModule()
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, shape: {param.shape}")
    print("----" * 10)
    model2 = MyModule2()
    for name, param in model2.named_parameters():
        print(f"Parameter name: {name}, shape: {param.shape}")
    print("----" * 10)
    model3 = MyModule3()
    for name, param in model3.named_parameters():
        print(f"Parameter name: {name}, shape: {param.shape}")
