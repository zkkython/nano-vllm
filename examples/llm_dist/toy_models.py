import torch
import torch.nn as nn


class ToyMLP(nn.Module):
    """一个非常简单的 MLP，用于 DP / TP / SP 等示例。

    参数规模刻意保持很小，方便在两张 GPU 上快速验证。
    """

    def __init__(self, dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ToyPipelineStage0(nn.Module):
    """流水线并行 Stage 0：输入嵌入 + 第一层线性。

    在 Rank 0 上运行。
    """

    def __init__(self, dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        return x


class ToyPipelineStage1(nn.Module):
    """流水线并行 Stage 1：第二层线性 + 输出。

    在 Rank 1 上运行。
    """

    def __init__(self, dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(x)


class ToyExpert(nn.Module):
    """Expert Parallel 示例中用到的简易 Expert。"""

    def __init__(self, dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.mlp = ToyMLP(dim=dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
