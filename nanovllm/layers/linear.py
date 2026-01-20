import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from nanovllm.layers.fp8 import linear_fp8


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
        quantization: str | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quantization = quantization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        transpose: bool = False,
        quantization: str | None = None,
    ):
        super().__init__(input_size, output_size, quantization=quantization)
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quantization == "fp8"
            else torch.get_default_dtype()
        )
        if not transpose:
            self.weight = nn.Parameter(
                torch.empty(self.output_size, self.input_size, dtype=weight_dtype)
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(self.input_size, self.output_size, dtype=weight_dtype)
            )
        self.weight.weight_loader = self.weight_loader
        if self.quantization == "fp8":
            if not transpose:
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        (self.output_size + 127) // 128, (self.input_size + 127) // 128
                    ),
                    requires_grad=False,
                )
            else:
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        (self.input_size + 127) // 128, (self.output_size + 127) // 128
                    ),
                    requires_grad=False,
                )

            self.weight_scale.weight_loader = self.weight_scale_loader
        else:
            self.register_parameter("weight_scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.transpose = transpose

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"[FP8_EXEC] Running Triton FP8 Kernel for {self.weight_scale.shape}")
        if not self.transpose:
            if self.weight_scale is not None:
                # print(f"[FP8_EXEC] Running Triton FP8 Kernel for {self.__class__.__name__}")
                return linear_fp8(x, self.weight, self.weight_scale, self.bias)
            return F.linear(x, self.weight, self.bias)
        else:
            return (
                x @ self.weight + self.bias
                if self.bias is not None
                else x @ self.weight
            )


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        transpose: bool = False,
        quantization: str | None = None,
    ):
        # 如果模型权重加载的时候没有转置，那么就是使用原来的
        # 如果转置了，那么直接在列维度切分就行，也就是tp_dim=1
        # print(f"quantization: {quantization}")
        if not transpose:
            super().__init__(input_size, output_size, 0, quantization=quantization)
        else:
            super().__init__(input_size, output_size, 1, quantization=quantization)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quantization == "fp8"
            else torch.get_default_dtype()
        )
        if not transpose:
            self.weight = nn.Parameter(
                torch.empty(
                    self.output_size_per_partition, self.input_size, dtype=weight_dtype
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    self.input_size, self.output_size_per_partition, dtype=weight_dtype
                )
            )
        self.weight.weight_loader = self.weight_loader
        if self.quantization == "fp8":
            if not transpose:
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        (self.output_size_per_partition + 127) // 128,
                        (self.input_size + 127) // 128,
                    ),
                    requires_grad=False,
                )
            else:
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        (self.input_size + 127) // 128,
                        (self.output_size_per_partition + 127) // 128,
                    ),
                    requires_grad=False,
                )
            self.weight_scale.weight_loader = self.weight_scale_loader
        else:
            self.register_parameter("weight_scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.transpose = transpose

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(
        #     f"ColumnParallelLinear forward, x.shape: {x.shape}, weight shape {self.weight.shape}"
        # )
        if self.weight_scale is not None:
            # print(f"[FP8_EXEC] Running Triton FP8 Kernel for {self.__class__.__name__}")
            return linear_fp8(
                x,
                self.weight,
                self.weight_scale,
                self.bias,
                transpose_weight=self.transpose,
            )

        if not self.transpose:

            return F.linear(x, self.weight, self.bias)
        else:
            return (
                x @ self.weight + self.bias
                if self.bias is not None
                else x @ self.weight
            )


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quantization: str | None = None,
    ):
        # input_size = hidden_size
        # output_sizes = [intermediate_size, intermediate_size] for Qwen3MLP
        self.output_sizes = output_sizes
        super().__init__(
            input_size, sum(output_sizes), bias=bias, quantization=quantization
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ):
        param_data = param.data
        # MergedColumnParallelLinear weight loading,
        # loaded_shard_id = 0 gate weight, 1 up weight for Qwen3MLP
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 获取tp_rank 对应的权重数据
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quantization: str | None = None,
    ):
        self.head_size = head_size  # 128
        self.total_num_heads = total_num_heads  # 16
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads  # 8
        tp_size = dist.get_world_size()
        self.num_heads = divide(
            self.total_num_heads, tp_size
        )  # 单个tp rank的head数量，如果tp_size=8, 则是16/8=2
        self.num_kv_heads = divide(
            self.total_num_kv_heads, tp_size
        )  # 单个tp rank的kv head数量，如果tp_size=8, 则是8/8=1
        input_size = hidden_size  # 1024
        output_size = (
            self.total_num_heads + 2 * self.total_num_kv_heads
        ) * self.head_size  # （16 + 2 * 8） * 128 = 4096, 相当于q,k,v三个矩阵的拼接结果
        super().__init__(input_size, output_size, bias, quantization=quantization)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size  # 256
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size  # 128
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size  # 128
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
        param_data = param_data.narrow(
            self.tp_dim, shard_offset, shard_size
        )  # 取出当前tp_rank对应的权重数据(q，k,v 合并在一起，通过不同行来区分)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[
            self.tp_rank
        ]  # 从模型文件中获取tp_rank 对应的权重数据
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        transpose: bool = False,
        quantization: str | None = None,
    ):
        if not transpose:
            super().__init__(input_size, output_size, 1, quantization=quantization)
        else:
            super().__init__(input_size, output_size, 0, quantization=quantization)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quantization == "fp8"
            else torch.get_default_dtype()
        )
        if not transpose:
            self.weight = nn.Parameter(
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=weight_dtype
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    self.input_size_per_partition, self.output_size, dtype=weight_dtype
                )
            )
        self.weight.weight_loader = self.weight_loader
        if self.quantization == "fp8":
            if not transpose:
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        (self.output_size + 127) // 128,
                        (self.input_size_per_partition + 127) // 128,
                    ),
                    requires_grad=False,
                )
            else:
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        (self.input_size_per_partition + 127) // 128,
                        (self.output_size + 127) // 128,
                    ),
                    requires_grad=False,
                )
            self.weight_scale.weight_loader = self.weight_scale_loader
        else:
            self.register_parameter("weight_scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
            # 预分配一个零偏置，避免 forward 过程中动态分配内存
            self.register_buffer(
                "zero_bias", torch.zeros(self.output_size), persistent=False
            )

        self.transpose = transpose

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_scale is not None:
            # print(f"[FP8_EXEC] Running Triton FP8 Kernel for {self.__class__.__name__}")
            y = linear_fp8(
                x,
                self.weight,
                self.weight_scale,
                self.bias,
                transpose_weight=self.transpose,
            )
            if self.tp_size > 1:
                dist.all_reduce(y)
            return y

        if not self.transpose:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        else:
            effective_bias = (
                self.bias
                if self.tp_rank == 0 and self.bias is not None
                else self.zero_bias
            )
            y = torch.addmm(effective_bias, x, self.weight)

        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
