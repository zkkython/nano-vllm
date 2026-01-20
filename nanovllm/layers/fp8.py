import torch
import triton
import triton.language as tl
from triton import Config
from typing import Tuple, Optional, Literal
from torch.nn import functional as F


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    s = s.to(tl.float32)
    s = tl.maximum(s, 1e-10)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


fp8_gemm_configs = [
    Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
]


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    B_PROJ: tl.constexpr,  # True if B is [N, K] (needs transpose), False if B is [K, N]
):
    # Grain size is fixed to 128 for DeepSeek-V3 FP8
    GRAIN: tl.constexpr = 128
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    k_steps = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]

    if B_PROJ:
        b_ptrs = b_ptr + offs_k[:, None] + offs_n[None, :] * K
    else:
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    # Scales for A: shape [M, K_blocks] where K_blocks = K // GRAIN
    k_blocks = tl.cdiv(K, GRAIN)
    n_blocks = tl.cdiv(N, GRAIN)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k_steps):
        k_curr = i * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k_curr),
            other=0.0,
        )
        if B_PROJ:
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k_curr) & (offs_n[None, :] < N),
                other=0.0,
            )
        else:
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k_curr) & (offs_n[None, :] < N),
                other=0.0,
            )

        # Load scales
        i_block = k_curr // GRAIN

        # A scale: [M, k_blocks]
        a_s = tl.load(a_s_ptr + offs_m * k_blocks + i_block, mask=offs_m < M, other=1.0)

        # B scale
        if B_PROJ:
            # B is [N, K], scale is [N_blocks, K_blocks]
            b_s_idx = (offs_n // GRAIN) * k_blocks + i_block
            b_s = tl.load(
                b_s_ptr + b_s_idx, mask=(offs_n // GRAIN) < n_blocks, other=1.0
            )
        else:
            # B is [K, N], scale is [K_blocks, N_blocks]
            b_s_idx = i_block * n_blocks + (offs_n // GRAIN)
            b_s = tl.load(
                b_s_ptr + b_s_idx, mask=(offs_n // GRAIN) < n_blocks, other=1.0
            )

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        a_ptrs += BLOCK_SIZE_K
        if B_PROJ:
            b_ptrs += BLOCK_SIZE_K
        else:
            b_ptrs += BLOCK_SIZE_K * N

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m_final = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_final = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create output mask to prevent out-of-bounds writes
    out_mask = (offs_m_final[:, None] < M) & (offs_n_final[None, :] < N)

    c_ptrs = c_ptr + offs_m_final[:, None] * N + offs_n_final[None, :]
    tl.store(c_ptrs, c, mask=out_mask)


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    bf16_weight: bool = False,  # Not used here but keeping interface
    transpose_b: bool = False,  # If True, B is [K, N]. If False, B is [N, K]
):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0) if not transpose_b else b.size(1)
    dtype = out_dtype if out_dtype is not None else torch.get_default_dtype()
    c = a.new_empty(*a.size()[:-1], N, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    # B_PROJ=True means physically [N, K] (needs transpose)
    # transpose_b=False means physically [N, K]
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K, B_PROJ=not transpose_b)
    return c


def linear_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    gemm_impl: Literal["bf16", "fp8"] = "bf16",
    block_size: int = 128,
    transpose_weight: bool = False,
) -> torch.Tensor:
    if gemm_impl == "bf16":
        # Dequantize weight and use normal linear
        w = weight_dequant(weight, weight_scale, block_size, dtype=x.dtype)
        if transpose_weight:
            return x @ w + bias if bias is not None else x @ w
        return F.linear(x, w, bias)
    else:
        # FP8 GEMM
        x_q, x_s = act_quant(x, block_size)
        y = fp8_gemm(
            x_q,
            x_s,
            weight,
            weight_scale,
            out_dtype=x.dtype,
            transpose_b=transpose_weight,
        )
        if bias is not None:
            y += bias
        return y
