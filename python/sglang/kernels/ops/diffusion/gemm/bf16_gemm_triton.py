# SPDX-License-Identifier: Apache-2.0
"""Plain bf16 GEMM for the projections that still run through rocBLAS.

Same precision class as ``torch.mm`` (bf16 in and out, fp32 accumulation); only
the reduction order differs, which the QKV epilogue already measured at 49.7 dB
against the previous path. Measured at S=16384 on the three shapes a DiT layer
issues, against rocBLAS with the module's own ``weight.t()`` view:

    out       [16384, 4096] @ [4096, 4096]       24.4 -> 19.1 ms   (1.28x, 28.8 TF)
    gate/up   [16384, 4096] @ [4096, 12288]      70.0 -> 50.9 ms   (1.38x, 32.4 TF)
    MLP down  [16384, 12288] @ [12288, 4096]    111.9 -> 69.8 ms   (1.60x, 23.6 TF)

The down projection is where rocBLAS collapses to 14.7 TF; the tiling below
holds 23.6 TF there, which is most of the win. Compared with the QKV epilogue
kernel (``qkv_projection_permuted_triton``) the difference is just the store:
this one writes the ordinary ``[M, N]`` layout.
"""

import torch
import triton
import triton.language as tl

_BM = 128
_BN = 128
# K=4096 prefers BK=64 (50.9 vs 67.8 ms), K=12288 prefers BK=128 (69.8 vs 71.3).
_BK_SMALL = 64
_BK_LARGE = 128
_BK_LARGE_MIN_K = 8192
_WARPS = 8
_STAGES = 2
_GROUP = 8


@triton.jit
def _bf16_gemm(
    A, B, C, M, N, K,
    stride_bk, stride_bn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    if GROUP > 1:
        group_sz = GROUP * num_pid_n
        gid = pid // group_sz
        first_m = gid * GROUP
        gsize = min(num_pid_m - first_m, GROUP)
        pid_m = first_m + ((pid % group_sz) % gsize)
        pid_n = (pid % group_sz) // gsize
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.float32)
        a_ptrs += BK
        b_ptrs += BK * stride_bk
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))


def _tile(k):
    return _BK_LARGE if k >= _BK_LARGE_MIN_K else _BK_SMALL


def can_use_bf16_gemm(x, weight, out_features):
    """Contiguous 2-D bf16 activation, plain bf16 weight, tiles that divide evenly."""
    if not (x.is_cuda and x.dtype is torch.bfloat16 and x.is_contiguous()):
        return False
    if not (weight.is_cuda and weight.dtype is torch.bfloat16 and weight.dim() == 2):
        return False
    m, k = x.shape
    if weight.shape != (out_features, k):
        return False
    return m % _BM == 0 and k % _tile(k) == 0 and out_features % _BN == 0


def bf16_gemm(x, weight_t, out):
    """``out[m, n] = x[m, :] @ weight_t[:, n]`` in bf16, where ``weight_t`` is ``weight.t()``."""
    m, k = x.shape
    n = out.shape[1]
    bk = _tile(k)
    grid = (triton.cdiv(m, _BM) * triton.cdiv(n, _BN),)
    _bf16_gemm[grid](
        x, weight_t, out, m, n, k, weight_t.stride(0), weight_t.stride(1),
        BM=_BM, BN=_BN, BK=bk, GROUP=_GROUP,
        num_warps=_WARPS, num_stages=_STAGES,
    )
    return out
