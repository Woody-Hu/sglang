# SPDX-License-Identifier: Apache-2.0
"""QKV projection whose epilogue stores dense ``[H, S, D]`` slabs.

The plain path issues one ``mm([S, C], [C, 3C])`` into a token-major buffer and
then pays a ``direct_copy`` per tensor so the flash attention backend sees dense
``[B, H, S, D]``. On gfx1151 that copy runs at a fraction of the machine's write
bandwidth, and the GEMM itself is ~1.3x slower through rocBLAS than through the
tiling below (measured at M=16384 / K=4096 / N=12288: 69.8 ms vs 54.0 ms).

Both costs disappear if the projection writes each head's columns as a
contiguous ``[S, D]`` slab: the block width equals ``D``, so one program covers
exactly one head and stores a dense ``[BM, D]`` block, and the attention backend
then receives an already-dense tensor.

Rounding differs from rocBLAS (fp32 accumulation order), so this is a quality
tier path, not a bit-exact one: callers compare the first output against the
reference projections before trusting it.
"""

import torch
import triton
import triton.language as tl

# Tuned on gfx1151 for M=16384 / K=4096 / N=12288 (P8/P9 microbenchmarks).
_BM = 128
_BK = 64
_GROUP = 8
_WARPS = 8
_STAGES = 2

# Eight rows per program keeps the 128-wide mean and the RoPE table load in one
# block without over-running the register budget.
_NORM_ROWS = 8


@triton.jit
def _qkv_gemm_permuted_store(
    A,
    B,
    OUT,
    M,
    K,
    N_OFF,
    stride_bk,
    stride_bn,
    ROW_OFF,
    STRIDE_HEAD,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """``A[M, K] @ B[K, N_OFF:N_OFF+HEADS*D]`` stored as ``OUT[h, ROW_OFF + m, :]``."""
    BN: tl.constexpr = D
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    if GROUP > 1:
        group_sz = GROUP * HEADS
        gid = pid // group_sz
        first_m = gid * GROUP
        gsize = min(num_pid_m - first_m, GROUP)
        pid_m = first_m + ((pid % group_sz) % gsize)
        pid_n = (pid % group_sz) // gsize
    else:
        pid_m = pid // HEADS
        pid_n = pid % HEADS

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_n = N_OFF + pid_n * BN + tl.arange(0, BN)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.float32)
        a_ptrs += BK
        b_ptrs += BK * stride_bk

    c_ptrs = (
        OUT
        + pid_n * STRIDE_HEAD
        + (ROW_OFF + offs_m)[:, None] * D
        + tl.arange(0, BN)[None, :]
    )
    tl.store(c_ptrs, acc.to(tl.bfloat16))


@triton.jit
def _qk_norm_rope_hsd(
    X,
    W,
    ROPE,
    P,
    S,
    ROWS,
    EPS,
    D: tl.constexpr,
    BM: tl.constexpr,
):
    """In place RMSNorm + complex RoPE on rows ``[P:P+S]`` of one ``[H, P+S, D]`` head.

    ``ROPE`` is the ``[S, D/2]`` complex table reinterpreted as ``[S, D]`` float32
    (real/imaginary interleaved), matching ``torch.view_as_complex``.
    """
    pid = tl.program_id(0)
    nblk = tl.cdiv(S, BM)
    head = pid // nblk
    offs_s = (pid % nblk) * BM + tl.arange(0, BM)
    mask = (offs_s < S)[:, None]
    offs_d = tl.arange(0, D)
    # ROWS is the physical row count per head: the buffer may be padded past
    # P + S so the attention kernel's key/value loop stays block aligned.
    row = X + head * ROWS * D + (P + offs_s)[:, None] * D + offs_d[None, :]
    x = tl.load(row, mask, 0.0).to(tl.float32)
    weight = tl.load(W + offs_d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + EPS)
    # The eager chain rounds the RMSNorm output to the activation dtype before
    # RoPE; keep that boundary so only the reduction order can differ.
    y = (x * inv[:, None]).to(X.dtype.element_ty).to(tl.float32)
    y = (y * weight[None, :]).to(X.dtype.element_ty).to(tl.float32)

    half: tl.constexpr = D // 2
    table = ROPE + offs_s[:, None] * D + 2 * tl.arange(0, half)[None, :]
    real, imag = tl.split(tl.reshape(y, (BM, half, 2)))
    cos = tl.load(table, mask, 0.0)
    sin = tl.load(table + 1, mask, 0.0)
    out = tl.join(tl.fma(real, cos, -imag * sin), tl.fma(imag, cos, real * sin))
    tl.store(row, tl.reshape(out, (BM, D)).to(X.dtype.element_ty), mask)


def can_use_qkv_projection_permuted(x, weight_t, heads, head_dim):
    """Dense bf16 input, bf16 ``[K, 3H*D]`` weight view, tiles that divide evenly."""
    if not (x.is_cuda and x.dtype is torch.bfloat16 and x.is_contiguous()):
        return False
    if not (weight_t.is_cuda and weight_t.dtype is torch.bfloat16):
        return False
    if head_dim != 128:
        return False
    m, k = x.shape
    return (
        weight_t.shape == (k, 3 * heads * head_dim)
        and m % _BM == 0
        and k % _BK == 0
    )


def qkv_projection_permuted(x, weight_t, q_buf, k_buf, v_buf, prefix):
    """Fill ``q_buf``/``k_buf``/``v_buf`` (``[H, S, D]``, ``[H, P+S, D]`` ...) with three launches.

    ``q_buf`` holds the target tokens only; ``k_buf``/``v_buf`` keep their first
    ``prefix`` rows untouched so the caller can drop the cached prefix in.
    """
    tokens, cols = x.shape
    heads, dim = q_buf.shape[1], q_buf.shape[3]
    rows = heads * dim
    grid = (triton.cdiv(tokens, _BM) * heads,)
    for buf, n_off, row_off in ((q_buf, 0, 0), (k_buf, rows, prefix), (v_buf, 2 * rows, prefix)):
        _qkv_gemm_permuted_store[grid](
            x,
            weight_t,
            buf,
            tokens,
            cols,
            n_off,
            weight_t.stride(0),
            weight_t.stride(1),
            row_off,
            buf.shape[2] * dim,
            HEADS=heads,
            D=dim,
            BM=_BM,
            BK=_BK,
            GROUP=_GROUP,
            num_warps=_WARPS,
            num_stages=_STAGES,
        )
    return q_buf, k_buf, v_buf


def qk_norm_rope_hsd(x, weight, rope, prefix, seq, eps, heads, head_dim, rows=None):
    """In place RMSNorm + complex RoPE on the ``[prefix:]`` rows of a dense ``[H, ROWS, D]`` buffer."""
    grid = (heads * triton.cdiv(seq, _NORM_ROWS),)
    _qk_norm_rope_hsd[grid](
        x,
        weight,
        rope.view(torch.float32),
        prefix,
        seq,
        prefix + seq if rows is None else rows,
        eps,
        D=head_dim,
        BM=_NORM_ROWS,
        num_warps=4,
    )
    return x
