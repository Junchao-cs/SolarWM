"""Fused camera-PRoPE projection of the head dimensions ``[96:128)``.

Switch: ``SOLARWM_H3_PROPE_FUSED=1``. Upstream (``torch_prope``) computes
``cat(x[..., :96], einsum('bsij,bhspj->bhspi', M, x[..., 96:].view(..., 8, 4)))``:
a slice copy, permute copies, a batched 4x4 GEMM and a full-width cat, each a pass over the
tensor. This kernel reads every row once and writes the result once: the first 96 channels
are copied, each of the eight 4-vectors of the last 32 channels is multiplied by the row's
4x4 matrix with FP32 accumulation and rounded to BF16 (the einsum's precision; the order
of the four-term sum may differ from the tensor-core kernel in the last FP32 bit, tier N1).

The backward is the same kernel with the transposed matrices (the matrices come from the
camera metadata and never require gradients). No atomics: deterministic.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("SOLARWM_H3_PROPE_FUSED", "0") == "1"
COUNTERS = {"fused": 0, "fused_backward": 0}


def prope_fused_enabled() -> bool:
    return _ENABLED


@triton.jit
def _prope_kernel(
    X,
    M,
    OUT,
    S,
    H,
    sxb,
    sxh,
    sxs,
    smb,
    sms,
    sob,
    soh,
    sos,
    TRANSPOSE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rmask = rows < S
    x_row = X + b * sxb + h * sxh + rows * sxs
    o_row = OUT + b * sob + h * soh + rows * sos
    # channels [0, 96): copied (64 + 32)
    c64 = tl.arange(0, 64)
    v = tl.load(x_row[:, None] + c64[None, :], mask=rmask[:, None], other=0.0)
    tl.store(o_row[:, None] + c64[None, :], v, mask=rmask[:, None])
    c32 = tl.arange(0, 32)
    v = tl.load(x_row[:, None] + 64 + c32[None, :], mask=rmask[:, None], other=0.0)
    tl.store(o_row[:, None] + 64 + c32[None, :], v, mask=rmask[:, None])
    # channels [96, 128): eight 4-vectors per row, y_i = sum_j M[i, j] x_j
    p = tl.arange(0, 8)
    base = x_row[:, None] + 96 + 4 * p[None, :]
    x0 = tl.load(base + 0, mask=rmask[:, None], other=0.0).to(tl.float32)
    x1 = tl.load(base + 1, mask=rmask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(base + 2, mask=rmask[:, None], other=0.0).to(tl.float32)
    x3 = tl.load(base + 3, mask=rmask[:, None], other=0.0).to(tl.float32)
    m_row = M + b * smb + rows * sms
    obase = o_row[:, None] + 96 + 4 * p[None, :]
    for i in tl.static_range(4):
        if TRANSPOSE:
            m0 = tl.load(m_row + 0 * 4 + i, mask=rmask, other=0.0).to(tl.float32)
            m1 = tl.load(m_row + 1 * 4 + i, mask=rmask, other=0.0).to(tl.float32)
            m2 = tl.load(m_row + 2 * 4 + i, mask=rmask, other=0.0).to(tl.float32)
            m3 = tl.load(m_row + 3 * 4 + i, mask=rmask, other=0.0).to(tl.float32)
        else:
            m0 = tl.load(m_row + i * 4 + 0, mask=rmask, other=0.0).to(tl.float32)
            m1 = tl.load(m_row + i * 4 + 1, mask=rmask, other=0.0).to(tl.float32)
            m2 = tl.load(m_row + i * 4 + 2, mask=rmask, other=0.0).to(tl.float32)
            m3 = tl.load(m_row + i * 4 + 3, mask=rmask, other=0.0).to(tl.float32)
        y = m0[:, None] * x0 + m1[:, None] * x1 + m2[:, None] * x2 + m3[:, None] * x3
        tl.store(obase + i, y.to(OUT.dtype.element_ty), mask=rmask[:, None])


def _launch(
    x: torch.Tensor, matrix: torch.Tensor, transpose: bool, out: torch.Tensor | None = None
):
    """``x`` is ``[B, H, S, 128]`` (unit channel stride), ``matrix`` ``[B|1, S, 4, 4]``."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    batch, heads, sequence, width = x.shape
    if width != 128:
        raise ValueError("fused PRoPE expects head_dim 128")
    if matrix.dtype != x.dtype:
        matrix = matrix.to(x.dtype)
    if matrix.stride(-1) != 1 or matrix.stride(-2) != 4:
        matrix = matrix.contiguous()
    if tuple(matrix.shape[-3:]) != (sequence, 4, 4) or matrix.shape[0] not in (1, batch):
        raise ValueError("fused PRoPE matrices are not aligned with the attention rows")
    if out is None:
        # BSHD memory, returned as the BHSD view (the layout torch_prope's cat produces after
        # the callers' transposes is irrelevant to values; attention takes either).
        out = torch.empty(
            (batch, sequence, heads, width), device=x.device, dtype=x.dtype
        ).transpose(1, 2)
    smb = 0 if matrix.shape[0] == 1 else matrix.stride(0)
    block = 32
    grid = (triton.cdiv(sequence, block), batch * heads)
    kernel = _prope_kernel_v2 if _V2 else _prope_kernel
    if _V2:
        block = 64
        grid = (triton.cdiv(sequence, block), batch * heads)
    kernel[grid](
        x,
        matrix,
        out,
        sequence,
        heads,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        smb,
        matrix.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        TRANSPOSE=transpose,
        BLOCK_S=block,
        num_warps=4,
    )
    return out


class _FusedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, matrix):
        ctx.save_for_backward(matrix)
        COUNTERS["fused"] += 1
        return _launch(x, matrix, False)

    @staticmethod
    def backward(ctx, grad):
        (matrix,) = ctx.saved_tensors
        COUNTERS["fused_backward"] += 1
        return _launch(grad, matrix, True), None


def prope_project(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """``cat(x[..., :96], _project(x[..., 96:128], matrix))`` for ``x`` ``[B, H, S, 128]``."""
    if matrix.requires_grad:
        raise RuntimeError("fused PRoPE matrices must not require gradients")
    return _FusedProjection.apply(x, matrix.detach())


# ------------------------------------------------------------------ SGF rotary + PRoPE in one pass
#
# ``_suffix(_apply_rotary_emb(x, cos, sin).transpose(1, 2), project(M)).transpose(1, 2)`` for
# ``x`` [B, S, H, 128] (BSHD): rotate-half MM-RoPE on channels [0, 96) with BF16 cos/sin,
# keeping every BF16 rounding of the eager chain (bitwise), and the camera projection of
# channels [96, 128) (as ``prope_project``). Rotary and PRoPE touch disjoint channels.


@triton.jit
def _bf16r(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rope_prope_kernel(
    X,
    COS,
    SIN,
    M,
    OUT,
    S,
    H,
    sxb,
    sxs,
    sxh,
    scs,
    smb,
    sms,
    sob,
    sos,
    soh,
    BACKWARD: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rmask = rows < S
    x_row = X + b * sxb + h * sxh + rows * sxs
    o_row = OUT + b * sob + h * soh + rows * sos
    c_row = rows * scs
    # rotary: channels [0, 48) and [48, 96)
    c48 = tl.arange(0, 64)
    cm = rmask[:, None] & (c48 < 48)[None, :]
    x1 = tl.load(x_row[:, None] + c48[None, :], mask=cm, other=0.0).to(tl.float32)
    x2 = tl.load(x_row[:, None] + 48 + c48[None, :], mask=cm, other=0.0).to(tl.float32)
    cos1 = _bf16r(tl.load(COS + c_row[:, None] + c48[None, :], mask=cm, other=0.0).to(tl.float32))
    cos2 = _bf16r(
        tl.load(COS + c_row[:, None] + 48 + c48[None, :], mask=cm, other=0.0).to(tl.float32)
    )
    sin1 = _bf16r(tl.load(SIN + c_row[:, None] + c48[None, :], mask=cm, other=0.0).to(tl.float32))
    sin2 = _bf16r(
        tl.load(SIN + c_row[:, None] + 48 + c48[None, :], mask=cm, other=0.0).to(tl.float32)
    )
    if BACKWARD:
        # y1 = x1*c1 - x2*s1 ; y2 = x2*c2 + x1*s2  =>  dx1 = g1*c1 + g2*s2 ; dx2 = g2*c2 - g1*s1
        y1 = _bf16r(_bf16r(x1 * cos1) + _bf16r(x2 * sin2))
        y2 = _bf16r(_bf16r(x2 * cos2) - _bf16r(x1 * sin1))
    else:
        y1 = _bf16r(_bf16r(x1 * cos1) + _bf16r(-x2 * sin1))
        y2 = _bf16r(_bf16r(x2 * cos2) + _bf16r(x1 * sin2))
    tl.store(o_row[:, None] + c48[None, :], y1.to(OUT.dtype.element_ty), mask=cm)
    tl.store(o_row[:, None] + 48 + c48[None, :], y2.to(OUT.dtype.element_ty), mask=cm)
    # camera projection: channels [96, 128)
    p = tl.arange(0, 8)
    base = x_row[:, None] + 96 + 4 * p[None, :]
    v0 = tl.load(base + 0, mask=rmask[:, None], other=0.0).to(tl.float32)
    v1 = tl.load(base + 1, mask=rmask[:, None], other=0.0).to(tl.float32)
    v2 = tl.load(base + 2, mask=rmask[:, None], other=0.0).to(tl.float32)
    v3 = tl.load(base + 3, mask=rmask[:, None], other=0.0).to(tl.float32)
    m_row = M + b * smb + rows * sms
    obase = o_row[:, None] + 96 + 4 * p[None, :]
    for i in tl.static_range(4):
        if BACKWARD:
            m0 = tl.load(m_row + 0 * 4 + i, mask=rmask, other=0.0).to(tl.float32)
            m1 = tl.load(m_row + 1 * 4 + i, mask=rmask, other=0.0).to(tl.float32)
            m2 = tl.load(m_row + 2 * 4 + i, mask=rmask, other=0.0).to(tl.float32)
            m3 = tl.load(m_row + 3 * 4 + i, mask=rmask, other=0.0).to(tl.float32)
        else:
            m0 = tl.load(m_row + i * 4 + 0, mask=rmask, other=0.0).to(tl.float32)
            m1 = tl.load(m_row + i * 4 + 1, mask=rmask, other=0.0).to(tl.float32)
            m2 = tl.load(m_row + i * 4 + 2, mask=rmask, other=0.0).to(tl.float32)
            m3 = tl.load(m_row + i * 4 + 3, mask=rmask, other=0.0).to(tl.float32)
        y = m0[:, None] * v0 + m1[:, None] * v1 + m2[:, None] * v2 + m3[:, None] * v3
        tl.store(obase + i, y.to(OUT.dtype.element_ty), mask=rmask[:, None])


def _rope_prope_launch(x, cos, sin, matrix, backward, out=None):
    """``x`` [B, S, H, 128] (unit channel stride), cos/sin [S, 96], matrix [B|1, S, 4, 4]."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    batch, sequence, heads, width = x.shape
    if width != 128 or tuple(cos.shape) != (sequence, 96) or cos.shape != sin.shape:
        raise ValueError("SGF rope+PRoPE expects head_dim 128 and [S, 96] cos/sin")
    cos = cos if cos.stride(-1) == 1 else cos.contiguous()
    sin = sin if sin.stride() == cos.stride() else sin.contiguous()
    if sin.stride() != cos.stride():
        cos, sin = cos.contiguous(), sin.contiguous()
    if matrix.dtype != x.dtype:
        matrix = matrix.to(x.dtype)
    if matrix.stride(-1) != 1 or matrix.stride(-2) != 4:
        matrix = matrix.contiguous()
    if tuple(matrix.shape[-3:]) != (sequence, 4, 4) or matrix.shape[0] not in (1, batch):
        raise ValueError("SGF PRoPE matrices are not aligned with the rows")
    if out is None:
        out = torch.empty((batch, sequence, heads, width), device=x.device, dtype=x.dtype)
    smb = 0 if matrix.shape[0] == 1 else matrix.stride(0)
    block = 32
    grid = (triton.cdiv(sequence, block), batch * heads)
    kernel = _rope_prope_kernel_v2 if _V2 else _rope_prope_kernel
    if _V2:
        block = 64
        grid = (triton.cdiv(sequence, block), batch * heads)
    kernel[grid](
        x,
        cos,
        sin,
        matrix,
        out,
        sequence,
        heads,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos.stride(0),
        smb,
        matrix.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BACKWARD=backward,
        BLOCK_S=block,
        num_warps=4,
    )
    return out


class _RopeProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, matrix):
        ctx.save_for_backward(cos, sin, matrix)
        COUNTERS["rope_prope"] = COUNTERS.get("rope_prope", 0) + 1
        return _rope_prope_launch(x, cos, sin, matrix, False)

    @staticmethod
    def backward(ctx, grad):
        cos, sin, matrix = ctx.saved_tensors
        COUNTERS["rope_prope_backward"] = COUNTERS.get("rope_prope_backward", 0) + 1
        return _rope_prope_launch(grad, cos, sin, matrix, True), None, None, None


def rope_prope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, matrix: torch.Tensor, out=None
):
    """BSHD ``x`` -> BSHD result of MM-RoPE on [0, 96) and the camera projection on [96, 128)."""
    if cos.requires_grad or sin.requires_grad or matrix.requires_grad:
        raise RuntimeError("SGF rope+PRoPE tables must not require gradients")
    if out is not None:
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("rope_prope(out=...) is for detached evaluations")
        COUNTERS["rope_prope"] = COUNTERS.get("rope_prope", 0) + 1
        return _rope_prope_launch(x, cos, sin, matrix, False, out=out)
    return _RopeProject.apply(x, cos, sin, matrix)


# ------------------------------------------------------ coalesced variants (SOLARWM_H3_PROPE_V2=1)
#
# Same arithmetic as the kernels above (the same four-term FP32 expression per output, so the
# results are bitwise equal to them), but the 32 camera channels of a row are loaded and stored
# as one contiguous tile and split into the four vector components in registers (tl.split /
# tl.join), instead of four stride-4 gathers and scatters of 2-byte elements.

_V2 = os.environ.get("SOLARWM_H3_PROPE_V2", "0") == "1"


@triton.jit
def _project32(t, m_row, rmask, TRANSPOSE: tl.constexpr, BLOCK_S: tl.constexpr):
    t4 = tl.reshape(t, [BLOCK_S, 8, 2, 2])
    a, b = tl.split(t4)
    x0, x2 = tl.split(a)
    x1, x3 = tl.split(b)
    if TRANSPOSE:
        m00 = tl.load(m_row + 0, mask=rmask, other=0.0).to(tl.float32)
        m01 = tl.load(m_row + 4, mask=rmask, other=0.0).to(tl.float32)
        m02 = tl.load(m_row + 8, mask=rmask, other=0.0).to(tl.float32)
        m03 = tl.load(m_row + 12, mask=rmask, other=0.0).to(tl.float32)
        m10 = tl.load(m_row + 1, mask=rmask, other=0.0).to(tl.float32)
        m11 = tl.load(m_row + 5, mask=rmask, other=0.0).to(tl.float32)
        m12 = tl.load(m_row + 9, mask=rmask, other=0.0).to(tl.float32)
        m13 = tl.load(m_row + 13, mask=rmask, other=0.0).to(tl.float32)
        m20 = tl.load(m_row + 2, mask=rmask, other=0.0).to(tl.float32)
        m21 = tl.load(m_row + 6, mask=rmask, other=0.0).to(tl.float32)
        m22 = tl.load(m_row + 10, mask=rmask, other=0.0).to(tl.float32)
        m23 = tl.load(m_row + 14, mask=rmask, other=0.0).to(tl.float32)
        m30 = tl.load(m_row + 3, mask=rmask, other=0.0).to(tl.float32)
        m31 = tl.load(m_row + 7, mask=rmask, other=0.0).to(tl.float32)
        m32 = tl.load(m_row + 11, mask=rmask, other=0.0).to(tl.float32)
        m33 = tl.load(m_row + 15, mask=rmask, other=0.0).to(tl.float32)
    else:
        m00 = tl.load(m_row + 0, mask=rmask, other=0.0).to(tl.float32)
        m01 = tl.load(m_row + 1, mask=rmask, other=0.0).to(tl.float32)
        m02 = tl.load(m_row + 2, mask=rmask, other=0.0).to(tl.float32)
        m03 = tl.load(m_row + 3, mask=rmask, other=0.0).to(tl.float32)
        m10 = tl.load(m_row + 4, mask=rmask, other=0.0).to(tl.float32)
        m11 = tl.load(m_row + 5, mask=rmask, other=0.0).to(tl.float32)
        m12 = tl.load(m_row + 6, mask=rmask, other=0.0).to(tl.float32)
        m13 = tl.load(m_row + 7, mask=rmask, other=0.0).to(tl.float32)
        m20 = tl.load(m_row + 8, mask=rmask, other=0.0).to(tl.float32)
        m21 = tl.load(m_row + 9, mask=rmask, other=0.0).to(tl.float32)
        m22 = tl.load(m_row + 10, mask=rmask, other=0.0).to(tl.float32)
        m23 = tl.load(m_row + 11, mask=rmask, other=0.0).to(tl.float32)
        m30 = tl.load(m_row + 12, mask=rmask, other=0.0).to(tl.float32)
        m31 = tl.load(m_row + 13, mask=rmask, other=0.0).to(tl.float32)
        m32 = tl.load(m_row + 14, mask=rmask, other=0.0).to(tl.float32)
        m33 = tl.load(m_row + 15, mask=rmask, other=0.0).to(tl.float32)
    y0 = m00[:, None] * x0 + m01[:, None] * x1 + m02[:, None] * x2 + m03[:, None] * x3
    y1 = m10[:, None] * x0 + m11[:, None] * x1 + m12[:, None] * x2 + m13[:, None] * x3
    y2 = m20[:, None] * x0 + m21[:, None] * x1 + m22[:, None] * x2 + m23[:, None] * x3
    y3 = m30[:, None] * x0 + m31[:, None] * x1 + m32[:, None] * x2 + m33[:, None] * x3
    out = tl.join(tl.join(y0, y2), tl.join(y1, y3))
    return tl.reshape(out, [BLOCK_S, 32])


@triton.jit
def _prope_kernel_v2(
    X,
    M,
    OUT,
    S,
    H,
    sxb,
    sxh,
    sxs,
    smb,
    sms,
    sob,
    soh,
    sos,
    TRANSPOSE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rmask = rows < S
    x_row = X + b * sxb + h * sxh + rows * sxs
    o_row = OUT + b * sob + h * soh + rows * sos
    c96 = tl.arange(0, 128)
    pmask = rmask[:, None] & (c96 < 96)[None, :]
    v = tl.load(x_row[:, None] + c96[None, :], mask=pmask, other=0.0)
    tl.store(o_row[:, None] + c96[None, :], v, mask=pmask)
    c32 = tl.arange(0, 32)
    t = tl.load(x_row[:, None] + 96 + c32[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    y = _project32(t, M + b * smb + rows * sms, rmask, TRANSPOSE, BLOCK_S)
    tl.store(o_row[:, None] + 96 + c32[None, :], y.to(OUT.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _rope_prope_kernel_v2(
    X,
    COS,
    SIN,
    M,
    OUT,
    S,
    H,
    sxb,
    sxs,
    sxh,
    scs,
    smb,
    sms,
    sob,
    sos,
    soh,
    BACKWARD: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rmask = rows < S
    x_row = X + b * sxb + h * sxh + rows * sxs
    o_row = OUT + b * sob + h * soh + rows * sos
    c_row = rows * scs
    c48 = tl.arange(0, 64)
    cm = rmask[:, None] & (c48 < 48)[None, :]
    x1 = tl.load(x_row[:, None] + c48[None, :], mask=cm, other=0.0).to(tl.float32)
    x2 = tl.load(x_row[:, None] + 48 + c48[None, :], mask=cm, other=0.0).to(tl.float32)
    cos1 = _bf16r(tl.load(COS + c_row[:, None] + c48[None, :], mask=cm, other=0.0).to(tl.float32))
    cos2 = _bf16r(
        tl.load(COS + c_row[:, None] + 48 + c48[None, :], mask=cm, other=0.0).to(tl.float32)
    )
    sin1 = _bf16r(tl.load(SIN + c_row[:, None] + c48[None, :], mask=cm, other=0.0).to(tl.float32))
    sin2 = _bf16r(
        tl.load(SIN + c_row[:, None] + 48 + c48[None, :], mask=cm, other=0.0).to(tl.float32)
    )
    if BACKWARD:
        y1 = _bf16r(_bf16r(x1 * cos1) + _bf16r(x2 * sin2))
        y2 = _bf16r(_bf16r(x2 * cos2) - _bf16r(x1 * sin1))
    else:
        y1 = _bf16r(_bf16r(x1 * cos1) + _bf16r(-x2 * sin1))
        y2 = _bf16r(_bf16r(x2 * cos2) + _bf16r(x1 * sin2))
    tl.store(o_row[:, None] + c48[None, :], y1.to(OUT.dtype.element_ty), mask=cm)
    tl.store(o_row[:, None] + 48 + c48[None, :], y2.to(OUT.dtype.element_ty), mask=cm)
    c32 = tl.arange(0, 32)
    t = tl.load(x_row[:, None] + 96 + c32[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    y = _project32(t, M + b * smb + rows * sms, rmask, BACKWARD, BLOCK_S)
    tl.store(o_row[:, None] + 96 + c32[None, :], y.to(OUT.dtype.element_ty), mask=rmask[:, None])


__all__ = ["COUNTERS", "prope_fused_enabled", "prope_project", "rope_prope"]
