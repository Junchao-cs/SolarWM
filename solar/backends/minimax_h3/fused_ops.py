"""Fused Triton kernels for the H3 block's memory-bound elementwise chains.

Switch ``SOLARWM_H3_FUSED=1`` (default off: upstream's eager torch ops).

* ``rmsnorm_modulate(x, weight, scale, shift, idx)`` =
  ``norm(x) * (1 + scale[idx]) + shift[idx]`` with ``norm = nn.RMSNorm`` (torch's composite:
  FP32 statistics, BF16 result, BF16 weight multiply). Every BF16 rounding point of the eager
  chain is kept; only the order of the per-row mean of squares differs (FP32 sum), so results
  can differ from eager by a BF16 ulp on rare elements (tier N1).
* ``gated_residual(residual, gate, idx, update)`` = ``residual + gate[idx] * update``: same
  roundings as eager (bitwise).

``scale``/``shift``/``gate`` are AdaLN tables (timesteps x modalities rows) computed from the
frozen time embedding; they never require gradients here, which the autograd functions check.
In deterministic mode both kernels are deterministic (no atomics).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("SOLARWM_H3_FUSED", "0") == "1"
COUNTERS = {"rmsnorm_modulate": 0, "gated_residual": 0, "qk_norm_rope": 0, "fallback": 0}


def fused_enabled() -> bool:
    return _ENABLED


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rmsnorm_modulate_fwd(
    X, W, SCALE, SHIFT, IDX, OUT, RSTD, N, inv_n, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # torch: mean = sum * (1/N) in FP32 (MeanOps), then add_(eps), then rsqrt.
    mean_sq = tl.sum(x * x, axis=0) * inv_n
    rstd = tl.math.rsqrt(mean_sq + eps)
    normed = _bf16(x * rstd)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    h = _bf16(normed * w)
    t = tl.load(IDX + row)
    s = tl.load(SCALE + t * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(SHIFT + t * N + cols, mask=mask, other=0.0).to(tl.float32)
    one_plus = _bf16(1.0 + s)
    m = _bf16(h * one_plus)
    out = m + b
    tl.store(OUT + row * N + cols, out.to(tl.bfloat16), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rmsnorm_modulate_bwd(
    DOUT, X, W, SCALE, IDX, RSTD, DX, N, BLOCK: tl.constexpr
):
    # Eager autograd of: n = bf16(xf * r); h = bf16(n * w); m = bf16(h * (1+s)); o = m + b
    # dm = do; dh = bf16(dm * (1+s)); dn = bf16(dh * w); dxf = dn * r + xf * dr, with
    # r = rsqrt(mean(xf^2) + eps): dr = -0.5 r^3 * d(mean)  and  d(mean)/dxf = 2 xf / N.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    do = tl.load(DOUT + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(IDX + row)
    s = tl.load(SCALE + t * N + cols, mask=mask, other=0.0).to(tl.float32)
    one_plus = _bf16(1.0 + s)
    dh = _bf16(do * one_plus)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dn = _bf16(dh * w)
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    dot = tl.sum(dn * x, axis=0)
    dx = dn * rstd - x * (rstd * rstd * rstd) * dot / N
    tl.store(DX + row * N + cols, dx.to(tl.bfloat16), mask=mask)


@triton.jit
def _gated_residual_fwd(RES, GATE, IDX, UPD, OUT, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    t = tl.load(IDX + row)
    g = tl.load(GATE + t * N + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(UPD + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    out = r + _bf16(g * u)
    tl.store(OUT + row * N + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _gated_residual_bwd(DOUT, GATE, IDX, DUPD, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    t = tl.load(IDX + row)
    g = tl.load(GATE + t * N + cols, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DOUT + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(DUPD + row * N + cols, (do * g).to(tl.bfloat16), mask=mask)


def _inv(n: int) -> float:
    return float(torch.tensor(1.0 / n, dtype=torch.float32).item())


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


class _RMSNormModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, shift, idx, eps):
        n = x.shape[-1]
        x2 = _rows(x.contiguous())
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(n)
        _rmsnorm_modulate_fwd[(rows,)](
            x2, weight, scale, shift, idx, out, rstd, n, _inv(n), eps, BLOCK=block, num_warps=8
        )
        ctx.save_for_backward(x2, weight, scale, idx, rstd)
        ctx.shape = x.shape
        COUNTERS["rmsnorm_modulate"] += 1
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad):
        x2, weight, scale, idx, rstd = ctx.saved_tensors
        n = x2.shape[-1]
        g2 = _rows(grad.contiguous())
        dx = torch.empty_like(x2)
        block = triton.next_power_of_2(n)
        _rmsnorm_modulate_bwd[(x2.shape[0],)](
            g2, x2, weight, scale, idx, rstd, dx, n, BLOCK=block, num_warps=8
        )
        return dx.view(ctx.shape), None, None, None, None, None


class _GatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, gate, idx, update):
        n = residual.shape[-1]
        r2 = _rows(residual.contiguous())
        u2 = _rows(update.contiguous())
        out = torch.empty_like(r2)
        block = 1024
        grid = (r2.shape[0], triton.cdiv(n, block))
        _gated_residual_fwd[grid](r2, gate, idx, u2, out, n, BLOCK=block, num_warps=4)
        ctx.save_for_backward(gate, idx)
        ctx.shape = residual.shape
        COUNTERS["gated_residual"] += 1
        return out.view(residual.shape)

    @staticmethod
    def backward(ctx, grad):
        gate, idx = ctx.saved_tensors
        n = grad.shape[-1]
        g2 = _rows(grad.contiguous())
        du = torch.empty_like(g2)
        block = 1024
        grid = (g2.shape[0], triton.cdiv(n, block))
        _gated_residual_bwd[grid](g2, gate, idx, du, n, BLOCK=block, num_warps=4)
        # d residual = grad (identity); d update = bf16(grad * gate[idx]) as eager.
        return grad, None, None, du.view(ctx.shape)


def _usable(*tensors: torch.Tensor) -> bool:
    return all(t.is_cuda and t.dtype == torch.bfloat16 for t in tensors)


def rmsnorm_modulate(norm: torch.nn.Module, x, scale, shift, idx):
    """``norm(x) * (1 + scale[idx]) + shift[idx]`` for ``nn.RMSNorm`` ``norm``."""

    weight = norm.weight
    eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    if (
        not _usable(x, weight, scale, shift)
        or scale.requires_grad
        or shift.requires_grad
        or weight.requires_grad
        or x.shape[-1] > 8192
        or tuple(norm.normalized_shape) != (x.shape[-1],)
    ):
        COUNTERS["fallback"] += 1
        h = norm(x)
        return h * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)
    # torch's composite adds eps in the input dtype (BF16-rounded) to the FP32 mean.
    eps_value = float(torch.tensor(eps, dtype=x.dtype).item())
    return _RMSNormModulate.apply(x, weight, scale.contiguous(), shift.contiguous(), idx, eps_value)


def gated_residual(residual, gate, idx, update):
    """``residual + gate[idx] * update``."""

    if not _usable(residual, gate, update) or gate.requires_grad or residual.shape != update.shape:
        COUNTERS["fallback"] += 1
        return residual + gate.index_select(0, idx) * update
    return _GatedResidual.apply(residual, gate.contiguous(), idx, update)


# ----------------------------------------------------------------------- q/k norm + rotary
#
# ``_apply_rotary_emb(norm(x), cos, sin)`` for x [B, S, H, 128] (rotary on dims [0, R)):
# per (row, head) RMSNorm over 128 with BF16 weight, then rotate-half on the first R dims
# with BF16 cos/sin, keeping every BF16 rounding point of the eager chain.


@triton.jit
def _qk_norm_rope_fwd(
    X, W, COS, SIN, OUT, RSTD, n_vec, heads, inv_d, eps,
    D: tl.constexpr, R: tl.constexpr, HALF: tl.constexpr, VB: tl.constexpr
):
    pid = tl.program_id(0)
    vec = pid * VB + tl.arange(0, VB)
    vmask = vec < n_vec
    cols = tl.arange(0, D)
    offs = vec[:, None] * D + cols[None, :]
    x = tl.load(X + offs, mask=vmask[:, None], other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=1) * inv_d + eps)
    w = tl.load(W + cols).to(tl.float32)
    h = _bf16(_bf16(x * rstd[:, None]) * w[None, :])
    # partner column for rotate-half: d<HALF -> d+HALF (negated); HALF<=d<R -> d-HALF
    partner = tl.where(cols < HALF, cols + HALF, cols - HALF)
    hp = tl.load(X + vec[:, None] * D + partner[None, :], mask=vmask[:, None] & (cols < R)[None, :], other=0.0).to(tl.float32)
    hp = _bf16(_bf16(hp * rstd[:, None]) * tl.load(W + partner).to(tl.float32)[None, :])
    rot = tl.where((cols < HALF)[None, :], -hp, hp)
    row = vec // heads
    rmask = vmask[:, None] & (cols < R)[None, :]
    c = _bf16(tl.load(COS + row[:, None] * R + cols[None, :], mask=rmask, other=0.0))
    sn = _bf16(tl.load(SIN + row[:, None] * R + cols[None, :], mask=rmask, other=0.0))
    rotated = _bf16(_bf16(h * c) + _bf16(rot * sn))
    out = tl.where((cols < R)[None, :], rotated, h)
    tl.store(OUT + offs, out.to(tl.bfloat16), mask=vmask[:, None])
    tl.store(RSTD + vec, rstd, mask=vmask)


@triton.jit
def _qk_norm_rope_bwd(
    G, X, W, COS, SIN, RSTD, DX, n_vec, heads, inv_d,
    D: tl.constexpr, R: tl.constexpr, HALF: tl.constexpr, VB: tl.constexpr
):
    pid = tl.program_id(0)
    vec = pid * VB + tl.arange(0, VB)
    vmask = vec < n_vec
    cols = tl.arange(0, D)
    offs = vec[:, None] * D + cols[None, :]
    g = tl.load(G + offs, mask=vmask[:, None], other=0.0).to(tl.float32)
    partner = tl.where(cols < HALF, cols + HALF, cols - HALF)
    rmask = vmask[:, None] & (cols < R)[None, :]
    gp = tl.load(G + vec[:, None] * D + partner[None, :], mask=rmask, other=0.0).to(tl.float32)
    row = vec // heads
    c = _bf16(tl.load(COS + row[:, None] * R + cols[None, :], mask=rmask, other=0.0))
    sp = _bf16(tl.load(SIN + row[:, None] * R + partner[None, :], mask=rmask, other=0.0))
    from_cos = _bf16(g * c)
    from_sin = _bf16(gp * sp)
    # d<HALF: dx1 contribution = +bf16(g[d+HALF]*s[d+HALF]); HALF<=d<R: -bf16(g[d-HALF]*s[d-HALF])
    from_sin = tl.where((cols < HALF)[None, :], from_sin, -from_sin)
    dh = tl.where((cols < R)[None, :], _bf16(from_cos + from_sin), g)
    w = tl.load(W + cols).to(tl.float32)
    dn = _bf16(dh * w[None, :])
    x = tl.load(X + offs, mask=vmask[:, None], other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + vec, mask=vmask, other=0.0)
    dot = tl.sum(dn * x, axis=1)
    dx = dn * rstd[:, None] - x * (rstd * rstd * rstd * dot * inv_d)[:, None]
    tl.store(DX + offs, dx.to(tl.bfloat16), mask=vmask[:, None])


class _QKNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cos, sin, eps):
        b, s, h, d = x.shape
        r = cos.shape[-1]
        xc = x.contiguous()
        out = torch.empty_like(xc)
        n_vec = b * s * h
        rstd = torch.empty(n_vec, device=x.device, dtype=torch.float32)
        vb = 16
        _qk_norm_rope_fwd[(triton.cdiv(n_vec, vb),)](
            xc, weight, cos, sin, out, rstd, n_vec, h, _inv(d), eps,
            D=d, R=r, HALF=r // 2, VB=vb, num_warps=4,
        )
        ctx.save_for_backward(xc, weight, cos, sin, rstd)
        COUNTERS["qk_norm_rope"] += 1
        return out

    @staticmethod
    def backward(ctx, grad):
        xc, weight, cos, sin, rstd = ctx.saved_tensors
        b, s, h, d = xc.shape
        r = cos.shape[-1]
        g = grad.contiguous()
        dx = torch.empty_like(xc)
        n_vec = b * s * h
        vb = 16
        _qk_norm_rope_bwd[(triton.cdiv(n_vec, vb),)](
            g, xc, weight, cos, sin, rstd, dx, n_vec, h, _inv(d),
            D=d, R=r, HALF=r // 2, VB=vb, num_warps=4,
        )
        return dx, None, None, None, None


def qk_norm_rope(norm: torch.nn.Module, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """``_apply_rotary_emb(norm(x), cos, sin)`` for ``x`` [B, S, H, D] and FP32 cos/sin [S, R]."""

    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

    weight = norm.weight
    eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    if (
        not _usable(x, weight)
        or weight.requires_grad
        or x.ndim != 4
        or x.shape[0] != 1
        or x.shape[-1] != 128
        or cos.shape != sin.shape
        or cos.ndim != 2
        or cos.shape[0] != x.shape[1]
        or cos.shape[-1] % 2
        or cos.shape[-1] > x.shape[-1]
        or cos.dtype != torch.float32
        or cos.requires_grad
        or sin.requires_grad
    ):
        COUNTERS["fallback"] += 1
        return _apply_rotary_emb(norm(x), cos, sin)
    eps_value = float(torch.tensor(eps, dtype=x.dtype).item())
    return _QKNormRope.apply(x, weight, cos.contiguous(), sin.contiguous(), eps_value)


def qk_norm(norm: torch.nn.Module, x: torch.Tensor):
    """``norm(x)`` (q/k RMSNorm over head_dim 128) with the fused kernel and no rotary
    channels (the SGF path rotates after its all-to-all, per attention segment)."""

    weight = norm.weight
    eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    if (
        not _usable(x, weight)
        or weight.requires_grad
        or x.ndim != 4
        or x.shape[0] != 1
        or x.shape[-1] != 128
    ):
        COUNTERS["fallback"] += 1
        return norm(x)
    eps_value = float(torch.tensor(eps, dtype=x.dtype).item())
    return _QKNorm.apply(x, weight, eps_value)


class _QKNorm(torch.autograd.Function):
    """The q/k norm kernels with zero rotary channels (R = 0: every cos/sin load is masked
    off, so ``x`` itself stands in as the never-read table pointer)."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        b, s, h, d = x.shape
        xc = x.contiguous()
        out = torch.empty_like(xc)
        n_vec = b * s * h
        rstd = torch.empty(n_vec, device=x.device, dtype=torch.float32)
        vb = 16
        _qk_norm_rope_fwd[(triton.cdiv(n_vec, vb),)](
            xc, weight, xc, xc, out, rstd, n_vec, h, _inv(d), eps,
            D=d, R=0, HALF=0, VB=vb, num_warps=4,
        )
        ctx.save_for_backward(xc, weight, rstd)
        COUNTERS["qk_norm"] = COUNTERS.get("qk_norm", 0) + 1
        return out

    @staticmethod
    def backward(ctx, grad):
        xc, weight, rstd = ctx.saved_tensors
        b, s, h, d = xc.shape
        g = grad.contiguous()
        dx = torch.empty_like(xc)
        n_vec = b * s * h
        vb = 16
        _qk_norm_rope_bwd[(triton.cdiv(n_vec, vb),)](
            g, xc, weight, xc, xc, rstd, dx, n_vec, h, _inv(d),
            D=d, R=0, HALF=0, VB=vb, num_warps=4,
        )
        return dx, None, None


__all__ = ["COUNTERS", "fused_enabled", "gated_residual", "qk_norm", "qk_norm_rope", "rmsnorm_modulate"]


# ----------------------------------------------------------------------------------- SwiGLU
#
# diffusers ``SwiGLU``: ``h, g = proj(x).chunk(2, -1); return h * silu(g)``. One kernel reads the
# projection output once and writes the product (eager: silu pass, mul pass, plus the chunk's cat
# in backward). Same arithmetic as torch's kernels: silu(g) = g / (1 + exp(-g)) in FP32 with the
# accurate exp and IEEE division, rounded to BF16, then the BF16 product; backward d_h = bf16(dy*s),
# d_s = bf16(dy*h), d_g = bf16(d_s * sig * (1 + g * (1 - sig))) written into one [M, 2N] buffer.
# Switch ``SOLARWM_H3_SWIGLU_FUSED=1``.

from triton.language.extra.cuda import libdevice as _libdevice  # noqa: E402

_SWIGLU = os.environ.get("SOLARWM_H3_SWIGLU_FUSED", "0") == "1"


@triton.jit
def _swiglu_fwd(H, OUT, M, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    h = tl.load(H + row * 2 * N + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(H + row * 2 * N + N + cols, mask=mask, other=0.0).to(tl.float32)
    s = _bf16(_libdevice.div_rn(g, 1.0 + _libdevice.exp(-g)))
    tl.store(OUT + row * N + cols, (h * s).to(tl.bfloat16), mask=mask)


@triton.jit
def _swiglu_bwd(DY, H, DH, M, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    dy = tl.load(DY + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(H + row * 2 * N + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(H + row * 2 * N + N + cols, mask=mask, other=0.0).to(tl.float32)
    e = _libdevice.exp(-g)
    s = _bf16(_libdevice.div_rn(g, 1.0 + e))
    d_h = dy * s
    d_s = _bf16(dy * h)
    sig = _libdevice.div_rn(1.0, 1.0 + e)
    d_g = d_s * sig * (1.0 + g * (1.0 - sig))
    tl.store(DH + row * 2 * N + cols, d_h.to(tl.bfloat16), mask=mask)
    tl.store(DH + row * 2 * N + N + cols, d_g.to(tl.bfloat16), mask=mask)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected):
        hc = projected.contiguous()
        n = hc.shape[-1] // 2
        m = hc.numel() // (2 * n)
        out = torch.empty((*hc.shape[:-1], n), device=hc.device, dtype=hc.dtype)
        block = 1024
        _swiglu_fwd[(m, triton.cdiv(n, block))](hc, out, m, n, BLOCK=block, num_warps=4)
        ctx.save_for_backward(hc)
        COUNTERS["swiglu"] = COUNTERS.get("swiglu", 0) + 1
        return out

    @staticmethod
    def backward(ctx, grad):
        (hc,) = ctx.saved_tensors
        g = grad.contiguous()
        n = hc.shape[-1] // 2
        m = hc.numel() // (2 * n)
        dh = torch.empty_like(hc)
        block = 1024
        _swiglu_bwd[(m, triton.cdiv(n, block))](g, hc, dh, m, n, BLOCK=block, num_warps=4)
        return dh


def swiglu(projected: torch.Tensor) -> torch.Tensor:
    if not projected.is_cuda or projected.dtype != torch.bfloat16 or projected.shape[-1] % 2:
        hidden, gate = projected.chunk(2, dim=-1)
        COUNTERS["fallback"] += 1
        return hidden * torch.nn.functional.silu(gate)
    return _SwiGLU.apply(projected)


def _fused_swiglu_forward(self, hidden_states):
    return swiglu(self.proj(hidden_states))


if _SWIGLU:
    from diffusers.models.activations import SwiGLU as _DiffusersSwiGLU

    _DiffusersSwiGLU.forward = _fused_swiglu_forward
