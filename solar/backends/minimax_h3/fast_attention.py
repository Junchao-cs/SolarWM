"""Unmasked BF16 attention through cuDNN's fused SDPA kernel (an FA2 alternative).

Same math as FlashAttention-2 (softmax(QK^T / sqrt(d)) V with BF16 inputs, FP32
accumulation and an FP32 log-sum-exp); on Hopper cuDNN's kernels are ~1.6x faster
than FA2 at H3's shapes. Layout is FlashAttention's ``[B, S, H, D]``.

cuDNN's backward is not deterministic. When deterministic algorithms are enabled,
the backward runs FlashAttention-2's deterministic kernel on cuDNN's own output and
log-sum-exp (same precision, same algorithm family); the forward is cuDNN in both
modes (it is deterministic).

Switch: ``SOLARWM_H3_ATTENTION=cudnn`` (default: upstream FlashAttention-2).
"""

from __future__ import annotations

import os

import torch

_BACKEND = os.environ.get("SOLARWM_H3_ATTENTION", "flash").strip().lower()
if _BACKEND not in {"flash", "cudnn"}:
    raise ValueError(f"SOLARWM_H3_ATTENTION must be flash or cudnn, got {_BACKEND!r}")

COUNTERS = {"cudnn_forward": 0, "cudnn_backward": 0, "deterministic_backward": 0}


_FORCE_CUDNN = [0]


class cudnn_scope:
    """Context: cuDNN attention inside (with SOLARWM_H3_ATTENTION=cudnn) even outside the
    stages that SOLARWM_H3_CUDNN_STAGES lists. Wrap a forward and its backward together, so
    that activation-checkpoint recomputation runs the same kernel as the forward."""

    def __enter__(self):
        _FORCE_CUDNN[0] += 1
        return self

    def __exit__(self, *exc):
        _FORCE_CUDNN[0] -= 1
        return False


def cudnn_enabled() -> bool:
    from .stage_scope import cudnn_active

    if _BACKEND != "cudnn":
        return False
    return cudnn_active() or bool(_FORCE_CUDNN[0])


def _bhsd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(1, 2)


class _CudnnAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value):
        scale = query.shape[-1] ** -0.5
        q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (query, key, value))
        result = torch.ops.aten._scaled_dot_product_cudnn_attention(
            _bhsd(q), _bhsd(k), _bhsd(v), None, True, 0.0, False, False, scale=scale
        )
        out, lse = result[0], result[1]
        ctx.save_for_backward(q, k, v, out, lse, result[6], result[7])
        ctx.scale = scale
        ctx.max_q, ctx.max_k = result[4], result[5]
        COUNTERS["cudnn_forward"] += 1
        # cuDNN writes BSHD memory; the transpose is a free view.
        return _bhsd(out)

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out, lse, seed, offset = ctx.saved_tensors
        grad = grad.contiguous()
        if torch.are_deterministic_algorithms_enabled():
            from flash_attn import flash_attn_interface as fai

            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            fai._flash_attn_backward(
                grad,
                q,
                k,
                v,
                _bhsd(out),
                lse.contiguous(),
                dq,
                dk,
                dv,
                0.0,
                ctx.scale,
                False,
                -1,
                -1,
                0.0,
                None,
                True,
                None,
            )
            COUNTERS["deterministic_backward"] += 1
            return dq, dk, dv
        dq, dk, dv = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
            _bhsd(grad),
            _bhsd(q),
            _bhsd(k),
            _bhsd(v),
            out,
            lse,
            seed,
            offset,
            None,
            None,
            None,
            ctx.max_q,
            ctx.max_k,
            0.0,
            False,
            scale=ctx.scale,
        )
        COUNTERS["cudnn_backward"] += 1
        return _bhsd(dq), _bhsd(dk), _bhsd(dv)


def cudnn_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Unmasked non-causal attention on ``[B, S, H, D]`` BF16 tensors."""

    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("H3 cuDNN attention expects BF16 q/k/v")
    return _CudnnAttention.apply(query, key, value)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """FA2 ``flash_attn_func(q, k, v)`` or its cuDNN equivalent, by switch."""

    if cudnn_enabled():
        return cudnn_attention(query, key, value)
    from flash_attn import flash_attn_func

    return flash_attn_func(query, key, value, dropout_p=0.0, causal=False)


__all__ = ["COUNTERS", "attention", "cudnn_attention", "cudnn_enabled"]
