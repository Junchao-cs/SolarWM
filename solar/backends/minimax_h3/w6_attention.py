"""Stage1 W6 attention as dense cuDNN attention per query run (switch ``SOLARWM_H3_W6=segments``).

The W6 visibility of a canonical ``[condition | audio | clean chunks | noisy chunks]``
document is constant over each (role, chunk) run of query rows, and the visible keys of a
run are a union of a few contiguous key spans. Instead of one FlexAttention kernel over
the 27%-dense block mask, each query run attends to exactly its visible keys with the
fused cuDNN kernel (see :mod:`fast_attention`): the same softmax(QK^T/sqrt(d))V over the
same visible pairs, BF16 inputs, FP32 accumulation.

Backward runs per run as well; key/value gradients of runs that share keys are summed in
FP32 and rounded to BF16 once. In deterministic mode the per-run backward is
FlashAttention-2's deterministic kernel on cuDNN's output and log-sum-exp.
"""

from __future__ import annotations

import os

import torch

from .layout import H3PackedLayout
from .mask import stage1_window_allows

_MODE = os.environ.get("SOLARWM_H3_W6", "flex").strip().lower()
if _MODE not in {"flex", "segments"}:
    raise ValueError(f"SOLARWM_H3_W6 must be flex or segments, got {_MODE!r}")

COUNTERS = {"forward": 0, "backward": 0}


def segments_enabled() -> bool:
    return _MODE == "segments"


class H3W6Plan:
    """Query runs ``(start, stop, key_spans)`` tiling one packed document."""

    __slots__ = ("segments", "sequence_length")

    def __init__(self, segments, sequence_length: int) -> None:
        self.segments = tuple(
            (int(a), int(b), tuple((int(s), int(e)) for s, e in spans)) for a, b, spans in segments
        )
        self.sequence_length = int(sequence_length)
        end = 0
        for start, stop, spans in self.segments:
            if start != end or stop <= start or not spans:
                raise ValueError("H3 W6 plan runs must tile the document and see keys")
            for s, e in spans:
                if not 0 <= s < e <= self.sequence_length:
                    raise ValueError("H3 W6 plan key span outside the document")
            end = stop
        if end != self.sequence_length:
            raise ValueError("H3 W6 plan must cover the whole document")


def build_w6_plan(layout: H3PackedLayout) -> H3W6Plan:
    if layout.stage != "stage1" or layout.window_chunks is None:
        raise ValueError("W6 plans require a Stage1 layout")
    roles = layout.row_roles.detach().to("cpu", torch.long)
    chunks = layout.target_video_chunk_ids.detach().to("cpu", torch.long)
    total = int(layout.sequence_length)
    changes = ((roles[1:] != roles[:-1]) | (chunks[1:] != chunks[:-1])).nonzero().flatten() + 1
    starts = [0, *changes.tolist()]
    ends = [*changes.tolist(), total]
    runs = [(s, e, int(roles[s]), int(chunks[s])) for s, e in zip(starts, ends, strict=True)]
    window = int(layout.window_chunks)
    segments = []
    for qs, qe, qrole, qchunk in runs:
        spans: list[tuple[int, int]] = []
        for ks, ke, krole, kchunk in runs:
            if stage1_window_allows(qrole, qchunk, krole, kchunk, window_chunks=window):
                if spans and spans[-1][1] == ks:
                    spans[-1] = (spans[-1][0], ke)
                else:
                    spans.append((ks, ke))
        segments.append((qs, qe, spans))
    return H3W6Plan(segments, total)


def _keys(tensor: torch.Tensor, spans) -> torch.Tensor:
    if len(spans) == 1:
        start, stop = spans[0]
        return tensor[:, start:stop]
    return torch.cat([tensor[:, start:stop] for start, stop in spans], dim=1)


def _bhsd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(1, 2)


class _W6Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, plan):
        scale = query.shape[-1] ** -0.5
        query, key, value = (t.contiguous() for t in (query, key, value))
        out = torch.empty_like(query)
        saved = []
        for start, stop, spans in plan.segments:
            result = torch.ops.aten._scaled_dot_product_cudnn_attention(
                _bhsd(query[:, start:stop]),
                _bhsd(_keys(key, spans)),
                _bhsd(_keys(value, spans)),
                None,
                True,
                0.0,
                False,
                False,
                scale=scale,
            )
            out[:, start:stop].copy_(_bhsd(result[0]))
            saved.append((result[1], result[4], result[5], result[6], result[7]))
        ctx.plan = plan
        ctx.scale = scale
        ctx.meta = [(item[1], item[2]) for item in saved]
        ctx.save_for_backward(
            query, key, value, out, *(t for item in saved for t in (item[0], item[3], item[4]))
        )
        COUNTERS["forward"] += 1
        return out

    @staticmethod
    def backward(ctx, grad):
        query, key, value, out, *rest = ctx.saved_tensors
        grad = grad.contiguous()
        deterministic = torch.are_deterministic_algorithms_enabled()
        dq = torch.empty_like(query)
        dk = torch.zeros(key.shape, device=key.device, dtype=torch.float32)
        dv = torch.zeros(value.shape, device=value.device, dtype=torch.float32)
        for index, (start, stop, spans) in enumerate(ctx.plan.segments):
            lse, seed, offset = rest[3 * index : 3 * index + 3]
            max_q, max_k = ctx.meta[index]
            q_seg, g_seg, o_seg = query[:, start:stop], grad[:, start:stop], out[:, start:stop]
            k_seg, v_seg = _keys(key, spans), _keys(value, spans)
            if deterministic:
                from flash_attn import flash_attn_interface as fai

                dq_seg = dq[:, start:stop]
                dk_seg, dv_seg = torch.empty_like(k_seg), torch.empty_like(v_seg)
                fai._flash_attn_backward(
                    g_seg,
                    q_seg,
                    k_seg.contiguous(),
                    v_seg.contiguous(),
                    o_seg,
                    lse.contiguous(),
                    dq_seg,
                    dk_seg,
                    dv_seg,
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
            else:
                dq_t, dk_t, dv_t = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
                    _bhsd(g_seg),
                    _bhsd(q_seg),
                    _bhsd(k_seg),
                    _bhsd(v_seg),
                    _bhsd(o_seg),
                    lse,
                    seed,
                    offset,
                    None,
                    None,
                    None,
                    max_q,
                    max_k,
                    0.0,
                    False,
                    scale=ctx.scale,
                )
                dq[:, start:stop].copy_(_bhsd(dq_t))
                dk_seg, dv_seg = _bhsd(dk_t), _bhsd(dv_t)
            cursor = 0
            for span_start, span_stop in spans:
                length = span_stop - span_start
                dk[:, span_start:span_stop] += dk_seg[:, cursor : cursor + length]
                dv[:, span_start:span_stop] += dv_seg[:, cursor : cursor + length]
                cursor += length
        COUNTERS["backward"] += 1
        return dq, dk.to(key.dtype), dv.to(value.dtype), None


def w6_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, plan: H3W6Plan):
    """``[B, S, H, D]`` BF16 W6 attention over the full sequence / local heads."""

    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("H3 W6 attention expects BF16 q/k/v")
    if int(query.shape[1]) != plan.sequence_length or key.shape[1] != query.shape[1]:
        raise ValueError("H3 W6 plan does not cover the attention sequence")
    return _W6Attention.apply(query, key, value, plan)


__all__ = ["COUNTERS", "H3W6Plan", "build_w6_plan", "segments_enabled", "w6_attention"]
