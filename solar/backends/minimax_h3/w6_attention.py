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




# --------------------------------------------------------------------------- prefix reuse
#
# In the W6 document the condition, audio and clean rows never attend to noisy rows,
# so within one AnyFlow step their keys/values at every layer are the same in all four
# model evaluations (same weights, same anchor/clean/audio/prompt inputs, same diagonal
# time conditioning). SOLARWM_H3_STAGE1_PREFIX_REUSE=1: the first detached evaluation
# keeps them per layer; the two later detached evaluations then run the noisy rows only
# and attend to the kept prefix keys/values. The trainable evaluation is unchanged.

_PREFIX_REUSE = os.environ.get("SOLARWM_H3_STAGE1_PREFIX_REUSE", "0") == "1"
REUSE_COUNTERS = {"captured_layers": 0, "suffix_layers": 0}


def prefix_reuse_enabled() -> bool:
    return _PREFIX_REUSE and segments_enabled()


class H3W6Suffix:
    """Static per-layout description of the noisy-row (suffix) document."""

    __slots__ = (
        "prefix_rows",
        "kept_rows",
        "suffix_rows",
        "segments",
        "camera_indices",
        "camera_select",
        "video_indices",
    )

    def __init__(self, plan: H3W6Plan, layout: H3PackedLayout) -> None:
        noisy = layout.noisy_video_indices.detach().to("cpu", torch.long)
        prefix = int(noisy[0])
        total = plan.sequence_length
        if not torch.equal(noisy, torch.arange(prefix, total)):
            raise ValueError("H3 W6 prefix reuse needs the noisy rows as one trailing block")
        segments = []
        kept = 0
        for start, stop, spans in plan.segments:
            if start < prefix:
                continue
            for span_start, span_stop in spans:
                if span_start < prefix < span_stop:
                    raise ValueError("H3 W6 key span crosses the prefix/noisy boundary")
                if span_stop <= prefix:
                    kept = max(kept, span_stop)
            segments.append((start - prefix, stop - prefix, spans))
        self.prefix_rows = prefix
        self.kept_rows = kept
        self.suffix_rows = total - prefix
        self.segments = tuple(segments)
        camera = layout.camera_video_indices.detach().to("cpu", torch.long)
        select = (camera >= prefix).nonzero().flatten()
        device = layout.camera_video_indices.device
        self.camera_select = select.to(device)
        self.camera_indices = (camera[select] - prefix).to(device)
        self.video_indices = torch.arange(self.suffix_rows, device=device)


class H3W6PrefixCache:
    """Per-step store of the prefix keys/values of every layer (local heads)."""

    __slots__ = ("layers",)

    def __init__(self) -> None:
        self.layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


class H3W6Run:
    """A W6 attention mask for one model evaluation: ``capture`` (full document, keep
    the prefix) or ``suffix`` (noisy rows only, read the kept prefix)."""

    __slots__ = ("plan", "suffix", "cache", "mode")

    def __init__(self, plan: H3W6Plan, suffix: H3W6Suffix, cache: H3W6PrefixCache, mode: str):
        if mode not in {"capture", "suffix"}:
            raise ValueError(f"unknown H3 W6 run mode {mode!r}")
        self.plan, self.suffix, self.cache, self.mode = plan, suffix, cache, mode

    @property
    def sequence_length(self) -> int:
        return self.plan.sequence_length if self.mode == "capture" else self.suffix.suffix_rows


def w6_run_attention(query, key, value, run: H3W6Run, layer: int):
    if torch.is_grad_enabled():
        raise RuntimeError("H3 W6 prefix reuse runs only in detached evaluations")
    if run.mode == "capture":
        out = w6_attention(query, key, value, run.plan)
        kept = run.suffix.kept_rows
        run.cache.layers[layer] = (key[:, :kept].clone(), value[:, :kept].clone())
        REUSE_COUNTERS["captured_layers"] += 1
        return out
    if int(query.shape[1]) != run.suffix.suffix_rows:
        raise ValueError("H3 W6 suffix evaluation does not cover the noisy rows")
    cached_key, cached_value = run.cache.layers[layer]
    prefix = run.suffix.prefix_rows
    scale = query.shape[-1] ** -0.5
    out = torch.empty_like(query)
    for start, stop, spans in run.suffix.segments:
        keys, values = [], []
        for span_start, span_stop in spans:
            if span_stop <= prefix:
                keys.append(cached_key[:, span_start:span_stop])
                values.append(cached_value[:, span_start:span_stop])
            else:
                keys.append(key[:, span_start - prefix : span_stop - prefix])
                values.append(value[:, span_start - prefix : span_stop - prefix])
        k_seg = keys[0] if len(keys) == 1 else torch.cat(keys, dim=1)
        v_seg = values[0] if len(values) == 1 else torch.cat(values, dim=1)
        result = torch.ops.aten._scaled_dot_product_cudnn_attention(
            _bhsd(query[:, start:stop]),
            _bhsd(k_seg),
            _bhsd(v_seg),
            None,
            True,  # same kernel as the full-document evaluation (which keeps the LSE)
            0.0,
            False,
            False,
            scale=scale,
        )
        out[:, start:stop].copy_(_bhsd(result[0]))
    REUSE_COUNTERS["suffix_layers"] += 1
    return out


__all__ = [
    "COUNTERS",
    "H3W6Plan",
    "H3W6PrefixCache",
    "H3W6Run",
    "H3W6Suffix",
    "build_w6_plan",
    "prefix_reuse_enabled",
    "segments_enabled",
    "w6_attention",
    "w6_run_attention",
]
