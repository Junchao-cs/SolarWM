"""Raw KV caching and per-window MM-RoPE for H3 self-gradient forcing.

Replay is one transformer forward over [condition, audio, clean, noisy]. At
each layer, small query groups attend to exactly their visible key spans.
Each group's raw Q/K receives its own local W6 positions before camera
PRoPE. This also handles H3's joint text/image/audio document: a global
position assignment to shared clean keys cannot express every local window.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from .torch_geometry import native_video_position_grid
from .torch_prope import _prepare_suffix_transforms, logd4_relative_viewmats, prope_qkv_separate

_PLAN_ENABLED = os.environ.get("SOLARWM_H3_SGF_PLAN", "0") == "1"
PLAN_COUNTERS = {"built": 0, "reused": 0}


class _SGFForwardPlan:
    """Per-forward memo of the attention segments' camera transforms and rotaries.

    A plain class (not a dataclass or container) so that FSDP/checkpoint argument
    traversal passes it by reference. Everything stored here depends only on the
    forward's control (layout, cameras, chunk, rotaries), never on a layer's q/k/v.
    """

    __slots__ = ("segments", "key")

    def __init__(self) -> None:
        self.segments = None
        self.key = None


@dataclass
class H3RawKVCache:
    """Each SP rank owns its head shard, before MM-RoPE and camera PRoPE."""

    layers: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = field(default_factory=dict)

    def history(self, layer: int, chunk_index: int):
        entries = self.layers.get(layer, [])
        expected = list(range(max(0, chunk_index - 5), chunk_index))
        if [entry[0] for entry in entries] != expected:
            raise RuntimeError(f"H3 KV history differs at layer={layer}, chunk={chunk_index}")
        return entries

    def commit(self, layer: int, chunk_index: int, key, value):
        if torch.is_grad_enabled():
            raise RuntimeError("H3 raw KV commits must be detached no-grad forwards")
        entries = self.history(layer, chunk_index)
        self.layers[layer] = [
            *entries,
            (chunk_index, key.detach().clone(), value.detach().clone()),
        ][-5:]

    def clear(self):
        self.layers.clear()


@dataclass(frozen=True)
class H3SGFAttention:
    layout: Any
    # Camera tensors include anchor slot zero and all 50 rollout video slots.
    viewmats: torch.Tensor
    intrinsics: torch.Tensor
    mode: str
    chunk_index: int = 0
    cache: H3RawKVCache | None = None
    commit_cache: bool = False
    # Populated once at the model boundary, before SP shards the input document.
    prefix_rotary: Any = None
    window_rotaries: Any = None
    # Layer-independent camera/rotary tensors of this forward (SOLARWM_H3_SGF_PLAN=1),
    # built by the first layer and reused by the other 49 (and by recomputation).
    plan: Any = None
    # Rollout prefix reuse (SOLARWM_H3_SGF_PREFIX_REUSE=1): H3SGFPrefixKV of the rollout.
    prefix_kv: Any = None


class H3SGFPrefixKV:
    """Per-rollout raw keys/values of the prefix rows (text, anchor, audio) per layer.

    In a rollout document the prefix rows attend only to prefix rows, with fixed
    rotary positions, cameras and time conditioning, so their raw (pre-rotary) keys
    and values are the same in every denoising and commit evaluation of the rollout.
    The first evaluation keeps them ('capture'); every later one runs the current
    chunk's rows only ('suffix') and reads them. A plain object: passed by reference.
    """

    __slots__ = ("layers", "mode", "window", "storage")

    def __init__(self) -> None:
        self.layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.mode = "capture"
        # SOLARWM_H3_SGF_KV_WINDOW=1: per layer (chunk index, keys, values, prepared rows) of
        # the current window's attention keys/values after MM-RoPE and camera PRoPE.
        self.window: dict[int, list] = {}
        # One allocation for every layer's window buffers, so that the rollout frees them as
        # a single block (100 separate ~115 MB segments fragmented the caching allocator).
        self.storage = None


_PREFIX_REUSE = os.environ.get("SOLARWM_H3_SGF_PREFIX_REUSE", "0") == "1"
PREFIX_REUSE_COUNTERS = {"captured_layers": 0, "suffix_layers": 0}

# SOLARWM_H3_SGF_KV_WINDOW=1 (with SGF_PLAN and SGF_PREFIX_REUSE): in a detached rollout the
# prefix and history rows of a chunk's attention window get the same rotary positions and
# camera transforms in all five evaluations of that chunk (four denoising steps and the
# commit). Their transformed keys/values are built once per chunk and layer into a
# per-layer buffer; each evaluation transforms and writes only the current chunk's rows.
_KV_WINDOW = os.environ.get("SOLARWM_H3_SGF_KV_WINDOW", "0") == "1"
_WINDOW_LAYERS = 50  # MiniMax-H3 transformer blocks
KV_WINDOW_COUNTERS = {"built": 0, "reused": 0}


def sgf_prefix_reuse_enabled() -> bool:
    from .stage_scope import n1_active

    return _PREFIX_REUSE and _PLAN_ENABLED and n1_active()


def prepare_sgf_rotaries(control: H3SGFAttention, rope):
    custom = getattr(control, "prepare_rotaries", None)
    if custom is not None:
        return custom(rope)
    from dataclasses import replace

    layout = control.layout
    prefix = int(layout.condition_indices.numel()) + int(layout.audio_indices.numel())
    rotations = {}
    for frames in (5, 10, 15, 20, 25, 30):
        positions = native_video_position_grid(
            frames,
            layout.latent_height,
            layout.latent_width,
            patch_size=layout.patch_size,
            origin=float(layout.text_indices.numel()),
        ).to(layout.position_ids.device)
        rotations[frames] = rope(positions)
    return replace(
        control,
        prefix_rotary=rope(layout.position_ids[:prefix]),
        window_rotaries=rotations,
        plan=_SGFForwardPlan() if _PLAN_ENABLED else None,
    )


def _slice_rotary(rotary, start, stop):
    # Native H3 returns [S, head_dim] cosine/sine buffers.
    return tuple(value[start:stop] for value in rotary)


def _cat_rotary(*rotaries):
    return tuple(torch.cat(parts, dim=0) for parts in zip(*rotaries, strict=True))


def sgf_attention(q, k, v, *, control: H3SGFAttention, layer: int, apply_rotary, attend):
    """Attend over full sequence / local heads after Ulysses all-to-all."""
    custom = getattr(control, "attend", None)
    if custom is not None:
        return custom(q, k, v, layer=layer, apply_rotary=apply_rotary, attend=attend)
    layout = control.layout
    condition_rows = int(layout.condition_indices.numel())
    prefix = condition_rows + int(layout.audio_indices.numel())
    rows = int(layout.rows_per_video_frame)
    chunk_rows = 5 * rows
    expected_video = 100 if control.mode == "replay" else 5
    suffix = control.prefix_kv is not None and control.prefix_kv.mode == "suffix"
    expected_rows = expected_video * rows + (0 if suffix else prefix)
    if q.shape[1] != expected_rows:
        raise ValueError("H3 SGF packed query shape differs from its rollout/replay layout")
    if control.prefix_rotary is None:
        raise ValueError("H3 SGF rotaries must be prepared before attention")
    if control.mode not in {"rollout", "replay"}:
        raise ValueError("H3 SGF attention mode must be rollout or replay")
    batch = q.shape[0]
    if control.plan is not None:
        return _planned_sgf_attention(
            q, k, v, control=control, layer=layer, apply_rotary=apply_rotary, attend=attend
        )

    prefix_view = torch.eye(4, device=q.device, dtype=torch.float32).repeat(batch, prefix, 1, 1)
    prefix_K = torch.eye(3, device=q.device, dtype=torch.float32).repeat(batch, prefix, 1, 1)
    anchor_indices = layout.condition_video_indices
    prefix_view[:, anchor_indices] = control.viewmats[:, :1]
    prefix_K[:, anchor_indices] = control.intrinsics[:, :1]

    def camera_for_frames(start, stop):
        return (
            control.viewmats[:, 1 + start : 1 + stop].repeat_interleave(rows, dim=1),
            control.intrinsics[:, 1 + start : 1 + stop].repeat_interleave(rows, dim=1),
        )

    def calculate(query, key, value, qr, kr, qcam, kcam):
        query = apply_rotary(query, *qr)
        key = apply_rotary(key, *kr)
        query, key, value, output_basis = prope_qkv_separate(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            query_viewmats=qcam[0],
            query_K=qcam[1],
            key_viewmats=kcam[0],
            key_K=kcam[1],
        )
        output = attend(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        return output_basis(output.transpose(1, 2)).transpose(1, 2)

    prefix_cam = (prefix_view, prefix_K)
    outputs = [
        calculate(
            q[:, :condition_rows],
            k[:, :condition_rows],
            v[:, :condition_rows],
            _slice_rotary(control.prefix_rotary, 0, condition_rows),
            _slice_rotary(control.prefix_rotary, 0, condition_rows),
            (prefix_view[:, :condition_rows], prefix_K[:, :condition_rows]),
            (prefix_view[:, :condition_rows], prefix_K[:, :condition_rows]),
        )
    ]
    if prefix > condition_rows:
        outputs.append(
            calculate(
                q[:, condition_rows:prefix],
                k[:, :prefix],
                v[:, :prefix],
                _slice_rotary(control.prefix_rotary, condition_rows, prefix),
                control.prefix_rotary,
                (prefix_view[:, condition_rows:prefix], prefix_K[:, condition_rows:prefix]),
                prefix_cam,
            )
        )

    chunk_specs = (
        [(False, i) for i in range(10)] + [(True, i) for i in range(10)]
        if control.mode == "replay"
        else [(True, control.chunk_index)]
    )
    for noisy, index in chunk_specs:
        first = max(0, index - 5)
        local_frames = (index - first + 1) * 5
        if control.mode == "replay":
            qstart = prefix + (50 * rows if noisy else 0) + index * chunk_rows
            clean_start = prefix + first * chunk_rows
            clean_stop = prefix + index * chunk_rows
            history_keys, history_values = (
                k[:, clean_start:clean_stop],
                v[:, clean_start:clean_stop],
            )
        else:
            if control.cache is None:
                raise ValueError("H3 rollout requires an explicit raw KV cache")
            qstart = prefix
            history = control.cache.history(layer, index)
            history_keys = (
                torch.cat([entry[1] for entry in history], dim=1) if history else k[:, :0]
            )
            history_values = (
                torch.cat([entry[2] for entry in history], dim=1) if history else v[:, :0]
            )
        qstop = qstart + chunk_rows
        key = torch.cat((k[:, :prefix], history_keys, k[:, qstart:qstop]), dim=1)
        value = torch.cat((v[:, :prefix], history_values, v[:, qstart:qstop]), dim=1)
        rotary = control.window_rotaries[local_frames]
        qrotary = _slice_rotary(rotary, (local_frames - 5) * rows, local_frames * rows)
        krotary = _cat_rotary(control.prefix_rotary, rotary)
        current_camera = camera_for_frames(index * 5, (index + 1) * 5)
        window_camera = camera_for_frames(first * 5, (index + 1) * 5)
        key_camera = tuple(
            torch.cat((p, w), dim=1) for p, w in zip(prefix_cam, window_camera, strict=True)
        )
        outputs.append(
            calculate(q[:, qstart:qstop], key, value, qrotary, krotary, current_camera, key_camera)
        )
        if control.mode == "rollout" and control.commit_cache:
            control.cache.commit(layer, index, k[:, qstart:qstop], v[:, qstart:qstop])
    return torch.cat(outputs, dim=1)


def _segment_specs(
    control: H3SGFAttention, rows: int, prefix: int, condition_rows: int, suffix: bool = False
):
    """Query/key row ranges of every attention segment, in the order of ``sgf_attention``."""
    chunk_rows = 5 * rows
    specs = []
    if not suffix:
        specs.append(("condition", 0, condition_rows, None))
        if prefix > condition_rows:
            specs.append(("audio", condition_rows, prefix, None))
    chunk_specs = (
        [(False, i) for i in range(10)] + [(True, i) for i in range(10)]
        if control.mode == "replay"
        else [(True, control.chunk_index)]
    )
    for noisy, index in chunk_specs:
        if control.mode == "replay":
            qstart = prefix + (50 * rows if noisy else 0) + index * chunk_rows
        else:
            qstart = 0 if suffix else prefix
        specs.append(("chunk", qstart, qstart + chunk_rows, index))
    return specs


def _build_plan(control: H3SGFAttention, dtype, device, batch: int, suffix: bool = False):
    """Exactly the camera/rotary tensors ``sgf_attention`` computes per layer, once."""
    layout = control.layout
    condition_rows = int(layout.condition_indices.numel())
    prefix = condition_rows + int(layout.audio_indices.numel())
    rows = int(layout.rows_per_video_frame)

    prefix_view = torch.eye(4, device=device, dtype=torch.float32).repeat(batch, prefix, 1, 1)
    prefix_K = torch.eye(3, device=device, dtype=torch.float32).repeat(batch, prefix, 1, 1)
    anchor_indices = layout.condition_video_indices
    prefix_view[:, anchor_indices] = control.viewmats[:, :1]
    prefix_K[:, anchor_indices] = control.intrinsics[:, :1]

    def camera_for_frames(start, stop):
        return (
            control.viewmats[:, 1 + start : 1 + stop].repeat_interleave(rows, dim=1),
            control.intrinsics[:, 1 + start : 1 + stop].repeat_interleave(rows, dim=1),
        )

    def transforms(qcam, kcam):
        # prope_qkv_separate's camera arithmetic (in the q/k dtype), without q/k/v.
        apply_q, _, apply_output = _prepare_suffix_transforms(
            viewmats=logd4_relative_viewmats(qcam[0].to(dtype)), Ks=qcam[1].to(dtype)
        )
        _, apply_kv, _ = _prepare_suffix_transforms(
            viewmats=logd4_relative_viewmats(kcam[0].to(dtype)), Ks=kcam[1].to(dtype)
        )
        return apply_q, apply_kv, apply_output

    def cast(rotary):
        # _apply_rotary_emb casts cos/sin to the activation dtype; cast once.
        return tuple(value.to(dtype) for value in rotary)

    segments = []
    for kind, qstart, qstop, index in _segment_specs(
        control, rows, prefix, condition_rows, suffix=suffix
    ):
        if kind == "condition":
            rot = _slice_rotary(control.prefix_rotary, 0, condition_rows)
            cam = (prefix_view[:, :condition_rows], prefix_K[:, :condition_rows])
            segments.append((kind, qstart, qstop, index, cast(rot), cast(rot), transforms(cam, cam)))
        elif kind == "audio":
            qrot = _slice_rotary(control.prefix_rotary, condition_rows, prefix)
            qcam = (prefix_view[:, condition_rows:prefix], prefix_K[:, condition_rows:prefix])
            segments.append(
                (
                    kind,
                    qstart,
                    qstop,
                    index,
                    cast(qrot),
                    cast(control.prefix_rotary),
                    transforms(qcam, (prefix_view, prefix_K)),
                )
            )
        else:
            first = max(0, index - 5)
            local_frames = (index - first + 1) * 5
            rotary = control.window_rotaries[local_frames]
            qrotary = _slice_rotary(rotary, (local_frames - 5) * rows, local_frames * rows)
            krotary = _cat_rotary(control.prefix_rotary, rotary)
            current_camera = camera_for_frames(index * 5, (index + 1) * 5)
            window_camera = camera_for_frames(first * 5, (index + 1) * 5)
            key_camera = tuple(
                torch.cat((p, w), dim=1)
                for p, w in zip((prefix_view, prefix_K), window_camera, strict=True)
            )
            segments.append(
                (
                    kind,
                    qstart,
                    qstop,
                    index,
                    cast(qrotary),
                    cast(krotary),
                    transforms(current_camera, key_camera),
                )
            )
    return tuple(segments)


def _n1() -> bool:
    from .stage_scope import n1_active

    return n1_active()


def _fused() -> bool:
    from .fused_prope import prope_fused_enabled

    return prope_fused_enabled()


def _suffix(features, transform):
    from .fused_prope import prope_fused_enabled, prope_project

    if prope_fused_enabled():
        return prope_project(features, transform.keywords["matrix"])
    return torch.cat((features[..., :96], transform(features[..., 96:128])), dim=-1)


def _planned_sgf_attention(q, k, v, *, control, layer, apply_rotary, attend):
    layout = control.layout
    condition_rows = int(layout.condition_indices.numel())
    prefix = condition_rows + int(layout.audio_indices.numel())
    rows = int(layout.rows_per_video_frame)
    chunk_rows = 5 * rows
    plan = control.plan
    prefix_kv = control.prefix_kv
    suffix = prefix_kv is not None and prefix_kv.mode == "suffix"
    if prefix_kv is not None and (control.mode != "rollout" or torch.is_grad_enabled()):
        raise RuntimeError("H3 SGF prefix reuse is for detached rollout evaluations only")
    key = (q.dtype, q.device, int(q.shape[0]))
    if plan.segments is None:
        plan.segments = _build_plan(control, q.dtype, q.device, int(q.shape[0]), suffix=suffix)
        plan.key = key
        PLAN_COUNTERS["built"] += 1
    else:
        if plan.key != key:
            raise RuntimeError("H3 SGF forward plan reused with a different q dtype/device/batch")
        PLAN_COUNTERS["reused"] += 1

    def calculate(query, key_, value, qr, kr, prope):
        apply_q, apply_kv, apply_output = prope
        if _fused():
            # SOLARWM_H3_PROPE_FUSED=1: MM-RoPE + camera PRoPE of q/k in one pass each.
            from .fused_prope import prope_project, rope_prope

            kv_matrix = apply_kv.keywords["matrix"]
            query = rope_prope(query, qr[0], qr[1], apply_q.keywords["matrix"])
            key_ = rope_prope(key_, kr[0], kr[1], kv_matrix)
            value = prope_project(value.transpose(1, 2), kv_matrix).transpose(1, 2)
            output = attend(query, key_, value)
            return prope_project(
                output.transpose(1, 2), apply_output.keywords["matrix"]
            ).transpose(1, 2)
        query = apply_rotary(query, *qr)
        key_ = apply_rotary(key_, *kr)
        query, key_, value = (
            _suffix(query.transpose(1, 2), apply_q),
            _suffix(key_.transpose(1, 2), apply_kv),
            _suffix(value.transpose(1, 2), apply_kv),
        )
        output = attend(query.transpose(1, 2), key_.transpose(1, 2), value.transpose(1, 2))
        return _suffix(output.transpose(1, 2), apply_output).transpose(1, 2)

    if suffix and _KV_WINDOW and control.mode == "rollout" and len(plan.segments) == 1:
        return _windowed_rollout_attention(
            q, k, v, control=control, layer=layer, apply_rotary=apply_rotary, attend=attend,
            prefix=prefix, rows=rows,
        )

    if _PIECES and control.mode == "replay" and torch.is_grad_enabled() and _n1():
        return _pieced_replay_attention(
            q, k, v, plan=plan, calculate=calculate, condition_rows=condition_rows,
            prefix=prefix, chunk_rows=chunk_rows,
        )

    outputs = []
    for kind, qstart, qstop, index, qr, kr, prope in plan.segments:
        if kind == "condition":
            outputs.append(
                calculate(
                    q[:, :condition_rows], k[:, :condition_rows], v[:, :condition_rows], qr, kr, prope
                )
            )
            continue
        if kind == "audio":
            outputs.append(
                calculate(q[:, condition_rows:prefix], k[:, :prefix], v[:, :prefix], qr, kr, prope)
            )
            continue
        first = max(0, index - 5)
        if control.mode == "replay":
            clean_start = prefix + first * chunk_rows
            clean_stop = prefix + index * chunk_rows
            history_keys, history_values = (
                k[:, clean_start:clean_stop],
                v[:, clean_start:clean_stop],
            )
        else:
            if control.cache is None:
                raise ValueError("H3 rollout requires an explicit raw KV cache")
            history = control.cache.history(layer, index)
            history_keys = (
                torch.cat([entry[1] for entry in history], dim=1) if history else k[:, :0]
            )
            history_values = (
                torch.cat([entry[2] for entry in history], dim=1) if history else v[:, :0]
            )
        if suffix:
            prefix_keys, prefix_values = prefix_kv.layers[layer]
            PREFIX_REUSE_COUNTERS["suffix_layers"] += 1
        else:
            prefix_keys, prefix_values = k[:, :prefix], v[:, :prefix]
            if prefix_kv is not None:
                prefix_kv.layers[layer] = (prefix_keys.clone(), prefix_values.clone())
                PREFIX_REUSE_COUNTERS["captured_layers"] += 1
        key_rows = torch.cat((prefix_keys, history_keys, k[:, qstart:qstop]), dim=1)
        value_rows = torch.cat((prefix_values, history_values, v[:, qstart:qstop]), dim=1)
        outputs.append(calculate(q[:, qstart:qstop], key_rows, value_rows, qr, kr, prope))
        if control.mode == "rollout" and control.commit_cache:
            control.cache.commit(layer, index, k[:, qstart:qstop], v[:, qstart:qstop])
    return torch.cat(outputs, dim=1)


# SOLARWM_H3_SGF_PIECES=1: in the replay with gradients, take every segment's rows from one
# split of q/k/v into [condition | audio | 20 chunks] instead of slicing the full tensors per
# segment. Same forward values; in the backward autograd accumulates gradients per piece and
# concatenates once, instead of materialising a zero-filled full-size gradient per slice
# (SliceBackward) and adding ~150 of them per layer.
_PIECES = os.environ.get("SOLARWM_H3_SGF_PIECES", "0") == "1"
PIECES_COUNTERS = {"replay_layers": 0}


def _pieced_replay_attention(q, k, v, *, plan, calculate, condition_rows, prefix, chunk_rows):
    sizes = [condition_rows, prefix - condition_rows] + [chunk_rows] * 20
    if sum(sizes) != q.shape[1] or k.shape[1] != q.shape[1]:
        raise RuntimeError("H3 SGF replay pieces do not tile the document")
    qp, kp, vp = (tensor.split(sizes, dim=1) for tensor in (q, k, v))
    outputs = []
    for kind, qstart, qstop, index, qr, kr, prope in plan.segments:
        if kind == "condition":
            outputs.append(calculate(qp[0], kp[0], vp[0], qr, kr, prope))
            continue
        if kind == "audio":
            outputs.append(
                calculate(qp[1], torch.cat(kp[:2], dim=1), torch.cat(vp[:2], dim=1), qr, kr, prope)
            )
            continue
        first = max(0, index - 5)
        current = 2 + (10 if qstart >= prefix + 10 * chunk_rows else 0) + index
        if qstart != prefix + (current - 2) * chunk_rows or qstop - qstart != chunk_rows:
            raise RuntimeError("H3 SGF replay piece does not match its segment")
        keys = torch.cat((kp[0], kp[1], *kp[2 + first : 2 + index], kp[current]), dim=1)
        values = torch.cat((vp[0], vp[1], *vp[2 + first : 2 + index], vp[current]), dim=1)
        outputs.append(calculate(qp[current], keys, values, qr, kr, prope))
    PIECES_COUNTERS["replay_layers"] += 1
    return torch.cat(outputs, dim=1)


def _rows_of(transform, start: int, stop: int):
    """A ``_project`` partial restricted to the camera matrices of rows ``[start, stop)``."""
    from functools import partial

    return partial(transform.func, matrix=transform.keywords["matrix"][:, start:stop])


def _suffix_into(features, transform, out):
    """``out.copy_(_suffix(features, transform))``; the fused kernel writes ``out`` directly."""
    from .fused_prope import _launch, prope_fused_enabled

    if prope_fused_enabled() and not torch.is_grad_enabled():
        _launch(features, transform.keywords["matrix"], False, out=out)
    else:
        out.copy_(_suffix(features, transform))


def _windowed_rollout_attention(q, k, v, *, control, layer, apply_rotary, attend, prefix, rows):
    """Suffix rollout evaluation of one chunk against its prepared attention window.

    Same operations as the planned path's chunk segment, split by key rows: the prefix and
    history rows of the window (identical in every evaluation of the chunk) are transformed
    once per chunk and layer into a per-layer buffer; the current chunk's rows are
    transformed and written into the buffer's tail on every evaluation.
    """
    plan = control.plan
    prefix_kv = control.prefix_kv
    ((_kind, qstart, qstop, index, qr, kr, prope),) = plan.segments
    apply_q, apply_kv, apply_output = prope
    chunk_rows = 5 * rows
    if (qstart, qstop) != (0, chunk_rows) or q.shape[1] != chunk_rows:
        raise RuntimeError("H3 SGF KV window expects exactly one chunk of query rows")
    total = int(kr[0].shape[0])
    kept = total - chunk_rows
    entry = prefix_kv.window.get(layer)
    if entry is None or entry[0] != index:
        if control.cache is None:
            raise ValueError("H3 rollout requires an explicit raw KV cache")
        history = control.cache.history(layer, index)
        prefix_keys, prefix_values = prefix_kv.layers[layer]
        if prefix_keys.shape[1] != prefix or kept != prefix + len(history) * chunk_rows:
            raise RuntimeError("H3 SGF KV window does not match the chunk's key layout")
        maximum = prefix + 6 * chunk_rows
        if entry is None:
            shape = (k.shape[0], maximum, k.shape[2], k.shape[3])
            storage = prefix_kv.storage
            if storage is None or tuple(storage.shape[2:]) != shape or layer >= storage.shape[0]:
                if storage is None:
                    storage = torch.empty(
                        (_WINDOW_LAYERS, 2, *shape), device=k.device, dtype=k.dtype
                    )
                    prefix_kv.storage = storage
                if tuple(storage.shape[2:]) != shape or layer >= storage.shape[0]:
                    raise RuntimeError("H3 SGF KV window storage does not fit this layer")
            keys, values = storage[layer, 0], storage[layer, 1]
        else:
            keys, values = entry[1], entry[2]
        raw_keys = torch.cat((prefix_keys, *(item[1] for item in history)), dim=1)
        raw_values = torch.cat((prefix_values, *(item[2] for item in history)), dim=1)
        kept_kv = _rows_of(apply_kv, 0, kept)
        if _fused():
            from .fused_prope import rope_prope

            rope_prope(raw_keys, kr[0][:kept], kr[1][:kept], kept_kv.keywords["matrix"],
                       out=keys[:, :kept])
        else:
            rotated = apply_rotary(raw_keys, kr[0][:kept], kr[1][:kept])
            _suffix_into(rotated.transpose(1, 2), kept_kv, keys[:, :kept].transpose(1, 2))
            del rotated
        _suffix_into(raw_values.transpose(1, 2), kept_kv, values[:, :kept].transpose(1, 2))
        del raw_keys, raw_values
        entry = [index, keys, values]
        prefix_kv.window[layer] = entry
        KV_WINDOW_COUNTERS["built"] += 1
    else:
        KV_WINDOW_COUNTERS["reused"] += 1
    keys, values = entry[1], entry[2]
    PREFIX_REUSE_COUNTERS["suffix_layers"] += 1
    chunk_kv = _rows_of(apply_kv, kept, total)
    if _fused():
        from .fused_prope import rope_prope

        rope_prope(k, kr[0][kept:], kr[1][kept:], chunk_kv.keywords["matrix"],
                   out=keys[:, kept:total])
        query = rope_prope(q, qr[0], qr[1], apply_q.keywords["matrix"])
    else:
        rotated = apply_rotary(k, kr[0][kept:], kr[1][kept:])
        _suffix_into(rotated.transpose(1, 2), chunk_kv, keys[:, kept:total].transpose(1, 2))
        query = _suffix(apply_rotary(q, *qr).transpose(1, 2), apply_q).transpose(1, 2)
    _suffix_into(v.transpose(1, 2), chunk_kv, values[:, kept:total].transpose(1, 2))
    output = attend(query, keys[:, :total], values[:, :total])
    output = _suffix(output.transpose(1, 2), apply_output).transpose(1, 2)
    if control.commit_cache:
        control.cache.commit(layer, index, k, v)
    return output
