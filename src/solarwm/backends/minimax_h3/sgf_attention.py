"""Raw KV caching and per-window MM-RoPE for H3 self-gradient forcing.

Replay is one transformer forward over [condition, audio, clean, noisy]. At
each layer, small query groups attend to exactly their visible key spans.
Each group's raw Q/K receives its own local W6 positions before camera
PRoPE. This also handles H3's joint text/image/audio document: a global
position assignment to shared clean keys cannot express every local window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .torch_geometry import native_video_position_grid
from .torch_prope import prope_qkv_separate


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


def prepare_sgf_rotaries(control: H3SGFAttention, rope):
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
        control, prefix_rotary=rope(layout.position_ids[:prefix]), window_rotaries=rotations
    )


def _slice_rotary(rotary, start, stop):
    # Native H3 returns [S, head_dim] cosine/sine buffers.
    return tuple(value[start:stop] for value in rotary)


def _cat_rotary(*rotaries):
    return tuple(torch.cat(parts, dim=0) for parts in zip(*rotaries, strict=True))


def sgf_attention(q, k, v, *, control: H3SGFAttention, layer: int, apply_rotary, attend):
    """Attend over full sequence / local heads after Ulysses all-to-all."""
    layout = control.layout
    condition_rows = int(layout.condition_indices.numel())
    prefix = condition_rows + int(layout.audio_indices.numel())
    rows = int(layout.rows_per_video_frame)
    chunk_rows = 5 * rows
    expected_video = 100 if control.mode == "replay" else 5
    if q.shape[1] != prefix + expected_video * rows:
        raise ValueError("H3 SGF packed query shape differs from its rollout/replay layout")
    if control.prefix_rotary is None:
        raise ValueError("H3 SGF rotaries must be prepared before attention")
    if control.mode not in {"rollout", "replay"}:
        raise ValueError("H3 SGF attention mode must be rollout or replay")
    batch = q.shape[0]

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
