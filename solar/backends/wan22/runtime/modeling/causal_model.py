# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 SolarWM Contributors.
#
# Modified from the Wan 2.2 architecture (see model.py in this directory, from
# which WanRMSNorm, WanLayerNorm, WanCrossAttention, rope_params and
# sinusoidal_embedding_1d are imported). Changes by SolarWM: causal attention
# over chunked history, camera conditioning through PRoPE, and the
# sequence-parallel paths the causal routes need.
from .attention import attention
from .camera_prope import prope_qkv, prope_qkv_separate
from .model import WanRMSNorm, WanLayerNorm, WanCrossAttention, rope_params, sinusoidal_embedding_1d
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from ..camera import normalize_camera_translation_transform
import torch.nn as nn
import torch
import copy
import math
import os

# Lazily-imported FlexAttention symbols. Some torch builds don't ship
# torch.nn.attention.flex_attention; we only require it when stage1 TF mode
# is actually used (the inference / KV-cache path doesn't need it).
try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )

    _HAS_FLEX_ATTENTION = True
    # Use static FlexAttention compilation. Cache-free validation
    # grows the active window through 80 distinct shapes, and also exercises
    # both the normal and PRoPE attention branches. PyTorch's default cache
    # limit of 8 silently falls back to dense SDPA after the eighth variant,
    # which is an OOM at 60 seconds. Dynamic shapes are not supported by the
    # torch 2.5 FlexAttention lowering, so retain static kernels and explicitly
    # budget enough variants for the complete rollout.
    if os.environ.get("SOLARWM_COMPILE_FLEX", "0") in {"1", "true", "yes"}:
        cache_limit = int(os.environ.get("SOLARWM_FLEX_CACHE_SIZE_LIMIT", "256"))
        accumulated_limit = int(os.environ.get("SOLARWM_FLEX_ACCUMULATED_CACHE_SIZE_LIMIT", "512"))
        if cache_limit < 160:
            raise ValueError(
                "SOLARWM_FLEX_CACHE_SIZE_LIMIT must be >=160 for 80-window "
                f"Stage1 validation, got {cache_limit}"
            )
        if accumulated_limit < cache_limit:
            raise ValueError(
                "SOLARWM_FLEX_ACCUMULATED_CACHE_SIZE_LIMIT must be >= "
                f"SOLARWM_FLEX_CACHE_SIZE_LIMIT, got {accumulated_limit} < {cache_limit}"
            )
        torch._dynamo.config.cache_size_limit = cache_limit
        torch._dynamo.config.accumulated_cache_size_limit = accumulated_limit
        flex_attention = torch.compile(flex_attention, dynamic=False)
except ImportError:  # pragma: no cover
    BlockMask = None
    create_block_mask = None
    flex_attention = None
    _HAS_FLEX_ATTENTION = False

from ..sequence_parallel import (
    get_sp_rank,
    get_sp_size,
    is_sequence_parallel_enabled,
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
    register_sequence_parallel_sequence_length,
)
from ..sequence_parallel import contiguous_sp_bounds


CAMERA_ATTENTION_MODES = frozenset({"parallel", "fused_prope"})


def normalize_camera_attention_mode(camera_attention_mode="parallel"):
    """Validate and normalize the camera-attention execution mode."""
    value = str(camera_attention_mode)
    if value not in CAMERA_ATTENTION_MODES:
        raise ValueError(
            "camera_attention_mode must be one of "
            f"{sorted(CAMERA_ATTENTION_MODES)}, got {camera_attention_mode!r}"
        )
    return value


def echorope_apply(
    x,
    grid_sizes,
    freqs,
    start_frame=0,
    frame_indices=None,
    seq_offsets=None,
):
    """Apply window-relative RoPE to self-attention Q/K.

    This is intentionally the normal Wan self-attention RoPE path, not the
    camera PRoPE/EProPE branch.

    Temporal positions are rebased inside the active attention window instead
    of using ever-growing global frame ids. Spatial RoPE remains the standard
    per-window ``h/w`` coordinate. The caller can either provide explicit
    per-frame temporal slots (``frame_indices``) or a contiguous range starting
    at ``start_frame``.

    Args:
        x: ``[B, L, num_heads, head_dim]``.
        grid_sizes: ``[B, 3]`` with valid ``(F, H, W)`` frame/grid sizes.
        freqs: complex RoPE table produced by ``rope_params`` and concatenated
            in Wan temporal/height/width layout.
        start_frame: Temporal slot used when ``frame_indices`` is omitted.
        frame_indices: Optional 1-D tensor/list of length ``F``.  Use this for
            stage1 teacher forcing (``[0..F-1, 0..F-1]`` for clean/noisy halves)
            and cache-free sliding-window validation (``[0..window_F-1]``).
        seq_offsets: Reserved for future token-sharded RoPE support.  The
            verified stage1 paths use unsharded RoPE coordinates (offset 0).

    Tail tokens beyond ``prod(grid_sizes[i])`` are FlexAttention padding and are
    copied through unrotated.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs_t, freqs_h, freqs_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    if frame_indices is not None:
        frame_indices = torch.as_tensor(frame_indices, device=x.device, dtype=torch.long)
    if seq_offsets is None:
        seq_offsets = [0] * x.shape[0]
    elif isinstance(seq_offsets, torch.Tensor):
        seq_offsets = [int(v) for v in seq_offsets.detach().cpu().tolist()]
    else:
        seq_offsets = [int(v) for v in seq_offsets]
    if len(seq_offsets) != x.shape[0]:
        raise ValueError(
            f"echorope_apply expected {x.shape[0]} seq_offsets, got {len(seq_offsets)}"
        )

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        if f == 0:
            if x.shape[1] == 0:
                output.append(x[i])
                continue
            raise ValueError("echorope_apply got f=0 for a non-empty input")

        seq_len = f * h * w
        seq_offset = seq_offsets[i]
        if seq_offset != 0:
            raise ValueError(
                "echorope_apply only supports seq_offset=0 in the verified "
                f"stage1 paths, got {seq_offset} for sample {i}"
            )
        valid_len = min(x.shape[1], seq_len)

        if frame_indices is None:
            temporal_idx = torch.arange(
                start_frame, start_frame + f, device=x.device, dtype=torch.long
            )
        else:
            if frame_indices.numel() != f:
                raise ValueError(
                    f"echorope frame_indices length {frame_indices.numel()} "
                    f"must equal grid frame count {f}"
                )
            temporal_idx = frame_indices

        if temporal_idx.numel() > 0:
            min_idx = int(temporal_idx.min().item())
            max_idx = int(temporal_idx.max().item())
            if min_idx < 0 or max_idx >= freqs_t.shape[0]:
                raise ValueError(
                    f"EchoRoPE temporal indices [{min_idx}, {max_idx}] are "
                    f"outside RoPE table length {freqs_t.shape[0]}"
                )

        freqs_i_full = torch.cat(
            [
                freqs_t[temporal_idx].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        if valid_len > 0:
            x_i = torch.view_as_complex(
                x[i, :valid_len].to(torch.float64).reshape(valid_len, n, -1, 2)
            )
            freqs_i = freqs_i_full[seq_offset : seq_offset + valid_len]
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        else:
            x_i = x[i, :0].to(torch.float64)
        if x.shape[1] > valid_len:
            x_i = torch.cat([x_i, x[i, valid_len:].to(x_i.dtype)])
        output.append(x_i)

    return torch.stack(output).type_as(x)


def block_relativistic_rope(
    x,
    grid_sizes,
    freqs,
    start_frame=0,
    relative_frame_indices=None,
):
    """Apply the SolarWM window/block-relative Wan RoPE.

    The tensor transform is intentionally kept separate from ``echorope_apply``
    so configurations can explicitly select the base Stage1 positional encoding.
    KV-cache callers select the base position allocation as well; this helper
    only performs the rotary transform for the positions it is given.
    """
    return echorope_apply(
        x,
        grid_sizes,
        freqs,
        start_frame=start_frame,
        frame_indices=relative_frame_indices,
    )


_FLEX_BLOCK_SIZE = 128


def _ordered_block_metadata(dense_mask):
    """Convert a block-level boolean matrix into ordered FlexAttention metadata."""
    dense_mask = dense_mask.to(dtype=torch.int32)
    num_blocks = dense_mask.sum(dim=-1).to(dtype=torch.int32, memory_format=torch.contiguous_format)
    indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True).to(
        dtype=torch.int32, memory_format=torch.contiguous_format
    )
    return num_blocks, indices


def _interval_full_any(kv_starts, kv_ends, start, end):
    if end <= start:
        empty = torch.zeros_like(kv_starts, dtype=torch.bool)
        return empty, empty.clone()
    full = (start <= kv_starts) & (kv_ends <= end)
    any_hit = (kv_starts < end) & (start < kv_ends)
    return full, any_hit


def _build_direct_block_mask(*, padded_total, device, mask_mod, iter_q_segments):
    """Build block metadata without create_block_mask's dense O(tokens^2) mask.

    Each yielded query segment has constant allowed KV intervals, so only an
    O(num_blocks^2)
    boolean matrix is needed before moving compact metadata to the GPU.
    """
    block_size = _FLEX_BLOCK_SIZE
    num_blocks = math.ceil(padded_total / block_size)
    kv_starts = torch.arange(num_blocks, dtype=torch.int64) * block_size
    kv_ends = torch.clamp(kv_starts + block_size, max=padded_total)
    full_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.bool)
    partial_mask = torch.zeros_like(full_mask)

    for q_block in range(num_blocks):
        q_start = q_block * block_size
        q_end = min(q_start + block_size, padded_total)
        segments = list(iter_q_segments(q_start, q_end))
        if not segments:
            raise RuntimeError(
                f"direct BlockMask builder produced no segment for [{q_start}, {q_end})"
            )

        row_full = torch.ones(num_blocks, dtype=torch.bool)
        row_any = torch.zeros(num_blocks, dtype=torch.bool)
        for segment_start, segment_end, intervals in segments:
            segment_full = torch.zeros(num_blocks, dtype=torch.bool)
            segment_any = torch.zeros(num_blocks, dtype=torch.bool)
            for allowed_start, allowed_end in intervals:
                interval_full, interval_any = _interval_full_any(
                    kv_starts, kv_ends, allowed_start, allowed_end
                )
                segment_full |= interval_full
                segment_any |= interval_any

            # The token diagonal is always legal, including right-padding rows.
            eye_any = (kv_starts < segment_end) & (segment_start < kv_ends)
            segment_any |= eye_any
            row_full &= segment_full
            row_any |= segment_any

        full_mask[q_block] = row_full
        partial_mask[q_block] = row_any & ~row_full

    partial_counts, partial_indices = _ordered_block_metadata(partial_mask)
    full_counts, full_indices = _ordered_block_metadata(full_mask)
    partial_counts = partial_counts[None, None].to(device)
    partial_indices = partial_indices[None, None].to(device)
    full_counts = full_counts[None, None].to(device)
    full_indices = full_indices[None, None].to(device)
    return BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
    )


def _iter_single_sequence_segments(
    q_start, q_end, *, total_length, attention_block_size, max_prior_chunks
):
    cursor = q_start
    while cursor < q_end:
        if cursor >= total_length:
            yield cursor, q_end, ()
            return
        chunk_index = cursor // attention_block_size
        segment_end = min(q_end, (chunk_index + 1) * attention_block_size, total_length)
        first_visible = max(0, chunk_index - max_prior_chunks)
        yield (
            cursor,
            segment_end,
            (
                (
                    first_visible * attention_block_size,
                    min((chunk_index + 1) * attention_block_size, total_length),
                ),
            ),
        )
        cursor = segment_end


def build_teacher_forcing_block_mask(
    *,
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    device: torch.device | str,
    max_prior_clean_chunks=None,
    sink_size: int = 0,
):
    """Chunkwise-causal teacher-forcing mask for ``[clean | noisy]`` tokens.

    With an explicit sink, both halves reproduce the streaming cache geometry:
    a fixed clean sink, a bounded recent-clean window, and the current chunk.
    With ``sink_size=0``, clean queries use the complete preceding clean prefix.
    Noisy chunk ``i`` never sees clean chunk ``i``, so the target cannot leak.
    ``sink_size`` is measured in latent frames and must contain whole chunks.
    """
    if not _HAS_FLEX_ATTENTION:
        raise RuntimeError(
            "build_teacher_forcing_block_mask requires torch.nn.attention.flex_attention; "
            "install a torch >= 2.5 build."
        )

    total_length = num_frames * frame_seqlen * 2
    # FlexAttention compiles sparse masks at block_size=128, so Q/K/V length
    # must be a multiple of 128. Pad tokens only see themselves via eye_mask.
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    clean_ends = num_frames * frame_seqlen
    if num_frame_per_block <= 0:
        raise ValueError(f"num_frame_per_block must be positive, got {num_frame_per_block}")
    attention_block_size = frame_seqlen * num_frame_per_block
    if num_frames % num_frame_per_block != 0:
        raise ValueError(
            f"num_frames={num_frames} must be divisible by "
            f"num_frame_per_block={num_frame_per_block}"
        )
    sink_size = int(sink_size)
    if sink_size < 0 or sink_size > num_frames:
        raise ValueError(f"sink_size must be in [0, {num_frames}], got {sink_size}")
    if sink_size % num_frame_per_block != 0:
        raise ValueError(
            f"sink_size={sink_size} must contain whole {num_frame_per_block}-frame chunks"
        )
    sink_tokens = sink_size * frame_seqlen

    num_chunks = num_frames // num_frame_per_block
    default_max_prior = max(0, num_chunks - 1)
    if max_prior_clean_chunks is None:
        max_prior_clean_chunks = default_max_prior
    else:
        max_prior_clean_chunks = int(max_prior_clean_chunks)
        if max_prior_clean_chunks < 0 or max_prior_clean_chunks > default_max_prior:
            raise ValueError(
                "model.max_prior_clean_chunks must be in "
                f"[0, {default_max_prior}] for a {num_chunks}-chunk window, "
                f"got {max_prior_clean_chunks}"
            )

    def attention_mask(b, h, q_idx, kv_idx):
        is_clean = q_idx < clean_ends
        clean_chunk = q_idx // attention_block_size
        first_clean_chunk = torch.clamp(clean_chunk - max_prior_clean_chunks, min=0)
        if sink_tokens > 0:
            recent_clean = (kv_idx >= first_clean_chunk * attention_block_size) & (
                kv_idx < (clean_chunk + 1) * attention_block_size
            )
            visible_clean_sink = (kv_idx < sink_tokens) & (
                kv_idx < (clean_chunk + 1) * attention_block_size
            )
            clean_mask = is_clean & (visible_clean_sink | recent_clean)
        else:
            # Without sink tokens, clean queries attend to all preceding clean keys.
            clean_mask = is_clean & (kv_idx < (clean_chunk + 1) * attention_block_size)

        is_noisy = (q_idx >= clean_ends) & (q_idx < total_length)
        noisy_chunk = (q_idx - clean_ends) // attention_block_size
        first_prior_chunk = torch.clamp(noisy_chunk - max_prior_clean_chunks, min=0)
        prior_clean = (kv_idx >= first_prior_chunk * attention_block_size) & (
            kv_idx < noisy_chunk * attention_block_size
        )
        prior_sink = (kv_idx < sink_tokens) & (kv_idx < noisy_chunk * attention_block_size)
        noisy_start = clean_ends + noisy_chunk * attention_block_size
        own_noisy = (kv_idx >= noisy_start) & (kv_idx < noisy_start + attention_block_size)
        noise_mask = is_noisy & (prior_sink | prior_clean | own_noisy)
        eye_mask = q_idx == kv_idx
        return eye_mask | clean_mask | noise_mask

    def iter_q_segments(q_start, q_end):
        cursor = q_start
        while cursor < q_end:
            if cursor < clean_ends:
                chunk_index = cursor // attention_block_size
                segment_end = min(q_end, (chunk_index + 1) * attention_block_size, clean_ends)
                if sink_tokens > 0:
                    first_clean_chunk = max(0, chunk_index - max_prior_clean_chunks)
                    intervals = (
                        (
                            0,
                            min(
                                sink_tokens,
                                (chunk_index + 1) * attention_block_size,
                            ),
                        ),
                        (
                            first_clean_chunk * attention_block_size,
                            (chunk_index + 1) * attention_block_size,
                        ),
                    )
                else:
                    intervals = ((0, (chunk_index + 1) * attention_block_size),)
            elif cursor < total_length:
                chunk_index = (cursor - clean_ends) // attention_block_size
                segment_end = min(
                    q_end,
                    clean_ends + (chunk_index + 1) * attention_block_size,
                    total_length,
                )
                first_clean_chunk = max(0, chunk_index - max_prior_clean_chunks)
                intervals = (
                    (0, min(sink_tokens, chunk_index * attention_block_size)),
                    (
                        first_clean_chunk * attention_block_size,
                        chunk_index * attention_block_size,
                    ),
                    (
                        clean_ends + chunk_index * attention_block_size,
                        clean_ends + (chunk_index + 1) * attention_block_size,
                    ),
                )
            else:
                segment_end = q_end
                intervals = ()
            yield cursor, segment_end, intervals
            cursor = segment_end

    block_mask = _build_direct_block_mask(
        padded_total=total_length + padded_length,
        device=device,
        mask_mod=attention_mask,
        iter_q_segments=iter_q_segments,
    )
    return block_mask, padded_length


def build_inference_window_block_mask(
    *,
    num_clean_frames: int,
    num_noisy_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    device: torch.device | str,
):
    """Cache-free chunk inference mask: standard chunk-causal lower-triangular.

    Token layout: [clean_history (Fc frames) | noisy_chunk (Fn frames)] —
    flattened to chunks of `num_frame_per_block` frames each. The noisy chunk
    is just the LAST chunk in this layout, so a unified chunk-causal mask
    (every chunk attends to itself + all earlier chunks) is exactly what we
    need. There is no need for the asymmetric "clean vs noisy half" mask used
    during TF training (where clean and noisy were two same-length halves
    paired chunk-by-chunk).

    Concretely: for every chunk i (0 = first clean, K-1 = noisy), q-tokens in
    chunk i attend to kv-tokens in chunks 0..i (inclusive). This naturally
    gives:
      * clean q in chunk i attends to clean chunks 0..i  (causal among clean).
      * noisy q (chunk K-1) attends to ALL clean chunks 0..K-2 + itself.

    Right-pads to a multiple of 128 for FlexAttention.
    """
    if not _HAS_FLEX_ATTENTION:
        raise RuntimeError(
            "build_inference_window_block_mask requires torch.nn.attention.flex_attention; "
            "install a torch >= 2.5 build."
        )

    total_length = (num_clean_frames + num_noisy_frames) * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128 - total_length
    attention_block_size = frame_seqlen * num_frame_per_block

    def attention_mask(b, h, q_idx, kv_idx):
        valid_q = q_idx < total_length
        chunk_index = q_idx // attention_block_size
        causal = kv_idx < torch.clamp((chunk_index + 1) * attention_block_size, max=total_length)
        eye_mask = q_idx == kv_idx  # covers right-padding rows
        return eye_mask | (valid_q & causal)

    num_chunks = math.ceil(total_length / attention_block_size)

    def iter_q_segments(q_start, q_end):
        return _iter_single_sequence_segments(
            q_start,
            q_end,
            total_length=total_length,
            attention_block_size=attention_block_size,
            max_prior_chunks=max(0, num_chunks - 1),
        )

    block_mask = _build_direct_block_mask(
        padded_total=total_length + padded_length,
        device=device,
        mask_mod=attention_mask,
        iter_q_segments=iter_q_segments,
    )
    return block_mask, padded_length


class CausalWanSelfAttention(nn.Module):
    """Self-attention with selectable Wan RoPE and optional fused camera PRoPE."""

    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=6,
        sink_size=1,
        qk_norm=True,
        eps=1e-6,
        frame_seq_length=880,
        rope_train_frames=None,
        use_echorope=True,
        camera_attention_mode="parallel",
        camera_translation_transform="linear",
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.frame_seq_length = frame_seq_length
        self.rope_train_frames = None if rope_train_frames is None else int(rope_train_frames)
        if self.rope_train_frames is not None and self.rope_train_frames <= 0:
            raise ValueError(f"rope_train_frames must be positive, got {self.rope_train_frames}")
        self.use_echorope = bool(use_echorope)
        self.camera_attention_mode = normalize_camera_attention_mode(camera_attention_mode)
        self.camera_translation_transform = normalize_camera_translation_transform(
            camera_translation_transform
        )
        self.max_attention_size = (
            39600 if local_attn_size == -1 else local_attn_size * frame_seq_length
        )

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.num_frame_per_block_attr = 3
        self.layer_idx = -1

    def _apply_fused_prope(
        self,
        q,
        k,
        v,
        cam_viewmats,
        cam_K,
        *,
        kv_cam_viewmats=None,
        kv_cam_K=None,
    ):
        """Apply SolarWM PRoPE after ordinary RoPE, before one attention pass.

        Inputs and outputs use the native self-attention layout ``[B,L,H,D]``.
        The returned output transform applies ``P`` to attention results in
        ``[B,H,L,D]`` layout before the ordinary ``self.o`` projection.
        """
        if self.camera_attention_mode != "fused_prope":
            return q, k, v, None
        if cam_viewmats is None and cam_K is None:
            return q, k, v, None
        if cam_viewmats is None or cam_K is None:
            raise ValueError("fused_prope requires both cam_viewmats and cam_K, or neither")
        if cam_viewmats.shape[0] != q.shape[0] or cam_K.shape[0] != q.shape[0]:
            raise ValueError(
                "fused_prope camera batch must match attention batch: "
                f"q={tuple(q.shape)}, viewmats={tuple(cam_viewmats.shape)}, "
                f"K={tuple(cam_K.shape)}"
            )
        if cam_viewmats.shape[1] != q.shape[1] or cam_K.shape[1] != q.shape[1]:
            raise ValueError(
                "fused_prope query camera tensors must cover the complete query "
                "sequence after SP head sharding/all-to-all: "
                f"q_len={q.shape[1]}, viewmats_len={cam_viewmats.shape[1]}, "
                f"K_len={cam_K.shape[1]}"
            )

        if kv_cam_viewmats is None and kv_cam_K is None:
            if q.shape != k.shape or k.shape != v.shape:
                raise ValueError(
                    "fused_prope attention with different query/KV lengths "
                    "requires explicit visible-KV camera tensors"
                )
            q_t, k_t, v_t, apply_fn_o = prope_qkv(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                viewmats=cam_viewmats,
                Ks=cam_K,
                camera_translation_transform=self.camera_translation_transform,
            )
        else:
            if kv_cam_viewmats is None or kv_cam_K is None:
                raise ValueError("fused_prope requires both visible-KV viewmats and K, or neither")
            q_t, k_t, v_t, apply_fn_o = prope_qkv_separate(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                q_viewmats=cam_viewmats,
                q_Ks=cam_K,
                kv_viewmats=kv_cam_viewmats,
                kv_Ks=kv_cam_K,
                camera_translation_transform=self.camera_translation_transform,
            )
        return (
            q_t.transpose(1, 2),
            k_t.transpose(1, 2),
            v_t.transpose(1, 2),
            apply_fn_o,
        )

    @staticmethod
    def _window_relative_positions(
        *,
        current_start_frame: int,
        num_new_frames: int,
        num_context_frames: int,
        num_query_memory_frames: int = 0,
        num_sink_frames: int = 0,
        pmax: int = 1024,
        num_frame_per_block: int = 3,
    ):
        """Allocate relative positions for a local attention span.

        For streaming/chunked generation, the query block is pinned at the tail
        of a finite temporal RoPE window and the visible context is placed
        immediately before it. During the initial bulk/teacher-forcing forward
        (more than one generation block), positions ``0..B-1`` are used for the
        query span and query memory is skipped.
        """
        is_bulk_forward = num_new_frames > num_frame_per_block
        q_last = min(current_start_frame + num_new_frames - 1, pmax - 1)
        q_start = q_last - num_new_frames + 1
        local_end = q_last
        local_start = local_end - num_context_frames + 1 if num_context_frames > 0 else q_last

        if is_bulk_forward or num_query_memory_frames <= 0:
            use_memory = False
            mem_start = -1
            mem_end = -1
        else:
            use_memory = True
            mem_end = local_start - 1
            mem_start = mem_end - num_query_memory_frames + 1

        return {
            "is_bulk_forward": is_bulk_forward,
            "use_memory": use_memory,
            "q_start": q_start,
            "q_last": q_last,
            "local_start": local_start,
            "local_end": local_end,
            "mem_start": mem_start,
            "mem_end": mem_end,
            "sink_start": 0,
            "num_sink_frames": num_sink_frames,
        }

    def _echorope_pmax_frames(self, freqs, min_frames: int = 1):
        """Finite temporal horizon used by EchoRoPE during KV-cache inference.

        ``freqs`` is usually much longer (1024) than the stage1 attention window
        (for example 21 latent frames). Relative RoPE positions are capped at
        the local window horizon, so long rollouts reuse local temporal slots
        instead of drifting toward the 1024-frame table limit.
        """
        requested = max(
            1,
            int(min_frames),
            int(self.max_attention_size // max(1, self.frame_seq_length)),
        )
        return min(requested, int(freqs.shape[0]))

    def _rope_q_and_window_k(
        self,
        *,
        q,
        k_window,
        grid_sizes,
        freqs,
        frame_seqlen: int,
        num_new_tokens: int,
    ):
        """Apply EchoRoPE to an already-sliced local K window and current Q.

        ``k_window`` is the active no-sink local attention span.  Positions are
        allocated relative to that span so that the newest query frames are at
        the tail of the window.  This avoids global temporal positions growing
        beyond the training horizon during long rollouts.
        """
        num_window_frames = k_window.shape[1] // frame_seqlen if frame_seqlen > 0 else 0
        num_new_frames = (
            num_new_tokens // frame_seqlen if frame_seqlen > 0 else int(grid_sizes[0][0].item())
        )
        if num_new_frames != int(grid_sizes[0][0].item()):
            raise ValueError(
                f"EchoRoPE expects chunk token count to be whole frames: "
                f"num_new_frames={num_new_frames}, grid F={int(grid_sizes[0][0].item())}, "
                f"num_new_tokens={num_new_tokens}, frame_seqlen={frame_seqlen}"
            )

        if self.use_echorope:
            pos = self._window_relative_positions(
                current_start_frame=max(0, num_window_frames - num_new_frames),
                num_new_frames=num_new_frames,
                num_context_frames=num_window_frames,
                num_query_memory_frames=0,
                num_sink_frames=0,
                pmax=self._echorope_pmax_frames(freqs, min_frames=num_new_frames),
                num_frame_per_block=getattr(self, "num_frame_per_block_attr", num_new_frames),
            )
        else:
            q_start = num_window_frames - num_new_frames
            pos = {
                "q_start": q_start,
                "q_last": q_start + num_new_frames - 1,
                "local_start": 0,
                "local_end": max(0, num_window_frames - 1),
            }
        if pos["q_start"] < 0 or pos["local_start"] < 0:
            raise ValueError(
                "EchoRoPE underflow while assigning window-relative positions: "
                f"B={num_new_frames}, R={num_window_frames}, "
                f"q_start={pos['q_start']}, "
                f"local_start={pos['local_start']}, "
                f"pmax={self._echorope_pmax_frames(freqs, min_frames=num_new_frames)}"
            )

        q_grid = grid_sizes.clone()
        q_grid[:, 0] = num_new_frames
        rope_fn = echorope_apply if self.use_echorope else block_relativistic_rope
        roped_query = rope_fn(q, q_grid, freqs, start_frame=pos["q_start"]).type_as(q)

        if num_window_frames > 0:
            k_grid = grid_sizes.clone()
            k_grid[:, 0] = num_window_frames
            roped_k = rope_fn(k_window, k_grid, freqs, start_frame=pos["local_start"]).type_as(
                k_window
            )
        else:
            # Empty KV windows should only occur for malformed cache states;
            # fail loudly instead of returning an all-zero attention context.
            raise RuntimeError("EchoRoPE attention got an empty K window")

        return roped_query, roped_k, pos

    def _rope_sink_local_and_query(
        self,
        *,
        q,
        k_sink,
        k_local,
        grid_sizes,
        freqs,
        frame_seqlen: int,
        num_new_tokens: int,
        current_start_frame: int,
    ):
        """Apply RoPE to a fixed sink and a top-aligned recent window.

        Sink keys stay at temporal positions starting from zero. For block-
        relative RoPE, recent keys follow physical positions until the configured
        training horizon, then freeze at its tail. A cache smaller than that
        horizon therefore leaves the configured block-relative RoPE gap.
        Camera PRoPE remains independent and follows physical-frame metadata.
        """
        num_new_frames = (
            num_new_tokens // frame_seqlen if frame_seqlen > 0 else int(grid_sizes[0][0].item())
        )
        num_sink_frames = k_sink.shape[1] // frame_seqlen if frame_seqlen > 0 else 0
        num_local_frames = k_local.shape[1] // frame_seqlen if frame_seqlen > 0 else 0

        if num_new_frames != int(grid_sizes[0][0].item()):
            raise ValueError(
                f"EchoRoPE expects query chunk to contain whole frames: "
                f"num_new_frames={num_new_frames}, grid F={int(grid_sizes[0][0].item())}"
            )

        rope_fn = echorope_apply if self.use_echorope else block_relativistic_rope
        if num_sink_frames > 0:
            sink_grid = grid_sizes.clone()
            sink_grid[:, 0] = num_sink_frames
            k_sink = rope_fn(k_sink, sink_grid, freqs, start_frame=0).type_as(k_sink)

        if self.use_echorope:
            pos = self._window_relative_positions(
                current_start_frame=max(0, num_sink_frames + num_local_frames - num_new_frames),
                num_new_frames=num_new_frames,
                num_context_frames=num_local_frames,
                num_query_memory_frames=0,
                num_sink_frames=num_sink_frames,
                pmax=self._echorope_pmax_frames(freqs, min_frames=num_new_frames),
                num_frame_per_block=getattr(self, "num_frame_per_block_attr", num_new_frames),
            )
        else:
            if self.rope_train_frames is None:
                q_last = num_sink_frames + num_local_frames - 1
            else:
                global_query_last = current_start_frame + num_new_frames - 1
                q_last = min(global_query_last, self.rope_train_frames - 1)
            q_start = q_last - num_new_frames + 1
            local_start = (
                q_last - num_local_frames + 1
                if num_local_frames > 0
                else max(num_sink_frames, q_start)
            )
            pos = {
                "q_start": q_start,
                "q_last": q_last,
                "local_start": local_start,
                "local_end": q_last if num_local_frames > 0 else local_start,
            }
        if pos["q_start"] < 0 or pos["local_start"] < 0:
            raise ValueError(
                "EchoRoPE underflow while assigning sink/local positions: "
                f"B={num_new_frames}, R={num_local_frames}, sink={num_sink_frames}, "
                f"q_start={pos['q_start']}, local_start={pos['local_start']}, "
                f"pmax={self._echorope_pmax_frames(freqs, min_frames=num_new_frames)}"
            )

        q_grid = grid_sizes.clone()
        q_grid[:, 0] = num_new_frames
        roped_query = rope_fn(q, q_grid, freqs, start_frame=pos["q_start"]).type_as(q)

        if num_local_frames > 0:
            local_grid = grid_sizes.clone()
            local_grid[:, 0] = num_local_frames
            k_local = rope_fn(k_local, local_grid, freqs, start_frame=pos["local_start"]).type_as(
                k_local
            )

        return roped_query, k_sink, k_local, pos

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        block_mask=None,
        frame_indices=None,
        cam_viewmats=None,
        cam_K=None,
        kv_cam_viewmats=None,
        kv_cam_K=None,
    ):
        """
        Args:
            x: Shape [B, L, C]
            seq_lens: Shape [B]
            grid_sizes: Shape [B, 3] containing (F, H, W)
            freqs: RoPE frequencies [1024, head_dim / 2]
            kv_cache: Dict with 'k', 'v', 'global_end_index', 'local_end_index'
            current_start: Current position in the global token sequence
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        sp_enabled = is_sequence_parallel_enabled()
        cache_head_parallel = sp_enabled and kv_cache is not None
        if cache_head_parallel:
            sp_size = get_sp_size()
            sp_rank = get_sp_rank()
            if n % sp_size != 0:
                raise RuntimeError(f"num_heads={n} must be divisible by sp_size={sp_size}")
            q = torch.chunk(q, sp_size, dim=2)[sp_rank].contiguous()
            k = torch.chunk(k, sp_size, dim=2)[sp_rank].contiguous()
            v = torch.chunk(v, sp_size, dim=2)[sp_rank].contiguous()
            n = q.shape[2]
        elif sp_enabled:
            q = sequence_model_parallel_all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = sequence_model_parallel_all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = sequence_model_parallel_all_to_all_4D(v, scatter_dim=2, gather_dim=1)
            n = q.shape[2]

        # ── Stage 1 masked training / cache-free inference path ───────────────
        # When a block_mask is supplied, we run a single FlexAttention pass over
        # the caller-provided token layout: admitted Stage1 teacher forcing uses
        # [clean | noisy], and validation uses [clean_history | noisy].
        # KV cache is intentionally ignored; cache writes happen only in the
        # explicit KV-cache inference/commit path.
        if block_mask is not None:
            assert _HAS_FLEX_ATTENTION, (
                "Stage1 masked causal path requires torch.nn.attention.flex_attention; "
                "install a torch >= 2.5 build."
            )
            # EchoRoPE with explicit per-frame indices. The caller passes
            # `frame_indices` covering the valid token layout (teacher-forcing
            # clean+noisy or inference clean-history+noisy), so q/k share
            # consistent window-relative phases.
            assert frame_indices is not None, "masked causal path requires frame_indices"
            sp_seq_offsets = None
            if sp_enabled:
                # The model entrypoint has already token-sharded x/e0.
                # all-to-all restores the full sequence length per head shard
                # for q/k/v, so offset=0 is the expected RoPE coordinate for both
                # Stage0.5 and Stage1 masked paths.
                sp_seq_offsets = [0] * q.shape[0]
            if self.use_echorope:
                roped_q = echorope_apply(
                    q,
                    grid_sizes,
                    freqs,
                    frame_indices=frame_indices,
                    seq_offsets=sp_seq_offsets,
                ).type_as(v)
                roped_k = echorope_apply(
                    k,
                    grid_sizes,
                    freqs,
                    frame_indices=frame_indices,
                    seq_offsets=sp_seq_offsets,
                ).type_as(v)
            else:
                roped_q = block_relativistic_rope(
                    q,
                    grid_sizes,
                    freqs,
                    relative_frame_indices=frame_indices,
                ).type_as(v)
                roped_k = block_relativistic_rope(
                    k,
                    grid_sizes,
                    freqs,
                    relative_frame_indices=frame_indices,
                ).type_as(v)

            attn_q, attn_k, attn_v, apply_fn_o = self._apply_fused_prope(
                roped_q, roped_k, v, cam_viewmats, cam_K
            )
            x_heads = flex_attention(
                query=attn_q.transpose(1, 2),
                key=attn_k.transpose(1, 2),
                value=attn_v.transpose(1, 2),
                block_mask=block_mask,
            )
            if apply_fn_o is not None:
                x_heads = apply_fn_o(x_heads)
            x_out = x_heads.transpose(1, 2)

            if cache_head_parallel:
                x_out = sequence_model_parallel_all_gather(x_out, dim=2)
            elif sp_enabled:
                x_out = sequence_model_parallel_all_to_all_4D(x_out, scatter_dim=1, gather_dim=2)
            x_out = x_out.flatten(2)
            x_out = self.o(x_out)
            return x_out, (current_start + s, s, None)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        if kv_cache is None:
            # Stage0.5 trains on the whole bidirectional window, so an empty
            # empty cache is equivalent to full attention over the current K/V.
            num_new_tokens = q.shape[1]
            roped_query, roped_k, _ = self._rope_q_and_window_k(
                q=q,
                k_window=k,
                grid_sizes=grid_sizes,
                freqs=freqs,
                frame_seqlen=frame_seqlen,
                num_new_tokens=num_new_tokens,
            )
            attn_q, attn_k, attn_v, apply_fn_o = self._apply_fused_prope(
                roped_query, roped_k, v, cam_viewmats, cam_K
            )
            x_out = attention(attn_q, attn_k, attn_v)
            if apply_fn_o is not None:
                x_out = apply_fn_o(x_out.transpose(1, 2)).transpose(1, 2)
            if cache_head_parallel:
                x_out = sequence_model_parallel_all_gather(x_out, dim=2)
            elif sp_enabled:
                x_out = sequence_model_parallel_all_to_all_4D(x_out, scatter_dim=1, gather_dim=2)
            x_out = self.o(x_out.flatten(2))
            return x_out, (current_start + num_new_tokens, num_new_tokens, None)

        current_end = current_start + q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = q.shape[1]

        cache_update_info = None
        is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0

        if (
            self.local_attn_size != -1
            and (current_end > kv_cache["global_end_index"].item())
            and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
        ):
            # === ROLLING MODE: cache full, evict oldest non-sink tokens ===
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = (
                kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            )
            local_end_index = (
                kv_cache["local_end_index"].item()
                + current_end
                - kv_cache["global_end_index"].item()
                - num_evicted_tokens
            )
            local_start_index = local_end_index - num_new_tokens

            temp_k = kv_cache["k"].detach().clone()
            temp_v = kv_cache["v"].detach().clone()
            temp_k[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_k[
                :,
                sink_tokens + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
            temp_v[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_v[
                :,
                sink_tokens + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()

            write_start_index = (
                max(local_start_index, sink_tokens) if is_recompute else local_start_index
            )
            roped_offset = max(0, write_start_index - local_start_index)
            write_len = max(0, local_end_index - write_start_index)
            if write_len > 0:
                temp_k[:, write_start_index:local_end_index] = k[
                    :, roped_offset : roped_offset + write_len
                ]
                temp_v[:, write_start_index:local_end_index] = v[
                    :, roped_offset : roped_offset + write_len
                ]

            cache_update_info = {
                "action": "roll_and_insert",
                "sink_tokens": sink_tokens,
                "num_rolled_tokens": num_rolled_tokens,
                "num_evicted_tokens": num_evicted_tokens,
                "local_start_index": local_start_index,
                "local_end_index": local_end_index,
                "write_start_index": write_start_index,
                "write_end_index": local_end_index,
                "new_k": k[:, roped_offset : roped_offset + write_len],
                "new_v": v[:, roped_offset : roped_offset + write_len],
                "current_end": current_end,
                "is_recompute": is_recompute,
            }
        else:
            # === DIRECT INSERT MODE: cache not yet full ===
            local_end_index = (
                kv_cache["local_end_index"].item()
                + current_end
                - kv_cache["global_end_index"].item()
            )
            local_start_index = local_end_index - num_new_tokens

            temp_k = kv_cache["k"].detach().clone()
            temp_v = kv_cache["v"].detach().clone()

            write_start_index = (
                max(local_start_index, sink_tokens) if is_recompute else local_start_index
            )
            if sink_recache_after_switch:
                write_start_index = local_start_index
            roped_offset = max(0, write_start_index - local_start_index)
            write_len = max(0, local_end_index - write_start_index)
            if write_len > 0:
                temp_k[:, write_start_index:local_end_index] = k[
                    :, roped_offset : roped_offset + write_len
                ]
                temp_v[:, write_start_index:local_end_index] = v[
                    :, roped_offset : roped_offset + write_len
                ]

            cache_update_info = {
                "action": "direct_insert",
                "local_start_index": local_start_index,
                "local_end_index": local_end_index,
                "write_start_index": write_start_index,
                "write_end_index": local_end_index,
                "new_k": k[:, roped_offset : roped_offset + write_len],
                "new_v": v[:, roped_offset : roped_offset + write_len],
                "current_end": current_end,
                "is_recompute": is_recompute,
            }

        # Attention: sink tokens + local window
        if sink_tokens > 0:
            local_budget = self.max_attention_size - sink_tokens
            active_sink_tokens = min(sink_tokens, local_end_index)
            k_sink_raw = temp_k[:, :active_sink_tokens]
            v_sink = temp_v[:, :active_sink_tokens]
            if local_budget > 0:
                local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                k_local_raw = temp_k[:, local_start_for_window:local_end_index]
                v_local = temp_v[:, local_start_for_window:local_end_index]
                roped_query, k_sink, k_local, _ = self._rope_sink_local_and_query(
                    q=q,
                    k_sink=k_sink_raw,
                    k_local=k_local_raw,
                    grid_sizes=grid_sizes,
                    freqs=freqs,
                    frame_seqlen=frame_seqlen,
                    num_new_tokens=num_new_tokens,
                    current_start_frame=current_start // frame_seqlen,
                )
                k_cat = torch.cat([k_sink, k_local], dim=1)
                v_cat = torch.cat([v_sink, v_local], dim=1)
            else:
                roped_query, k_sink, _, _ = self._rope_sink_local_and_query(
                    q=q,
                    k_sink=k_sink_raw,
                    k_local=temp_k[:, :0],
                    grid_sizes=grid_sizes,
                    freqs=freqs,
                    frame_seqlen=frame_seqlen,
                    num_new_tokens=num_new_tokens,
                    current_start_frame=current_start // frame_seqlen,
                )
                k_cat = k_sink
                v_cat = v_sink
            visible_cam_viewmats = None
            visible_cam_K = None
            if kv_cam_viewmats is not None or kv_cam_K is not None:
                if kv_cam_viewmats is None or kv_cam_K is None:
                    raise ValueError("fused_prope KV-cache requires both shared camera tensors")
                cam_sink = kv_cam_viewmats[:, :active_sink_tokens]
                K_sink = kv_cam_K[:, :active_sink_tokens]
                if local_budget > 0:
                    cam_local = kv_cam_viewmats[:, local_start_for_window:local_end_index]
                    K_local = kv_cam_K[:, local_start_for_window:local_end_index]
                    visible_cam_viewmats = torch.cat([cam_sink, cam_local], dim=1)
                    visible_cam_K = torch.cat([K_sink, K_local], dim=1)
                else:
                    visible_cam_viewmats = cam_sink
                    visible_cam_K = K_sink
            attn_q, attn_k, attn_v, apply_fn_o = self._apply_fused_prope(
                roped_query,
                k_cat,
                v_cat,
                cam_viewmats,
                cam_K,
                kv_cam_viewmats=visible_cam_viewmats,
                kv_cam_K=visible_cam_K,
            )
            x = attention(attn_q, attn_k, attn_v)
            if apply_fn_o is not None:
                x = apply_fn_o(x.transpose(1, 2)).transpose(1, 2)
        else:
            window_start = max(0, local_end_index - self.max_attention_size)
            k_window_raw = temp_k[:, window_start:local_end_index]
            roped_query, roped_k_window, _ = self._rope_q_and_window_k(
                q=q,
                k_window=k_window_raw,
                grid_sizes=grid_sizes,
                freqs=freqs,
                frame_seqlen=frame_seqlen,
                num_new_tokens=num_new_tokens,
            )
            visible_cam_viewmats = None
            visible_cam_K = None
            if kv_cam_viewmats is not None or kv_cam_K is not None:
                if kv_cam_viewmats is None or kv_cam_K is None:
                    raise ValueError("fused_prope KV-cache requires both shared camera tensors")
                visible_cam_viewmats = kv_cam_viewmats[:, window_start:local_end_index]
                visible_cam_K = kv_cam_K[:, window_start:local_end_index]
            attn_q, attn_k, attn_v, apply_fn_o = self._apply_fused_prope(
                roped_query,
                roped_k_window,
                temp_v[:, window_start:local_end_index],
                cam_viewmats,
                cam_K,
                kv_cam_viewmats=visible_cam_viewmats,
                kv_cam_K=visible_cam_K,
            )
            x = attention(attn_q, attn_k, attn_v)
            if apply_fn_o is not None:
                x = apply_fn_o(x.transpose(1, 2)).transpose(1, 2)

        if cache_head_parallel:
            x = sequence_model_parallel_all_gather(x, dim=2)
        elif sp_enabled:
            x = sequence_model_parallel_all_to_all_4D(x, scatter_dim=1, gather_dim=2)

        x = x.flatten(2)
        x = self.o(x)
        return x, (current_end, local_end_index, cache_update_info)


class CausalPropeSelfAttention(nn.Module):
    """PRoPE self-attention with optional KV cache for camera-controlled inference."""

    def __init__(
        self,
        dim,
        attn_dim,
        num_heads,
        window_size=(-1, -1),
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        frame_seq_length=880,
        camera_translation_transform="linear",
    ):
        assert dim % num_heads == 0
        assert attn_dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.window_size = window_size
        self.frame_seq_length = frame_seq_length
        self.camera_translation_transform = normalize_camera_translation_transform(
            camera_translation_transform
        )
        self.max_attention_size = (
            39600 if local_attn_size == -1 else local_attn_size * frame_seq_length
        )

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)

        self.norm_q = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x,
        cam_viewmats,
        cam_K,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        cache_update_policy="commit_detached",
        block_mask=None,
    ):
        """
        Args:
            x: Shape [B, L, C]
            cam_viewmats: Camera view matrices
            cam_K: Camera intrinsics
            kv_cache: Optional KV cache dict. When None, runs full attention over current chunk.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q_proj(x)).view(b, s, n, d)
        k = self.norm_k(self.k_proj(x)).view(b, s, n, d)
        v = self.v_proj(x).view(b, s, n, d)

        # Apply PRoPE (Positional Rotary Position Embedding from camera parameters)
        q_t, k_t, v_t, apply_fn_o = prope_qkv(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            viewmats=cam_viewmats,
            Ks=cam_K,
            camera_translation_transform=self.camera_translation_transform,
        )
        proped_q = q_t.transpose(1, 2)
        proped_k = k_t.transpose(1, 2)
        proped_v = v_t.transpose(1, 2)

        sp_enabled = is_sequence_parallel_enabled()
        cache_head_parallel = sp_enabled and kv_cache is not None
        if cache_head_parallel:
            sp_size = get_sp_size()
            sp_rank = get_sp_rank()
            if proped_q.shape[2] % sp_size != 0:
                raise RuntimeError(
                    f"PRoPE num_heads={proped_q.shape[2]} must be divisible by sp_size={sp_size}"
                )
            proped_q = torch.chunk(proped_q, sp_size, dim=2)[sp_rank].contiguous()
            proped_k = torch.chunk(proped_k, sp_size, dim=2)[sp_rank].contiguous()
            proped_v = torch.chunk(proped_v, sp_size, dim=2)[sp_rank].contiguous()
        elif sp_enabled:
            proped_q = sequence_model_parallel_all_to_all_4D(proped_q, scatter_dim=2, gather_dim=1)
            proped_k = sequence_model_parallel_all_to_all_4D(proped_k, scatter_dim=2, gather_dim=1)
            proped_v = sequence_model_parallel_all_to_all_4D(proped_v, scatter_dim=2, gather_dim=1)

        if block_mask is not None:
            # ── Stage 1 teacher-forcing / cache-free validation path ──────────
            # PRoPE is already baked into proped_q/k by prope_qkv; we just feed
            # them through flex_attention with the caller-provided block mask.
            assert _HAS_FLEX_ATTENTION, (
                "Stage1 masked causal path requires torch.nn.attention.flex_attention; "
                "install a torch >= 2.5 build."
            )
            x_out = flex_attention(
                query=proped_q.transpose(1, 2),
                key=proped_k.transpose(1, 2),
                value=proped_v.transpose(1, 2),
                block_mask=block_mask,
            ).transpose(1, 2)
        elif kv_cache is None:
            # No cache: full attention over current chunk
            x_out = attention(proped_q, proped_k, proped_v)
        else:
            # KV cache mode with rolling cache support
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            num_new_tokens = s
            current_end = current_start + num_new_tokens
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            is_recompute = (current_end <= kv_cache["global_end_index"].item()) and (
                current_start > 0
            )
            attn_k = kv_cache["k"]
            attn_v = kv_cache["v"]

            if (
                self.local_attn_size != -1
                and (current_end > kv_cache["global_end_index"].item())
                and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
            ):
                # === ROLLING MODE ===
                num_evicted_tokens = (
                    num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                )
                num_rolled_tokens = (
                    kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                )
                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens

                write_start_index = (
                    max(local_start_index, sink_tokens) if is_recompute else local_start_index
                )
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if cache_update_policy == "none":
                    # No-commit forwards (training loss / validation denoise) must
                    # still attend to the current chunk. Build a temporary rolled
                    # cache with this chunk's PRoPE K/V, but leave kv_cache state
                    # untouched so later clean-context commits are well-defined.
                    attn_k = kv_cache["k"].detach().clone()
                    attn_v = kv_cache["v"].detach().clone()
                    attn_k[:, sink_tokens : sink_tokens + num_rolled_tokens] = attn_k[
                        :,
                        sink_tokens + num_evicted_tokens : sink_tokens
                        + num_evicted_tokens
                        + num_rolled_tokens,
                    ].clone()
                    attn_v[:, sink_tokens : sink_tokens + num_rolled_tokens] = attn_v[
                        :,
                        sink_tokens + num_evicted_tokens : sink_tokens
                        + num_evicted_tokens
                        + num_rolled_tokens,
                    ].clone()
                    if write_len > 0:
                        attn_k[:, write_start_index:local_end_index] = proped_k[
                            :, roped_offset : roped_offset + write_len
                        ]
                        attn_v[:, write_start_index:local_end_index] = proped_v[
                            :, roped_offset : roped_offset + write_len
                        ]
                else:
                    with torch.no_grad():
                        kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache[
                            "k"
                        ][
                            :,
                            sink_tokens + num_evicted_tokens : sink_tokens
                            + num_evicted_tokens
                            + num_rolled_tokens,
                        ].clone()
                        kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache[
                            "v"
                        ][
                            :,
                            sink_tokens + num_evicted_tokens : sink_tokens
                            + num_evicted_tokens
                            + num_rolled_tokens,
                        ].clone()

                    if write_len > 0:
                        with torch.no_grad():
                            kv_cache["k"][:, write_start_index:local_end_index] = proped_k[
                                :, roped_offset : roped_offset + write_len
                            ].detach()
                            kv_cache["v"][:, write_start_index:local_end_index] = proped_v[
                                :, roped_offset : roped_offset + write_len
                            ].detach()
            else:
                # === DIRECT INSERT MODE ===
                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                )
                local_start_index = local_end_index - num_new_tokens

                write_start_index = (
                    max(local_start_index, sink_tokens) if is_recompute else local_start_index
                )
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if cache_update_policy == "none":
                    attn_k = kv_cache["k"].detach().clone()
                    attn_v = kv_cache["v"].detach().clone()
                    if write_len > 0:
                        attn_k[:, write_start_index:local_end_index] = proped_k[
                            :, roped_offset : roped_offset + write_len
                        ]
                        attn_v[:, write_start_index:local_end_index] = proped_v[
                            :, roped_offset : roped_offset + write_len
                        ]
                else:
                    if write_len > 0:
                        with torch.no_grad():
                            kv_cache["k"][:, write_start_index:local_end_index] = proped_k[
                                :, roped_offset : roped_offset + write_len
                            ].detach()
                            kv_cache["v"][:, write_start_index:local_end_index] = proped_v[
                                :, roped_offset : roped_offset + write_len
                            ].detach()

            # Attention: sink tokens + local window
            if sink_tokens > 0:
                local_budget = self.max_attention_size - sink_tokens
                k_sink = attn_k[:, :sink_tokens]
                v_sink = attn_v[:, :sink_tokens]
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    k_local = attn_k[:, local_start_for_window:local_end_index]
                    v_local = attn_v[:, local_start_for_window:local_end_index]
                    k_cat = torch.cat([k_sink, k_local], dim=1)
                    v_cat = torch.cat([v_sink, v_local], dim=1)
                else:
                    k_cat = k_sink
                    v_cat = v_sink
                x_out = attention(proped_q, k_cat, v_cat)
            else:
                window_start = max(0, local_end_index - self.max_attention_size)
                x_out = attention(
                    proped_q,
                    attn_k[:, window_start:local_end_index],
                    attn_v[:, window_start:local_end_index],
                )

            if not is_recompute and cache_update_policy != "none":
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

        if cache_head_parallel:
            x_out = sequence_model_parallel_all_gather(x_out, dim=2)
        elif sp_enabled:
            x_out = sequence_model_parallel_all_to_all_4D(x_out, scatter_dim=1, gather_dim=2)

        # Apply inverse PRoPE
        x = apply_fn_o(x_out.transpose(1, 2)).transpose(1, 2)
        x = x.flatten(2)
        x = self.out_proj(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.add_control_adapter = kwargs.get("add_control_adapter", False)
        self.cam_method = kwargs.get("cam_method", None)
        self.attn_compress = kwargs.get("attn_compress", 1)
        self.layer_idx = kwargs.get("layer_idx", None)
        cam_self_attn_layers = kwargs.get("cam_self_attn_layers", None)
        frame_seq_length = int(kwargs.get("frame_seq_length", 880))
        rope_train_frames = kwargs.get("rope_train_frames", None)
        use_echorope = bool(kwargs.get("use_echorope", True))
        self.camera_attention_mode = normalize_camera_attention_mode(
            kwargs.get("camera_attention_mode", "parallel")
        )
        self.camera_translation_transform = normalize_camera_translation_transform(
            kwargs.get("camera_translation_transform", "linear")
        )

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim,
            num_heads,
            local_attn_size,
            sink_size,
            qk_norm,
            eps,
            frame_seq_length=frame_seq_length,
            rope_train_frames=rope_train_frames,
            use_echorope=use_echorope,
            camera_attention_mode=self.camera_attention_mode,
            camera_translation_transform=self.camera_translation_transform,
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )

        # PRoPE self-attention branch for camera control
        add_cam_attn = self.add_control_adapter and self.cam_method == "prope"
        if add_cam_attn and cam_self_attn_layers is not None:
            add_cam_attn = self.layer_idx in cam_self_attn_layers
        self.camera_attention_enabled = bool(add_cam_attn)
        if add_cam_attn and self.camera_attention_mode == "parallel":
            self.cam_self_attn = CausalPropeSelfAttention(
                dim,
                dim // self.attn_compress,
                num_heads,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                qk_norm=qk_norm,
                eps=eps,
                frame_seq_length=frame_seq_length,
                camera_translation_transform=self.camera_translation_transform,
            )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        kv_cache,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        cam_viewmats=None,
        cam_K=None,
        sink_recache_after_switch=False,
        cache_update_policy="commit_detached",
        block_mask=None,
        frame_indices=None,
        kv_cam_viewmats=None,
        kv_cam_K=None,
    ):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        attn_input = (
            self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
        ).flatten(1, 2)
        y, cache_update_info = self.self_attn(
            attn_input,
            seq_lens,
            grid_sizes,
            freqs,
            kv_cache,
            current_start,
            cache_start,
            sink_recache_after_switch,
            block_mask=block_mask,
            frame_indices=frame_indices,
            cam_viewmats=(
                cam_viewmats
                if self.camera_attention_enabled and self.camera_attention_mode == "fused_prope"
                else None
            ),
            cam_K=(
                cam_K
                if self.camera_attention_enabled and self.camera_attention_mode == "fused_prope"
                else None
            ),
            kv_cam_viewmats=(
                kv_cam_viewmats
                if self.camera_attention_enabled and self.camera_attention_mode == "fused_prope"
                else None
            ),
            kv_cam_K=(
                kv_cam_K
                if self.camera_attention_enabled and self.camera_attention_mode == "fused_prope"
                else None
            ),
        )

        # PRoPE camera attention (parallel branch)
        if hasattr(self, "cam_self_attn") and cam_viewmats is not None and cam_K is not None:
            prope_kv_cache = None
            if kv_cache is not None and "prope_k" in kv_cache:
                prope_kv_cache = {
                    "k": kv_cache["prope_k"],
                    "v": kv_cache["prope_v"],
                    "global_end_index": kv_cache["prope_global_end_index"],
                    "local_end_index": kv_cache["prope_local_end_index"],
                }
            y = y + self.cam_self_attn(
                attn_input,
                cam_viewmats,
                cam_K,
                seq_lens,
                grid_sizes,
                freqs,
                kv_cache=prope_kv_cache,
                current_start=current_start,
                cache_start=cache_start,
                cache_update_policy=cache_update_policy,
                block_mask=block_mask,
            )

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & FFN
        x = x + self.cross_attn(
            self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
        )
        y = self.ffn(
            (
                self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]
            ).flatten(1, 2)
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)

        return x, cache_update_info


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
        )
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    """Wan diffusion backbone for causal camera-controlled training and inference."""

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["CausalWanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=6,
        sink_size=1,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        cam_method="prope",
        attn_compress=1,
        cam_self_attn_layers=None,
        frame_seq_length=880,
        max_prior_clean_chunks=None,
        rope_train_frames=None,
        use_echorope=True,
        camera_attention_mode="parallel",
        camera_translation_transform="linear",
        flow_objective="flow_matching",
        anyflow_gate=0.25,
        anyflow_deltatime_type="r",
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.frame_seq_length = frame_seq_length
        self.max_prior_clean_chunks = (
            None if max_prior_clean_chunks is None else int(max_prior_clean_chunks)
        )
        self.rope_train_frames = None if rope_train_frames is None else int(rope_train_frames)
        self.use_echorope = bool(use_echorope)
        self.camera_attention_mode = normalize_camera_attention_mode(camera_attention_mode)
        self.camera_translation_transform = normalize_camera_translation_transform(
            camera_translation_transform
        )
        self.flow_objective = str(flow_objective)
        if self.flow_objective not in {"flow_matching", "anyflow_forward_map"}:
            raise ValueError(f"unsupported flow_objective={self.flow_objective!r}")
        self.anyflow_deltatime_type = str(anyflow_deltatime_type)
        if self.flow_objective == "anyflow_forward_map" and self.anyflow_deltatime_type != "r":
            raise ValueError("SolarWM AnyFlow currently supports anyflow_deltatime_type='r' only")
        self._num_frame_per_block = 1
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # transformer blocks
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    add_control_adapter=add_control_adapter,
                    cam_method=cam_method,
                    attn_compress=attn_compress,
                    layer_idx=layer_idx,
                    cam_self_attn_layers=cam_self_attn_layers,
                    frame_seq_length=frame_seq_length,
                    rope_train_frames=self.rope_train_frames,
                    use_echorope=self.use_echorope,
                    camera_attention_mode=self.camera_attention_mode,
                    camera_translation_transform=self.camera_translation_transform,
                )
                for layer_idx in range(num_layers)
            ]
        )
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers
            block.self_attn.num_frame_per_block_attr = self.num_frame_per_block

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        self.init_weights()
        # Create the second time MLP only for AnyFlow, after all base modules
        # have been initialized. Deepcopy consumes no RNG, so the default and
        # explicit flow_matching construction order remains identical.
        if self.flow_objective == "anyflow_forward_map":
            self.delta_embedding = copy.deepcopy(self.time_embedding)
            self.register_buffer(
                "anyflow_gate_buffer",
                torch.tensor([float(anyflow_gate)], dtype=torch.float32),
                persistent=False,
            )

    @property
    def uses_anyflow(self):
        return self.flow_objective == "anyflow_forward_map"

    def initialize_anyflow_delta_from_time(self):
        """Reset delta embedding from the currently loaded time embedding."""
        if not self.uses_anyflow:
            return False
        self.delta_embedding.load_state_dict(self.time_embedding.state_dict(), strict=True)
        return True

    def _time_condition(self, timestep, reference, r_timestep=None):
        """Return the base time embedding or the AnyFlow mixed (t,r) embedding."""
        time_emb = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(reference)
        )
        if not self.uses_anyflow:
            return time_emb
        if r_timestep is None:
            raise ValueError("AnyFlow model forward requires r_timestep")
        if r_timestep.shape != timestep.shape:
            raise ValueError(
                f"r_timestep shape {tuple(r_timestep.shape)} != timestep shape {tuple(timestep.shape)}"
            )
        delta_emb = self.delta_embedding(
            sinusoidal_embedding_1d(self.freq_dim, r_timestep.flatten()).type_as(reference)
        )
        gate = self.anyflow_gate_buffer.to(device=time_emb.device, dtype=time_emb.dtype)
        return (1.0 - gate) * time_emb + gate * delta_emb

    @property
    def num_frame_per_block(self):
        return self._num_frame_per_block

    @num_frame_per_block.setter
    def num_frame_per_block(self, value):
        self._num_frame_per_block = int(value)
        if hasattr(self, "blocks"):
            for block in self.blocks:
                block.self_attn.num_frame_per_block_attr = int(value)

    def _infer_num_frame_per_block_from_t(self, t):
        """Best-effort chunk size inference for KV-cache inference calls."""
        if t is None or t.ndim < 2 or self.frame_seq_length <= 0:
            return self.num_frame_per_block
        frames = int(t.shape[1]) // int(self.frame_seq_length)
        return max(1, frames)

    @staticmethod
    def _slice_current_camera_tokens(
        tensor,
        *,
        name: str,
        current_start: int,
        num_new_tokens: int,
    ):
        """Accept either current-chunk or full-trajectory camera tensors."""
        if tensor.shape[1] == num_new_tokens:
            return tensor
        end = int(current_start) + int(num_new_tokens)
        if current_start >= 0 and tensor.shape[1] >= end:
            return tensor[:, current_start:end]
        raise ValueError(
            f"fused_prope {name} must cover the current query chunk: "
            f"camera_len={tensor.shape[1]}, current_start={current_start}, "
            f"num_new_tokens={num_new_tokens}"
        )

    def _stage_fused_camera_cache(
        self,
        kv_cache,
        *,
        cam_viewmats,
        cam_K,
        current_start: int,
        num_new_tokens: int,
        frame_seqlen: int,
    ):
        """Stage one shared camera-metadata cache matching the raw K/V layout.

        Cached fused PRoPE selects separate camera transforms for the current
        query and visible K/V. K/V uses a fixed rolling buffer per block, so the
        buffer's token layout is staged once on the first cache dictionary and
        reused in every transformer block. No camera collective or per-block
        camera cache is introduced.
        """
        if cam_viewmats is None or cam_K is None:
            raise ValueError("fused_prope camera-conditioned KV-cache requires both viewmats and K")
        if not kv_cache:
            raise ValueError("fused_prope KV-cache requires a non-empty block cache")

        current_start = int(current_start)
        num_new_tokens = int(num_new_tokens)
        cam_viewmats = self._slice_current_camera_tokens(
            cam_viewmats,
            name="viewmats",
            current_start=current_start,
            num_new_tokens=num_new_tokens,
        )
        cam_K = self._slice_current_camera_tokens(
            cam_K,
            name="K",
            current_start=current_start,
            num_new_tokens=num_new_tokens,
        )
        if cam_viewmats.shape[0] != cam_K.shape[0]:
            raise ValueError(
                "fused_prope viewmats/K batch mismatch: "
                f"{tuple(cam_viewmats.shape)} vs {tuple(cam_K.shape)}"
            )

        cache = kv_cache[0]
        cache_size = int(cache["k"].shape[1])
        batch = int(cache["k"].shape[0])
        if cam_viewmats.shape[0] != batch:
            raise ValueError(
                "fused_prope camera/cache batch mismatch: "
                f"camera={cam_viewmats.shape[0]}, cache={batch}"
            )
        global_end = int(cache["global_end_index"].item())
        previous_local_end = int(cache["local_end_index"].item())
        current_end = current_start + num_new_tokens
        sink_tokens = int(self.sink_size) * int(frame_seqlen)
        is_recompute = current_end <= global_end and current_start > 0

        state_key = "_fused_prope_camera_metadata"
        state = cache.get(state_key)
        if state is None:
            if previous_local_end > 0:
                raise RuntimeError(
                    "fused_prope found populated raw K/V history without shared "
                    "camera metadata; reset/rebuild the cache before enabling camera"
                )
            state = {
                "viewmats": cam_viewmats.new_zeros(batch, cache_size, 4, 4),
                "K": cam_K.new_zeros(batch, cache_size, 3, 3),
            }
            cache[state_key] = state
        expected_view_shape = (batch, cache_size, 4, 4)
        expected_K_shape = (batch, cache_size, 3, 3)
        if tuple(state["viewmats"].shape) != expected_view_shape:
            raise RuntimeError(
                "fused_prope shared viewmat cache shape mismatch: "
                f"expected={expected_view_shape}, got={tuple(state['viewmats'].shape)}"
            )
        if tuple(state["K"].shape) != expected_K_shape:
            raise RuntimeError(
                "fused_prope shared K cache shape mismatch: "
                f"expected={expected_K_shape}, got={tuple(state['K'].shape)}"
            )
        if state["viewmats"].device != cam_viewmats.device or state["K"].device != cam_K.device:
            raise RuntimeError("fused_prope shared camera cache is on the wrong device")

        temp_viewmats = state["viewmats"].detach().clone()
        temp_K = state["K"].detach().clone()
        rolling = (
            self.local_attn_size != -1
            and current_end > global_end
            and num_new_tokens + previous_local_end > cache_size
        )
        if rolling:
            num_evicted_tokens = num_new_tokens + previous_local_end - cache_size
            num_rolled_tokens = previous_local_end - num_evicted_tokens - sink_tokens
            local_end_index = previous_local_end + current_end - global_end - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            temp_viewmats[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_viewmats[
                :,
                sink_tokens + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
            temp_K[:, sink_tokens : sink_tokens + num_rolled_tokens] = temp_K[
                :,
                sink_tokens + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
        else:
            local_end_index = previous_local_end + current_end - global_end
            local_start_index = local_end_index - num_new_tokens

        write_start_index = (
            max(local_start_index, sink_tokens) if is_recompute else local_start_index
        )
        camera_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_viewmats[:, write_start_index:local_end_index] = cam_viewmats[
                :, camera_offset : camera_offset + write_len
            ]
            temp_K[:, write_start_index:local_end_index] = cam_K[
                :, camera_offset : camera_offset + write_len
            ]

        update = {
            "state": state,
            "temp_viewmats": temp_viewmats,
            "temp_K": temp_K,
        }
        return cam_viewmats, cam_K, temp_viewmats, temp_K, update

    @staticmethod
    def _commit_fused_camera_cache(update):
        """Commit staged shared metadata only after all blocks succeed."""
        if update is None:
            return
        with torch.no_grad():
            update["state"]["viewmats"].copy_(update["temp_viewmats"])
            update["state"]["K"].copy_(update["temp_K"])

    # ──────────────────────────────────────────────────────────────────
    # Stage 1 standard teacher-forcing path (single forward + FlexAttention)
    # ──────────────────────────────────────────────────────────────────
    def forward_train_tf(
        self,
        x,  # noisy: tensor [B, C_in, F, H, W] OR list of [C_in, F, H, W]
        clean_x,  # clean: same layout as x
        t,  # noisy per-token timestep   [B, F * frame_seq_length]
        aug_t,  # clean per-token timestep   [B, F * frame_seq_length]
        context,  # list/tensor of [L, text_dim]
        seq_len,
        num_frame_per_block,
        y=None,
        y_camera=None,
        r=None,
        aug_r=None,
    ):
        """One-shot teacher-forcing forward over [clean | noisy] tokens.

        Processing steps:
          - Patch-embed clean and noisy independently, concat along token dim.
          - Build a chunkwise-causal TF mask (clean q sees prior+self clean
            chunks; noisy q sees strict-prior clean + self noisy chunk).
          - Run all transformer blocks once with the mask. KV caches are NOT
            allocated and NOT touched.
          - Discard the clean half of the output, return only the noisy half.

        Camera viewmats/K (in y_camera) are concatenated to cover both halves.
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        self.num_frame_per_block = int(num_frame_per_block)

        # Bring (B, C_in, F, H, W) into the canonical list-of-clip form used by
        # the existing inference forward, so we can reuse the patch_embedding /
        # text_embedding plumbing.
        def _as_list(z):
            if isinstance(z, list):
                return z
            assert z.dim() == 5, f"expected (B, C, F, H, W), got {tuple(z.shape)}"
            return [u for u in z]

        x_list = _as_list(x)
        clean_list = _as_list(clean_x)
        if self.model_type == "i2v" and y is None:
            raise ValueError("Wan I2V teacher forcing requires y")
        if y is not None:
            y_list = _as_list(y)
            if any(item.shape[1] != x_list[index].shape[1] for index, item in enumerate(y_list)):
                raise ValueError("Wan I2V teacher-forcing y must cover the full target window")
            x_list = [torch.cat([u, v], dim=0) for u, v in zip(x_list, y_list)]
            clean_list = [torch.cat([u, v], dim=0) for u, v in zip(clean_list, y_list)]

        # patch embed
        x_pe = [self.patch_embedding(u.unsqueeze(0)) for u in x_list]
        clean_pe = [self.patch_embedding(u.unsqueeze(0)) for u in clean_list]
        # grid_sizes (for noisy half) tracks (F_lat_after_patch, H_after_patch, W_after_patch)
        grid_sizes_noisy = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x_pe])
        x_tokens = [u.flatten(2).transpose(1, 2) for u in x_pe]
        clean_tokens = [u.flatten(2).transpose(1, 2) for u in clean_pe]

        seq_lens_noisy = torch.tensor([u.size(1) for u in x_tokens], dtype=torch.long)
        assert seq_lens_noisy.max() <= seq_len

        # Pad ragged sequences (Wan2.2 path uses uniform shape so this is no-op normally).
        x_padded = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_lens_noisy[0] - u.size(1), u.size(2))], dim=1)
                for u in x_tokens
            ]
        )
        clean_padded = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_lens_noisy[0] - u.size(1), u.size(2))], dim=1)
                for u in clean_tokens
            ]
        )

        # Token-dim cat: [clean | noisy]
        full = torch.cat([clean_padded, x_padded], dim=1)  # (B, 2*F*s, dim)

        # Time embeddings: clean uses aug_t, noisy uses t. Both are per-token
        # (B, F*s); concat them along the token dim and run a single embedding +
        # projection so e0 has the same token-axis layout as `full`.
        t_full = torch.cat([aug_t, t], dim=1)  # (B, 2*F*s)
        if self.uses_anyflow:
            if r is None:
                raise ValueError("AnyFlow teacher forcing requires noisy r")
            if aug_r is None:
                aug_r = aug_t
            r_full = torch.cat([aug_r, r], dim=1)
        else:
            r_full = None
        e_full = self._time_condition(t_full, full, r_full)  # (B*2F*s, dim)
        e0 = (
            self.time_projection(e_full)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t_full.shape)
        )  # (B, 2F*s, 6, dim)
        # Head modulation: same as inference forward, only over noisy tokens.
        e_for_head = self._time_condition(t, full, r).unflatten(
            dim=0, sizes=t.shape
        )  # (B, F*s, dim)

        # text embedding (same for clean & noisy)
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )

        # Camera viewmats / K: duplicate to cover both halves with the SAME trajectory
        cam_viewmats = None
        cam_K = None
        if y_camera is not None and isinstance(y_camera, dict):
            cam_viewmats = y_camera["viewmats"]
            cam_K = y_camera["K"]
            cam_viewmats = torch.cat([cam_viewmats, cam_viewmats], dim=1)
            cam_K = torch.cat([cam_K, cam_K], dim=1)

        # Build the FlexAttention block mask once per call (cached on the model).
        F_lat = int(grid_sizes_noisy[0][0].item())
        h_lat = int(grid_sizes_noisy[0][1].item())
        w_lat = int(grid_sizes_noisy[0][2].item())
        s_per_frame = h_lat * w_lat
        max_prior_clean_chunks = self.max_prior_clean_chunks
        mask_key = (
            "tf",
            max_prior_clean_chunks,
            int(self.sink_size),
            F_lat,
            s_per_frame,
            num_frame_per_block,
            str(device),
        )
        if not hasattr(self, "_tf_block_mask_cache"):
            self._tf_block_mask_cache = {}
        cached = self._tf_block_mask_cache.get(mask_key)
        if cached is None:
            block_mask, padded_length = build_teacher_forcing_block_mask(
                num_frames=F_lat,
                frame_seqlen=s_per_frame,
                num_frame_per_block=num_frame_per_block,
                device=device,
                max_prior_clean_chunks=max_prior_clean_chunks,
                sink_size=int(self.sink_size),
            )
            self._tf_block_mask_cache[mask_key] = (block_mask, padded_length)
        else:
            block_mask, padded_length = cached

        # FlexAttention requires Q/K/V length to be the mask length, so right-pad
        # the token sequence (and time modulation, viewmats) to match.
        if padded_length > 0:
            full = torch.cat(
                [full, full.new_zeros(full.shape[0], padded_length, full.shape[2])],
                dim=1,
            )
            # e0 is (B, 2F*s, 6, dim); pad along token dim by padded_length
            e0 = torch.cat(
                [e0, e0.new_zeros(e0.shape[0], padded_length, e0.shape[2], e0.shape[3])],
                dim=1,
            )
            if cam_viewmats is not None:
                eye_vm = torch.eye(4, device=cam_viewmats.device, dtype=cam_viewmats.dtype)
                eye_K = torch.eye(3, device=cam_K.device, dtype=cam_K.dtype)
                pad_vm = (
                    eye_vm.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(cam_viewmats.shape[0], padded_length, -1, -1)
                )
                pad_K = (
                    eye_K.unsqueeze(0).unsqueeze(0).expand(cam_K.shape[0], padded_length, -1, -1)
                )
                cam_viewmats = torch.cat([cam_viewmats, pad_vm], dim=1)
                cam_K = torch.cat([cam_K, pad_K], dim=1)

        # grid_sizes describes ONLY the valid (clean+noisy) 2F frames. The pad
        # tokens at the tail are handled inside echorope_apply by the
        # `x[i, seq_len:]` passthrough — they are NOT rotated. Mask only allows
        # them to attend to themselves (eye_mask), so their RoPE phase doesn't
        # matter.
        grid_sizes_full = grid_sizes_noisy.clone()
        grid_sizes_full[:, 0] = 2 * F_lat

        # frame_indices: clean section uses [0..F-1], noisy section uses [0..F-1].
        # No entries for pad tokens — echorope_apply only RoPEs the
        # first 2*F*s tokens (= grid_sizes_full[0].prod() = 2F * H * W).
        frame_indices = torch.cat(
            [
                torch.arange(F_lat, device=device),
                torch.arange(F_lat, device=device),
            ]
        )

        # Keep the full [clean | noisy | Flex pad] layout contiguous when
        # sharding. Attention all-to-all reconstructs this exact global order,
        # so the full block mask and EchoRoPE frame_indices remain valid.
        sp_enabled = is_sequence_parallel_enabled()
        sp_full_length = full.shape[1]
        if sp_enabled:
            sp_size = get_sp_size()
            sp_rank = get_sp_rank()
            if self.num_heads % sp_size != 0:
                raise RuntimeError(
                    f"num_heads={self.num_heads} must be divisible by sp_size={sp_size}"
                )
            shard_start, shard_length = contiguous_sp_bounds(sp_full_length, sp_size, sp_rank)
            full = full.narrow(1, shard_start, shard_length).contiguous()
            e0 = e0.narrow(1, shard_start, shard_length).contiguous()
            if cam_viewmats is not None and self.camera_attention_mode == "parallel":
                cam_viewmats = cam_viewmats.narrow(1, shard_start, shard_length).contiguous()
                cam_K = cam_K.narrow(1, shard_start, shard_length).contiguous()

        block_kwargs = dict(
            e=e0,
            seq_lens=seq_lens_noisy,
            grid_sizes=grid_sizes_full,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            cam_viewmats=cam_viewmats,
            cam_K=cam_K,
            cache_update_policy="none",
            block_mask=block_mask,
            frame_indices=frame_indices,
            kv_cache=None,
            crossattn_cache=None,
            current_start=0,
            cache_start=0,
        )

        x_tok = full
        for block in self.blocks:
            x_tok, _ = block(x_tok, **block_kwargs)

        if sp_enabled:
            x_tok = sequence_model_parallel_all_gather(x_tok, dim=1)
            if x_tok.shape[1] != sp_full_length:
                raise RuntimeError(
                    "Stage1 TF SP gather returned an unexpected sequence length: "
                    f"got {x_tok.shape[1]}, expected {sp_full_length}"
                )

        # Drop padding and clean half
        if padded_length > 0:
            x_tok = x_tok[:, : 2 * F_lat * s_per_frame]
        # noisy half is the second half along token dim
        x_tok = x_tok[:, x_tok.shape[1] // 2 :]

        # head expects e_per_frame at noisy-frame granularity
        x_tok = self.head(x_tok, e_for_head.unsqueeze(2))
        x_tok = self.unpatchify(x_tok, grid_sizes_noisy)
        return torch.stack(x_tok)

    # ──────────────────────────────────────────────────────────────────
    # Stage 1 cache-free inference (chunk recompute over a sliding window)
    # ──────────────────────────────────────────────────────────────────
    def forward_inference_window(
        self,
        noisy_chunk,  # tensor [B, C_in, Fn, H, W]   noisy chunk being denoised
        clean_history,  # tensor [B, C_in, Fc, H, W]  prior clean (Fc=0 allowed)
        t,  # noisy per-token timestep   [B, Fn * frame_seq_length]
        context,
        seq_len,
        num_frame_per_block,
        y=None,
        y_camera_window=None,  # dict with viewmats/K covering (Fc + Fn) × frame_seq_length
        clean_history_timestep=None,  # scalar or tensor timestep for clean-history tokens
        r=None,  # noisy per-token target timestep for flow-map sampling
        clean_history_r_timestep=None,
    ):
        """Re-run the transformer over [clean_history | noisy_chunk] for ONE
        denoising step. No KV cache used; the entire window is re-computed
        from scratch each call. This trades flops for simplicity / numerical
        certainty, intended for validation rollout. The training-time TF mask
        is replaced with `build_inference_window_block_mask` because the
        clean / noisy halves are no longer same-length-paired.
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        self.num_frame_per_block = int(num_frame_per_block)

        def _as_list(z):
            if isinstance(z, list):
                return z
            assert z.dim() == 5, f"expected (B, C, F, H, W), got {tuple(z.shape)}"
            return [u for u in z]

        noisy_list = _as_list(noisy_chunk)
        Fc = (
            clean_history.shape[2]
            if (clean_history is not None and clean_history.shape[2] > 0)
            else 0
        )
        clean_list = _as_list(clean_history) if Fc > 0 else None

        if self.model_type == "i2v" and y is None:
            raise ValueError("Wan I2V cache-free inference requires y for the active window")
        if y is not None:
            y_list = _as_list(y)
            Fn_input = noisy_list[0].shape[1]
            if any(item.shape[1] != Fc + Fn_input for item in y_list):
                raise ValueError(
                    "Wan I2V inference y must cover [clean_history | noisy_chunk]: "
                    f"expected {Fc + Fn_input} latent frames"
                )
            noisy_list = [torch.cat([u, v[:, Fc:]], dim=0) for u, v in zip(noisy_list, y_list)]
            if clean_list is not None:
                clean_list = [torch.cat([u, v[:, :Fc]], dim=0) for u, v in zip(clean_list, y_list)]

        noisy_pe = [self.patch_embedding(u.unsqueeze(0)) for u in noisy_list]
        grid_sizes_noisy = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in noisy_pe]
        )
        noisy_tokens = [u.flatten(2).transpose(1, 2) for u in noisy_pe]
        seq_lens_noisy = torch.tensor([u.size(1) for u in noisy_tokens], dtype=torch.long)
        assert seq_lens_noisy.max() <= seq_len

        if clean_list is not None:
            clean_pe = [self.patch_embedding(u.unsqueeze(0)) for u in clean_list]
            clean_tokens_list = [u.flatten(2).transpose(1, 2) for u in clean_pe]
        else:
            clean_tokens_list = None

        noisy_padded = torch.cat(noisy_tokens, dim=0)
        if clean_tokens_list is not None:
            clean_padded = torch.cat(clean_tokens_list, dim=0)
            full = torch.cat([clean_padded, noisy_padded], dim=1)
        else:
            full = noisy_padded

        Fn = int(grid_sizes_noisy[0][0].item())
        h_lat = int(grid_sizes_noisy[0][1].item())
        w_lat = int(grid_sizes_noisy[0][2].item())
        s_per_frame = h_lat * w_lat

        # Clean-history timestep defaults to 0 (matches TF training aug_t=0),
        # but validation can pass a small context-noise timestep just like the
        # cache-commit path. Noisy half uses caller-provided `t`.
        if Fc > 0:
            B = t.shape[0]
            if clean_history_timestep is None:
                aug_t = torch.zeros(B, Fc * s_per_frame, device=t.device, dtype=t.dtype)
            elif torch.is_tensor(clean_history_timestep):
                cht = clean_history_timestep.to(device=t.device, dtype=t.dtype)
                if cht.ndim == 0:
                    aug_t = torch.full(
                        (B, Fc * s_per_frame), float(cht.item()), device=t.device, dtype=t.dtype
                    )
                elif cht.shape == (B, Fc):
                    aug_t = (
                        cht.unsqueeze(-1).expand(B, Fc, s_per_frame).reshape(B, Fc * s_per_frame)
                    )
                elif cht.shape == (B, Fc * s_per_frame):
                    aug_t = cht
                else:
                    raise ValueError(
                        "clean_history_timestep must be scalar, [B,Fc], or "
                        f"[B,Fc*frame_seq_length], got {tuple(cht.shape)}"
                    )
            else:
                aug_t = torch.full(
                    (B, Fc * s_per_frame),
                    float(clean_history_timestep),
                    device=t.device,
                    dtype=t.dtype,
                )
            t_full = torch.cat([aug_t, t], dim=1)
        else:
            t_full = t

        if self.uses_anyflow:
            if r is None:
                raise ValueError("AnyFlow inference window requires noisy r")
            # Clean history is a fixed point of the flow-map update: t_clean
            # and r_clean are identical (normally context_noise).
            if Fc > 0:
                if clean_history_r_timestep is not None:
                    raise ValueError(
                        "clean_history_r_timestep is not independently configurable; "
                        "AnyFlow requires r_clean=t_clean"
                    )
                r_full = torch.cat([aug_t, r], dim=1)
            else:
                r_full = r
        else:
            r_full = None

        e_full = self._time_condition(t_full, full, r_full)
        e0 = (
            self.time_projection(e_full)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t_full.shape)
        )
        # head modulation source is the noisy half only
        e_for_head = self._time_condition(t, full, r).unflatten(dim=0, sizes=t.shape)

        # text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )

        # Camera viewmats / K: caller passes a window covering (Fc + Fn) frames
        # at token granularity. If absent, leave as None.
        cam_viewmats = None
        cam_K = None
        if y_camera_window is not None and isinstance(y_camera_window, dict):
            cam_viewmats = y_camera_window["viewmats"]
            cam_K = y_camera_window["K"]
            assert cam_viewmats.shape[1] == (Fc + Fn) * s_per_frame, (
                f"y_camera_window length {cam_viewmats.shape[1]} != "
                f"(Fc + Fn) * frame_seqlen = ({Fc} + {Fn}) * {s_per_frame}"
            )

        # Build & cache the FlexAttention block mask per (Fc, Fn) shape.
        mask_key = (Fc, Fn, s_per_frame, num_frame_per_block, str(device))
        if not hasattr(self, "_inf_block_mask_cache"):
            self._inf_block_mask_cache = {}
        cached = self._inf_block_mask_cache.get(mask_key)
        if cached is None:
            block_mask, padded_length = build_inference_window_block_mask(
                num_clean_frames=Fc,
                num_noisy_frames=Fn,
                frame_seqlen=s_per_frame,
                num_frame_per_block=num_frame_per_block,
                device=device,
            )
            self._inf_block_mask_cache[mask_key] = (block_mask, padded_length)
        else:
            block_mask, padded_length = cached

        if padded_length > 0:
            full = torch.cat(
                [full, full.new_zeros(full.shape[0], padded_length, full.shape[2])],
                dim=1,
            )
            e0 = torch.cat(
                [e0, e0.new_zeros(e0.shape[0], padded_length, e0.shape[2], e0.shape[3])],
                dim=1,
            )
            if cam_viewmats is not None:
                eye_vm = torch.eye(4, device=cam_viewmats.device, dtype=cam_viewmats.dtype)
                eye_K = torch.eye(3, device=cam_K.device, dtype=cam_K.dtype)
                pad_vm = (
                    eye_vm.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(cam_viewmats.shape[0], padded_length, -1, -1)
                )
                pad_K = (
                    eye_K.unsqueeze(0).unsqueeze(0).expand(cam_K.shape[0], padded_length, -1, -1)
                )
                cam_viewmats = torch.cat([cam_viewmats, pad_vm], dim=1)
                cam_K = torch.cat([cam_K, pad_K], dim=1)

        # grid_sizes describes Fc + Fn valid frames; pad tokens are passed
        # through unrotated by echorope_apply.
        grid_sizes_full = grid_sizes_noisy.clone()
        grid_sizes_full[:, 0] = Fc + Fn

        # frame_indices: window-relative positions [0..Fc+Fn-1]. This is the
        # core of EchoRoPE — clean and noisy tokens get
        # consecutive RoPE phases inside the current window. The *training*
        # path always sees Fc=Fn=T_lat; here the noisy chunk slides to
        # [Fc..Fc+Fn-1] regardless of where in the global timeline we are.
        frame_indices = torch.arange(Fc + Fn, device=device)

        # Cache-free validation uses the same contiguous token sharding as TF.
        # This handles both Fc=0 and Fc>0 without introducing global RoPE offsets:
        # all-to-all restores the complete window before EchoRoPE is applied.
        sp_enabled = is_sequence_parallel_enabled()
        sp_full_length = full.shape[1]
        if sp_enabled:
            sp_size = get_sp_size()
            sp_rank = get_sp_rank()
            if self.num_heads % sp_size != 0:
                raise RuntimeError(
                    f"num_heads={self.num_heads} must be divisible by sp_size={sp_size}"
                )
            shard_start, shard_length = contiguous_sp_bounds(sp_full_length, sp_size, sp_rank)
            full = full.narrow(1, shard_start, shard_length).contiguous()
            e0 = e0.narrow(1, shard_start, shard_length).contiguous()
            if cam_viewmats is not None and self.camera_attention_mode == "parallel":
                cam_viewmats = cam_viewmats.narrow(1, shard_start, shard_length).contiguous()
                cam_K = cam_K.narrow(1, shard_start, shard_length).contiguous()

        block_kwargs = dict(
            e=e0,
            seq_lens=seq_lens_noisy,
            grid_sizes=grid_sizes_full,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            cam_viewmats=cam_viewmats,
            cam_K=cam_K,
            cache_update_policy="none",
            block_mask=block_mask,
            frame_indices=frame_indices,
            kv_cache=None,
            crossattn_cache=None,
            current_start=0,
            cache_start=0,
        )

        x_tok = full
        for block in self.blocks:
            x_tok, _ = block(x_tok, **block_kwargs)

        if sp_enabled:
            x_tok = sequence_model_parallel_all_gather(x_tok, dim=1)
            if x_tok.shape[1] != sp_full_length:
                raise RuntimeError(
                    "Stage1 cache-free SP gather returned an unexpected sequence length: "
                    f"got {x_tok.shape[1]}, expected {sp_full_length}"
                )

        # Drop padding, keep only the noisy half
        if padded_length > 0:
            x_tok = x_tok[:, : (Fc + Fn) * s_per_frame]
        x_tok = x_tok[:, Fc * s_per_frame :]  # noisy half

        x_tok = self.head(x_tok, e_for_head.unsqueeze(2))
        x_tok = self.unpatchify(x_tok, grid_sizes_noisy)
        return torch.stack(x_tok)

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        y_camera=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=0,
        cache_update_policy="commit_detached",
        r=None,
        **kwargs,
    ):
        """
        Causal inference with KV caching.
        See Algorithm 2 of CausVid (https://arxiv.org/abs/2412.07772).

        Args:
            x: List of input video tensors [C_in, F, H, W]
            t: Timestep tensor [B, L]
            context: List of text embeddings [L, C]
            seq_len: Maximum sequence length for positional encoding
            y: Optional conditional video inputs (I2V mode)
            y_camera: Camera parameters dict {'viewmats': ..., 'K': ...}
            kv_cache: List of KV cache dicts per transformer block
            crossattn_cache: List of cross-attention cache dicts
            current_start: Current position in global token sequence
            cache_start: Cache start position
            cache_update_policy: Cache update strategy ('commit_detached' or 'none')

        Returns:
            Stacked output tensors [B, C_out, F, H/8, W/8]
        """
        training_mode = kwargs.pop("training_mode", None)
        if training_mode is not None:
            if training_mode == "inference_window":
                clean_history = kwargs.pop("clean_history", None)
                num_frame_per_block = kwargs.pop("num_frame_per_block", None)
                clean_history_timestep = kwargs.pop("clean_history_timestep", None)
                clean_history_r_timestep = kwargs.pop("clean_history_r_timestep", None)
                if kwargs:
                    raise TypeError(f"unexpected Wan inference-window arguments: {sorted(kwargs)}")
                if num_frame_per_block is None:
                    raise ValueError("Wan inference window requires num_frame_per_block")
                return self.forward_inference_window(
                    noisy_chunk=x,
                    clean_history=clean_history,
                    t=t,
                    context=context,
                    seq_len=seq_len,
                    num_frame_per_block=num_frame_per_block,
                    y=y,
                    y_camera_window=y_camera,
                    clean_history_timestep=clean_history_timestep,
                    r=r,
                    clean_history_r_timestep=clean_history_r_timestep,
                )
            if training_mode != "teacher_forcing":
                raise ValueError(f"unsupported Wan training_mode={training_mode!r}")
            clean_x = kwargs.pop("clean_x", None)
            aug_t = kwargs.pop("aug_t", None)
            num_frame_per_block = kwargs.pop("num_frame_per_block", None)
            aug_r = kwargs.pop("aug_r", None)
            if kwargs:
                raise TypeError(f"unexpected Wan teacher-forcing arguments: {sorted(kwargs)}")
            if clean_x is None or aug_t is None or num_frame_per_block is None:
                raise ValueError(
                    "Wan teacher forcing requires clean_x, aug_t, and num_frame_per_block"
                )
            return self.forward_train_tf(
                x=x,
                clean_x=clean_x,
                t=t,
                aug_t=aug_t,
                context=context,
                seq_len=seq_len,
                num_frame_per_block=num_frame_per_block,
                y=y,
                y_camera=y_camera,
                r=r,
                aug_r=aug_r,
            )
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if kv_cache is not None:
            self.num_frame_per_block = self._infer_num_frame_per_block_from_t(t)

        if self.model_type == "i2v" and y is None:
            raise ValueError("Wan I2V forward requires y")
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # patch embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embedding
        e = self._time_condition(t, x, r)
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )

        # camera parameters
        if y_camera is not None and isinstance(y_camera, dict):
            cam_viewmats = y_camera["viewmats"]
            cam_K = y_camera["K"]
        else:
            cam_viewmats = None
            cam_K = None

        kv_cam_viewmats = None
        kv_cam_K = None
        fused_camera_cache_update = None
        if (
            self.camera_attention_mode == "fused_prope"
            and kv_cache is not None
            and (cam_viewmats is not None or cam_K is not None)
        ):
            if cam_viewmats is None or cam_K is None:
                raise ValueError("fused_prope requires both camera viewmats and K, or neither")
            frame_seqlen = int(math.prod(grid_sizes[0][1:]).item())
            (
                cam_viewmats,
                cam_K,
                kv_cam_viewmats,
                kv_cam_K,
                fused_camera_cache_update,
            ) = self._stage_fused_camera_cache(
                kv_cache,
                cam_viewmats=cam_viewmats,
                cam_K=cam_K,
                current_start=current_start,
                num_new_tokens=x.shape[1],
                frame_seqlen=frame_seqlen,
            )

        sp_enabled = is_sequence_parallel_enabled()
        cache_head_parallel = sp_enabled and kv_cache is not None
        sp_seq_len_orig = x.shape[1]
        if sp_enabled and not cache_head_parallel:
            sp_size = get_sp_size()
            sp_rank = get_sp_rank()
            if self.num_heads % sp_size != 0:
                raise RuntimeError(
                    f"num_heads={self.num_heads} must be divisible by sp_size={sp_size}"
                )
            num_frames = int(grid_sizes[0][0].item())
            frame_seqlen = int(math.prod(grid_sizes[0][1:]).item())
            if sp_seq_len_orig != num_frames * frame_seqlen:
                raise RuntimeError(
                    "Stage0.5 SP requires a whole-frame token layout: "
                    f"sequence={sp_seq_len_orig} frames={num_frames} frame_seqlen={frame_seqlen}"
                )
            frames_per_rank, remainder = divmod(num_frames, sp_size)
            local_frames = frames_per_rank + int(sp_rank < remainder)
            frame_start = sp_rank * frames_per_rank + min(sp_rank, remainder)
            token_start = frame_start * frame_seqlen
            local_tokens = local_frames * frame_seqlen
            x = x.narrow(1, token_start, local_tokens).contiguous()
            # ``t`` is expanded to one entry per spatial token by the trainer,
            # so e0 has the same token axis as x.  Slicing it by frame count
            # would feed only the first frame's repeated t=0 embedding to an
            # entire SP shard (405 tokens/frame at 480p), producing noise-like
            # validation and an inflated Stage0.5 loss.
            e0 = e0.narrow(1, token_start, local_tokens).contiguous()
            register_sequence_parallel_sequence_length(local_tokens)
            if cam_viewmats is not None and self.camera_attention_mode == "parallel":
                cam_viewmats = cam_viewmats.narrow(1, token_start, local_tokens).contiguous()
                cam_K = cam_K.narrow(1, token_start, local_tokens).contiguous()

        block_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            cam_viewmats=cam_viewmats,
            cam_K=cam_K,
            kv_cam_viewmats=kv_cam_viewmats,
            kv_cam_K=kv_cam_K,
            cache_update_policy=cache_update_policy,
        )

        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
            block_kwargs.update(
                {
                    "kv_cache": kv_cache[block_index] if kv_cache is not None else None,
                    "crossattn_cache": crossattn_cache[block_index]
                    if crossattn_cache is not None
                    else None,
                    "current_start": current_start,
                    "cache_start": cache_start,
                }
            )
            x, block_cache_update_info = block(x, **block_kwargs)
            if kv_cache is not None:
                cache_update_infos.append((block_index, block_cache_update_info))

        # Apply deferred cache updates
        if kv_cache is not None and cache_update_infos and cache_update_policy != "none":
            self._apply_cache_updates(kv_cache, cache_update_infos)
            if self.camera_attention_mode == "fused_prope" and cam_viewmats is None:
                # A cache populated without camera conditioning cannot later be
                # interpreted as camera-conditioned history. Drop stale metadata
                # so a later mixed-mode call asks for a cache rebuild.
                kv_cache[0].pop("_fused_prope_camera_metadata", None)
            else:
                self._commit_fused_camera_cache(fused_camera_cache_update)

        if sp_enabled and not cache_head_parallel:
            x = sequence_model_parallel_all_gather(x, dim=1)
            if x.shape[1] != sp_seq_len_orig:
                x = x[:, :sp_seq_len_orig]

        # head & unpatchify
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """Apply deferred cache updates collected from all transformer blocks.

        Standard self-attention stores un-roped K values in the cache. RoPE is
        applied dynamically during attention using the selected base or Echo
        position allocation for the current visible window.
        """
        with torch.no_grad():
            for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
                if update_info is not None:
                    cache = kv_cache[block_index]

                    if update_info["action"] == "roll_and_insert":
                        sink_tokens = update_info["sink_tokens"]
                        num_rolled_tokens = update_info["num_rolled_tokens"]
                        num_evicted_tokens = update_info["num_evicted_tokens"]
                        write_start_index = update_info.get(
                            "write_start_index", update_info["local_start_index"]
                        )
                        write_end_index = update_info.get(
                            "write_end_index", update_info["local_end_index"]
                        )
                        new_k = update_info["new_k"].detach()
                        new_v = update_info["new_v"].detach()

                        cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["k"][
                            :,
                            sink_tokens + num_evicted_tokens : sink_tokens
                            + num_evicted_tokens
                            + num_rolled_tokens,
                        ].clone()
                        cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["v"][
                            :,
                            sink_tokens + num_evicted_tokens : sink_tokens
                            + num_evicted_tokens
                            + num_rolled_tokens,
                        ].clone()

                        if write_end_index > write_start_index and new_k.shape[1] == (
                            write_end_index - write_start_index
                        ):
                            cache["k"][:, write_start_index:write_end_index] = new_k
                            cache["v"][:, write_start_index:write_end_index] = new_v

                    elif update_info["action"] == "direct_insert":
                        write_start_index = update_info.get(
                            "write_start_index", update_info["local_start_index"]
                        )
                        write_end_index = update_info.get(
                            "write_end_index", update_info["local_end_index"]
                        )
                        new_k = update_info["new_k"].detach()
                        new_v = update_info["new_v"].detach()

                        if write_end_index > write_start_index and new_k.shape[1] == (
                            write_end_index - write_start_index
                        ):
                            cache["k"][:, write_start_index:write_end_index] = new_k
                            cache["v"][:, write_start_index:write_end_index] = new_v

                is_recompute = (
                    False if update_info is None else update_info.get("is_recompute", False)
                )
                if not is_recompute:
                    kv_cache[block_index]["global_end_index"].fill_(current_end)
                    kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def unpatchify(self, x, grid_sizes):
        """Reconstruct video tensors from patch embeddings."""
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """Initialize model parameters using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        nn.init.zeros_(self.head.head.weight)
