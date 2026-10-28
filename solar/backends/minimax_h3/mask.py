"""MiniMax-H3 attention visibility contracts without framework imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import (
    AUDIO_ROLE,
    CLEAN_VIDEO_ROLE,
    CONDITION_ROLE,
    NOISY_VIDEO_ROLE,
    H3PackedLayout,
)


@dataclass(frozen=True)
class FullAttentionMask:
    """A non-materialized all-visible square mask."""

    sequence_length: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.sequence_length, self.sequence_length

    def allows(self, query_index: int, key_index: int) -> bool:
        if not 0 <= query_index < self.sequence_length:
            raise IndexError(query_index)
        if not 0 <= key_index < self.sequence_length:
            raise IndexError(key_index)
        return True


def stage0p5_full_attention(layout: H3PackedLayout) -> FullAttentionMask:
    """Describe Stage0.5 full attention without allocating an ``S**2`` array."""

    if layout.stage != "stage0p5":
        raise ValueError("Stage0.5 full attention requires a Stage0.5 layout")
    return FullAttentionMask(layout.sequence_length)


def stage1_window_allows(
    query_role: int,
    query_chunk: int,
    key_role: int,
    key_chunk: int,
    *,
    window_chunks: int = 6,
) -> bool:
    """Return the Stage1 six-chunk visibility decision."""

    if window_chunks <= 0:
        raise ValueError(f"window_chunks must be positive, got {window_chunks}")
    if query_role == CONDITION_ROLE:
        return key_role == CONDITION_ROLE
    if query_role == AUDIO_ROLE:
        return key_role in (CONDITION_ROLE, AUDIO_ROLE)
    if query_role == CLEAN_VIDEO_ROLE:
        if key_role in (CONDITION_ROLE, AUDIO_ROLE):
            return True
        return key_role == CLEAN_VIDEO_ROLE and query_chunk - key_chunk >= 0
    if query_role == NOISY_VIDEO_ROLE:
        if key_role in (CONDITION_ROLE, AUDIO_ROLE):
            return True
        if key_role == CLEAN_VIDEO_ROLE:
            delta = query_chunk - key_chunk
            return 1 <= delta < window_chunks
        return key_role == NOISY_VIDEO_ROLE and query_chunk == key_chunk
    return False


def build_stage1_window_mask(layout: H3PackedLayout) -> np.ndarray:
    """Materialize a dense boolean six-chunk mask."""

    if layout.stage != "stage1" or layout.window_chunks is None:
        raise ValueError("six-chunk mask requires a Stage1 layout")
    if not layout.clean_video_indices.size or not layout.noisy_video_indices.size:
        raise ValueError("Stage1 mask requires explicit clean and noisy video rows")
    roles = layout.row_roles
    chunks = layout.target_video_chunk_ids
    q_role, k_role = roles[:, None], roles[None, :]
    delta = chunks[:, None] - chunks[None, :]
    condition_q = q_role == CONDITION_ROLE
    condition_k = k_role == CONDITION_ROLE
    audio_q = q_role == AUDIO_ROLE
    audio_k = k_role == AUDIO_ROLE
    clean_q = q_role == CLEAN_VIDEO_ROLE
    clean_k = k_role == CLEAN_VIDEO_ROLE
    noisy_q = q_role == NOISY_VIDEO_ROLE
    noisy_k = k_role == NOISY_VIDEO_ROLE
    common_k = condition_k | audio_k
    return (
        (condition_q & condition_k)
        | (audio_q & common_k)
        | (clean_q & (common_k | (clean_k & (delta >= 0))))
        | (
            noisy_q
            & (
                common_k
                | (clean_k & (delta >= 1) & (delta < int(layout.window_chunks)))
                | (noisy_k & (delta == 0))
            )
        )
    )


__all__ = [
    "FullAttentionMask",
    "build_stage1_window_mask",
    "stage0p5_full_attention",
    "stage1_window_allows",
]
