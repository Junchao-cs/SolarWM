"""Extend the renderer's encoded silence without changing shared H3 backend contracts."""

from __future__ import annotations

from typing import Any


def extend_silence(silence: Any, length: int, *, edge: int = 20) -> Any:
    """Preserve AudioVAE start/end transients and repeat a steady middle column.

    Encoded silence is constant away from its first and last ``edge`` columns.
    A shorter clip is cropped; an equal-length clip reuses the input tensor.
    """
    import torch

    columns = int(silence.shape[-1])
    if length <= columns:
        return silence[..., :length] if length < columns else silence
    middle = silence[..., columns // 2 : columns // 2 + 1].expand(
        *silence.shape[:-1], length - 2 * edge
    )
    return torch.cat(
        (silence[..., :edge], middle, silence[..., columns - edge :]), dim=-1
    ).contiguous()
