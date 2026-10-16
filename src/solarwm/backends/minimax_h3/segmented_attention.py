"""Segmented H3 attention and bounded per-layer key/value caches.

Adapted from SolarRenderer, Copyright 2026 SolarRenderer authors, Apache-2.0.
Moved into the shared H3 backend without changing the attention or cache algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class H3ChunkKVCache:
    """Per layer, the keys and values (head-local, after rotary) of committed rows, under
    ``(kind, chunk)``: a chunk's
    structure (``"structure"``) and its clean frames (``"clean"``). Each kind keeps its last
    ``window`` chunks."""

    layers: dict[int, dict[tuple[str, int], tuple[Any, Any]]] = field(default_factory=dict)
    window: int = 6

    def get(self, layer: int, name: tuple[str, int]) -> tuple[Any, Any]:
        try:
            return self.layers[layer][name]
        except KeyError as exc:
            raise RuntimeError(f"{name} was never committed at layer {layer}") from exc

    def put(self, layer: int, name: tuple[str, int], key: Any, value: Any) -> None:
        import torch

        if torch.is_grad_enabled():
            raise RuntimeError("a key/value commit runs without a graph")
        entries = self.layers.setdefault(layer, {})
        entries[name] = (key.detach().clone(), value.detach().clone())
        kind, chunk = name
        for old in [n for n in entries if n[0] == kind and n[1] <= chunk - self.window]:
            entries.pop(old)


@dataclass(frozen=True)
class H3SegmentPlan:
    """Query segments in document order; each attends to the union of its key spans and cached
    chunks."""

    segments: tuple[
        tuple[
            int,
            int,
            tuple[tuple[int, int], ...],
            tuple[tuple[str, int], ...],
            tuple[str, int] | None,
        ],
        ...,
    ]
    sequence_length: int
    cache: H3ChunkKVCache | None = None

    def __post_init__(self) -> None:
        end = 0
        for start, stop, spans, cached, commit in self.segments:
            if start != end or stop <= start:
                raise ValueError("segments must tile the document in order")
            if not spans and (not cached):
                raise ValueError(f"segment [{start}, {stop}) sees nothing")
            if (cached or commit is not None) and self.cache is None:
                raise ValueError("cached or committed chunks need a cache")
            end = stop
        if end != self.sequence_length:
            raise ValueError("segments must cover the whole document")


def segment_attention(query: Any, key: Any, value: Any, plan: H3SegmentPlan, layer: int) -> Any:
    """``[B, S, H, D]`` attention over full sequence / local heads (after any Ulysses
    exchange), segment by segment."""
    import torch

    if query.device.type == "cuda":
        from flash_attn import flash_attn_func
    else:

        def flash_attn_func(q, k, v, dropout_p=0.0, causal=False):
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            return out.transpose(1, 2)

    if int(query.shape[1]) != plan.sequence_length:
        raise ValueError(
            f"segment plan covers {plan.sequence_length} rows, attention got {query.shape[1]}"
        )
    outputs = []
    for start, stop, spans, cached, commit in plan.segments:
        keys = [key[:, a:b] for a, b in spans]
        values = [value[:, a:b] for a, b in spans]
        for chunk in cached:
            k, v = plan.cache.get(layer, chunk)
            keys.append(k)
            values.append(v)
        outputs.append(
            flash_attn_func(
                query[:, start:stop].contiguous(),
                torch.cat(keys, dim=1).contiguous(),
                torch.cat(values, dim=1).contiguous(),
                dropout_p=0.0,
                causal=False,
            )
        )
        if commit is not None:
            plan.cache.put(layer, commit, key[:, start:stop], value[:, start:stop])
    return torch.cat(outputs, dim=1)


__all__ = ["H3ChunkKVCache", "H3SegmentPlan", "segment_attention"]
