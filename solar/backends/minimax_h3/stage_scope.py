"""Stage scope of ``SOLARWM_H3_ATTENTION=cudnn``.

``SOLARWM_H3_CUDNN_STAGES`` (comma list, default: every stage) names the stages in which cuDNN
attention replaces FlashAttention-2. Stage2's student gradient is a difference of two
near-identical score evaluations (critic vs teacher) and follows the attention kernel's rounding,
so the verified configuration keeps FlashAttention-2 there outside the scopes in
``stage2_runtime``. The runtime sets the stage once at construction, before the model is built.
"""

from __future__ import annotations

import os

_RAW = os.environ.get("SOLARWM_H3_CUDNN_STAGES", "").strip()
_CUDNN_STAGES = frozenset(s.strip() for s in _RAW.split(",") if s.strip()) if _RAW else None
_CURRENT: str | None = None


def set_stage(stage: str) -> None:
    global _CURRENT
    _CURRENT = str(stage)


def cudnn_active() -> bool:
    return _CUDNN_STAGES is None or _CURRENT is None or _CURRENT in _CUDNN_STAGES


__all__ = ["cudnn_active", "set_stage"]
