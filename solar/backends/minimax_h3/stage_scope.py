"""Stage scope of the switches that change rounding (tier N1).

``SOLARWM_H3_N1_STAGES`` (comma list, default: every stage) names the stages in which the N1 switches
(cuDNN attention, FUSED kernels, PROPE_FUSED, LORA_ADDMM, SGF prefix reuse / pieces) act; in the other stages
those code paths fall back to upstream's (the bitwise switches stay active). The runtime sets the stage once
at construction, before the model is built.
"""

from __future__ import annotations

import os

_RAW = os.environ.get("SOLARWM_H3_N1_STAGES", "").strip()
_STAGES = frozenset(s.strip() for s in _RAW.split(",") if s.strip()) if _RAW else None
_CURRENT: str | None = None


def set_stage(stage: str) -> None:
    global _CURRENT
    _CURRENT = str(stage)


def n1_active() -> bool:
    return _STAGES is None or _CURRENT is None or _CURRENT in _STAGES


# SOLARWM_H3_CUDNN_STAGES: stages in which SOLARWM_H3_ATTENTION=cudnn acts (default: every stage).
# Stage2's student gradient is a difference of two near-identical score evaluations (critic vs teacher)
# and follows the attention kernel's rounding; outside the listed stages upstream's FA2 runs.
_RAW_CUDNN = os.environ.get("SOLARWM_H3_CUDNN_STAGES", "").strip()
_CUDNN_STAGES = (
    frozenset(s.strip() for s in _RAW_CUDNN.split(",") if s.strip()) if _RAW_CUDNN else None
)


def cudnn_active() -> bool:
    if not n1_active():
        return False
    return _CUDNN_STAGES is None or _CURRENT is None or _CURRENT in _CUDNN_STAGES


__all__ = ["cudnn_active", "n1_active", "set_stage"]
