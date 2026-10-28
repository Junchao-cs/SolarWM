"""Backend-neutral training lifecycle with strict numeric gates."""

from .engine import (
    BatchIdentity,
    GradientStatus,
    JsonlEventSink,
    MicrobatchResult,
    StepPolicy,
    TrainingEngine,
    TrainingRuntime,
)

__all__ = [
    "BatchIdentity",
    "GradientStatus",
    "JsonlEventSink",
    "MicrobatchResult",
    "StepPolicy",
    "TrainingEngine",
    "TrainingRuntime",
]
