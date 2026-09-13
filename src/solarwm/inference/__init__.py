"""The generation engine shared verbatim by CLI inference and validation."""

from .comparison import (
    ComparisonValidationRecord,
    encode_compare_mp4,
    publish_comparison_complete,
    publish_comparison_partition,
)
from .engine import (
    GeneratedFile,
    GeneratedSample,
    InferenceAdapter,
    InferenceCase,
    InferenceEngine,
    InferenceSummary,
    run_validation,
)

__all__ = [
    "ComparisonValidationRecord",
    "GeneratedFile",
    "GeneratedSample",
    "InferenceAdapter",
    "InferenceCase",
    "InferenceEngine",
    "InferenceSummary",
    "encode_compare_mp4",
    "publish_comparison_complete",
    "publish_comparison_partition",
    "run_validation",
]
