from __future__ import annotations

from pathlib import Path

import pytest

import solar.backends.ltx25.torch_validation as validation_module
from solar.backends.ltx25.torch_validation import (
    TrainingValidation,
    _partition_validation_cases,
    _validation_passes,
)
from solar.errors import BackendContractError
from solar.inference import InferenceCase


def test_validation_passes_accept_named_weight_roles() -> None:
    passes = _validation_passes(
        [
            {"name": "live_balanced64x2_153f", "weights": "live"},
            {"name": "ema_balanced64x2_153f", "weights": "ema"},
        ]
    )

    assert [(item.name, item.weights) for item in passes] == [
        ("live_balanced64x2_153f", "live"),
        ("ema_balanced64x2_153f", "ema"),
    ]


def test_validation_passes_keep_short_form() -> None:
    passes = _validation_passes(["live", "ema"])

    assert [(item.name, item.weights) for item in passes] == [
        ("live", "live"),
        ("ema", "ema"),
    ]


def test_validation_cases_cover_complete_logical_dp_waves() -> None:
    cases = tuple(range(16))

    assert _partition_validation_cases(cases, dp_rank=0, dp_world_size=4) == (0, 4, 8, 12)
    assert _partition_validation_cases(cases, dp_rank=3, dp_world_size=4) == (3, 7, 11, 15)


def test_validation_cases_reject_partial_logical_dp_wave() -> None:
    with pytest.raises(BackendContractError, match="complete logical-DP waves"):
        _partition_validation_cases(tuple(range(15)), dp_rank=0, dp_world_size=4)


@pytest.mark.parametrize(
    "passes",
    (
        [{"name": "live", "weights": "teacher"}],
        [{"name": "../live", "weights": "live"}],
        [
            {"name": "live", "weights": "live"},
            {"name": "live", "weights": "ema"},
        ],
    ),
)
def test_validation_passes_reject_invalid_routes(passes: object) -> None:
    with pytest.raises(BackendContractError):
        _validation_passes(passes)


def test_training_validation_freezes_then_loads_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        InferenceCase(0, "sample-0", "", 0, 10, "camera-0"),
        InferenceCase(1, "sample-1", "", 1, 11, "camera-1"),
    )
    calls = 0

    def build_cases(*_args: object, **_kwargs: object) -> tuple[InferenceCase, ...]:
        nonlocal calls
        calls += 1
        return cases

    monkeypatch.setattr(validation_module, "inference_cases", build_cases)
    validation = TrainingValidation.__new__(TrainingValidation)
    validation.source = object()
    validation.plan = object()
    validation.sample_count = 2
    validation.plan_path = tmp_path / "validation/frozen-plan.json"
    validation.config = {
        "model": {"camera_translation_transform": "linear"},
        "validation": {"selection_seed": 42},
    }
    validation.plan_key = "1" * 64

    first, first_source, first_digest = validation._resolve_cases()
    second, second_source, second_digest = validation._resolve_cases()

    assert first == second == cases
    assert first_source == "created"
    assert second_source == "loaded"
    assert first_digest == second_digest
    assert calls == 1
