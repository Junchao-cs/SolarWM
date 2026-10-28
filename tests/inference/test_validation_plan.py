from __future__ import annotations

import json
from pathlib import Path

import pytest

from solar.errors import BackendContractError
from solar.inference import InferenceCase
from solar.inference.validation_plan import (
    load_validation_plan,
    publish_validation_plan,
    validation_plan_key,
)


def _cases() -> tuple[InferenceCase, ...]:
    return tuple(
        InferenceCase(
            slot=slot,
            sample_id=f"sample-{slot}",
            prompt=f"prompt-{slot}",
            start_frame=slot,
            noise_seed=42 + slot,
            camera_fingerprint=f"camera-{slot}",
            metadata={"key": f"key-{slot}", "artifact_valid": True},
        )
        for slot in range(2)
    )


def _config() -> dict[str, object]:
    return {
        "model": {"family": "test"},
        "data": {"test_index": "test-index.jsonl.gz"},
        "distributed": {"world_size": 2},
        "validation": {"sample_count": 2, "selection_seed": 42},
    }


def test_validation_plan_is_created_once_and_loaded_exactly(tmp_path: Path) -> None:
    path = tmp_path / "validation/frozen-plan.json"
    key = validation_plan_key("test", _config())

    first_digest = publish_validation_plan(
        path,
        backend="test",
        plan_key=key,
        cases=_cases(),
    )
    first_bytes = path.read_bytes()
    second_digest = publish_validation_plan(
        path,
        backend="test",
        plan_key=key,
        cases=_cases(),
    )

    assert first_digest == second_digest
    assert path.read_bytes() == first_bytes
    assert (
        load_validation_plan(
            path,
            backend="test",
            plan_key=key,
            expected_count=2,
        )
        == _cases()
    )


def test_validation_plan_key_ignores_gcs_prefetch_tuning_only() -> None:
    config = _config()
    key = validation_plan_key("test", config)
    tuned = {**config, "data": {**config["data"], "gcs_prefetch_shards": 32}}
    drifted = {**config, "data": {**config["data"], "test_index": "other.jsonl.gz"}}

    assert validation_plan_key("test", tuned) == key
    assert validation_plan_key("test", drifted) != key


def test_validation_plan_preserves_repeated_samples_in_distinct_slots(tmp_path: Path) -> None:
    cases = list(_cases())
    cases[1] = InferenceCase(
        slot=1,
        sample_id=cases[0].sample_id,
        prompt=cases[0].prompt,
        start_frame=cases[0].start_frame,
        noise_seed=43,
        camera_fingerprint=cases[0].camera_fingerprint,
        metadata={"key": cases[0].sample_id, "artifact_valid": True},
    )
    path = tmp_path / "frozen-plan.json"
    key = validation_plan_key("test", _config())

    publish_validation_plan(path, backend="test", plan_key=key, cases=cases)

    assert load_validation_plan(
        path,
        backend="test",
        plan_key=key,
        expected_count=2,
    ) == tuple(cases)


def test_validation_plan_rejects_config_or_payload_drift(tmp_path: Path) -> None:
    path = tmp_path / "frozen-plan.json"
    key = validation_plan_key("test", _config())
    publish_validation_plan(path, backend="test", plan_key=key, cases=_cases())

    with pytest.raises(BackendContractError, match="config identity"):
        load_validation_plan(
            path,
            backend="test",
            plan_key="0" * 64,
            expected_count=2,
        )

    value = json.loads(path.read_text(encoding="utf-8"))
    value["cases"][0]["sample_id"] = "drifted"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(BackendContractError, match="payload digest"):
        load_validation_plan(
            path,
            backend="test",
            plan_key=key,
            expected_count=2,
        )


def test_validation_plan_rejects_noncontiguous_slots(tmp_path: Path) -> None:
    cases = list(_cases())
    cases[1] = InferenceCase(
        slot=3,
        sample_id="sample-3",
        prompt="prompt-3",
        start_frame=3,
        noise_seed=45,
        camera_fingerprint="camera-3",
    )
    with pytest.raises(BackendContractError, match="contiguous"):
        publish_validation_plan(
            tmp_path / "frozen-plan.json",
            backend="test",
            plan_key=validation_plan_key("test", _config()),
            cases=cases,
        )
