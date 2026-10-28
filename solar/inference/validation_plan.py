"""Immutable validation-case plans shared across training validations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from solar.errors import BackendContractError
from solar.runtime.create_only import write_file_create_only
from solar.runtime.serialization import canonical_json_bytes

from .engine import InferenceCase

_SCHEMA = "solarwm.validation-plan.v1"


def validation_plan_key(backend: str, config: Mapping[str, Any]) -> str:
    """Bind a frozen plan to every config field that can affect case identity."""

    data = config.get("data")
    if isinstance(data, Mapping):
        data = dict(data)
        # Node-local I/O lookahead changes only when shard reads are issued. It
        # does not change the selected validation rows, starts, or noise.
        data.pop("gcs_prefetch_shards", None)
    payload = {
        "schema": "solarwm.validation-plan-key.v1",
        "backend": str(backend),
        "model": config.get("model"),
        "data": data,
        "distributed": config.get("distributed"),
        "validation": config.get("validation"),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validated_cases(raw: object, *, expected_count: int) -> tuple[InferenceCase, ...]:
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise BackendContractError(
            "frozen validation plan case count differs: "
            f"expected={expected_count} actual={len(raw) if isinstance(raw, list) else 'invalid'}"
        )
    cases: list[InferenceCase] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {
            "slot",
            "sample_id",
            "prompt",
            "start_frame",
            "noise_seed",
            "camera_fingerprint",
            "metadata",
        }:
            raise BackendContractError("frozen validation plan contains an invalid case")
        metadata = item["metadata"]
        if not isinstance(metadata, Mapping):
            raise BackendContractError("frozen validation case metadata must be a mapping")
        cases.append(
            InferenceCase(
                slot=int(item["slot"]),
                sample_id=str(item["sample_id"]),
                prompt=str(item["prompt"]),
                start_frame=int(item["start_frame"]),
                noise_seed=int(item["noise_seed"]),
                camera_fingerprint=str(item["camera_fingerprint"]),
                metadata=dict(metadata),
            )
        )
    if [case.slot for case in cases] != list(range(expected_count)):
        raise BackendContractError("frozen validation plan slots are not contiguous and ordered")
    return tuple(cases)


def validation_plan_payload(
    *,
    backend: str,
    plan_key: str,
    cases: Sequence[InferenceCase],
) -> bytes:
    ordered = tuple(cases)
    if not backend or len(plan_key) != 64:
        raise BackendContractError("frozen validation plan lacks a backend/config identity")
    if [case.slot for case in ordered] != list(range(len(ordered))):
        raise BackendContractError(
            "validation cases must be contiguous and ordered before freezing"
        )
    body = {
        "schema": _SCHEMA,
        "backend": backend,
        "plan_key": plan_key,
        "case_count": len(ordered),
        "cases": [asdict(case) for case in ordered],
    }
    body_bytes = canonical_json_bytes(body)
    return canonical_json_bytes(
        {
            **body,
            "payload_digest": hashlib.sha256(body_bytes).hexdigest(),
        }
    )


def load_validation_plan(
    path: str | Path,
    *,
    backend: str,
    plan_key: str,
    expected_count: int,
) -> tuple[InferenceCase, ...] | None:
    source = Path(path)
    if not source.exists():
        return None
    if not source.is_file() or source.is_symlink():
        raise BackendContractError(f"frozen validation plan is not a regular file: {source}")
    try:
        value = json.loads(source.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendContractError(f"cannot read frozen validation plan {source}: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema") != _SCHEMA:
        raise BackendContractError("frozen validation plan schema differs")
    if value.get("backend") != backend or value.get("plan_key") != plan_key:
        raise BackendContractError("frozen validation plan config identity differs")
    if int(value.get("case_count", -1)) != expected_count:
        raise BackendContractError("frozen validation plan declared case count differs")
    payload_digest = value.get("payload_digest")
    body = {key: item for key, item in value.items() if key != "payload_digest"}
    actual_digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if payload_digest != actual_digest:
        raise BackendContractError("frozen validation plan payload digest differs")
    return _validated_cases(value.get("cases"), expected_count=expected_count)


def publish_validation_plan(
    path: str | Path,
    *,
    backend: str,
    plan_key: str,
    cases: Sequence[InferenceCase],
) -> str:
    """Create the immutable plan, or accept an identical race winner."""

    target = Path(path)
    payload = validation_plan_payload(backend=backend, plan_key=plan_key, cases=cases)
    try:
        write_file_create_only(
            target,
            payload,
            error_type=BackendContractError,
            label="frozen validation plan",
        )
    except BackendContractError:
        try:
            existing = target.read_bytes()
        except OSError:
            raise
        if existing != payload:
            raise
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "load_validation_plan",
    "publish_validation_plan",
    "validation_plan_key",
    "validation_plan_payload",
]
