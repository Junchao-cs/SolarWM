from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from solar.checkpoint import (
    CheckpointContract,
    CheckpointTransaction,
    assert_resume_compatible,
    store,
    verify_checkpoint,
)
from solar.errors import CheckpointError
from solar.runtime.serialization import canonical_json_bytes


def _contract(**changes: object) -> CheckpointContract:
    values: dict[str, object] = {
        "family": "wan22_ti2v_5b",
        "stage": "stage2",
        "causal_mode": "self_gradient_forcing",
        "objective": "flow_matching",
        "objective_variant": "",
        "camera_translation_transform": "linear",
        "parameterization": "full",
        "sp_size": 1,
        "data_generation": "index-digest",
        "extras": {"student_steps": 5},
    }
    values.update(changes)
    return CheckpointContract(**values)  # type: ignore[arg-type]


def test_transaction_requires_and_verifies_stage2_roles(tmp_path: Path) -> None:
    target = tmp_path / "step-000100"
    with CheckpointTransaction(target) as transaction:
        (transaction.path / "model.pt").write_bytes(b"student")
        (transaction.path / "critic.pt").write_bytes(b"critic")
        verified = transaction.commit(
            step=100,
            contract=_contract(),
            required_components=("model.pt", "critic.pt"),
            metadata={"resolved_config_digest": "a" * 64},
        )
    assert verified.step == 100
    assert {record.path for record in verified.files} == {"model.pt", "critic.pt"}
    assert all(not hasattr(record, "digest") for record in verified.files)
    assert verify_checkpoint(target).manifest_digest == verified.manifest_digest


def test_transaction_fails_closed_when_critic_is_missing(tmp_path: Path) -> None:
    with CheckpointTransaction(tmp_path / "step") as transaction:
        (transaction.path / "model.pt").write_bytes(b"student")
        with pytest.raises(CheckpointError, match="critic"):
            transaction.commit(
                step=1,
                contract=_contract(),
                required_components=("model.pt", "critic.pt"),
                metadata={},
            )


def test_verification_detects_payload_size_drift(tmp_path: Path) -> None:
    target = tmp_path / "step"
    with CheckpointTransaction(target) as transaction:
        (transaction.path / "model.pt").write_bytes(b"before")
        transaction.commit(
            step=1,
            contract=_contract(stage="stage0p5", causal_mode="bidirectional"),
            required_components=("model.pt",),
            metadata={},
        )
    (target / "model.pt").write_bytes(b"after-longer")
    with pytest.raises(CheckpointError, match="size differs"):
        verify_checkpoint(target)


def test_v2_manifest_records_only_component_path_and_positive_size(tmp_path: Path) -> None:
    target = tmp_path / "step"
    with CheckpointTransaction(target) as transaction:
        (transaction.path / "model.pt").write_bytes(b"payload")
        verified = transaction.commit(
            step=1,
            contract=_contract(stage="stage0p5", causal_mode="bidirectional"),
            required_components=("model.pt",),
            metadata={},
        )
    assert verified.files == (store.ComponentFile(path="model.pt", size=7),)
    manifest = json.loads((target / "checkpoint-manifest.json").read_bytes())
    assert manifest["files"] == [{"path": "model.pt", "size": 7}]


def test_transaction_rejects_empty_component_payload(tmp_path: Path) -> None:
    with CheckpointTransaction(tmp_path / "step") as transaction:
        (transaction.path / "model.pt").write_bytes(b"")
        with pytest.raises(CheckpointError, match="size must be positive"):
            transaction.commit(
                step=1,
                contract=_contract(stage="stage0p5", causal_mode="bidirectional"),
                required_components=("model.pt",),
                metadata={},
            )


def test_verification_rejects_non_v2_schema(tmp_path: Path) -> None:
    target = tmp_path / "step"
    with CheckpointTransaction(target) as transaction:
        (transaction.path / "model.pt").write_bytes(b"payload")
        transaction.commit(
            step=1,
            contract=_contract(stage="stage0p5", causal_mode="bidirectional"),
            required_components=("model.pt",),
            metadata={},
        )
    manifest = json.loads((target / "checkpoint-manifest.json").read_bytes())
    manifest["schema"] = "solar.checkpoint.unsupported"
    manifest_bytes = canonical_json_bytes(manifest)
    (target / "checkpoint-manifest.json").write_bytes(manifest_bytes)
    complete = json.loads((target / "COMPLETE.json").read_bytes())
    complete["manifest_digest"] = hashlib.blake2s(manifest_bytes).hexdigest()
    (target / "COMPLETE.json").write_bytes(canonical_json_bytes(complete))
    with pytest.raises(CheckpointError, match="unknown or mismatched"):
        verify_checkpoint(target)


def test_exact_resume_rejects_camera_transform_drift() -> None:
    with pytest.raises(CheckpointError, match="camera_translation_transform"):
        assert_resume_compatible(_contract(), _contract(camera_translation_transform="logd4"))
