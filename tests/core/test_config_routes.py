from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from solar.config import load_config
from solar.config.loader import canonical_json
from solar.errors import ConfigurationError


def _write(tmp_path: Path, values: dict) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _base() -> dict:
    return {
        "schema": "solarwm.run.v1",
        "action": "train",
        "name": "unit",
        "model": {"family": "wan22_ti2v_5b", "camera_translation_transform": "linear"},
        "data": {"pixel_frames": 81},
        "distributed": {"sp_size": 1},
        "train": {
            "stage": "stage0p5",
            "causal_mode": "bidirectional",
            "objective": "flow_matching",
        },
        "runtime": {"seed": 7},
    }


def test_load_and_override_has_stable_digest(tmp_path: Path) -> None:
    path = _write(tmp_path, _base())
    first = load_config(path, ["runtime.seed=9"])
    second = load_config(path, ["runtime.seed=9"])
    assert first.values["runtime"]["seed"] == 9
    assert first.resolved_digest == second.resolved_digest


def test_canonical_json_round_trip_preserves_scientific_notation_floats(
    tmp_path: Path,
) -> None:
    values = _base()
    values["train"]["optimizer"] = {"lr": 5e-5, "eps": 1e-8}
    path = tmp_path / "resolved.json"
    path.write_bytes(canonical_json(values))

    resolved = load_config(path)

    assert resolved.values["train"]["optimizer"] == {"lr": 5e-5, "eps": 1e-8}
    assert resolved.source_digest == resolved.resolved_digest


def test_unregistered_stage_is_rejected(tmp_path: Path) -> None:
    values = _base()
    values["train"]["stage"] = "unregistered"
    values["train"]["causal_mode"] = "teacher_forcing"
    values["train"]["objective"] = "anyflow_forward_map"
    with pytest.raises(ConfigurationError, match="unsupported route"):
        load_config(_write(tmp_path, values))


def test_h3_requires_logd4(tmp_path: Path) -> None:
    values = _base()
    values["model"]["family"] = "minimax_h3"
    values["data"]["pixel_frames"] = 158
    with pytest.raises(ConfigurationError, match="requires logd4"):
        load_config(_write(tmp_path, values))


def test_unknown_top_level_key_fails(tmp_path: Path) -> None:
    values = _base()
    values["typo"] = True
    with pytest.raises(ConfigurationError, match="unknown top-level"):
        load_config(_write(tmp_path, values))


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "duplicate.yaml"
    config.write_text(
        """schema: solarwm.run.v1
action: train
name: first
name: second
model: {}
data: {}
runtime: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="duplicate YAML key"):
        load_config(config)


def test_non_json_yaml_values_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "timestamp.yaml"
    config.write_text(
        """schema: solarwm.run.v1
action: train
name: timestamp
model: {}
data: {}
runtime:
  created: 2026-08-20
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="canonical JSON-compatible"):
        load_config(config)
