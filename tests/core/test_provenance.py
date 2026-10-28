from __future__ import annotations

from pathlib import Path

import pytest

from solar.errors import ConfigurationError
from solar.runtime.provenance import build_launch_manifest, reject_inline_secrets


def _config() -> dict:
    return {
        "schema": "solarwm.run.v1",
        "action": "train",
        "name": "test",
        "model": {"family": "wan22_ti2v_5b"},
        "data": {"index": "indexes/train.jsonl", "generation": "g1"},
        "runtime": {},
    }


def test_inline_credentials_are_rejected_but_token_files_are_not() -> None:
    reject_inline_secrets({"gcs_token_file": "/path/to/token"})
    with pytest.raises(ConfigurationError, match="inline credential"):
        reject_inline_secrets({"access_token": "sensitive"})


def test_manifest_identity_is_deterministic(tmp_path: Path) -> None:
    values = _config()
    kwargs = {
        "config": values,
        "source_config": tmp_path / "run.yaml",
        "source_digest": "a" * 64,
        "resolved_digest": "b" * 64,
        "route": "wan22_ti2v_5b:stage0p5:bidirectional:flow_matching",
        "repository": tmp_path,
    }
    left = build_launch_manifest(**kwargs)
    right = build_launch_manifest(**kwargs)
    assert left == right
    assert left["source"]["available"] is False
    assert left["launch_identity_digest"]


def test_enforced_image_must_be_immutable(tmp_path: Path) -> None:
    values = _config()
    values["runtime"] = {"enforce_image": True, "image": "repo:latest"}
    with pytest.raises(ConfigurationError, match="requires a digest-pinned image"):
        build_launch_manifest(
            config=values,
            source_config=tmp_path / "run.yaml",
            source_digest="a" * 64,
            resolved_digest="b" * 64,
            route="wan22_ti2v_5b:stage0p5:bidirectional:flow_matching",
            repository=tmp_path,
        )
