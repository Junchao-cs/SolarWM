from __future__ import annotations

from solar.backends.wan22.runtime import readiness


def test_normal_training_probe_does_not_require_inference_diffusers(monkeypatch) -> None:
    monkeypatch.setattr(readiness.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        readiness.metadata,
        "version",
        lambda distribution: "0.38.0" if distribution == "diffusers" else "1.0",
    )

    versions, issues = readiness._dependency_versions(online=True, transformer=True)

    assert versions["diffusers"] == "0.38.0"
    assert not [issue for issue in issues if issue.code == "dependency_version_mismatch"]
