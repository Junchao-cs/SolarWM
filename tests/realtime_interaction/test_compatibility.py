"""Catch core changes that invalidate the realtime runtime's source pins."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
H3 = ROOT / "solar/backends/minimax_h3"
WEB = ROOT / "deploy/solar_wm/solarwm_web"


@pytest.mark.parametrize(
    ("pin", "source"),
    [
        ("model", H3 / "model.py"),
        ("layout", H3 / "layout.py"),
        ("torch_layout", H3 / "torch_layout.py"),
        ("sgf_rollout", H3 / "sgf_rollout.py"),
        ("sgf_attention", H3 / "sgf_attention.py"),
        ("torch_flow", H3 / "torch_flow.py"),
        ("distributed", H3 / "distributed.py"),
        ("canonical_all_gather", WEB / "acceleration/attention/all_gather.py"),
    ],
    ids=lambda value: value.name if isinstance(value, Path) else value,
)
def test_realtime_source_pin_matches_current_checkout(pin: str, source: Path) -> None:
    pins = json.loads((WEB / "acceleration/compatibility.json").read_text())

    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins["digests"][pin], (
        f"{source.relative_to(ROOT)} no longer matches the qualified realtime runtime. "
        "Revalidate the acceleration integration before changing its compatibility pins."
    )


@pytest.mark.parametrize("name", ["first_window.pt", "steady.pt"])
def test_realtime_vae_calibration_asset_is_intact(name: str) -> None:
    tables = WEB / "vae/tables"
    manifest = json.loads((tables / "assets.json").read_text())
    data = (tables / name).read_bytes()

    assert len(data) == manifest[name]["bytes"]
    assert hashlib.sha256(data).hexdigest() == manifest[name]["sha256"]
