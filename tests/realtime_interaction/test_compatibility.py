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


@pytest.mark.parametrize("chunk_index", [0, 1, 5, 7])
def test_packed_history_attention_matches_native_across_window_boundary(chunk_index):
    from dataclasses import replace

    import torch
    from torch.nn.functional import scaled_dot_product_attention

    from deploy.solar_wm.solarwm_web.acceleration.attention import history
    from solar.backends.minimax_h3 import sgf_attention as native
    from solar.backends.minimax_h3.layout import build_stage0p5_layout

    layout = build_stage0p5_layout([0, 1, 1], 5, 2, 2, 3).to("cpu")
    generator = torch.Generator().manual_seed(42)
    cache = native.H3RawKVCache()
    packed = history.PackedHistoryCache()
    with torch.no_grad():
        for index in range(chunk_index):
            key, value = [torch.randn(1, 5, 2, 128, generator=generator) for _ in range(2)]
            cache.commit(0, index, key, value)
            packed.commit(0, index, key, value)
    entries = packed.history(0, chunk_index)
    if entries:
        packed.packed_history_by_layer[0] = (
            chunk_index,
            tuple(torch.cat([entry[index] for entry in entries], dim=1) for index in (1, 2)),
        )
    views = torch.eye(4).repeat(1, 51, 1, 1)
    views[0, :, 0, 3] = torch.linspace(0, 1, 51)
    control = native.H3SGFAttention(
        layout,
        views,
        torch.eye(3).repeat(1, 51, 1, 1),
        "rollout",
        chunk_index=chunk_index,
        cache=cache,
        commit_cache=True,
    )

    def rope(positions):
        angles = positions.float().sum(-1, keepdim=True).expand(-1, 128) * 0.01
        return angles.cos(), angles.sin()

    def attend(q, k, v):
        return scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    control = native.prepare_sgf_rotaries(control, rope)
    q, k, v = [
        torch.randn(1, layout.sequence_length, 2, 128, generator=generator) for _ in range(3)
    ]
    kwargs = dict(layer=0, apply_rotary=lambda x, c, s: x * c[None, :, None], attend=attend)
    with torch.no_grad():
        expected = native.sgf_attention(q, k, v, control=control, **kwargs)
        actual = history.candidate_sgf_attention(
            q, k, v, control=replace(control, cache=packed), **kwargs
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for original, accelerated in zip(cache.layers[0], packed.layers[0], strict=True):
        assert original[0] == accelerated[0]
        for a, b in zip(original[1:], accelerated[1:], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_realtime_attention_scope_accepts_sources_and_restores_hooks():
    from deploy.solar_wm.solarwm_web.acceleration.attention import history
    from deploy.solar_wm.solarwm_web.acceleration.graph.inputs import _check_sources
    from solar.backends.minimax_h3 import sgf_attention, sgf_rollout

    _check_sources()
    original_attention = sgf_attention.sgf_attention
    original_cache = sgf_rollout.H3RawKVCache
    with history.installed_candidate():
        assert sgf_attention.sgf_attention is history.candidate_sgf_attention
        assert sgf_rollout.H3RawKVCache is history.PackedHistoryCache
    assert sgf_attention.sgf_attention is original_attention
    assert sgf_rollout.H3RawKVCache is original_cache
