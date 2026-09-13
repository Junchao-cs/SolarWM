from pathlib import Path

import pytest

from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.backends.minimax_h3.full_length import exported_fps, full_length_geometry
from solarwm.config import load_config
from solarwm.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "configs/examples/minimax_h3/infer-stage2-source-length-sp8.yaml"


@pytest.mark.parametrize(
    "frames,decoded,latents,rollout",
    [(81, 90, 27, 30), (158, 158, 47, 50), (957, 957, 282, 285), (960, 974, 287, 290)],
)
def test_source_length_codec_boundaries(frames, decoded, latents, rollout):
    g = full_length_geometry(frames)
    assert (g.decoded_frames, g.decode_latents, g.rollout_latents) == (decoded, latents, rollout)
    assert len(g.camera_indices) == rollout
    assert g.camera_indices[:7] == (0, 1, 5, 9, 13, 17, 18)
    assert all(0 <= index < frames for index in g.camera_indices)
    assert list(g.camera_indices) == sorted(g.camera_indices)
    assert g.decoded_frames >= frames and g.decoded_frames - 17 < frames
    assert exported_fps(16.0, "source") == 16.0
    assert frames / exported_fps(16.0, "source") == frames / 16


@pytest.mark.parametrize("frames", [True, 0, 21, 158.5])
def test_geometry_rejects_invalid_source_lengths(frames):
    with pytest.raises(ValueError):
        full_length_geometry(frames)


def test_full_length_sp8_is_explicit_and_inference_only():
    config = load_config(EXAMPLE).mutable_copy()
    assert validate_h3_config(config).sequence_parallel_size == 8
    config["inference"]["length_policy"] = "fixed"
    with pytest.raises(ConfigurationError, match="SP4"):
        validate_h3_config(config)
    config["inference"]["length_policy"] = "source"
    config["action"] = "train"
    with pytest.raises(ConfigurationError, match="only supported"):
        validate_h3_config(config)


@pytest.mark.parametrize("field,value", [("world_size", 16), ("sequence_parallel_size", 4)])
def test_source_length_rejects_another_topology(field, value):
    config = load_config(EXAMPLE).mutable_copy()
    config["distributed"][field] = value
    with pytest.raises(ConfigurationError):
        validate_h3_config(config)


def test_full_length_rollout_uses_every_chunk_and_keeps_training_strict(monkeypatch):
    from types import SimpleNamespace

    import torch

    from solarwm.backends.minimax_h3 import sgf_rollout as sgf

    seen = []

    def forward(student, inputs, noisy, timestep, **kwargs):
        seen.append((kwargs["chunk_index"], kwargs.get("commit_cache", False)))
        return torch.ones_like(noisy) * kwargs["chunk_index"]

    monkeypatch.setattr(sgf, "h3_student_forward", forward)
    inputs = SimpleNamespace(
        camera_viewmats=torch.zeros(1, 61, 4, 4),
        camera_K=torch.zeros(1, 61, 3, 3),
        sp_enabled=False,
    )
    noise = torch.zeros(1, 1, 60, 1, 1)
    result = sgf.h3_sgf_rollout(
        student=None, inputs=inputs, noise=noise, exit_index=3, inference_full_length=True
    )
    assert result.cache_target.shape[2] == 60
    assert result.cache_target.flatten().tolist() == [float(i) for i in range(12) for _ in range(5)]
    assert [i for i, commit in seen if commit] == list(range(12))
    assert len(seen) == 12 * 5
    with pytest.raises(ValueError, match="exactly 50"):
        sgf.h3_sgf_rollout(student=None, inputs=inputs, noise=noise, exit_index=3)
    with pytest.raises(ValueError, match="exit_index=3"):
        sgf.h3_sgf_rollout(
            student=None, inputs=inputs, noise=noise, exit_index=2, inference_full_length=True
        )
