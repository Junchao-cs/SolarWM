from pathlib import Path

import pytest

from solar.backends.minimax_h3.config import validate_h3_config
from solar.backends.minimax_h3.full_length import exported_fps, full_length_geometry
from solar.config import load_config
from solar.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "configs/solar_wm/minimax_h3/infer-stage2-source-length-sp8.yaml"


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


@pytest.mark.parametrize("sp", [1, 2, 4, 8])
def test_source_length_accepts_independent_single_sp_groups(sp):
    config = load_config(EXAMPLE).mutable_copy()
    config["distributed"].update(world_size=sp, sequence_parallel_size=sp)
    assert validate_h3_config(config).sequence_parallel_size == sp


def test_full_length_rollout_uses_every_chunk_and_keeps_training_strict(monkeypatch):
    from types import SimpleNamespace

    import torch

    from solar.backends.minimax_h3 import sgf_rollout as sgf

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


@pytest.mark.parametrize("memory", ["absent", None, {}, {"enabled": False}])
def test_generation_without_memory_skips_adapter_loading(monkeypatch, memory):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch.distributed as dist

    from solar.backends.minimax_h3 import (
        artifacts,
        full_inference,
        lora,
        memory_inference,
        optional,
        runtime,
        weights,
    )

    config = load_config(EXAMPLE).mutable_copy()
    if memory != "absent":
        config["inference"]["memory"] = memory
    model = Mock()
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(
        optional, "load_transformer", lambda *_args, **_kwargs: SimpleNamespace(transformer=model)
    )
    monkeypatch.setattr(runtime, "_base_model_load_receipt", lambda *_args: {})
    monkeypatch.setattr(lora, "inject_h3_lora", lambda *_args, **_kwargs: (model, None))
    monkeypatch.setattr(weights, "load_initial_weights", lambda *_args: "stage2:ema:step=3900")
    attach = Mock(side_effect=AssertionError("Memory adapter must remain unloaded"))
    monkeypatch.setattr(memory_inference, "attach_memory", attach)
    monkeypatch.setattr(
        artifacts, "load_silence_latents", Mock(side_effect=RuntimeError("condition-load-reached"))
    )
    args = SimpleNamespace(
        config=config,
        checkpoint="/unused",
        weight_source="ema",
        expected_step=3900,
        silence="/unused",
    )
    with pytest.raises(RuntimeError, match="condition-load-reached"):
        full_inference.generate(args)
    attach.assert_not_called()
