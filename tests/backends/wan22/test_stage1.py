from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


def test_stage1_training_defers_cleanup_to_explicit_caller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from solar.backends.wan22.runtime import stage1

    runtime = SimpleNamespace(topology=SimpleNamespace(raw_rank=0))
    calls: list[str] = []

    class FakeEngine:
        def __init__(self, runtime_arg: object, policy: object, *, event_sink: object) -> None:
            assert runtime_arg is runtime

        def run(self) -> int:
            calls.append("run")
            return 1

    monkeypatch.setenv("SOLARWM_TORCHRUN_LIFECYCLE_OWNER", "caller")
    monkeypatch.setattr(stage1, "build_stage1_runtime", lambda _: runtime)
    monkeypatch.setattr(stage1, "TrainingEngine", FakeEngine)
    monkeypatch.setattr(stage1, "cleanup_torchrun", lambda: calls.append("cleanup"))
    config = {
        "train": {"max_steps": 1, "grad_accum": 1},
        "runtime": {"output_dir": str(tmp_path)},
    }

    assert stage1.run_stage1_training(config) == 0
    assert calls == ["run"]


def test_per_block_sampler_consumes_the_full_frame_draw() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.stage1 import (
        sample_per_block_timestep_indices,
    )

    actual_generator = torch.Generator().manual_seed(1234)
    actual = sample_per_block_timestep_indices(
        2,
        6,
        3,
        num_train_timesteps=1000,
        device="cpu",
        generator=actual_generator,
    )
    actual_next = torch.randint(0, 1000, (4,), generator=actual_generator)

    reference_generator = torch.Generator().manual_seed(1234)
    reference = torch.randint(0, 1000, (2, 6), generator=reference_generator)
    blocks = reference.reshape(2, 2, 3)
    blocks[:, :, 1:] = blocks[:, :, :1]
    reference_next = torch.randint(0, 1000, (4,), generator=reference_generator)

    assert torch.equal(actual, blocks.reshape(2, 6))
    assert torch.equal(actual_next, reference_next)


def test_teacher_forcing_adapter_uses_the_root_module_forward() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.components import WanDiffusion

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class RootModule:
        def __call__(self, *args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            return torch.zeros((1, 2, 3, 4, 5))

    diffusion = WanDiffusion(
        RootModule(),
        timestep_shift=5.0,
        frame_sequence_length=4,
        num_output_frames=3,
    )
    result = diffusion.forward_train_tf(
        torch.zeros((1, 3, 2, 4, 5)),
        torch.ones((1, 3, 2, 4, 5)),
        {"prompt_embeds": torch.zeros((1, 2, 3))},
        {"viewmats": object(), "K": object()},
        torch.zeros((1, 12)),
        torch.zeros((1, 12)),
        num_frame_per_block=3,
    )
    assert tuple(result.shape) == (1, 3, 2, 4, 5)
    assert len(calls) == 1
    assert calls[0][1]["training_mode"] == "teacher_forcing"
    assert tuple(calls[0][0][0].shape) == (1, 2, 3, 4, 5)
    assert tuple(calls[0][1]["clean_x"].shape) == (1, 2, 3, 4, 5)


def test_wan_adapter_applies_logd4_once_at_the_model_boundary() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.components import WanDiffusion

    calls: list[dict[str, object]] = []

    class RootModule:
        def __call__(self, *_args: object, **kwargs: object) -> object:
            calls.append(kwargs)
            return torch.zeros((1, 2, 1, 1, 1))

    diffusion = WanDiffusion(
        RootModule(),
        timestep_shift=5.0,
        frame_sequence_length=1,
        num_output_frames=1,
        camera_translation_transform="logd4",
    )
    viewmats = torch.eye(4).reshape(1, 1, 4, 4)
    viewmats[..., 0, 3] = 3.0
    camera = {"viewmats": viewmats, "K": torch.eye(3).reshape(1, 1, 3, 3)}
    diffusion(
        torch.zeros((1, 1, 2, 1, 1)),
        {"prompt_embeds": torch.zeros((1, 1, 1))},
        camera,
        torch.zeros((1, 1)),
    )

    transformed = calls[0]["y_camera"]
    assert isinstance(transformed, dict)
    assert torch.equal(transformed["K"], camera["K"])
    assert torch.allclose(
        transformed["viewmats"][..., 0, 3],
        torch.full((1, 1), torch.log(torch.tensor(4.0)).item() / 4.0),
    )
    assert torch.equal(camera["viewmats"], viewmats)


def test_causal_root_forward_dispatches_teacher_forcing_inside_wrapper() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.modeling.causal_model import CausalWanModel

    model = CausalWanModel.__new__(CausalWanModel)
    torch.nn.Module.__init__(model)
    seen: dict[str, object] = {}

    def fake_forward_train_tf(self: object, **kwargs: object) -> str:
        seen.update(kwargs)
        return "teacher-output"

    model.forward_train_tf = MethodType(fake_forward_train_tf, model)
    result = model.forward(
        "noisy",
        "timestep",
        "context",
        12,
        y="official-y",
        y_camera="camera",
        r="r",
        training_mode="teacher_forcing",
        clean_x="clean",
        aug_t="augmentation",
        num_frame_per_block=3,
    )
    assert result == "teacher-output"
    assert seen == {
        "x": "noisy",
        "clean_x": "clean",
        "t": "timestep",
        "aug_t": "augmentation",
        "context": "context",
        "seq_len": 12,
        "num_frame_per_block": 3,
        "y": "official-y",
        "y_camera": "camera",
        "r": "r",
        "aug_r": None,
    }


def test_inference_window_adapter_uses_the_root_module_forward() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.components import WanDiffusion

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class RootModule:
        def __call__(self, *args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            return torch.zeros((1, 2, 3, 4, 5))

    diffusion = WanDiffusion(
        RootModule(),
        timestep_shift=5.0,
        frame_sequence_length=4,
        num_output_frames=3,
    )
    result = diffusion.forward_inference_window(
        torch.zeros((1, 3, 2, 4, 5)),
        torch.ones((1, 6, 2, 4, 5)),
        {"prompt_embeds": torch.zeros((1, 2, 3))},
        {"viewmats": object(), "K": object()},
        torch.zeros((1, 12)),
        num_frame_per_block=3,
        i2v_y=torch.zeros((1, 9, 20, 4, 5)),
        clean_history_timestep=0.25,
        r_timestep_tokens=torch.ones((1, 12)),
    )
    assert tuple(result.shape) == (1, 3, 2, 4, 5)
    assert len(calls) == 1
    assert calls[0][1]["training_mode"] == "inference_window"
    assert tuple(calls[0][0][0].shape) == (1, 2, 3, 4, 5)
    assert tuple(calls[0][1]["clean_history"].shape) == (1, 2, 6, 4, 5)
    assert tuple(calls[0][1]["y"].shape) == (1, 20, 9, 4, 5)
    assert calls[0][1]["clean_history_timestep"] == 0.25


def test_causal_root_forward_dispatches_inference_window_inside_wrapper() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.modeling.causal_model import CausalWanModel

    model = CausalWanModel.__new__(CausalWanModel)
    torch.nn.Module.__init__(model)
    seen: dict[str, object] = {}

    def fake_forward_inference_window(self: object, **kwargs: object) -> str:
        seen.update(kwargs)
        return "window-output"

    model.forward_inference_window = MethodType(fake_forward_inference_window, model)
    result = model.forward(
        "noisy",
        "timestep",
        "context",
        12,
        y="official-y",
        y_camera="camera-window",
        r="target-timestep",
        training_mode="inference_window",
        clean_history="clean-history",
        num_frame_per_block=3,
        clean_history_timestep=0.25,
    )
    assert result == "window-output"
    assert seen == {
        "noisy_chunk": "noisy",
        "clean_history": "clean-history",
        "t": "timestep",
        "context": "context",
        "seq_len": 12,
        "num_frame_per_block": 3,
        "y": "official-y",
        "y_camera_window": "camera-window",
        "clean_history_timestep": 0.25,
        "r": "target-timestep",
        "clean_history_r_timestep": None,
    }
