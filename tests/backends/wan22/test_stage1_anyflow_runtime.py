from __future__ import annotations

import copy
from pathlib import Path

import pytest

from solar.errors import BackendContractError


def test_negative_embedding_loader_validates_the_real_tensor_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.stage1_anyflow import (
        load_negative_prompt_embedding,
    )

    path = tmp_path / "conditioning" / "wan_negemb_cn.pth"
    path.parent.mkdir()
    torch.save({"negative_prompt_embeds": torch.ones((2, 3))}, path)
    config = {
        "model": {
            "family": "wan22_ti2v_5b",
            "base_path": str(tmp_path),
            "assets": {"transformer_config": "builtin"},
        },
        "train": {"anyflow_negative_embedding": "conditioning/wan_negemb_cn.pth"},
    }

    loaded = load_negative_prompt_embedding(config, device="cpu")

    assert loaded.shape == (1, 2, 3)
    assert loaded.dtype == torch.bfloat16

    torch.save({"wrong": torch.ones((2, 3))}, path)
    with pytest.raises(BackendContractError, match="must contain negative_prompt_embeds"):
        load_negative_prompt_embedding(config, device="cpu")


def test_anyflow_v1_5_uses_exact_five_forward_graph_and_anchor_mask() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.anyflow import bounded_difference_timesteps
    from solar.backends.wan22.runtime.scheduler import FlowMatchScheduler
    from solar.backends.wan22.runtime.stage1_anyflow import (
        anyflow_v15_objective,
    )

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    calls: list[tuple[object, object, bool]] = []

    def model_u(sample: object, timestep: object, endpoint: object, condition: object):
        del condition
        calls.append((timestep.clone(), endpoint.clone(), torch.is_grad_enabled()))
        index = len(calls) - 1
        values = (parameter, 2.0, 4.0, 8.0, 4.0)
        value = values[index]
        if torch.is_tensor(value):
            return value.expand_as(sample)
        return torch.full_like(sample, value)

    clean = torch.zeros((4, 2, 1, 1, 1))
    mask = torch.ones((4, 2), dtype=torch.bool)
    mask[:, 0] = False
    result = anyflow_v15_objective(
        clean=clean,
        condition={"prompt_embeds": torch.zeros((4, 1, 1))},
        model_u=model_u,
        scheduler=FlowMatchScheduler(shift=5.0),
        loss_mask=mask,
        logical_dp_rank=0,
        logical_dp_world_size=1,
        shift=5.0,
        num_train_timesteps=1000,
        epsilon=5.0,
        diffusion_ratio=0.5,
        consistency_ratio=0.25,
        guidance=3.0,
        negative_prompt_embeds=torch.zeros((1, 1, 1)),
        generator=torch.Generator().manual_seed(123),
        noise=torch.ones_like(clean),
    )

    assert result.forward_count == 5
    assert len(calls) == 5
    assert [grad for _, _, grad in calls] == [True, False, False, False, False]
    assert result.sample_type.tolist() == [0, 0, 1, 2]
    assert torch.equal(result.timestep[:, 0], torch.zeros(4))
    assert torch.equal(result.endpoint_timestep[:, 0], torch.zeros(4))
    assert torch.allclose(result.target[:, 1], torch.full_like(result.target[:, 1], -1.0))

    plus, minus = bounded_difference_timesteps(
        result.timestep,
        result.endpoint_timestep,
        epsilon=5.0,
        num_train_timesteps=1000,
    )
    derivative = 4.0 / (plus - minus).clamp_min(1.0e-6)
    expected = 1.0 + (result.timestep - result.endpoint_timestep) * derivative
    assert torch.allclose(result.prediction[..., 0, 0, 0], expected)
    result.loss.backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert parameter.grad.abs().item() > 0


def test_anyflow_v1_5_unguided_path_uses_four_forwards() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.scheduler import FlowMatchScheduler
    from solar.backends.wan22.runtime.stage1_anyflow import (
        anyflow_v15_objective,
    )

    calls = 0

    def model_u(sample: object, *_: object) -> object:
        nonlocal calls
        calls += 1
        return torch.zeros_like(sample)

    clean = torch.zeros((1, 2, 1, 1, 1))
    result = anyflow_v15_objective(
        clean=clean,
        condition={"prompt_embeds": torch.zeros((1, 1, 1))},
        model_u=model_u,
        scheduler=FlowMatchScheduler(shift=5.0),
        loss_mask=torch.tensor([[False, True]]),
        logical_dp_rank=0,
        logical_dp_world_size=1,
        shift=5.0,
        num_train_timesteps=1000,
        epsilon=5.0,
        diffusion_ratio=1.0,
        consistency_ratio=0.0,
        guidance=1.0,
        negative_prompt_embeds=None,
        generator=torch.Generator().manual_seed(5),
        noise=torch.ones_like(clean),
    )
    assert calls == result.forward_count == 4


def test_anyflow_v1_5_default_noise_preserves_clean_stride_rng_order() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.scheduler import FlowMatchScheduler
    from solar.backends.wan22.runtime.stage1_anyflow import anyflow_v15_objective

    clean = torch.zeros((1, 3, 2, 2, 2)).permute(0, 2, 1, 3, 4)
    assert not clean.is_contiguous()
    torch.manual_seed(123)
    result = anyflow_v15_objective(
        clean=clean,
        condition={"prompt_embeds": torch.zeros((1, 1, 1))},
        model_u=lambda sample, *_: torch.zeros_like(sample),
        scheduler=FlowMatchScheduler(shift=5.0),
        loss_mask=torch.tensor([[False, True]]),
        logical_dp_rank=0,
        logical_dp_world_size=1,
        shift=5.0,
        num_train_timesteps=1000,
        epsilon=5.0,
        diffusion_ratio=1.0,
        consistency_ratio=0.0,
        guidance=1.0,
        negative_prompt_embeds=None,
    )
    torch.manual_seed(123)
    torch.rand(1)
    torch.rand(1)
    expected = torch.randn_like(clean)
    assert torch.equal(result.noise, expected)
    assert result.noise.stride() == clean.stride()


def test_anyflow_adaptive_rescale_matches_global_diffusion_mean() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.stage1_anyflow import (
        adaptive_rescale_non_diffusion_losses,
    )

    local = torch.tensor([2.0, 8.0], requires_grad=True)
    mask = torch.tensor([True, False])
    gathered_losses = torch.tensor([2.0, 8.0, 6.0, 4.0])
    gathered_masks = torch.tensor([True, False, True, False])
    calls = 0

    def gather(value: object) -> object:
        nonlocal calls
        calls += 1
        return gathered_losses if value.dtype.is_floating_point else gathered_masks

    scaled = adaptive_rescale_non_diffusion_losses(local, mask, gather_fn=gather)
    assert calls == 2
    assert scaled[0].item() == pytest.approx(2.0)
    assert scaled[1].item() == pytest.approx(4.0, rel=2.0e-6)
    scaled.sum().backward()
    assert local.grad is not None


def test_fm_to_anyflow_upgrade_initializes_exactly_four_delta_tensors() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.checkpoint import (
        _load_stage0p5_state_into_anyflow,
    )

    class ToyAnyFlow(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.time_embedding = torch.nn.Sequential(
                torch.nn.Linear(2, 2),
                torch.nn.SiLU(),
                torch.nn.Linear(2, 2),
            )
            self.delta_embedding = copy.deepcopy(self.time_embedding)
            self.output = torch.nn.Linear(2, 2)

    source = ToyAnyFlow()
    with torch.no_grad():
        for value in source.time_embedding.parameters():
            value.fill_(3.0)
        for value in source.output.parameters():
            value.fill_(7.0)
    fm_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if not key.startswith("delta_embedding.")
    }
    target = ToyAnyFlow()
    delta = _load_stage0p5_state_into_anyflow(target, fm_state)
    assert len(delta) == 4
    for key in delta:
        time_key = key.replace("delta_embedding", "time_embedding", 1)
        assert torch.equal(target.state_dict()[key], target.state_dict()[time_key])

    broken = dict(fm_state)
    broken.pop("output.bias")
    with pytest.raises(BackendContractError, match="exactly four missing"):
        _load_stage0p5_state_into_anyflow(ToyAnyFlow(), broken)


def test_teacher_forcing_adapter_propagates_anyflow_r_and_aug_r() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.components import WanDiffusion

    seen: dict[str, object] = {}

    class Root:
        def __call__(self, value: object, **kwargs: object) -> object:
            seen.update(kwargs)
            return torch.zeros_like(value)

    diffusion = WanDiffusion(
        Root(),
        timestep_shift=5.0,
        frame_sequence_length=2,
        num_output_frames=3,
    )
    value = torch.zeros((1, 3, 1, 1, 1))
    r = torch.ones((1, 6))
    aug_r = torch.full((1, 6), 2.0)
    diffusion.forward_train_tf(
        value,
        value,
        {"prompt_embeds": torch.zeros((1, 1, 1))},
        {"viewmats": object(), "K": object()},
        torch.zeros((1, 6)),
        torch.zeros((1, 6)),
        num_frame_per_block=3,
        r_timestep_tokens=r,
        augmentation_r_timestep_tokens=aug_r,
    )
    assert seen["r"] is r
    assert seen["aug_r"] is aug_r
