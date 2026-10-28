from __future__ import annotations

from types import SimpleNamespace

import pytest

from solar.renderer.core import _Prepared
from solar.renderer.dmd import H3DMDInputs, H3OmniDMDCore

torch = pytest.importorskip("torch")


class RecordingVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.1 * (i + 1))) for i in range(4)]
        )
        self.calls = []

    def forward(self, **kwargs):
        index = len(self.calls)
        self.calls.append((torch.is_grad_enabled(), kwargs["hidden_states"].detach().clone()))
        return torch.ones_like(kwargs["hidden_states"]) * self.values[index], None


def small_inputs(model=None):
    core = H3OmniDMDCore(model, torch.zeros(1, 32, 8), torch.device("cpu"), {})
    layout = core.layout(torch.tensor([1, 1]), anchors=(), grid=(2, 2), frames=2)
    clean = torch.zeros(1, 24, 2, 2, 2)
    audio = torch.zeros(1, 8, 32)
    prepared = _Prepared(
        clean=clean,
        video_noise=torch.zeros_like(clean),
        video_t=torch.tensor(0.5),
        audio_t=torch.tensor(0.5),
        audio_rows=audio,
        condition_rows=[],
        prompt=torch.zeros(1, 2, 5120),
        layout=layout,
        viewmats=None,
        K=None,
        frame_ids=None,
        observation_degrees=None,
        plan=None,
    )
    return core, H3DMDInputs(prepared, audio, 2, (2, 2))


@pytest.mark.parametrize("exit_index", range(4))
def test_dmd_rollout_differentiates_only_selected_exit(exit_index):
    model = RecordingVelocity()
    core, inputs = small_inputs(model)
    output = core.rollout(
        inputs,
        model=model,
        forwards=4,
        exit_index=exit_index,
        generator=torch.Generator().manual_seed(731),
    )
    assert output.shape == (1, 24, 2, 2, 2)
    assert torch.isfinite(output).all()
    assert [enabled for enabled, _ in model.calls] == [False] * exit_index + [True]
    output.square().mean().backward()
    for index, parameter in enumerate(model.values):
        if index == exit_index:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad) and parameter.grad != 0
        else:
            assert parameter.grad is None
    if exit_index:
        assert not torch.equal(model.calls[0][1], model.calls[1][1])


def test_dmd_rollout_seed_and_no_grad_contract():
    results = []
    for _ in range(2):
        model = RecordingVelocity()
        core, inputs = small_inputs(model)
        with torch.no_grad():
            results.append(
                core.rollout(
                    inputs,
                    model=model,
                    forwards=4,
                    exit_index=3,
                    generator=torch.Generator().manual_seed(97),
                )
            )
        assert not results[-1].requires_grad
        assert all(not enabled for enabled, _ in model.calls)
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.parametrize("forwards,exit_index", [(0, 0), (4, -1), (4, 4)])
def test_dmd_rollout_rejects_invalid_exit(forwards, exit_index):
    core, inputs = small_inputs()
    with pytest.raises(ValueError, match="exit_index"):
        core.rollout(inputs, model=None, forwards=forwards, exit_index=exit_index)


def test_dmd_score_noise_is_reproducible_bounded_and_data_ward():
    core, _ = small_inputs()
    clean = torch.randn(1, 24, 2, 2, 2, requires_grad=True)
    first = core.score_inputs(clean, torch.Generator().manual_seed(31))
    second = core.score_inputs(clean, torch.Generator().manual_seed(31))
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    noisy, noise, video_t, audio_t = first
    assert 0.019 <= video_t <= 0.981 and 0.019 <= audio_t <= 0.981
    torch.testing.assert_close(noisy + (1 - video_t) * (clean - noise), clean)
    noisy.sum().backward()
    torch.testing.assert_close(clean.grad, torch.full_like(clean, float(video_t)))


def conditioned_inputs(*, degree=1.0, drop_free=False):
    core, inputs = small_inputs()
    observations = (
        SimpleNamespace(
            latents=torch.full((24, 2, 2, 2), 2.0), frames=(0, 1), degree=degree, boxes=None
        ),
        SimpleNamespace(
            latents=torch.full((24, 1, 2, 2), 3.0), frames=(0,), degree=degree, boxes=None
        ),
        SimpleNamespace(
            latents=torch.full((24, 1, 2, 2), 4.0), frames=None, degree=degree, boxes=None
        ),
    )
    batch = SimpleNamespace(
        target_latents=inputs.prepared.clean[0],
        anchor_latents=None,
        visual_observations=observations,
        prompt_embeds=torch.zeros(2, 5120),
        text_token_tags=torch.tensor([1, 1]),
        text_prompt_embeds=torch.zeros(2, 5120),
        text_prompt_tags=torch.tensor([1, 1]),
    )
    plan = (
        SimpleNamespace(keeps_observation=lambda i: i != 2, use_anchor=False) if drop_free else None
    )
    keep = None if plan is None else plan.keeps_observation
    specs, degrees, rows = core.observation_rows(batch, torch.Generator().manual_seed(33), keep)
    from dataclasses import replace

    prepared = replace(
        inputs.prepared,
        condition_rows=[rows],
        observation_degrees=degrees,
        plan=plan,
        layout=core.layout(
            torch.tensor([1, 1]), anchors=(), observations=specs, grid=(2, 2), frames=2
        ),
    )
    return core, replace(inputs, prepared=prepared), batch


@pytest.mark.parametrize("drop_free", [False, True])
@pytest.mark.parametrize("degree", [1.0, 0.999])
def test_structure_free_teacher_keeps_non_structure_conditions(drop_free, degree):
    core, inputs, batch = conditioned_inputs(drop_free=drop_free, degree=degree)
    rng_before = torch.get_rng_state().clone()
    free = core.structure_free(inputs, batch)
    assert torch.equal(torch.get_rng_state(), rng_before)
    expected = inputs.prepared.condition_rows[0][:, 2:]
    torch.testing.assert_close(free.prepared.condition_rows[0], expected, rtol=0, atol=0)
    assert free.prepared.prompt is inputs.prepared.prompt
    assert free.prepared.clean is inputs.prepared.clean
    assert free.prepared.layout.source.num_observation_rows == (1 if drop_free else 2)
    assert inputs.prepared.layout.source.num_observation_rows == (3 if drop_free else 4)


def test_structure_guidance_changes_only_teacher_structure_component(monkeypatch):
    core, inputs, batch = conditioned_inputs()
    calls = []

    def score(model, prepared, noisy, video_t, audio_t, audio_noise):
        calls.append(prepared)
        observed = prepared.prepared.layout.source.num_observation_rows
        return torch.ones_like(noisy) * observed

    monkeypatch.setattr(core, "score_forward", score)
    noisy = torch.full((1, 24, 2, 2, 2), 7.0)
    result = core.teacher_answer(
        None, inputs, batch, noisy, torch.tensor(0.5), torch.tensor(0.7), None, guidance=2
    )
    torch.testing.assert_close(result, torch.full_like(result, 10.0))
    assert len(calls) == 2
    calls.clear()
    ordinary = core.teacher_answer(
        None, inputs, batch, noisy, torch.tensor(0.5), torch.tensor(0.7), None, guidance=1
    )
    torch.testing.assert_close(ordinary, torch.full_like(result, 9.0))
    assert len(calls) == 1


class ScalarVelocity(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, **kwargs):
        return torch.ones_like(kwargs["hidden_states"]) * self.weight, None


class TinyAdapter:
    def __init__(self, model):
        self.parameter_by_key = {"weight": model.weight}
        self.parameters = tuple(self.parameter_by_key.values())

    def state_dict(self):
        return {key: value.detach() for key, value in self.parameter_by_key.items()}

    def load_state_dict(self, state, *, broadcast=False):
        assert set(state) == set(self.parameter_by_key)
        with torch.no_grad():
            for key, value in state.items():
                self.parameter_by_key[key].copy_(value)

    def metadata(self):
        return {"synthetic_scalar_adapter": 1}


class TinyReader:
    current_plan_fingerprint = "fixed-small-training-fixture"

    def __init__(self):
        self.position = 0

    def next(self):
        self.position += 1
        return SimpleNamespace(sample_id=f"sample-{self.position}")

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def small_runtime(tmp_path, monkeypatch):
    from solar.backends.minimax_h3.fsdp import finite_clip_norm
    from solar.renderer.dmd_runtime import H3OmniDMDRuntime, build_optimizer

    runtime = object.__new__(H3OmniDMDRuntime)
    runtime.torch, runtime.dist = torch, torch.distributed
    runtime.device, runtime.rank, runtime.is_main = torch.device("cpu"), 0, True
    runtime.topology = SimpleNamespace(raw_world_size=1, raw_rank=0, local_rank=0)
    runtime.model = ScalarVelocity(0.2)
    runtime.teacher = ScalarVelocity(0.7).requires_grad_(False)
    critic = ScalarVelocity(0.1)
    runtime.lora, critic_lora = TinyAdapter(runtime.model), TinyAdapter(critic)
    optimizer_cfg = {"learning_rate": 0.01, "betas": [0.0, 0.99], "gradient_clip": 10.0}
    runtime.optimizer = build_optimizer(runtime.lora, optimizer_cfg)
    critic_optimizer = build_optimizer(critic_lora, optimizer_cfg)
    runtime.scheduler = torch.optim.lr_scheduler.LambdaLR(runtime.optimizer, lambda _: 1.0)
    runtime.extra_roles = {
        "critic": {
            "model": critic,
            "lora": critic_lora,
            "optimizer": critic_optimizer,
            "scheduler": torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lambda _: 1.0),
        }
    }
    runtime.dmd = {
        "mode": "adapt_plug",
        "forwards": 4,
        "critic_updates_per_student": 2,
        "structure_guidance": 1.0,
    }
    runtime.mode, runtime.stage = "adapt_plug", "stage0p5"
    runtime.train_cfg = {
        "stage": "stage0p5",
        "objective": "dmd",
        "max_steps": 10,
        "optimizer": optimizer_cfg,
        "critic_optimizer": optimizer_cfg,
        "observations": {"patterns": {"shared": 1.0}},
        "dmd": runtime.dmd,
    }
    runtime.checkpoint_cfg = {"ema": {"enabled": False}}
    runtime.runtime_cfg = {"output_dir": str(tmp_path)}
    runtime.config = {
        "train": runtime.train_cfg,
        "checkpoint": runtime.checkpoint_cfg,
        "distributed": {"sequence_parallel_size": 1},
    }
    runtime._global_step, runtime.student_step, runtime._weights_id = 0, 0, "base-fixture"
    runtime.ema = None
    runtime.reader = TinyReader()
    runtime.core, inputs = small_inputs(runtime.model)
    runtime.core.prepare_inputs = lambda batch: inputs
    runtime._next_batch = runtime.reader.next
    runtime._phase = lambda name: None
    runtime._finite_clip_norm = finite_clip_norm
    runtime.contract = runtime._make_contract(
        encoder_profile={"profile": "fixture"},
        silence_profile={"profile": "fixture"},
        base_model={"digest": "fixture"},
        config=runtime.config,
    )
    cuda_rng = torch.Generator().manual_seed(100)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device=None: cuda_rng.get_state())
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda value, device=None: cuda_rng.set_state(value)
    )
    return runtime


def test_dmd_updates_critic_each_step_and_student_after_warmup(tmp_path, monkeypatch):
    runtime = small_runtime(tmp_path, monkeypatch)
    student_start = runtime.model.weight.detach().clone()
    teacher_start = runtime.teacher.weight.detach().clone()
    critic = runtime.extra_roles["critic"]
    critic_start = critic["model"].weight.detach().clone()
    for step in range(6):
        stats = runtime.train_outer_step()
        assert stats["student_updated"] == (1.0 if step in (2, 4) else 0.0)
        assert torch.isfinite(torch.tensor(stats["loss"]))
        if step < 2:
            torch.testing.assert_close(runtime.model.weight, student_start, rtol=0, atol=0)
    assert runtime.global_step == 6 and runtime.student_step == 2
    assert runtime.reader.position == 8
    assert not torch.equal(runtime.model.weight, student_start)
    assert not torch.equal(critic["model"].weight, critic_start)
    torch.testing.assert_close(runtime.teacher.weight, teacher_start, rtol=0, atol=0)
    assert runtime.teacher.weight.grad is None
    assert runtime.optimizer.state[runtime.model.weight]["step"] == 2
    assert critic["optimizer"].state[critic["model"].weight]["step"] == 6
    assert runtime.model.weight.grad is None and critic["model"].weight.grad is None


def test_dmd_checkpoint_resume_reproduces_next_updates(tmp_path, monkeypatch):
    from solar.checkpoint import verify_checkpoint

    torch.manual_seed(193)
    runtime = small_runtime(tmp_path, monkeypatch)
    for _ in range(3):
        runtime.train_outer_step()
    runtime.save_checkpoint(3)
    checkpoint = tmp_path / "checkpoint_model_000003"
    verified = verify_checkpoint(checkpoint)
    assert "critic.pt" in {record.path for record in verified.files}
    runtime.train_outer_step()
    runtime.train_outer_step()
    expected_student = runtime.model.weight.detach().clone()
    expected_critic = runtime.extra_roles["critic"]["model"].weight.detach().clone()
    expected_position = runtime.reader.position
    resumed = small_runtime(tmp_path, monkeypatch)
    resumed.load_checkpoint(str(checkpoint))
    assert resumed.global_step == 3 and resumed.student_step == 1
    resumed.train_outer_step()
    resumed.train_outer_step()
    torch.testing.assert_close(resumed.model.weight, expected_student, rtol=0, atol=0)
    torch.testing.assert_close(
        resumed.extra_roles["critic"]["model"].weight, expected_critic, rtol=0, atol=0
    )
    assert resumed.reader.position == expected_position
    assert resumed.student_step == 2


def test_renderer_resume_contract_pins_recipe_and_parent_bytes(tmp_path, monkeypatch):
    from copy import deepcopy

    from solar.checkpoint import assert_resume_compatible
    from solar.errors import CheckpointError

    runtime = small_runtime(tmp_path, monkeypatch)
    control = tmp_path / "control.safetensors"
    control.write_bytes(b"first-control-version")
    config = deepcopy(runtime.config)
    config["train"]["dmd"]["control"] = str(control)

    def contract():
        return runtime._make_contract(
            encoder_profile={},
            silence_profile={},
            base_model={},
            config=config,
        )

    original = contract()
    assert original.family == "minimax_h3_renderer"
    config["train"]["max_steps"] = 100
    assert_resume_compatible(original, contract())
    config["train"]["dmd"]["structure_guidance"] = 2.0
    with pytest.raises(CheckpointError, match="contract mismatch"):
        assert_resume_compatible(original, contract())
    config["train"]["dmd"]["structure_guidance"] = 1.0
    control.write_bytes(b"other-control-version")
    with pytest.raises(CheckpointError, match="contract mismatch"):
        assert_resume_compatible(original, contract())


def test_dmd_nonfinite_gradient_does_not_advance_optimizer(tmp_path, monkeypatch):
    runtime = small_runtime(tmp_path, monkeypatch)
    runtime.model.weight.grad = torch.tensor(float("nan"))
    before = runtime.model.weight.detach().clone()
    with pytest.raises(FloatingPointError, match="non-finite"):
        runtime._finish_update(
            runtime.model, runtime.lora, runtime.optimizer, runtime.scheduler, 10.0
        )
    torch.testing.assert_close(runtime.model.weight, before, rtol=0, atol=0)
    assert runtime.optimizer.state_dict()["state"] == {}
    assert runtime.scheduler.last_epoch == 0


def test_dmd_resume_rejects_missing_critic_before_changing_student(tmp_path, monkeypatch):
    from solar.errors import BackendContractError

    runtime = small_runtime(tmp_path, monkeypatch)
    for _ in range(3):
        runtime.train_outer_step()
    runtime.save_checkpoint(3)
    checkpoint = tmp_path / "checkpoint_model_000003"
    (checkpoint / "critic.pt").unlink()
    resumed = small_runtime(tmp_path, monkeypatch)
    before = resumed.model.weight.detach().clone()
    with pytest.raises(BackendContractError, match=r"critic\.pt"):
        resumed.load_checkpoint(str(checkpoint))
    torch.testing.assert_close(resumed.model.weight, before, rtol=0, atol=0)
    assert resumed.global_step == 0


def test_peer_nonfinite_is_reported_before_backward():
    from types import SimpleNamespace

    from solar.renderer.runtime import RendererTrainingRuntime

    runtime = RendererTrainingRuntime.__new__(RendererTrainingRuntime)
    runtime.torch = torch
    runtime.device = torch.device("cpu")
    calls = []

    def reduce_peer_status(flag, *, op):
        calls.append(op)
        assert flag.item() == 1
        flag.zero_()

    runtime.dist = SimpleNamespace(
        is_initialized=lambda: True,
        all_reduce=reduce_peer_status,
        ReduceOp=SimpleNamespace(MIN="MIN"),
    )
    with pytest.raises(FloatingPointError, match="peer failed"):
        runtime._check_finite(True, "peer failed")
    assert calls == ["MIN"]
