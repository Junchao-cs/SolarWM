"""Public causal routes, cross-stage weights and paired SGF checkpoint lifecycle."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.errors import (
    BackendContractError,
    CheckpointError,
    ConfigurationError,
    DataContractError,
)

EXAMPLES = Path(__file__).resolve().parents[3] / "configs/examples/minimax_h3"


def _config(stage):
    return yaml.safe_load(
        (EXAMPLES / f"{stage}-158f-lora384-w6-sp{4 if stage == 'stage2' else 2}.yaml").read_text()
    )


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_causal_examples_select_exact_stage_and_geometry(stage):
    config = _config(stage)
    resolved = validate_h3_config(config)
    assert resolved.stage == stage
    assert resolved.encoded_latents == 47
    assert config["validation"]["rollout_latents"] == 50
    config["data"]["train_target_latents"] = 50
    with pytest.raises(ConfigurationError, match="train_target_latents"):
        validate_h3_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "student_rope_mode", "native_absolute"),
        ("model", "score_rope_mode", "sliding_local"),
        ("train", "tail_camera_policy", "zero"),
        ("train", "audio_condition_policy", "resample"),
        ("train", "critic_updates_per_student", 1),
    ],
)
def test_sgf_rejects_semantic_drift(section, key, value):
    config = _config("stage2")
    config[section][key] = value
    with pytest.raises(ConfigurationError, match=key):
        validate_h3_config(config)


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_causal_inference_examples_use_the_training_contract(stage):
    config = yaml.safe_load(
        (EXAMPLES / f"infer-{stage}-158f-sp{4 if stage == 'stage2' else 2}.yaml").read_text()
    )
    assert validate_h3_config(config).action == "infer"
    assert config["validation"]["num_inference_steps"] == 4


def _lora():
    torch = pytest.importorskip("torch")
    from solarwm.backends.minimax_h3.lora import H3LoRARuntime

    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    lora = H3LoRARuntime(
        model=model,
        targets=("weight",),
        parameter_by_key=OrderedDict(weight=model.weight),
        peft_config=None,
        peft_module=SimpleNamespace(__version__="0.20.0"),
        base_identity={"architecture": "unit-test"},
        rank=384,
        alpha=384,
    )
    return model, lora


@pytest.mark.parametrize("step", [20, 196])
def test_sgf_checkpoint_restores_both_optimizers_rng_reader_and_delayed_ema(
    tmp_path, monkeypatch, step
):
    torch = pytest.importorskip("torch")
    from solarwm.backends.minimax_h3.ema import H3ShardedEMA
    from solarwm.backends.minimax_h3.optimizer import FP32MasterAdamW
    from solarwm.backends.minimax_h3.runtime import H3TrainingRuntime, _checkpoint_contract
    from solarwm.backends.minimax_h3.stage2_runtime import H3SGFTrainingRuntime
    from solarwm.checkpoint import verify_checkpoint

    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: None)

    class Reader:
        cursor = 17

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, values):
            self.cursor = values["cursor"]

    runtime = H3TrainingRuntime.__new__(H3TrainingRuntime)
    runtime.torch, runtime.dist = torch, torch.distributed
    runtime.stage, runtime.rank, runtime.is_main = "stage2", 0, True
    runtime.device = torch.device("cpu")
    runtime.topology = SimpleNamespace(raw_world_size=1, raw_rank=0)
    runtime.model, runtime.lora = _lora()
    critic, critic_lora = _lora()
    runtime.optimizer = FP32MasterAdamW(runtime.lora.parameters, lr=2e-6, betas=(0.0, 0.999))
    critic_optimizer = FP32MasterAdamW(critic_lora.parameters, lr=4e-7, betas=(0.0, 0.999))
    runtime.scheduler = torch.optim.lr_scheduler.LambdaLR(runtime.optimizer, lambda _: 1.0)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lambda _: 1.0)
    runtime.extra_roles = {
        "critic": {
            "model": critic,
            "lora": critic_lora,
            "optimizer": critic_optimizer,
            "scheduler": critic_scheduler,
        }
    }
    runtime.reader = Reader()
    runtime.student_step = (step - 1) // 5
    runtime._global_step, runtime._weights_id = step, "parent"
    runtime.runtime_cfg = {"output_dir": str(tmp_path)}
    runtime.checkpoint_cfg = _config("stage2")["checkpoint"]
    runtime.contract = _checkpoint_contract(
        encoder_profile={}, silence_profile={}, base_model={}, config=_config("stage2")
    )
    for model, optimizer, scheduler in (
        (runtime.model, runtime.optimizer, runtime.scheduler),
        (critic, critic_optimizer, critic_scheduler),
    ):
        model.weight.float().square().sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    runtime.ema = H3ShardedEMA(runtime.model, decay=0.99, device="cpu") if step >= 196 else None
    expected = runtime.model.weight.detach().clone(), critic.weight.detach().clone()
    runtime.save_checkpoint(step)
    path = tmp_path / f"checkpoint_model_{step:06d}"
    assert verify_checkpoint(path).step == step
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    with torch.no_grad():
        runtime.model.weight.zero_()
        critic.weight.zero_()
    runtime.reader.cursor = 999
    runtime.student_step = 0
    runtime.ema = None
    runtime.load_checkpoint(str(path))
    H3SGFTrainingRuntime._check_ema_boundary(runtime)
    assert runtime.student_step == (step - 1) // 5
    assert runtime.reader.cursor == 17
    assert torch.equal(runtime.model.weight, expected[0])
    assert torch.equal(critic.weight, expected[1])
    assert runtime.optimizer.state[runtime.model.weight]["step"] == 1
    assert critic_optimizer.state[critic.weight]["step"] == 1
    assert (random.random(), float(np.random.rand())) == expected_rng[:2]
    assert torch.equal(torch.rand(3), expected_rng[2])


def test_stage_transition_validates_all_tensors_before_mutation(tmp_path):
    torch = pytest.importorskip("torch")
    from solarwm.backends.minimax_h3.weights import load_initial_weights

    (tmp_path / "COMPLETE.json").write_text("{}")
    (tmp_path / "checkpoint-manifest.json").write_text(
        json.dumps(
            {
                "step": 7000,
                "contract": {
                    "family": "minimax_h3",
                    "stage": "stage0p5",
                    "camera_translation_transform": "logd4",
                    "parameterization": "peft-lora-r384-alpha384",
                    "extras": {
                        "encoder_profile": {"pixel_frames": 158, "height": 768, "width": 1344}
                    },
                },
            }
        )
    )
    model, lora = _lora()
    before = model.weight.detach().clone()
    torch.save(
        {"schema": "solarwm.minimax-h3-ema.v1", "shadow": {"weight": torch.ones(3)}},
        tmp_path / "ema.pt",
    )
    spec = {"path": str(tmp_path), "stage": "stage0p5", "weight_source": "ema"}
    with pytest.raises(BackendContractError, match="descriptor"):
        load_initial_weights(spec, lora)
    assert torch.equal(model.weight, before)
    torch.save(
        {"schema": "solarwm.minimax-h3-ema.v1", "shadow": {"weight": torch.full((2, 2), 1.234)}},
        tmp_path / "ema.pt",
    )
    assert load_initial_weights(spec, lora) == "stage0p5:ema:step=7000"
    assert torch.equal(model.weight, torch.full((2, 2), 1.234, dtype=torch.bfloat16))
    with pytest.raises(BackendContractError, match="contract"):
        load_initial_weights({**spec, "stage": "stage1"}, lora)


def test_stage2_checkpoint_rejects_changed_teacher_before_resume():
    from dataclasses import replace

    from solarwm.backends.minimax_h3.runtime import _checkpoint_contract
    from solarwm.checkpoint import assert_resume_compatible

    contract = _checkpoint_contract(
        encoder_profile={}, silence_profile={}, base_model={}, config=_config("stage2")
    )
    first = replace(
        contract,
        extras={**contract.extras, "initialization": {"teacher": "stage0p5:ema:step=10500"}},
    )
    changed = replace(
        contract,
        extras={**contract.extras, "initialization": {"teacher": "stage0p5:ema:step=7000"}},
    )
    with pytest.raises(CheckpointError, match="extras"):
        assert_resume_compatible(first, changed)


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_validation_camera_cadence_and_tail_are_audited(tmp_path, stage):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    from solarwm.backends.minimax_h3.geometry import latent_aligned_pixel_indices
    from solarwm.backends.minimax_h3.validation_inputs import _load_cameras

    count = 170 if stage == "stage1" else 47
    views = torch.eye(4).repeat(count, 1, 1)
    views[:, 0, 3] = torch.arange(count) * 0.01
    row = {
        "validation_start_frame": 70,
        "camera_convention": "relative_w2c+normalized_K",
        "camera_alignment": "latent",
    }
    key = "source_frame_indices" if stage == "stage1" else "source_latent_frame_indices"
    indices = (
        torch.arange(170) if stage == "stage1" else torch.tensor(latent_aligned_pixel_indices(158))
    )
    values = {"viewmats": views, "K": torch.eye(3).repeat(count, 1, 1), key: indices + 70}
    path = tmp_path / "camera.safetensors"
    save_file(values, str(path))
    selected, _ = _load_cameras(path, row, stage=stage)
    assert selected.shape == (50 if stage == "stage1" else 47, 4, 4)
    if stage == "stage1":
        assert float(selected[-1, 0, 3]) == pytest.approx(1.66)
    values[key][-1] += 1
    save_file(values, str(path))
    with pytest.raises(DataContractError, match=r"camera|indices"):
        _load_cameras(path, row, stage=stage)
