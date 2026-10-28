from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from solar.config import load_config
from solar.errors import ConfigurationError
from solar.renderer.config import validate_config
from solar.renderer.training import main

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "solar_renderer"
RECIPES = (
    "control_lora",
    "structure_guided_4step",
    "causal_control_lora",
    "causal_structure_guided_4step",
)


@pytest.mark.parametrize("recipe", RECIPES)
def test_renderer_recipes_are_separate_from_camera_training_routes(recipe):
    config = CONFIGS / f"{recipe}.yaml"
    resolved = load_config(config, route_validator=validate_config)
    assert resolved.values["model"]["family"] == "minimax_h3_renderer"
    with pytest.raises(ConfigurationError):
        load_config(config)


@pytest.mark.parametrize(
    "recipe,override,reason",
    [
        ("control_lora", "model.transformer_subfolder=transformer", "Ref2VA"),
        ("control_lora", "model.adapter.rank=128", "rank-384"),
        ("control_lora", "distributed.world_size=7", "divisible"),
        ("control_lora", "train.global_batch_size=8", "global_batch_size"),
        ("control_lora", "runtime.output_dir=relative/run", "absolute"),
        ("control_lora", "runtime.output_dir=gs://bucket/run", "filesystem"),
        ("control_lora", "validation.smoke_step=1", "inference command"),
        ("control_lora", "train.fsdp.saved_attention_blocks=-1", "saved_attention_blocks"),
        ("control_lora", "train.fsdp.saved_attention_blocks=51", "saved_attention_blocks"),
        ("control_lora", "train.fsdp.saved_attention_blocks=true", "saved_attention_blocks"),
        ("control_lora", "checkpoint.save_every_steps=-1", "save_every_steps"),
        ("control_lora", "distributed.context_parallel_size=2", "context_parallel_size"),
        ("control_lora", "train.optimizer.learning_rate=-0.01", "learning_rate"),
        ("control_lora", "train.optimizer.gradient_clip=0", "gradient_clip"),
        ("control_lora", "train.optimizer.betas=[0.9,1]", "betas"),
        ("control_lora", "train.optimizer.warmup_steps=-1", "warmup_steps"),
        ("causal_control_lora", "checkpoint.initialization.student.weight_source=ema", "live"),
        ("structure_guided_4step", "train.dmd.critic_rank=0", "critic_rank"),
        ("structure_guided_4step", "train.dmd.critic_rank=true", "critic_rank"),
        ("structure_guided_4step", "train.dmd.init_lora_alpha=-1", "init_lora_alpha"),
        ("structure_guided_4step", "train.dmd.mode=pdmd", "adapt_plug or causal"),
        ("structure_guided_4step", "train.dmd.forwards=3", "four evaluations"),
        ("structure_guided_4step", "train.dmd.quality_guidance=1.5", "unsupported DMD"),
        ("structure_guided_4step", "checkpoint.ema.enabled=true", "without EMA"),
        ("causal_structure_guided_4step", "train.dmd.causal_stream.window=5", "W6/C5"),
    ],
)
def test_invalid_recipes_fail_before_loading_a_model(recipe, override, reason):
    with pytest.raises(ConfigurationError, match=reason):
        load_config(CONFIGS / f"{recipe}.yaml", [override], route_validator=validate_config)


@pytest.mark.parametrize(
    "world,sp,shard,reason",
    [
        (8, 4, 8, "one logical sample"),
        (16, 8, None, "one logical sample"),
        (8, 8, 0, "positive divisor"),
        (8, 8, 3, "positive divisor"),
        (8, 8, True, "positive divisor"),
    ],
)
def test_causal_rollout_cannot_share_fsdp_collectives_across_samples(world, sp, shard, reason):
    config = load_config(
        CONFIGS / "causal_structure_guided_4step.yaml", route_validator=validate_config
    ).mutable_copy()
    config["distributed"].update(world_size=world, sequence_parallel_size=sp)
    config["train"]["global_batch_size"] = world // sp
    config["train"]["fsdp"]["frozen_base_shard_size"] = shard
    with pytest.raises(ConfigurationError, match=reason):
        validate_config(config)


@pytest.mark.parametrize("world,sp,shard", [(16, 8, 8), (8, 8, None)])
def test_causal_rollout_accepts_fsdp_groups_with_one_sample(world, sp, shard):
    config = load_config(
        CONFIGS / "causal_structure_guided_4step.yaml", route_validator=validate_config
    ).mutable_copy()
    config["distributed"].update(world_size=world, sequence_parallel_size=sp)
    config["train"]["global_batch_size"] = world // sp
    config["train"]["fsdp"]["frozen_base_shard_size"] = shard
    validate_config(config)


@pytest.mark.parametrize("recipe", RECIPES)
def test_resolve_preserves_overrides_and_never_starts_training(
    recipe, tmp_path, capsys, monkeypatch
):
    output = tmp_path / "unused-training-output"

    def fail_training(config):
        pytest.fail("configuration resolution started training")

    monkeypatch.setattr("solar.renderer.training.run_control_training", fail_training)
    assert (
        main(
            [
                "--config",
                str(CONFIGS / f"{recipe}.yaml"),
                "--set",
                f"runtime.output_dir={output}",
                "--set",
                "model.checkpoint_path=/models/MiniMax-H3",
                "--resolve",
            ]
        )
        == 0
    )
    config = json.loads(capsys.readouterr().out)
    assert config["runtime"]["output_dir"] == str(output)
    assert config["model"]["checkpoint_path"] == "/models/MiniMax-H3"
    assert Path(config["data"]["silence_latents_path"]).is_absolute()
    assert Path(config["data"]["encoder_contract_path"]).is_absolute()
    assert not output.exists()


@pytest.mark.parametrize(
    "module,args",
    [
        ("solar.renderer", ["--help"]),
        ("solar.renderer.prepare_h3", ["--help"]),
        ("solar.renderer.training", ["--help"]),
        (
            "solar.renderer.training",
            [
                "--config",
                str(CONFIGS / "control_lora.yaml"),
                "--resolve",
            ],
        ),
    ],
)
def test_cli_discovery_and_configuration_do_not_import_model_dependencies(module, args, tmp_path):
    script = """
import importlib.abc
import runpy
import sys

class BlockModelDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'diffusers', 'transformers', 'peft'}:
            raise AssertionError('model dependency imported during CLI discovery: ' + fullname)
        return None

sys.meta_path.insert(0, BlockModelDependencies())
sys.argv = [sys.argv[1], *sys.argv[2:]]
runpy.run_module(sys.argv[0], run_name='__main__')
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script, module, *args],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout or json.loads(result.stdout)["action"] == "train"
