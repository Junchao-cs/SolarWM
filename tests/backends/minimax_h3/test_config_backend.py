from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from solar.backends import load_backend
from solar.backends.minimax_h3.config import validate_h3_config
from solar.config import load_config
from solar.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "configs/solar_wm/minimax_h3/stage0p5-158f-lora384-sp2.yaml"
PREENCODE = ROOT / "configs/solar_wm/minimax_h3/preencode-158f.yaml"
INFER = ROOT / "configs/solar_wm/minimax_h3/infer-158f-lora384-sp2.yaml"


def _config() -> dict[str, object]:
    return load_config(EXAMPLE).mutable_copy()


def test_example_resolves_to_exact_stable_contract() -> None:
    resolved = load_config(EXAMPLE)
    contract = validate_h3_config(resolved.values)
    assert contract.stage == "stage0p5"
    assert contract.pixel_frames == 158
    assert contract.encoded_latents == 47
    assert contract.sequence_parallel_size == 2
    assert contract.adapter_rank == 384
    assert contract.camera_translation_transform == "logd4"
    assert resolved.values["data"]["gcs_prefetch_shards"] == 32


@pytest.mark.parametrize(
    "example",
    (
        "stage0p5-158f-lora384-sp2.yaml",
        "stage1-158f-lora384-w6-sp2.yaml",
        "stage2-158f-lora384-w6-sp4.yaml",
    ),
)
def test_h3_training_examples_set_shuffle_buffer(example: str) -> None:
    config = load_config(EXAMPLE.parent / example)
    assert config.values["data"]["shuffle_buffer"] == 32768


def test_h3_rejects_an_invalid_shard_prefetch_depth() -> None:
    config = _config()
    config["data"]["gcs_prefetch_shards"] = -1
    with pytest.raises(ConfigurationError, match="gcs_prefetch_shards"):
        validate_h3_config(config)


def test_h3_validation_sample_count_is_required_and_positive() -> None:
    missing = _config()
    del missing["validation"]["sample_count"]
    with pytest.raises(ConfigurationError, match=r"validation\.sample_count"):
        validate_h3_config(missing)

    invalid = _config()
    invalid["validation"]["sample_count"] = 0
    with pytest.raises(ConfigurationError, match=r"validation\.sample_count"):
        validate_h3_config(invalid)


def test_h3_training_smoke_validation_can_be_disabled() -> None:
    config = _config()
    assert config["validation"]["smoke_step"] == 0
    validate_h3_config(config)
    config["validation"]["smoke_step"] = -1
    with pytest.raises(ConfigurationError, match="non-negative"):
        validate_h3_config(config)


def test_h3_training_is_preencoded_only() -> None:
    config = _config()
    config["data"]["input_mode"] = "raw_online"
    with pytest.raises(ConfigurationError, match=r"data\.input_mode"):
        validate_h3_config(config)


def test_removed_h3_checkpoint_digest_is_rejected() -> None:
    config = _config()
    model = config["model"]
    assert isinstance(model, dict)
    model["checkpoint_digest"] = "ignored"
    with pytest.raises(ConfigurationError, match="removed content-digest fields"):
        validate_h3_config(config)


def test_example_uses_the_public_release_or_publishable_placeholders() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    for forbidden in ("/private/workspace", "/scratch", "s3://", "${"):
        assert forbidden not in text
    assert "/path/to/SolarWM-Data/releases-v1" in text
    loaded = yaml.safe_load(text)

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and (
            "path" in key or "root" in key or "index" in key or key == "output_dir"
        ):
            assert value.startswith("/path/to/") or not value.startswith("/")

    visit(loaded)


def test_validator_rejects_camera_and_batch_drift() -> None:
    config = _config()
    model = config["model"]
    assert isinstance(model, dict)
    model["camera_prope_head_dim_start"] = 95
    with pytest.raises(ConfigurationError, match="head_dim_start"):
        validate_h3_config(config)

    # The world size is a deployment choice, so changing it alone is not the error --
    # leaving global_batch_size behind is. That identity is what keeps a resized run
    # honest about the batch it actually takes.
    config = _config()
    distributed = config["distributed"]
    assert isinstance(distributed, dict)
    distributed["world_size"] = 128
    with pytest.raises(ConfigurationError, match="global batch mismatch"):
        validate_h3_config(config)

    # ... and the same change WITH a matching batch is accepted.
    config = _config()
    distributed = config["distributed"]
    assert isinstance(distributed, dict)
    train = config["train"]
    assert isinstance(train, dict)
    distributed["world_size"] = 128
    train["global_batch_size"] = 64
    validate_h3_config(config)

    # SP=2 is architectural and stays enforced.
    config = _config()
    distributed = config["distributed"]
    assert isinstance(distributed, dict)
    distributed["sequence_parallel_size"] = 4
    with pytest.raises(ConfigurationError, match="sequence_parallel_size=2"):
        validate_h3_config(config)


def test_backend_runtime_entries_delegate_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    backend = load_backend("minimax_h3")
    backend.validate_config(config)
    from solar.backends.minimax_h3 import preencode_runner, runtime

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        runtime,
        "run_training",
        lambda value: calls.append(("train", value)) or 11,
    )
    monkeypatch.setattr(
        runtime,
        "run_inference",
        lambda value: calls.append(("infer", value)) or 12,
    )
    monkeypatch.setattr(
        preencode_runner,
        "run_preencode",
        lambda value: calls.append(("preencode", value)) or 13,
    )
    assert backend.train(config) == 11

    infer = load_config(INFER).mutable_copy()
    assert backend.infer(infer) == 12

    preencode = load_config(PREENCODE).mutable_copy()
    assert backend.preencode(preencode) == 13
    assert [name for name, _ in calls] == ["train", "infer", "preencode"]


def test_standalone_inference_example_is_strict_sp2_fixed_plan() -> None:
    config = load_config(INFER).mutable_copy()
    contract = validate_h3_config(config)
    assert contract.action == "infer"
    assert contract.sequence_parallel_size == 2
    assert config["distributed"]["world_size"] == 8
    assert config["distributed"]["sequence_parallel_size"] == 2
    assert config["validation"]["sample_count"] == 16
    assert config["validation"]["selection_seed"] == 42
    assert config["validation"]["noise_seed"] == 42
    assert config["checkpoint"]["resume_from"].startswith("/path/to/")
    broken = copy.deepcopy(config)
    broken["checkpoint"].pop("resume_from")
    with pytest.raises(ConfigurationError, match="resume_from"):
        validate_h3_config(broken)


def test_h3_bucket_transport_requires_local_staged_indexes() -> None:
    config = _config()
    data = config["data"]
    assert isinstance(data, dict)
    data["transport"] = {
        "kind": "gcs",
        "root": "gs://example-bucket",
        "cache_dir": "/path/to/cache",
        "cache_max_gib": 512,
    }
    data.pop("index_root")
    with pytest.raises(ConfigurationError, match="index_root"):
        validate_h3_config(config)
    data["index_root"] = "/path/to/staged-controls"
    validate_h3_config(config)


def test_lazy_plugin_import_does_not_require_torch() -> None:
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('H3 contract plugin imported torch')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from solar.backends import load_backend
backend = load_backend('minimax_h3')
assert backend.family == 'minimax_h3'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
