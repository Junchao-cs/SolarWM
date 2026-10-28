from __future__ import annotations

import copy
from pathlib import Path

import pytest

from solar.backends.ltx25 import create_backend, official, torch_data
from solar.backends.ltx25.config import validate_ltx25_config
from solar.backends.ltx25.official import OfficialLTX25Provider
from solar.config import load_config
from solar.errors import BackendContractError, ConfigurationError

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "configs" / "solar_wm" / "ltx25"


@pytest.mark.parametrize(
    "name,action,input_mode",
    [
        ("stage0p5-train-153f-lora384-sp2.yaml", "train", "preencoded"),
        ("stage0p5-infer-153f.yaml", "infer", "preencoded"),
        ("stage0p5-train-online-153f-unpaired.yaml", "train", "raw_online"),
        ("preencode-153f.yaml", "preencode", "raw"),
    ],
)
def test_public_examples_resolve_and_validate(name: str, action: str, input_mode: str) -> None:
    config = load_config(EXAMPLES / name).values
    contract = validate_ltx25_config(config)
    assert contract.action == action
    assert contract.input_mode == input_mode
    assert "checkpoint_digest" not in config["model"]
    codec = config["model"]["codec"]
    assert "video_vae_digest" not in codec
    assert "gemma4_digest" not in codec


@pytest.mark.parametrize(
    "section,field",
    [
        ("model", "checkpoint_digest"),
        ("codec", "video_vae_digest"),
        ("codec", "gemma4_digest"),
    ],
)
def test_removed_asset_digest_fields_are_rejected(section: str, field: str) -> None:
    values = load_config(EXAMPLES / "stage0p5-train-153f-lora384-sp2.yaml").mutable_copy()
    target = values["model"] if section == "model" else values["model"]["codec"]
    target[field] = "ignored"
    with pytest.raises(ConfigurationError, match="removed content-digest fields"):
        validate_ltx25_config(values)


def test_training_example_has_explicit_topology() -> None:
    config = load_config(EXAMPLES / "stage0p5-train-153f-lora384-sp2.yaml").values
    contract = validate_ltx25_config(config)
    assert contract.sequence_parallel_size == 2
    assert config["train"]["gradient_accumulation_steps"] == 2
    assert config["train"]["fsdp"]["activation_checkpointed_blocks"] == 48
    assert "root" not in config["data"]
    assert config["data"]["transport"] == {
        "kind": "local",
        "root": "/path/to/SolarWM-Data/releases-v1",
    }
    assert config["data"]["gcs_prefetch_shards"] == 32


def test_ltx_training_examples_share_the_shard_prefetch_contract() -> None:
    for name in (
        "stage0p5-train-153f-lora384-sp2.yaml",
        "stage0p5-train-online-153f-unpaired.yaml",
    ):
        values = load_config(EXAMPLES / name).mutable_copy()
        assert values["data"]["gcs_prefetch_shards"] == 32
        values["data"]["gcs_prefetch_shards"] = -1
        with pytest.raises(ConfigurationError, match="gcs_prefetch_shards"):
            validate_ltx25_config(values)


def test_flat_data_root_is_rejected() -> None:
    values = load_config(EXAMPLES / "stage0p5-infer-153f.yaml").mutable_copy()
    values["data"]["root"] = values["data"]["transport"]["root"]
    del values["data"]["transport"]
    with pytest.raises(ConfigurationError, match=r"data\.transport\.root"):
        validate_ltx25_config(values)


def test_gcs_transport_requires_and_accepts_a_bounded_local_cache() -> None:
    values = load_config(EXAMPLES / "stage0p5-infer-153f.yaml").mutable_copy()
    values["data"]["transport"] = {
        "kind": "gcs",
        "root": "gs://example-bucket/ltx25",
        "cache_dir": "/var/cache/solarwm/ltx25",
        "cache_max_gib": 256,
    }
    assert validate_ltx25_config(values).action == "infer"

    del values["data"]["transport"]["cache_dir"]
    with pytest.raises(ConfigurationError, match="cache_dir"):
        validate_ltx25_config(values)


def test_inference_rejects_nonstandard_adapter_checkpoint_format() -> None:
    values = load_config(EXAMPLES / "stage0p5-infer-153f.yaml").mutable_copy()
    values["model"]["adapter_checkpoint_format"] = "unsupported_v1"
    with pytest.raises(ConfigurationError, match="inference_transaction_v1"):
        validate_ltx25_config(values)


def test_geometry_drift_is_rejected() -> None:
    values = load_config(EXAMPLES / "stage0p5-train-153f-lora384-sp2.yaml").mutable_copy()
    values["data"]["pixel_frames"] = 145
    with pytest.raises(ConfigurationError, match="pixel_frames"):
        validate_ltx25_config(values)


def test_training_validation_sample_count_is_required() -> None:
    values = load_config(EXAMPLES / "stage0p5-train-153f-lora384-sp2.yaml").mutable_copy()
    del values["validation"]["sample_count"]
    with pytest.raises(ConfigurationError, match="sample_count"):
        validate_ltx25_config(values)


def test_online_behavior_status_is_fixed() -> None:
    values = load_config(EXAMPLES / "stage0p5-train-online-153f-unpaired.yaml").mutable_copy()
    values["data"]["behavior_status"] = "unsupported"
    with pytest.raises(ConfigurationError, match="unpaired"):
        validate_ltx25_config(values)


def test_embedded_raw_online_route_is_implemented_but_remains_unpaired() -> None:
    values = load_config(EXAMPLES / "stage0p5-train-online-153f-unpaired.yaml").values
    route = next(
        check
        for check in OfficialLTX25Provider().readiness(values, "train")
        if check.name == "provider.route.train.raw_online"
    )
    assert route.status == "pass"
    assert values["data"]["behavior_status"] == "unpaired"


def test_raw_online_and_preencode_examples_share_frozen_reader_and_codec_contract() -> None:
    online = load_config(EXAMPLES / "stage0p5-train-online-153f-unpaired.yaml").values
    offline = load_config(EXAMPLES / "preencode-153f.yaml").values
    for field in ("seed", "num_workers", "shuffle_buffer"):
        assert online["data"][field] == offline["data"][field]
    assert online["data"]["online_codec_protocol"] == offline["preencode"]["codec_protocol"]
    assert (
        online["data"]["behavior_status"] == offline["preencode"]["behavior_status"] == "unpaired"
    )
    assert online["validation"]["passes"] == ["live", "ema"]


def test_hsdp_global_batch_formula_fails_closed() -> None:
    values = load_config(EXAMPLES / "stage0p5-train-153f-lora384-sp2.yaml").mutable_copy()
    values["distributed"]["world_size"] = 8
    with pytest.raises(BackendContractError, match="global batch"):
        validate_ltx25_config(values)


def test_one_node_topology_can_resolve_as_full_shard() -> None:
    values = load_config(EXAMPLES / "stage0p5-train-153f-lora384-sp2.yaml").mutable_copy()
    values["distributed"]["world_size"] = 8
    values["train"]["global_batch_size"] = 8
    values["train"]["fsdp"]["sharding_strategy"] = "FULL_SHARD"
    contract = validate_ltx25_config(values)
    assert contract.global_batch_size == 8


def test_embedded_backend_resolves_and_missing_runtime_assets_fail_closed(
    tmp_path: Path,
) -> None:
    backend = create_backend(family="ltx25_video")
    values = load_config(EXAMPLES / "stage0p5-train-153f-lora384-sp2.yaml").mutable_copy()
    values["runtime"]["output_dir"] = str(tmp_path / "run")
    backend.validate_config(values)
    with pytest.raises(BackendContractError, match="readiness failed"):
        backend.train(values)
    report = tmp_path / "run" / "ltx25-readiness.rank-00000.json"
    assert report.is_file()
    assert (
        '"provider":{"entrypoint":"solar.backends.ltx25.official:create_provider"'
        in report.read_text()
    )
    with pytest.raises(BackendContractError, match="cannot serve"):
        create_backend(family="minimax_h3")


def test_config_copy_remains_plain_data() -> None:
    values = load_config(EXAMPLES / "stage0p5-infer-153f.yaml").values
    assert validate_ltx25_config(copy.deepcopy(values)).action == "infer"


def test_official_inference_reads_the_recipe_test_index_directly(monkeypatch) -> None:
    calls = []
    marker = object()

    def source(config):
        calls.append(config)
        return marker

    monkeypatch.setattr(torch_data, "IndexedPreencodedSource", source)
    config = {"data": {}}
    assert official._inference_data_source(config) is marker
    assert calls == [config]
