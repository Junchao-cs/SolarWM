from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solarwm.backends.wan22.runtime.checkpoint import (
    _assert_optimizer_names_match_module,
    normalize_model_state,
    normalize_optimizer_state,
    optimizer_parameter_names,
)
from solarwm.backends.wan22.runtime.data import CAMERA_CONVENTION
from solarwm.backends.wan22.runtime.preencoded import (
    I2V_A14B_153F_VERSION,
    TI2V_5B_81F_VERSION,
    TI2V_5B_153F_VERSION,
    TI2V_5B_720P_153F_VERSION,
    WINDOW_HASH_NAMESPACE_81F,
    WINDOW_HASH_NAMESPACE_153F,
    _is_skippable_camera_guard_error,
    decode_preencoded_sample,
    expected_81f_window_start,
    expected_153f_window_start,
)
from solarwm.config import load_config
from solarwm.data.archive import RawSample
from solarwm.data.sampling import SamplePlan
from solarwm.errors import BackendContractError, DataContractError

ROOT = Path(__file__).resolve().parents[3]
TI2V_153F_CONFIG = ROOT / "configs/examples/wan22_ti2v_5b/train_stage0p5_fm_153f.yaml"
TI2V_81F_CONFIG = ROOT / "configs/examples/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml"
A14B_153F_CONFIG = ROOT / "configs/examples/wan22_i2v_a14b/train_stage0p5_fm_153f.yaml"


def test_only_camera_magnitude_rejections_are_skippable() -> None:
    assert _is_skippable_camera_guard_error(
        DataContractError("relative translation exceeds configured guard")
    )
    assert _is_skippable_camera_guard_error(
        DataContractError("camera matrix exceeds configured absolute guard")
    )
    assert not _is_skippable_camera_guard_error(
        DataContractError("preencoding tensors digest differs")
    )


def test_frozen_153f_window_assignment_covers_six_and_single_window_sources() -> None:
    long_form = {
        "source_sample_id": "SOLARWM/abot/example",
        "source_dataset": "abot",
        "window_count": 6,
        "window_index": 0,
    }
    starts = []
    for index in range(6):
        long_form["window_index"] = index
        starts.append(expected_153f_window_start(long_form, 960))
    assert starts == [0, 161, 323, 484, 646, 807]

    ordinary = {
        "source_sample_id": "SOLARWM/dl3dv/example",
        "source_dataset": "dl3dv",
        "window_count": 1,
        "window_index": 0,
    }
    assert expected_153f_window_start(ordinary, 200) == expected_153f_window_start(ordinary, 200)


def test_frozen_81f_window_assignment_matches_release_geometry() -> None:
    preencoding = {"window_count": 12, "window_index": 0}
    starts = []
    for index in range(12):
        preencoding["window_index"] = index
        starts.append(expected_81f_window_start(preencoding, 960))
    assert starts == [0, 80, 160, 240, 320, 400, 479, 559, 639, 719, 799, 879]


def _ti2v_81f_preencoded_config() -> dict:
    config = load_config(TI2V_81F_CONFIG).mutable_copy()
    config["data"].update(
        encoding="preencoded",
        preencode_schema="solarwm.wan22_ti2v_5b.480p.81f.v1",
        preencode_window_assignment="materialized_index_v1",
        random_start=False,
    )
    return config


def _preencoded_sample(config: dict) -> RawSample:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    data = config["data"]
    family = config["model"]["family"]
    a14b = family == "wan22_i2v_a14b"
    pixel_frames = int(data["pixel_frames"])
    is_81f = pixel_frames == 81
    is_720p = data["preencode_schema"] == "solarwm.wan22_ti2v_5b.720p.153f.v1"
    version = (
        I2V_A14B_153F_VERSION
        if a14b
        else TI2V_5B_81F_VERSION
        if is_81f
        else TI2V_5B_720P_153F_VERSION
        if is_720p
        else TI2V_5B_153F_VERSION
    )
    preencoding = {
        "version": version,
        "dtype": "bfloat16",
        "pixel_frames": pixel_frames,
        "latent_frames": int(data["latent_frames"]),
        "latent_shape": list(data["latent_shape"]),
        "prompt_shape": [512, 4096],
        "target_h": int(data["height"]),
        "target_w": int(data["width"]),
        "output_fps": 16.0,
        "source_fps": 16.0,
        "source_sample_id": "SOLARWM/dl3dv/example",
        "source_dataset": "dl3dv",
        "window_count": 2 if is_81f else 1,
        "window_index": 0,
        "window_hash_namespace": (
            WINDOW_HASH_NAMESPACE_81F if is_81f else WINDOW_HASH_NAMESPACE_153F
        ),
    }
    start = (
        expected_81f_window_start(preencoding, 200)
        if is_81f
        else expected_153f_window_start(preencoding, 200)
    )
    preencoding.update(
        start_frame=start,
        source_frame_first=start,
        source_frame_last=start + pixel_frames - 1,
    )
    camera = np.repeat(np.eye(4, dtype=np.float32)[None], 200, axis=0)
    camera_payload = io.BytesIO()
    np.savez(camera_payload, c2w=camera)
    tensors = {
        "latents": torch.zeros(tuple(data["latent_shape"]), dtype=torch.bfloat16),
        "prompt_embeds": torch.zeros((512, 4096), dtype=torch.bfloat16),
    }
    if a14b:
        tensors["i2v_y"] = torch.zeros(tuple(data["i2v_y_shape"]), dtype=torch.bfloat16)
    tensor_payload = safetensors.save(tensors)
    sample_id = f"SOLARWM/dl3dv/example/latent-{pixel_frames}f-w00"
    key = f"dl3dv__example__latent{pixel_frames}f_w00"
    plan = SamplePlan(
        sample_id=sample_id,
        key=key,
        shard="shards/part-000000.tar",
        row_ordinal=0,
        repeat_ordinal=0,
        epoch=0,
        start_frame=start,
        source_frame_indices=tuple(range(start, start + pixel_frames)),
        reader_rank=0,
        worker_id=0,
    )
    manifest = {
        "sample_id": sample_id,
        "key": key,
        "video": {"num_frames": 200, "fps": 16.0},
        "prompt": {"text": "frozen caption"},
        "preencoding": preencoding,
        "camera": {
            "convention": CAMERA_CONVENTION,
            "dtype": "float32",
            "finite": True,
            "magnitude_audit_seconds": 10.0,
            "max_camera_abs": 20.0,
            "max_rel_translation": 20.0,
            "shape": [200, 4, 4],
        },
    }
    return RawSample(
        plan=plan,
        index_values={},
        caption="frozen caption",
        scene="scene",
        manifest=manifest,
        members={
            "preencoded_member": tensor_payload,
            "camera_member": camera_payload.getvalue(),
        },
    )


def test_153f_tensor_artifact_decodes_without_online_codecs() -> None:
    torch = pytest.importorskip("torch")
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    decoded = decode_preencoded_sample(_preencoded_sample(config), config)
    assert tuple(decoded.latents.shape) == (39, 48, 30, 54)
    assert tuple(decoded.prompt_embeds.shape) == (512, 4096)
    assert decoded.latents.dtype == torch.bfloat16
    assert tuple(decoded.camera["viewmats"].shape) == (39 * 405, 4, 4)
    assert tuple(decoded.camera["K"].shape) == (39 * 405, 3, 3)


def test_81f_tensor_artifact_decodes_without_online_codecs() -> None:
    torch = pytest.importorskip("torch")
    config = _ti2v_81f_preencoded_config()
    decoded = decode_preencoded_sample(_preencoded_sample(config), config)
    assert tuple(decoded.latents.shape) == (21, 48, 30, 54)
    assert tuple(decoded.prompt_embeds.shape) == (512, 4096)
    assert decoded.latents.dtype == torch.bfloat16
    assert tuple(decoded.camera["viewmats"].shape) == (21 * 405, 4, 4)
    assert tuple(decoded.camera["K"].shape) == (21 * 405, 3, 3)


def test_published_153f_config_uses_materialized_index_windows() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    assert config["data"]["preencode_window_assignment"] == "materialized_index_v1"


def test_materialized_index_accepts_bound_window_after_public_identity_rename() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    sample = _preencoded_sample(config)
    deterministic = int(sample.manifest["preencoding"]["start_frame"])
    materialized = (deterministic + 1) % (200 - 153 + 1)
    decode_preencoded_sample(_replace_window(sample, start=materialized), config)


def test_deterministic_window_mode_still_rejects_non_hash_assignment() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    config["data"]["preencode_window_assignment"] = "deterministic_hash_v1"
    sample = _preencoded_sample(config)
    deterministic = int(sample.manifest["preencoding"]["start_frame"])
    materialized = (deterministic + 1) % (200 - 153 + 1)
    with pytest.raises(DataContractError, match="deterministic assignment"):
        decode_preencoded_sample(_replace_window(sample, start=materialized), config)


def test_720p_153f_tensor_artifact_decodes() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    config["model"]["frame_sequence_length"] = 880
    config["data"].update(
        preencode_schema="solarwm.wan22_ti2v_5b.720p.153f.v1",
        latent_shape=[39, 48, 44, 80],
        height=704,
        width=1280,
    )
    decoded = decode_preencoded_sample(_preencoded_sample(config), config)
    assert tuple(decoded.latents.shape) == (39, 48, 44, 80)
    assert tuple(decoded.camera["viewmats"].shape) == (39 * 880, 4, 4)


def test_preencoded_artifact_accepts_configured_version_alias() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    sample = _replace_preencoding(_preencoded_sample(config), version="materialized-v0")
    config["data"]["preencode_version_aliases"] = ["materialized-v0"]
    decode_preencoded_sample(sample, config)


def test_preencoded_artifact_accepts_configured_window_namespace_alias() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    sample = _preencoded_sample(config)
    preencoding = dict(sample.manifest["preencoding"])
    preencoding["window_hash_namespace"] = "materialized-window-v0"
    start = expected_153f_window_start(preencoding, 200)
    preencoding.update(
        start_frame=start,
        source_frame_first=start,
        source_frame_last=start + 152,
    )
    manifest = dict(sample.manifest)
    manifest["preencoding"] = preencoding
    aliased = RawSample(
        plan=SamplePlan(
            sample_id=sample.plan.sample_id,
            key=sample.plan.key,
            shard=sample.plan.shard,
            row_ordinal=sample.plan.row_ordinal,
            repeat_ordinal=sample.plan.repeat_ordinal,
            epoch=sample.plan.epoch,
            start_frame=start,
            source_frame_indices=tuple(range(start, start + 153)),
            reader_rank=sample.plan.reader_rank,
            worker_id=sample.plan.worker_id,
        ),
        index_values=sample.index_values,
        caption=sample.caption,
        scene=sample.scene,
        manifest=manifest,
        members=sample.members,
    )
    config["data"]["preencode_window_namespace_aliases"] = ["materialized-window-v0"]
    decode_preencoded_sample(aliased, config)


def test_a14b_153f_artifact_requires_and_decodes_official_i2v_y() -> None:
    torch = pytest.importorskip("torch")
    config = load_config(A14B_153F_CONFIG).mutable_copy()
    decoded = decode_preencoded_sample(_preencoded_sample(config), config)
    assert tuple(decoded.latents.shape) == (39, 16, 60, 104)
    assert decoded.i2v_y is not None
    assert tuple(decoded.i2v_y.shape) == (39, 20, 60, 104)
    assert decoded.i2v_y.dtype == torch.bfloat16
    assert tuple(decoded.camera["viewmats"].shape) == (39 * 1560, 4, 4)


def test_preencoded_artifact_rejects_version_drift() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    sample = _preencoded_sample(config)
    manifest = dict(sample.manifest)
    manifest["preencoding"] = dict(manifest["preencoding"], version="wrong-v2")
    with pytest.raises(Exception, match=r"preencoding\.version"):
        decode_preencoded_sample(
            RawSample(
                plan=sample.plan,
                index_values=sample.index_values,
                caption=sample.caption,
                scene=sample.scene,
                manifest=manifest,
                members=sample.members,
            ),
            config,
        )


def _replace_preencoding(sample: RawSample, **updates: object) -> RawSample:
    manifest = dict(sample.manifest)
    manifest["preencoding"] = dict(manifest["preencoding"], **updates)
    return RawSample(
        plan=sample.plan,
        index_values=sample.index_values,
        caption=sample.caption,
        scene=sample.scene,
        manifest=manifest,
        members=sample.members,
    )


def _replace_window(sample: RawSample, *, start: int) -> RawSample:
    preencoding = dict(
        sample.manifest["preencoding"],
        start_frame=start,
        source_frame_first=start,
        source_frame_last=start + 152,
    )
    manifest = dict(sample.manifest, preencoding=preencoding)
    plan = SamplePlan(
        sample_id=sample.plan.sample_id,
        key=sample.plan.key,
        shard=sample.plan.shard,
        row_ordinal=sample.plan.row_ordinal,
        repeat_ordinal=sample.plan.repeat_ordinal,
        epoch=sample.plan.epoch,
        start_frame=start,
        source_frame_indices=tuple(range(start, start + 153)),
        reader_rank=sample.plan.reader_rank,
        worker_id=sample.plan.worker_id,
    )
    return RawSample(
        plan=plan,
        index_values=sample.index_values,
        caption=sample.caption,
        scene=sample.scene,
        manifest=manifest,
        members=sample.members,
    )


def test_materialized_window_rejects_source_identity_drift() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    sample = _replace_preencoding(
        _preencoded_sample(config),
        source_sample_id="SOLARWM/dl3dv/different",
    )
    with pytest.raises(DataContractError, match="source_sample_id differs"):
        decode_preencoded_sample(sample, config)


def test_preencoded_reader_treats_tensor_digest_as_offline_metadata() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    sample = _preencoded_sample(config)
    payload = sample.members["preencoded_member"]
    sample = _replace_preencoding(
        sample,
        tensors_digest="0" * 64,
        encoder_contract_digest="0" * 64,
    )
    decode_preencoded_sample(sample, config)

    corrupted = RawSample(
        plan=sample.plan,
        index_values=sample.index_values,
        caption=sample.caption,
        scene=sample.scene,
        manifest=sample.manifest,
        members={**sample.members, "preencoded_member": payload + b"corrupt"},
    )
    with pytest.raises(DataContractError, match="cannot decode preencoded Wan tensors"):
        decode_preencoded_sample(corrupted, config)

    without_contract = _replace_preencoding(sample, encoder_contract_digest=None)
    decode_preencoded_sample(without_contract, config)


def test_current_preencode_schema_uses_readable_tensor_contract() -> None:
    config = load_config(TI2V_153F_CONFIG).mutable_copy()
    sample = _replace_preencoding(
        _preencoded_sample(config),
        version=config["data"]["preencode_schema"],
        encoder_contract_digest=None,
    )
    decode_preencoded_sample(sample, config)


def test_preencoded_dataloader_materializes_the_index_before_worker_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import preencoded
    from solarwm.data.index import IndexRow

    rows = (
        IndexRow.from_mapping(
            0,
            {
                "sample_id": "sample",
                "key": "sample",
                "shard": "shards/part-000000.tar",
                "num_frames": 81,
                "fps": 16.0,
                "start_frame": 0,
            },
        ),
    )
    calls = {"read": 0}

    def read_once(_path: object) -> tuple[object, ...]:
        calls["read"] += 1
        return rows

    monkeypatch.setattr(preencoded, "resolve_index_path", lambda *_args: object())
    monkeypatch.setattr(preencoded, "read_index", read_once)
    monkeypatch.setattr(preencoded, "_rows_with_fixed_starts", tuple)
    monkeypatch.setattr(preencoded, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(preencoded, "TarShardReader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(preencoded, "RawSampleReader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(preencoded, "build_shard_prefetcher", lambda *_args, **_kwargs: None)
    loader = preencoded.build_preencoded_dataloader(
        {
            "data": {
                "num_workers": 1,
                "train_index": "index.jsonl.gz",
                "pixel_frames": 81,
                "fps": 16.0,
                "seed": 42,
                "shuffle_buffer": 1,
                "partition_mode": "global_occurrence",
                "transport": {"root": "/read-only-data"},
            },
            "train": {"micro_batch_size": 1},
        },
        SimpleNamespace(
            dp_rank=0,
            dp_world_size=1,
            node_id=0,
            node_count=1,
            local_dp_rank=0,
            local_dp_world_size=1,
            local_rank=0,
        ),
    )
    assert loader.rows == rows
    assert calls == {"read": 1}


def test_stateful_preencoded_reader_keeps_multiworker_prefetch_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import preencoded
    from solarwm.data.index import IndexRow

    torch = pytest.importorskip("torch")
    rows = tuple(
        IndexRow.from_mapping(
            ordinal,
            {
                "sample_id": f"sample-{ordinal}",
                "key": f"sample-{ordinal}",
                "shard": f"shards/part-{ordinal // 2:06d}.tar",
                "num_frames": 81,
                "fps": 16.0,
                "start_frame": 0,
            },
        )
        for ordinal in range(12)
    )

    class Shards:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Shards:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    class Reader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def materialize(plan: SamplePlan) -> SamplePlan:
            return plan

    monkeypatch.setattr(preencoded, "resolve_index_path", lambda *_args: "train.jsonl")
    monkeypatch.setattr(preencoded, "read_index", lambda *_args: rows)
    monkeypatch.setattr(preencoded, "_rows_with_fixed_starts", tuple)
    monkeypatch.setattr(preencoded, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(preencoded, "TarShardReader", Shards)
    monkeypatch.setattr(preencoded, "RawSampleReader", Reader)
    monkeypatch.setattr(preencoded, "build_shard_prefetcher", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(preencoded, "decode_preencoded_sample", lambda plan, _config: plan)
    monkeypatch.setattr(
        preencoded,
        "collate_preencoded_samples",
        lambda plans: {
            "sample_ids": tuple(plan.sample_id for plan in plans),
            "start_frames": tuple(plan.start_frame for plan in plans),
        },
    )
    config = {
        "data": {
            "num_workers": 2,
            "prefetch_factor": 3,
            "train_index": "train.jsonl",
            "pixel_frames": 81,
            "fps": 16.0,
            "seed": 42,
            "shuffle_buffer": 2,
            "partition_mode": "global_occurrence",
            "transport": {"root": "/read-only-data"},
        },
        "train": {"micro_batch_size": 1},
    }
    topology = SimpleNamespace(
        dp_rank=0,
        dp_world_size=1,
        node_id=0,
        node_count=1,
        local_dp_rank=0,
        local_dp_world_size=1,
        local_rank=0,
    )

    uninterrupted = preencoded.build_preencoded_dataloader(config, topology)
    cpu_rng = torch.random.get_rng_state().clone()
    uninterrupted._start_loader()
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    for _ in range(7):
        next(uninterrupted)
    assert uninterrupted._loader.num_workers == 2
    assert uninterrupted._loader.prefetch_factor == 3
    state = uninterrupted.checkpoint_state_dict()
    assert uninterrupted._iterator is None
    expected = [next(uninterrupted) for _ in range(13)]
    uninterrupted.close()

    resumed = preencoded.build_preencoded_dataloader(config, topology)
    resumed.load_state_dict(state)
    actual = [next(resumed) for _ in range(13)]
    resumed.close()

    assert actual == expected


def test_stateful_preencoded_reader_fails_after_an_entire_timed_out_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import preencoded
    from solarwm.data.index import IndexRow
    from solarwm.data.transport import GCSDownloadTimeout

    pytest.importorskip("torch")
    rows = tuple(
        IndexRow.from_mapping(
            ordinal,
            {
                "sample_id": f"sample-{ordinal}",
                "key": f"sample-{ordinal}",
                "shard": "shards/part-000000.tar",
                "num_frames": 81,
                "fps": 16.0,
                "start_frame": 0,
            },
        )
        for ordinal in range(3)
    )

    class Shards:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Shards:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    class Reader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def materialize(plan: SamplePlan) -> SamplePlan:
            return plan

    monkeypatch.setattr(preencoded, "resolve_index_path", lambda *_args: "train.jsonl")
    monkeypatch.setattr(preencoded, "read_index", lambda *_args: rows)
    monkeypatch.setattr(preencoded, "_rows_with_fixed_starts", tuple)
    monkeypatch.setattr(preencoded, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(preencoded, "TarShardReader", Shards)
    monkeypatch.setattr(preencoded, "RawSampleReader", Reader)
    monkeypatch.setattr(preencoded, "build_shard_prefetcher", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        preencoded,
        "decode_preencoded_sample",
        lambda *_args: (_ for _ in ()).throw(GCSDownloadTimeout("injected timeout")),
    )
    config = {
        "data": {
            "num_workers": 0,
            "train_index": "train.jsonl",
            "pixel_frames": 81,
            "fps": 16.0,
            "seed": 42,
            "shuffle_buffer": 1,
            "partition_mode": "global_occurrence",
            "transport": {"root": "/read-only-data"},
        },
        "train": {"micro_batch_size": 1},
    }
    topology = SimpleNamespace(
        dp_rank=0,
        dp_world_size=1,
        node_id=0,
        node_count=1,
        local_dp_rank=0,
        local_dp_world_size=1,
        local_rank=0,
    )
    reader = preencoded.build_preencoded_dataloader(config, topology)

    with pytest.raises(RuntimeError, match="emitted no samples in epoch 0"):
        next(reader)
    reader.close()


def test_preencoded_stream_prepares_the_planned_shard_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import preencoded
    from solarwm.data.index import IndexRow

    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample",
            "key": "sample",
            "shard": "shards/part-000000.tar",
            "num_frames": 81,
            "fps": 16.0,
            "start_frame": 0,
        },
    )
    events: list[tuple[str, SamplePlan | None]] = []

    class Prefetcher:
        @staticmethod
        def prepare(plan: SamplePlan) -> None:
            events.append(("prepare", plan))

        @staticmethod
        def close() -> None:
            events.append(("close", None))

    class Shards:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Shards:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    class Reader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def materialize(plan: SamplePlan) -> SamplePlan:
            events.append(("materialize", plan))
            return plan

    monkeypatch.setattr(preencoded, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        preencoded,
        "build_shard_prefetcher",
        lambda *_args, **_kwargs: Prefetcher(),
    )
    monkeypatch.setattr(preencoded, "TarShardReader", Shards)
    monkeypatch.setattr(preencoded, "RawSampleReader", Reader)
    monkeypatch.setattr(preencoded, "decode_preencoded_sample", lambda sample, _config: sample)
    monkeypatch.setattr(preencoded, "collate_preencoded_samples", tuple)
    topology = SimpleNamespace(
        dp_rank=0,
        dp_world_size=1,
        node_id=0,
        node_count=1,
        local_dp_rank=0,
        local_dp_world_size=1,
        local_rank=0,
    )
    stream = preencoded.iter_preencoded_batches(
        {
            "data": {
                "transport": {"kind": "gcs", "root": "gs://dataset"},
                "pixel_frames": 81,
                "fps": 16.0,
                "seed": 42,
                "shuffle_buffer": 1,
                "partition_mode": "node_shard",
                "gcs_prefetch_shards": 32,
            },
            "train": {"micro_batch_size": 1},
        },
        topology,
        rows=(row,),
    )
    assert next(stream) == (events[0][1],)
    stream.close()
    assert [name for name, _ in events] == ["prepare", "materialize", "close"]
    assert events[0][1] == events[1][1]


def test_model_and_optimizer_names_normalize_as_one_contract() -> None:
    model = normalize_model_state(
        {"model.block.weight": object(), "model.head.bias": object()},
        field="generator",
    )
    assert tuple(model) == ("block.weight", "head.bias")
    optimizer = normalize_optimizer_state(
        {
            "state": {
                "model.block.weight": {"step": 10},
                "model.head.bias": {"step": 10},
            },
            "param_groups": [
                {
                    "lr": 5.0e-5,
                    "params": ["model.block.weight", "model.head.bias"],
                }
            ],
        }
    )
    assert tuple(optimizer["state"]) == ("block.weight", "head.bias")
    assert optimizer["param_groups"][0]["params"] == ["block.weight", "head.bias"]
    assert optimizer_parameter_names(optimizer) == ("block.weight", "head.bias")


def test_optimizer_name_normalization_rejects_mixed_prefixes() -> None:
    with pytest.raises(BackendContractError, match="mixes model-prefixed"):
        normalize_optimizer_state(
            {
                "state": {"model.block.weight": {}, "head.bias": {}},
                "param_groups": [{"params": ["model.block.weight", "head.bias"]}],
            }
        )


def test_optimizer_names_must_match_runtime_model_without_receipt() -> None:
    class Module:
        @staticmethod
        def named_parameters() -> list[tuple[str, object]]:
            return [
                (
                    "block._fsdp_wrapped_module._checkpoint_wrapped_module.weight",
                    object(),
                ),
                ("head.bias", object()),
            ]

    state = {
        "state": {
            "model.block.weight": {"step": 10},
            "model.head.bias": {"step": 10},
        },
        "param_groups": [
            {
                "lr": 5.0e-5,
                "params": ["model.block.weight", "model.head.bias"],
            }
        ],
    }
    _assert_optimizer_names_match_module(state, Module())

    reordered = {
        **state,
        "param_groups": [
            {
                "lr": 5.0e-5,
                "params": ["model.head.bias", "model.block.weight"],
            }
        ],
    }
    with pytest.raises(BackendContractError, match="differ from the runtime model"):
        _assert_optimizer_names_match_module(reordered, Module())


def test_optimizer_names_must_be_explicit_without_receipt() -> None:
    class Module:
        @staticmethod
        def named_parameters() -> list[tuple[str, object]]:
            return [("block.weight", object())]

    with pytest.raises(BackendContractError, match="must use explicit names"):
        _assert_optimizer_names_match_module(
            {"state": {0: {"step": 10}}, "param_groups": [{"params": [0]}]},
            Module(),
        )
