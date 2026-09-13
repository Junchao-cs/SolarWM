from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from solarwm.backends.ltx25 import torch_raw as ltx_raw
from solarwm.backends.ltx25.inference import build_inference_plan
from solarwm.backends.ltx25.torch_data import (
    IndexedPreencodedSource,
    VerifiedShardResolver,
    verified_resolver_from_config,
)
from solarwm.backends.ltx25.torch_raw import RawIndexedStream, RawInferenceSource
from solarwm.data.index import IndexRow
from solarwm.data.transport import GCSResolver, LocalResolver
from solarwm.errors import BackendContractError
from solarwm.runtime import Topology


def _raw_row() -> dict[str, object]:
    return {
        "sample_id": "sample-7",
        "key": "source-key-7",
        "shard": "dataset/shards/part-00007.tar",
        "shard_generation": "1712345678901234",
        "shard_size": 123456,
        "shard_md5_b64": "AAAAAAAAAAAAAAAAAAAAAA==",
        "shard_digest": "a" * 64,
        "video_member": "source-key-7.mp4",
        "camera_member": "source-key-7.camera.npz",
        "intrinsics_member": "source-key-7.intrinsics.npy",
        "start_frame": 7,
        "source_frame_indices": list(range(7, 160)),
        "num_frames": 200,
        "fps": 24.0,
        "caption": "A fixed validation caption.",
    }


def test_raw_inference_cases_do_not_require_a_rank_owned_reader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    row = _raw_row()
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    for name in ("WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    source = RawInferenceSource(
        {
            "data": {
                "transport": {
                    "kind": "local",
                    "root": str(tmp_path / "payloads"),
                },
                "index_root": str(tmp_path),
                "index": index.name,
            }
        },
        object(),  # The codec is not touched while fixed cases are enumerated.
    )
    try:
        case = source.case_for_row(
            source.rows[0],
            slot=0,
            plan=build_inference_plan(),
            camera_translation_transform="linear",
        )
        assert case.sample_id == row["sample_id"]
        assert case.start_frame == 7
        assert case.camera_fingerprint == source.case_fingerprint("sample-7")
        assert case.metadata["camera_translation_transform"] == "linear"
        assert len(case.camera_fingerprint) == 64
    finally:
        source.close()


def test_raw_recipe_test_rows_use_the_deterministic_first_window(
    tmp_path: Path,
) -> None:
    row = _raw_row()
    del row["start_frame"]
    del row["source_frame_indices"]
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    config = {
        "data": {
            "transport": {"kind": "local", "root": str(tmp_path / "payloads")},
            "index_root": str(tmp_path),
            "index": index.name,
        }
    }

    source = RawInferenceSource(config, object())
    try:
        normalized = source.rows[0]
        assert normalized.values["start_frame"] == 0
        assert normalized.values["source_frame_indices"] == list(range(153))
        assert (
            source.case_for_row(
                normalized,
                slot=0,
                plan=build_inference_plan(),
                camera_translation_transform="linear",
            ).start_frame
            == 0
        )
    finally:
        source.close()

    stream = RawIndexedStream(config, logical_dp=True, initialize_readers=False)
    try:
        assert stream.rows[0].values["start_frame"] == 0
        assert stream.rows[0].values["source_frame_indices"] == list(range(153))
    finally:
        stream.close()


def test_raw_training_rejects_a_partial_source_window(tmp_path: Path) -> None:
    row = _raw_row()
    del row["source_frame_indices"]
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    config = {
        "data": {
            "transport": {"kind": "local", "root": str(tmp_path / "payloads")},
            "index_root": str(tmp_path),
            "index": index.name,
        }
    }
    with pytest.raises(BackendContractError, match="partial source window"):
        RawIndexedStream(config, logical_dp=True, initialize_readers=False)


def test_raw_online_training_uses_the_shared_shard_prefetcher(monkeypatch) -> None:
    row = IndexRow.from_mapping(0, _raw_row())
    calls: list[str] = []

    class Shards:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    class Reader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class Prefetcher:
        def close(self) -> None:
            calls.append("close")

    def build(*_args, **kwargs):
        assert kwargs["rows"] == (row,)
        assert kwargs["node_leader"] is True
        calls.append("build")
        return Prefetcher()

    monkeypatch.setattr(ltx_raw, "read_index", lambda _path: (row,))
    monkeypatch.setattr(ltx_raw, "resolve_index_path", lambda *_args: Path("/index.jsonl"))
    monkeypatch.setattr(ltx_raw, "verified_resolver_from_config", lambda _data: object())
    monkeypatch.setattr(ltx_raw, "TarShardReader", Shards)
    monkeypatch.setattr(ltx_raw, "RawSampleReader", Reader)
    monkeypatch.setattr(ltx_raw, "build_shard_prefetcher", build)
    monkeypatch.setattr(
        ltx_raw.Topology,
        "from_environ",
        classmethod(lambda cls, _sp: Topology(1, 0, 1, 0, sp_size=1)),
    )
    stream = RawIndexedStream(
        {
            "data": {
                "index": "index.jsonl",
                "transport": {"kind": "gcs", "root": "gs://dataset"},
                "num_workers": 1,
                "seed": 42,
                "shuffle_buffer": 1,
                "partition_mode": "node_shard",
                "gcs_prefetch_shards": 32,
            },
            "distributed": {"sequence_parallel_size": 1},
        },
        logical_dp=True,
    )
    stream.close()
    assert calls == ["build", "close"]


def test_raw_online_reader_resume_preserves_the_next_plans(monkeypatch) -> None:
    rows = []
    for ordinal in range(12):
        values = _raw_row()
        values.update(
            {
                "sample_id": f"sample-{ordinal}",
                "key": f"key-{ordinal}",
                "shard": f"dataset/shards/part-{ordinal:05d}.tar",
            }
        )
        rows.append(IndexRow.from_mapping(ordinal, values))

    class Shards:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(ltx_raw, "read_index", lambda _path: tuple(rows))
    monkeypatch.setattr(ltx_raw, "resolve_index_path", lambda *_args: Path("/index.jsonl"))
    monkeypatch.setattr(ltx_raw, "verified_resolver_from_config", lambda _data: object())
    monkeypatch.setattr(ltx_raw, "TarShardReader", Shards)
    monkeypatch.setattr(ltx_raw, "RawSampleReader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(ltx_raw, "build_shard_prefetcher", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ltx_raw.RawIndexedStream, "_materialize", lambda _self, plan: plan)
    monkeypatch.setattr(
        ltx_raw.Topology,
        "from_environ",
        classmethod(lambda cls, _sp: Topology(1, 0, 1, 0, sp_size=1)),
    )
    config = {
        "data": {
            "index": "index.jsonl",
            "transport": {"kind": "gcs", "root": "gs://dataset"},
            "num_workers": 2,
            "seed": 42,
            "shuffle_buffer": 4,
            "partition_mode": "global_occurrence",
        },
        "distributed": {"sequence_parallel_size": 1},
    }

    uninterrupted = RawIndexedStream(config, logical_dp=True)
    resumed = RawIndexedStream(config, logical_dp=True)
    try:
        for _ in range(7):
            uninterrupted.next()
        resumed.load_state_dict(uninterrupted.state_dict())
        expected = [uninterrupted.next() for _ in range(20)]
        observed = [resumed.next() for _ in range(20)]
        assert observed == expected
    finally:
        uninterrupted.close()
        resumed.close()


def test_raw_case_fingerprint_binds_source_controls(tmp_path: Path) -> None:
    first = _raw_row()
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(first) + "\n")
    config = {
        "data": {
            "transport": {
                "kind": "local",
                "root": str(tmp_path / "payloads"),
            },
            "index_root": str(tmp_path),
            "index": index.name,
        }
    }
    source = RawInferenceSource(config, object())
    try:
        original = source.case_fingerprint("sample-7")
    finally:
        source.close()

    changed = _raw_row()
    changed["shard_generation"] = "1712345678901235"
    index.write_text(json.dumps(changed) + "\n")
    source = RawInferenceSource(config, object())
    try:
        assert source.case_fingerprint("sample-7") != original
    finally:
        source.close()


def test_local_shard_uses_declared_size_without_content_binding(tmp_path: Path) -> None:
    payload = b"immutable indexed shard"
    shard = tmp_path / "dataset" / "part-00000.tar"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(payload)
    values = {
        "sample_id": "sample",
        "key": "key",
        "shard": "dataset/part-00000.tar",
        "shard_generation": "digest:" + hashlib.blake2s(payload).hexdigest(),
        "shard_size": len(payload),
        "shard_md5_b64": base64.b64encode(
            hashlib.md5(payload, usedforsecurity=False).digest()
        ).decode(),
        "shard_digest": hashlib.blake2s(payload).hexdigest(),
    }
    row = IndexRow.from_mapping(0, values)
    assert VerifiedShardResolver(LocalResolver(tmp_path)).resolve(row) == shard

    shard.write_bytes(b"alternate indexed shard")
    assert VerifiedShardResolver(LocalResolver(tmp_path)).resolve(row) == shard


def test_random_access_preencoded_source_uses_the_verified_shard_resolver(
    tmp_path: Path,
) -> None:
    row = {
        "sample_id": "sample",
        "key": "key",
        "shard": "dataset/part-00000.tar",
        "start_frame": 0,
        "source_frame_indices": list(range(153)),
    }
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    source = IndexedPreencodedSource(
        {
            "data": {
                "transport": {"kind": "local", "root": str(tmp_path)},
                "index_root": str(tmp_path),
                "index": index.name,
            }
        }
    )
    try:
        assert isinstance(source.shards.resolver, VerifiedShardResolver)
    finally:
        source.close()


def test_ltx_gcs_transport_builds_the_shared_bounded_cache(tmp_path: Path) -> None:
    resolver = verified_resolver_from_config(
        {
            "transport": {
                "kind": "gcs",
                "root": "gs://example-bucket/ltx25",
                "cache_dir": str(tmp_path / "cache"),
                "cache_max_gib": 0.25,
            }
        }
    )

    assert isinstance(resolver.delegate, GCSResolver)
    assert resolver.delegate.root == "gs://example-bucket/ltx25"
    assert resolver.delegate.max_bytes == int(0.25 * 1024**3)
