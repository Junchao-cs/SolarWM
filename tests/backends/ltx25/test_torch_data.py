from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from solar.backends.ltx25 import torch_data as ltx_data
from solar.backends.ltx25.artifact import TENSOR_SPECS
from solar.backends.ltx25.geometry import STABLE_GEOMETRY
from solar.backends.ltx25.torch_data import (
    PreencodedBatchSource,
    TorchBatch,
    _camera_magnitude_rejection,
    _validate_tensors,
)
from solar.data.index import IndexRow
from solar.data.sampling import SamplePlan
from solar.errors import BackendContractError
from solar.runtime import Topology


def _plan() -> SamplePlan:
    indices = tuple(range(STABLE_GEOMETRY.pixel_frames))
    return SamplePlan(
        sample_id="sample",
        key="sample",
        shard="sample.tar",
        row_ordinal=0,
        repeat_ordinal=0,
        epoch=0,
        start_frame=0,
        source_frame_indices=indices,
        reader_rank=0,
        worker_id=0,
    )


def _tensors() -> dict[str, torch.Tensor]:
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}
    tensors = {
        name: torch.zeros(spec.shape, dtype=dtypes[spec.dtype])
        for name, spec in TENSOR_SPECS.items()
    }
    tensors["prompt_attention_mask"].fill_(1)
    tensors["relative_w2c"].copy_(torch.eye(4).expand(20, -1, -1))
    tensors["camera_K"].copy_(torch.eye(3).expand(20, -1, -1))
    tensors["source_indices"].copy_(torch.arange(153))
    tensors["camera_source_indices"].copy_(
        tensors["source_indices"].index_select(
            0, torch.tensor(STABLE_GEOMETRY.camera_pixel_indices)
        )
    )
    return tensors


def test_reader_accepts_relative_w2c_inverse_roundoff() -> None:
    tensors = _tensors()
    tensors["relative_w2c"][0, 0, 1] = 5e-9
    _validate_tensors(tensors, plan=_plan())


def test_reader_rejects_nonidentity_relative_w2c_origin() -> None:
    tensors = _tensors()
    tensors["relative_w2c"][0, 0, 3] = 1e-3
    with pytest.raises(BackendContractError, match="row zero must be identity"):
        _validate_tensors(tensors, plan=_plan())


def test_reader_applies_camera_magnitude_guard() -> None:
    tensors = _tensors()
    tensors["relative_w2c"][1, 0, 3] = -21.0
    batch = TorchBatch(
        sample_id="sample",
        start_frame=0,
        plan_fingerprint="fingerprint",
        video_latent=tensors["video_latent"].unsqueeze(0),
        first_frame_latent=tensors["first_frame_latent"].unsqueeze(0),
        video_prompt_embeds=tensors["video_prompt_embeds"].unsqueeze(0),
        prompt_attention_mask=tensors["prompt_attention_mask"].unsqueeze(0),
        relative_w2c=tensors["relative_w2c"].unsqueeze(0),
        camera_k=tensors["camera_K"].unsqueeze(0),
        source_indices=tensors["source_indices"].unsqueeze(0),
        camera_source_indices=tensors["camera_source_indices"].unsqueeze(0),
    )
    rejection = _camera_magnitude_rejection(
        batch,
        source_fps=24.0,
        max_rel_translation=20.0,
        max_camera_abs=20.0,
    )
    assert rejection is not None
    assert rejection.startswith("camera exceeds max_rel_translation")


def test_preencoded_reader_treats_tensor_digest_as_offline_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample",
            "key": "sample",
            "shard": "sample.tar",
            "start_frame": 0,
            "source_frame_indices": list(range(STABLE_GEOMETRY.pixel_frames)),
            "preencoded_member": "sample.ltx25.safetensors",
            "tensor_digest": "0" * 64,
        },
    )
    tensors = {
        name: torch.zeros(1)
        for name in (
            "video_latent",
            "first_frame_latent",
            "video_prompt_embeds",
            "prompt_attention_mask",
            "relative_w2c",
            "camera_K",
            "source_indices",
            "camera_source_indices",
        )
    }

    class Shards:
        @staticmethod
        def read(_: IndexRow, member: str) -> bytes:
            assert member == "sample.ltx25.safetensors"
            return b"serialized tensor payload"

    monkeypatch.setattr(
        ltx_data,
        "_metadata",
        lambda _: {
            "schema": ltx_data.PREENCODE_SCHEMA,
            "version": ltx_data.PREENCODE_VERSION,
            "sample_id": "sample",
        },
    )
    monkeypatch.setattr(ltx_data, "load_safetensors", lambda _: tensors)
    monkeypatch.setattr(ltx_data, "_validate_tensors", lambda *_args, **_kwargs: None)

    batch = ltx_data.load_preencoded_row(row, shards=Shards())
    assert batch.sample_id == "sample"


def test_preencoded_recipe_row_uses_the_tensor_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample",
            "key": "sample",
            "shard": "sample.tar",
            "preencoded_member": "sample.ltx25.safetensors",
        },
    )
    tensors = _tensors()
    tensors["source_indices"].copy_(torch.arange(7, 160))
    tensors["camera_source_indices"].copy_(
        tensors["source_indices"].index_select(
            0, torch.tensor(STABLE_GEOMETRY.camera_pixel_indices)
        )
    )

    class Shards:
        @staticmethod
        def read(_: IndexRow, member: str) -> bytes:
            assert member == "sample.ltx25.safetensors"
            return b"serialized tensor payload"

    monkeypatch.setattr(
        ltx_data,
        "_metadata",
        lambda _: {
            "schema": ltx_data.PREENCODE_SCHEMA,
            "version": ltx_data.PREENCODE_VERSION,
            "sample_id": "sample",
        },
    )
    monkeypatch.setattr(ltx_data, "load_safetensors", lambda _: tensors)
    batch = ltx_data.load_preencoded_row(row, shards=Shards())
    assert batch.start_frame == 7
    assert batch.source_indices[0].tolist() == list(range(7, 160))


def _source_rows(count: int = 8) -> tuple[IndexRow, ...]:
    indices = list(range(STABLE_GEOMETRY.pixel_frames))
    return tuple(
        IndexRow.from_mapping(
            ordinal,
            {
                "sample_id": f"sample-{ordinal}",
                "key": f"sample-{ordinal}",
                "shard": f"shard-{ordinal}.tar",
                "num_frames": STABLE_GEOMETRY.pixel_frames,
                "start_frame": 0,
                "source_frame_indices": indices,
            },
        )
        for ordinal in range(count)
    )


def _source_config(*, transport_kind: str = "local") -> dict[str, Any]:
    transport = (
        {"kind": "local", "root": "/dataset"}
        if transport_kind == "local"
        else {
            "kind": "gcs",
            "root": "gs://dataset",
            "cache_dir": "/cache",
            "cache_max_gib": 1,
        }
    )
    return {
        "model": {"checkpoint_path": "/models/ltx25-transformer.safetensors"},
        "distributed": {"sequence_parallel_size": 1},
        "data": {
            "index": "index.jsonl",
            "transport": transport,
            "num_workers": 2,
            "prefetch_factor": 2,
            "shuffle_buffer": 1,
            "partition_mode": "global_occurrence",
        },
    }


def _batch(plan: SamplePlan) -> TorchBatch:
    empty = torch.empty(0)
    return TorchBatch(
        sample_id=plan.sample_id,
        start_frame=plan.start_frame,
        plan_fingerprint=f"plan-{plan.sample_id}",
        video_latent=empty,
        first_frame_latent=empty,
        video_prompt_embeds=empty,
        prompt_attention_mask=empty,
        relative_w2c=empty,
        camera_k=empty,
        source_indices=empty,
        camera_source_indices=empty,
    )


def _patch_source_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    *,
    resolver: Any = None,
) -> None:
    class DummyShards:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(ltx_data, "read_index", lambda _: _source_rows())
    monkeypatch.setattr(ltx_data, "resolve_index_path", lambda *_: Path("/dataset/index.jsonl"))
    monkeypatch.setattr(
        ltx_data,
        "verified_resolver_from_config",
        lambda _: resolver if resolver is not None else object(),
    )
    monkeypatch.setattr(ltx_data, "TarShardReader", DummyShards)
    monkeypatch.setattr(ltx_data, "load_preencoded_row", loader)
    monkeypatch.setattr(
        ltx_data.Topology,
        "from_environ",
        classmethod(lambda cls, _: Topology(1, 0, 1, 0, sp_size=1)),
    )


def test_preencoded_source_prefetches_concurrently_without_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    release = threading.Event()
    parallel = threading.Event()
    active = 0
    worker_readers: dict[int, set[int]] = {}

    def load(_: IndexRow, *, plan: SamplePlan, shards: Any) -> TorchBatch:
        nonlocal active
        with lock:
            worker_readers.setdefault(plan.worker_id, set()).add(id(shards))
            active += 1
            if active >= 2:
                parallel.set()
        assert release.wait(2)
        with lock:
            active -= 1
        return _batch(plan)

    _patch_source_dependencies(monkeypatch, load)
    source = PreencodedBatchSource(_source_config())
    try:
        assert parallel.wait(1)
        release.set()
        assert [source.next().sample_id for _ in range(4)] == [
            "sample-0",
            "sample-1",
            "sample-2",
            "sample-3",
        ]
        assert set(worker_readers) == {0, 1}
        assert all(len(readers) == 1 for readers in worker_readers.values())
        assert len(set.union(*worker_readers.values())) == 2
    finally:
        release.set()
        source.close()


def test_preencoded_source_decouples_materializers_from_sampler_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load(_: IndexRow, *, plan: SamplePlan, shards: Any) -> TorchBatch:
        del shards
        return _batch(plan)

    _patch_source_dependencies(monkeypatch, load)
    config = _source_config()
    config["data"]["reader_threads"] = 6
    source = PreencodedBatchSource(config)
    try:
        assert len(source.streams) == 2
        assert source._executor is not None
        assert source._executor._max_workers == 6
    finally:
        source.close()


def test_preencoded_source_defaults_to_full_pending_materialization_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load(_: IndexRow, *, plan: SamplePlan, shards: Any) -> TorchBatch:
        del shards
        return _batch(plan)

    _patch_source_dependencies(monkeypatch, load)
    source = PreencodedBatchSource(_source_config())
    try:
        assert source._executor is not None
        assert source._executor._max_workers == 32
        assert source._tar_cache_size == 4
    finally:
        source.close()


def test_preencoded_source_rejects_fewer_materializers_than_sampler_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_source_dependencies(monkeypatch, lambda *_args, **_kwargs: _batch(_plan()))
    config = _source_config()
    config["data"]["reader_threads"] = 1
    with pytest.raises(BackendContractError, match="reader_threads"):
        PreencodedBatchSource(config)


def test_preencoded_source_prefetches_gcs_shards_beyond_active_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    four_shards = threading.Event()
    lock = threading.Lock()
    resolved: set[str] = set()

    class Resolver:
        def resolve(self, row: IndexRow) -> Path:
            with lock:
                resolved.add(row.shard)
                if len(resolved) >= 4:
                    four_shards.set()
            return Path("/cache") / row.shard

    def load(_: IndexRow, *, plan: SamplePlan, shards: Any) -> TorchBatch:
        del shards
        assert release.wait(2)
        return _batch(plan)

    _patch_source_dependencies(monkeypatch, load, resolver=Resolver())
    source = PreencodedBatchSource(_source_config(transport_kind="gcs"))
    try:
        assert four_shards.wait(1)
    finally:
        release.set()
        source.close()


def test_node_shard_gcs_lookahead_is_deep_and_leader_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    six_shards = threading.Event()
    lock = threading.Lock()
    resolved: set[str] = set()

    class Resolver:
        def resolve(self, row: IndexRow) -> Path:
            with lock:
                resolved.add(row.shard)
                if len(resolved) >= 6:
                    six_shards.set()
            return Path("/cache") / row.shard

    def load(_: IndexRow, *, plan: SamplePlan, shards: Any) -> TorchBatch:
        del shards
        assert release.wait(2)
        return _batch(plan)

    _patch_source_dependencies(monkeypatch, load, resolver=Resolver())
    config = _source_config(transport_kind="gcs")
    config["data"].update(
        {
            "partition_mode": "node_shard",
            "gcs_prefetch_shards": 6,
        }
    )
    leader = PreencodedBatchSource(config)
    try:
        assert six_shards.wait(1)
    finally:
        release.set()
        leader.close()

    resolved.clear()
    release.clear()
    monkeypatch.setattr(
        ltx_data.Topology,
        "from_environ",
        classmethod(lambda cls, _: Topology(2, 1, 2, 1, sp_size=1)),
    )
    follower = PreencodedBatchSource(config)
    try:
        assert follower._shard_prefetcher is None
        assert not resolved
    finally:
        release.set()
        follower.close()


def test_node_shard_lookahead_follows_worker_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class Prefetcher:
        def schedule(self, plan: SamplePlan) -> None:
            calls.append(plan.worker_id)

        def wait(self, _plan: SamplePlan) -> None:
            pass

        def close(self) -> None:
            pass

    def load(_: IndexRow, *, plan: SamplePlan, shards: Any) -> TorchBatch:
        del shards
        return _batch(plan)

    class Resolver:
        def resolve(self, row: IndexRow) -> Path:
            return Path("/cache") / row.shard

    _patch_source_dependencies(monkeypatch, load, resolver=Resolver())
    monkeypatch.setattr(
        ltx_data,
        "build_shard_prefetcher",
        lambda *_args, **_kwargs: Prefetcher(),
    )
    config = _source_config(transport_kind="gcs")
    config["data"].update(
        {
            "partition_mode": "node_shard",
            "gcs_prefetch_shards": 6,
        }
    )
    source = PreencodedBatchSource(config)
    try:
        assert calls
        assert set(calls) == {0}
    finally:
        source.close()


def test_node_shard_lookahead_fills_gaps_between_worker_zero_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Resolver:
        def resolve(self, row: IndexRow) -> Path:
            return Path("/cache") / row.shard

    def load(_: IndexRow, *, plan: SamplePlan, shards: Any) -> TorchBatch:
        del shards
        return _batch(plan)

    _patch_source_dependencies(monkeypatch, load, resolver=Resolver())
    monkeypatch.setattr(ltx_data, "read_index", lambda _: _source_rows(32))
    monkeypatch.setattr(
        ltx_data.Topology,
        "from_environ",
        classmethod(lambda cls, _: Topology(4, 0, 4, 0, sp_size=1)),
    )
    config = _source_config(transport_kind="gcs")
    config["data"].update(
        {
            "partition_mode": "node_shard",
            "gcs_prefetch_shards": 2,
        }
    )
    source = PreencodedBatchSource(config)
    try:
        assert source._shard_prefetcher is not None
        assert set(source._shard_prefetcher._futures) == {
            f"shard-{index}.tar" for index in range(10)
        }
    finally:
        source.close()


def test_preencoded_source_resume_preserves_prefetched_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Resolver:
        def resolve(self, row: IndexRow) -> Path:
            return Path("/cache") / row.shard

    def load(_: IndexRow, *, plan: SamplePlan, shards: Any) -> TorchBatch:
        del shards
        return _batch(plan)

    _patch_source_dependencies(monkeypatch, load, resolver=Resolver())
    config = _source_config(transport_kind="gcs")
    first = PreencodedBatchSource(config)
    second: PreencodedBatchSource | None = None
    try:
        assert [first.next().sample_id for _ in range(3)] == [
            "sample-0",
            "sample-1",
            "sample-2",
        ]
        state = first.state_dict()
        expected = [first.next().sample_id for _ in range(5)]
        second = PreencodedBatchSource(config)
        second.load_state_dict(state)
        assert [second.next().sample_id for _ in range(5)] == expected
    finally:
        first.close()
        if second is not None:
            second.close()
