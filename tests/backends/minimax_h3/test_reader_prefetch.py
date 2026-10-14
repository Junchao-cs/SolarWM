from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from solarwm.backends.minimax_h3.artifacts import (
    H3CameraFilterError,
    H3PlanMultiplexer,
    H3PreencodedStream,
    h3_encoder_contract,
)
from solarwm.data import CanonicalSampler, IndexRow, ReaderIdentity, SamplingConfig, ShardPrefetcher
from solarwm.runtime import Topology


def rows():
    return tuple(
        IndexRow.from_mapping(
            i,
            {
                "sample_id": f"sample-{i}",
                "key": f"sample-{i}",
                "shard": f"shards/{i}.tar",
                "num_frames": 158,
                "start_frame": 0,
            },
        )
        for i in range(16)
    )


def test_cached_h3_demand_bypasses_a_busy_prefetch_executor(tmp_path):
    records = rows()
    sampler = CanonicalSampler(
        records,
        SamplingConfig(seed=42, pixel_frames=158, random_start=False),
        ReaderIdentity(rank=0, world_size=1),
    )
    plans = tuple(sampler.iter_epoch(0))
    started, release = Event(), Event()
    cached = tmp_path / "cached.tar"
    cached.write_bytes(b"cached")

    class Resolver:
        def resolve(self, row):
            if row.shard == plans[0].shard:
                started.set()
                assert release.wait(10)
            return cached

    resolver = Resolver()
    prefetcher = ShardPrefetcher(records, resolver, max_workers=1)
    stream = object.__new__(H3PreencodedStream)
    stream.rows = records
    stream.plan = SimpleNamespace(
        next_plan=lambda: plans[1],
        peek_plans=lambda count: (plans[2],),
        num_workers=1,
    )
    stream._prefetch_depth = 2
    stream._shard_prefetcher = prefetcher
    stream.read_artifact = lambda row, plan: (resolver.resolve(row), plan)
    consumer = ThreadPoolExecutor(max_workers=1)
    try:
        prefetcher.schedule_plans((plans[0],))
        assert started.wait(2)
        result = consumer.submit(stream._next_once).result(timeout=2)
        assert result == (cached, plans[1])
        assert not release.is_set()
    finally:
        release.set()
        consumer.shutdown(wait=True)
        prefetcher.close()


@pytest.mark.parametrize("rank", range(8))
def test_each_logical_h3_reader_prefetches_once_per_sp_group(tmp_path, rank):
    contract = h3_encoder_contract(encoder_identity="test")
    encoder = tmp_path / "encoder.json"
    encoder.write_text(json.dumps(contract.as_dict()))
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(dict(row.values)) + "\n" for row in rows()))
    stream = H3PreencodedStream(
        root="gs://test-bucket",
        index=str(index),
        topology=Topology(8, rank, 8, rank, 2),
        seed=42,
        encoder_contract_path=str(encoder),
        cache_dir=str(tmp_path / "cache"),
        gcs_prefetch_shards=4,
    )
    try:
        assert (stream._shard_prefetcher is not None) == (rank % 2 == 0)
        before = stream.state_dict()
        preview = stream.plan.peek_plans(100)
        assert stream.state_dict() == before
        assert len(preview) == len(stream.plan.workers[0].plans)
        assert tuple(stream.plan.next_plan() for _ in preview) == preview
        assert stream.plan.peek_plans(1) == ()
    finally:
        stream.close()


@pytest.mark.parametrize("shuffle_buffer", (4, 32768))
def test_prefetch_preserves_filtered_multiworker_stream_across_resume(shuffle_buffer):
    records = tuple(
        IndexRow.from_mapping(
            i,
            {
                "sample_id": f"sample-{i}",
                "key": f"sample-{i}",
                "shard": f"shards/{i % 17}.tar",
                "num_frames": 174,
            },
        )
        for i in range(257)
    )
    sampling = SamplingConfig(
        seed=42,
        pixel_frames=158,
        random_start=True,
        shuffle_buffer=shuffle_buffer,
        partition_mode="node_shard",
    )
    scheduled = []

    def build(prefetch):
        stream = object.__new__(H3PreencodedStream)
        stream.rows = records
        stream.plan = H3PlanMultiplexer(
            records,
            sampling,
            Topology(8, 0, 8, 0, 2),
            num_workers=8,
            state_schema="test.h3-workers.v1",
        )
        stream.encoder_profile = {"format": "test"}
        stream.fixed_validation = False
        stream._prefetch_depth = 5
        stream._shard_prefetcher = (
            SimpleNamespace(schedule_plans=scheduled.append) if prefetch else None
        )

        def read(row, plan):
            if row.ordinal % 11 == 0:
                raise H3CameraFilterError("filtered")
            return plan

        stream.read_artifact = read
        return stream

    baseline, prefetched = build(False), build(True)
    for _ in range(23):
        assert prefetched.next() == baseline.next()
    state = prefetched.state_dict()
    assert state == baseline.state_dict()
    resumed = build(True)
    resumed.load_state_dict(state)
    for _ in range(150):
        before = resumed.state_dict()
        preview = resumed.plan.peek_plans(40)
        assert resumed.state_dict() == before
        replay = build(False)
        replay.load_state_dict(before)
        assert preview == tuple(replay.plan.next_plan() for _ in preview)
        assert resumed.next() == prefetched.next() == baseline.next()
    assert resumed.state_dict() == prefetched.state_dict() == baseline.state_dict()
    assert max(w["epoch"] for w in resumed.state_dict()["workers"]) > 0
    assert scheduled
    assert all(len(plans) <= 5 for plans in scheduled)
    assert all(len({p.shard for p in plans}) == len(plans) for plans in scheduled)


def test_explicit_frontier_cancels_obsolete_queue_and_deduplicates(tmp_path):
    records = rows()
    plans = tuple(
        CanonicalSampler(
            records,
            SamplingConfig(seed=42, pixel_frames=158, random_start=False),
            ReaderIdentity(rank=0, world_size=1),
        ).iter_epoch(0)
    )
    started, release = Event(), Event()
    calls = []
    cached = tmp_path / "cached.tar"
    cached.write_bytes(b"cached")

    class Resolver:
        def resolve(self, row):
            calls.append(row.shard)
            if row.shard == plans[0].shard:
                started.set()
                assert release.wait(10)
            return cached

    prefetcher = ShardPrefetcher(records, Resolver(), max_workers=1)
    try:
        prefetcher.schedule_plans((plans[0],))
        assert started.wait(2)
        running = prefetcher._futures[plans[0].shard]
        prefetcher.schedule_plans((plans[1], plans[2], plans[1]))
        obsolete = [prefetcher._futures[p.shard] for p in plans[1:3]]
        prefetcher.schedule_plans((plans[3], plans[3]))
        assert all(future.cancelled() for future in obsolete)
        assert prefetcher._futures[plans[0].shard] is running
        assert len(prefetcher._futures) == 2
        release.set()
        for future in tuple(prefetcher._futures.values()):
            future.result(timeout=2)
        prefetcher.schedule_plans(())
        assert not prefetcher._futures
        assert calls == [plans[0].shard, plans[3].shard]
    finally:
        release.set()
        prefetcher.close()
