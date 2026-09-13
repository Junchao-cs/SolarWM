from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from solarwm.data.index import IndexRow
from solarwm.data.prefetch import ShardPrefetcher, build_shard_prefetcher, warm_shard_cache
from solarwm.data.sampling import CanonicalSampler, ReaderIdentity, SamplePlan, SamplingConfig
from solarwm.data.transport import GCSDownloadTimeout
from solarwm.errors import DataContractError


def _rows(count: int = 12) -> tuple[IndexRow, ...]:
    return tuple(
        IndexRow(
            ordinal=index,
            sample_id=f"sample-{index}",
            key=f"sample-{index}",
            shard=f"raw/shard-{index:03d}.tar",
            epoch_repeats=1,
            values={"num_frames": 96, "fps": 16.0},
        )
        for index in range(count)
    )


def _sampler(rows: tuple[IndexRow, ...]) -> CanonicalSampler:
    return CanonicalSampler(
        rows,
        SamplingConfig(
            seed=42,
            pixel_frames=81,
            random_start=True,
            shuffle_buffer=4,
            partition_mode="node_shard",
        ),
        ReaderIdentity(rank=0, world_size=1),
    )


class _Resolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, row: IndexRow) -> Path:
        self.calls.append(row.shard)
        return Path("/cache") / row.shard


def test_node_shard_prefetch_does_not_change_sample_plans() -> None:
    rows = _rows()
    expected = tuple(_sampler(rows).iter_epoch(0))
    sampler = _sampler(rows)
    resolver = _Resolver()
    prefetcher = ShardPrefetcher(
        rows,
        resolver,
        max_workers=4,
        sampler=sampler,
        lookahead_shards=4,
    )

    try:
        observed = []
        for plan in sampler.iter_epoch(0):
            prefetcher.prepare(plan)
            observed.append(plan)
    finally:
        prefetcher.close()

    assert tuple(observed) == expected


def test_node_shard_prefetch_schedules_the_configured_frontier() -> None:
    rows = _rows()
    sampler = _sampler(rows)
    resolver = _Resolver()
    prefetcher = ShardPrefetcher(
        rows,
        resolver,
        max_workers=4,
        sampler=sampler,
        lookahead_shards=6,
    )
    first = next(sampler.iter_epoch(0))

    try:
        prefetcher.schedule(first)
        ordered = [group[0].shard for _, group in sampler.node_shard_groups(0)]
        start = ordered.index(first.shard)
        expected = set(ordered[start : start + 6])
        assert len(prefetcher._futures) == 6
        assert set(prefetcher._futures) == expected
    finally:
        prefetcher.close()


@pytest.mark.parametrize(
    ("transport_kind", "partition_mode", "node_leader", "expected"),
    (
        ("gcs", "node_shard", True, True),
        ("gcs", "node_shard", False, False),
        ("local", "node_shard", True, False),
        ("gcs", "global_occurrence", True, False),
    ),
)
def test_node_shard_prefetch_is_gcs_leader_only(
    transport_kind: str,
    partition_mode: str,
    node_leader: bool,
    expected: bool,
) -> None:
    rows = _rows()
    prefetcher = build_shard_prefetcher(
        {
            "transport": {"kind": transport_kind},
            "partition_mode": partition_mode,
            "gcs_prefetch_shards": 32,
        },
        rows=rows,
        sampler=_sampler(rows),
        resolver=_Resolver(),
        node_leader=node_leader,
    )
    try:
        assert (prefetcher is not None) is expected
    finally:
        if prefetcher is not None:
            prefetcher.close()


def test_failed_prefetch_leaves_the_authoritative_read_and_plan_unchanged() -> None:
    rows = _rows(1)
    sampler = _sampler(rows)
    plan = next(sampler.iter_epoch(0))

    class RetryResolver:
        calls = 0

        def resolve(self, _row: IndexRow) -> Path:
            self.calls += 1
            if self.calls == 1:
                raise OSError("temporary read failure")
            return Path("/cache/raw/shard-000.tar")

    resolver = RetryResolver()
    prefetcher = ShardPrefetcher(
        rows,
        resolver,
        max_workers=1,
        sampler=sampler,
        lookahead_shards=1,
    )
    try:
        prefetcher.prepare(plan)
        prefetcher.prepare(plan)
    finally:
        prefetcher.close()

    assert resolver.calls == 2


def test_timed_out_prefetch_is_returned_to_the_sample_skip_boundary() -> None:
    rows = _rows(1)
    sampler = _sampler(rows)
    plan = next(sampler.iter_epoch(0))

    class TimeoutResolver:
        def resolve(self, _row: IndexRow) -> Path:
            raise GCSDownloadTimeout("injected timeout")

    prefetcher = ShardPrefetcher(
        rows,
        TimeoutResolver(),
        max_workers=1,
        sampler=sampler,
        lookahead_shards=1,
    )
    try:
        with pytest.raises(GCSDownloadTimeout, match="injected timeout"):
            prefetcher.prepare(plan)
    finally:
        prefetcher.close()


def test_prefetch_resolves_again_when_cached_file_was_removed(tmp_path: Path) -> None:
    rows = _rows(1)
    sampler = _sampler(rows)
    plan = next(sampler.iter_epoch(0))
    cached = tmp_path / "raw" / "shard-000.tar"

    class CacheResolver:
        calls = 0

        def resolve(self, _row: IndexRow) -> Path:
            self.calls += 1
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(b"shard")
            return cached

    resolver = CacheResolver()
    prefetcher = ShardPrefetcher(
        rows,
        resolver,
        max_workers=1,
        sampler=sampler,
        lookahead_shards=1,
    )
    try:
        prefetcher.prepare(plan)
        assert resolver.calls == 1
        cached.unlink()
        prefetcher.prepare(plan)
    finally:
        prefetcher.close()

    assert resolver.calls == 2
    assert cached.is_file()


def test_prefetch_rejects_plan_identity_drift() -> None:
    rows = _rows(1)
    sampler = _sampler(rows)
    plan = next(sampler.iter_epoch(0))
    drifted = SamplePlan(
        sample_id="different",
        key=plan.key,
        shard=plan.shard,
        row_ordinal=plan.row_ordinal,
        repeat_ordinal=plan.repeat_ordinal,
        epoch=plan.epoch,
        start_frame=plan.start_frame,
        source_frame_indices=plan.source_frame_indices,
        reader_rank=plan.reader_rank,
        worker_id=plan.worker_id,
    )
    prefetcher = ShardPrefetcher(
        rows,
        _Resolver(),
        max_workers=1,
        sampler=sampler,
        lookahead_shards=1,
    )
    try:
        with pytest.raises(DataContractError, match="identity drift"):
            prefetcher.prepare(drifted)
    finally:
        prefetcher.close()


@pytest.mark.parametrize("queued", (False, True))
def test_prefetch_wait_budget_includes_executor_queue(queued: bool) -> None:
    rows = _rows(2)
    plans = {plan.row_ordinal: plan for plan in _sampler(rows).iter_epoch(0)}
    started, release = Event(), Event()

    class SlowResolver:
        max_elapsed_seconds = 0.05
        calls = 0

        def resolve(self, row: IndexRow) -> Path:
            self.calls += 1
            started.set()
            assert release.wait(10), "test failed to release downloader"
            return Path("/cache") / row.shard

    resolver = SlowResolver()
    prefetcher = ShardPrefetcher(rows, resolver, max_workers=1)
    try:
        prefetcher.schedule(plans[0])
        assert started.wait(5)
        plan = plans[1 if queued else 0]
        with pytest.raises(GCSDownloadTimeout, match="prefetch wait"):
            prefetcher.prepare(plan)
        assert resolver.calls == 1
        if queued:
            assert plan.shard not in prefetcher._futures
        else:
            assert plan.shard in prefetcher._futures
            with pytest.raises(GCSDownloadTimeout, match="prefetch wait"):
                prefetcher.prepare(plan)
            assert resolver.calls == 1
    finally:
        release.set()
        prefetcher.close()


def _warm_rows(count: int = 6) -> tuple[IndexRow, ...]:
    return tuple(
        IndexRow.from_mapping(
            index,
            {
                "sample_id": f"sample-{index}",
                "key": f"key-{index}",
                "shard": f"shards/{index}.tar",
                "shard_size": 4,
            },
        )
        for index in range(count)
    )


def test_explicit_cache_warmup_deduplicates_without_sampling(tmp_path: Path) -> None:
    import random

    rows = _warm_rows()
    calls = []
    progress = []

    class Resolver:
        def resolve(self, row: IndexRow) -> Path:
            calls.append(row.shard)
            path = tmp_path / Path(row.shard).name
            path.write_bytes(b"data")
            return path

    rng = random.getstate()
    result = warm_shard_cache(
        (*rows, *rows),
        Resolver(),
        max_bytes=24,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert random.getstate() == rng
    assert result.shards == 6 and result.bytes == 24
    assert sorted(calls) == sorted(row.shard for row in rows)
    assert progress == [(index, 6) for index in range(1, 7)]


def test_cache_warmup_rejects_oversized_working_set_before_resolving() -> None:
    resolver = _Resolver()
    with pytest.raises(DataContractError, match="working set needs"):
        warm_shard_cache(_warm_rows(), resolver, max_bytes=23)
    assert resolver.calls == []


def test_cache_warmup_never_exceeds_four_inflight_downloads(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    release, all_started = Event(), Event()
    lock = Lock()
    active = peak = 0

    class Resolver:
        def resolve(self, row: IndexRow) -> Path:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 4:
                    all_started.set()
            assert release.wait(10)
            path = tmp_path / Path(row.shard).name
            path.write_bytes(b"data")
            with lock:
                active -= 1
            return path

    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(warm_shard_cache, _warm_rows(12), Resolver(), max_bytes=48)
        try:
            assert all_started.wait(5)
            assert peak == 4
        finally:
            release.set()
        assert future.result(timeout=5).shards == 12
    assert peak == 4


def test_cache_warmup_timeout_does_not_silently_report_ready() -> None:
    class Resolver:
        def resolve(self, row: IndexRow) -> Path:
            raise GCSDownloadTimeout("injected storage outage")

    with pytest.raises(GCSDownloadTimeout):
        warm_shard_cache(_warm_rows(), Resolver(), max_bytes=24, timeout_seconds=0.01)


def test_cache_warmup_rejects_a_resolver_result_after_the_deadline() -> None:
    import time

    class Resolver:
        def resolve(self, row: IndexRow) -> Path:
            del row
            time.sleep(0.02)
            return Path("/cache/late.tar")

    with pytest.raises(GCSDownloadTimeout, match="time budget exhausted"):
        warm_shard_cache(_warm_rows(1), Resolver(), max_bytes=4, timeout_seconds=0.01)


def test_cache_warmup_recovers_from_a_transient_timeout(tmp_path: Path) -> None:
    class Resolver:
        calls = 0

        def resolve(self, row: IndexRow) -> Path:
            self.calls += 1
            if self.calls == 1:
                raise GCSDownloadTimeout("injected transient timeout")
            path = tmp_path / Path(row.shard).name
            path.write_bytes(b"data")
            return path

    resolver = Resolver()
    result = warm_shard_cache(_warm_rows(1), resolver, max_bytes=4, timeout_seconds=5)
    assert result.shards == 1
    assert resolver.calls == 2
