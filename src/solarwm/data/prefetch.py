"""Asynchronous shard preparation behind an unchanged sample plan."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solarwm.errors import DataContractError

from .index import IndexRow
from .sampling import CanonicalSampler, SamplePlan
from .transport import GCSDownloadTimeout, GCSResolver, ObjectIdentity, ShardResolver


@dataclass(frozen=True)
class CacheWarmupResult:
    shards: int
    bytes: int
    elapsed_seconds: float


def warm_shard_cache(
    rows: Sequence[IndexRow],
    resolver: ShardResolver,
    *,
    max_bytes: int,
    max_workers: int = 4,
    timeout_seconds: float = 7200.0,
    progress: Callable[[int, int], None] | None = None,
) -> CacheWarmupResult:
    """Optionally materialize an explicit working set before training starts.

    This preparation step is independent of models, stages and sampling. The
    caller supplies the desired index (or a node's planned subset); no reader
    cursors or RNG states are created or advanced. Ordinary training does not
    call it. A working set larger than the cache is rejected before any I/O,
    since evicting its beginning while warming its end would not prepare it.
    """

    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
        or isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise DataContractError("cache warmup requires positive workers, capacity and timeout")
    if max_bytes > int(getattr(resolver, "max_bytes", max_bytes)):
        raise DataContractError("cache warmup capacity exceeds the resolver cache capacity")
    unique: dict[str, IndexRow] = {}
    sizes: dict[str, int] = {}
    for row in rows:
        size = ObjectIdentity.from_row(row).size
        if row.shard in sizes and sizes[row.shard] != size:
            raise DataContractError(f"cache warmup has conflicting sizes for {row.shard}")
        unique.setdefault(row.shard, row)
        sizes[row.shard] = size
    total_bytes = sum(sizes.values())
    if total_bytes > max_bytes:
        raise DataContractError(
            f"cache warmup working set needs {total_bytes} bytes, capacity is {max_bytes}; "
            "provide a smaller planned index or a larger cache"
        )
    started = time.monotonic()
    deadline = started + timeout_seconds
    stopped = threading.Event()

    def resolve(row: IndexRow) -> Path:
        while True:
            if stopped.is_set():
                raise GCSDownloadTimeout("cache warmup cancelled")
            if time.monotonic() >= deadline:
                raise GCSDownloadTimeout(f"cache warmup time budget exhausted at {row.shard}")
            try:
                path = (
                    resolver.resolve(row, deadline=deadline)
                    if isinstance(resolver, GCSResolver)
                    else resolver.resolve(row)
                )
                if time.monotonic() >= deadline:
                    raise GCSDownloadTimeout(f"cache warmup time budget exhausted at {row.shard}")
                return path
            except GCSDownloadTimeout:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                stopped.wait(min(1.0, remaining))

    pending: set[Future[Path]] = set()
    iterator = iter(unique.values())
    completed = 0
    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="solarwm-cache-warmup")
    try:
        for _ in range(min(max_workers, len(unique))):
            pending.add(pool.submit(resolve, next(iterator)))
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GCSDownloadTimeout("cache warmup time budget exhausted")
            done, pending = wait(
                pending,
                timeout=remaining,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                raise GCSDownloadTimeout("cache warmup time budget exhausted")
            for future in done:
                future.result()
                completed += 1
                if progress is not None:
                    progress(completed, len(unique))
                row = next(iterator, None)
                if row is not None:
                    pending.add(pool.submit(resolve, row))
    finally:
        stopped.set()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    return CacheWarmupResult(completed, total_bytes, time.monotonic() - started)


def shard_prefetch_depth(data: Mapping[str, Any]) -> int:
    """Return a validated node-level shard lookahead depth."""

    raw = data.get("gcs_prefetch_shards", 0)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise DataContractError("data.gcs_prefetch_shards must be an integer")
    if raw < 0:
        raise DataContractError("data.gcs_prefetch_shards must be non-negative")
    return raw


class ShardPrefetcher:
    """Resolve future shards without selecting or modifying sample plans."""

    def __init__(
        self,
        rows: Sequence[IndexRow],
        resolver: ShardResolver,
        *,
        max_workers: int,
        sampler: CanonicalSampler | None = None,
        lookahead_shards: int = 0,
    ) -> None:
        if max_workers < 1:
            raise DataContractError("shard prefetch workers must be positive")
        if lookahead_shards < 0:
            raise DataContractError("shard lookahead must be non-negative")
        if lookahead_shards and sampler is None:
            raise DataContractError("node shard lookahead requires a sampler")
        self.rows = tuple(rows)
        self.resolver = resolver
        self.sampler = sampler
        self.lookahead_shards = int(lookahead_shards)
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="solarwm-shard-prefetch",
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future[Path]] = {}
        self._epoch_rows: dict[int, tuple[IndexRow, ...]] = {}
        self._epoch_positions: dict[int, dict[str, int]] = {}
        self._scheduled_until: dict[int, int] = {}
        self._closed = False

    def _row(self, plan: SamplePlan) -> IndexRow:
        if not 0 <= plan.row_ordinal < len(self.rows):
            raise DataContractError("sample plan row ordinal is invalid")
        row = self.rows[plan.row_ordinal]
        if (plan.sample_id, plan.key, plan.shard) != (row.sample_id, row.key, row.shard):
            raise DataContractError("sample plan identity drift during shard prefetch")
        return row

    def _schedule_row(self, row: IndexRow) -> Future[Path]:
        with self._lock:
            if self._closed:
                raise DataContractError("shard prefetcher is closed")
            future = self._futures.get(row.shard)
            if future is not None and future.done():
                reusable = False
                if not future.cancelled():
                    with suppress(Exception):
                        reusable = future.result().is_file()
                if not reusable:
                    self._futures.pop(row.shard, None)
                    future = None
            if future is None:
                future = self._executor.submit(self.resolver.resolve, row)
                self._futures[row.shard] = future
            return future

    def _node_shard_order(
        self,
        epoch: int,
    ) -> tuple[tuple[IndexRow, ...], dict[str, int]]:
        rows = self._epoch_rows.get(epoch)
        positions = self._epoch_positions.get(epoch)
        if rows is None or positions is None:
            if self.sampler is None:
                raise DataContractError("node shard lookahead requires a sampler")
            groups = self.sampler.node_shard_groups(epoch)
            rows = tuple(group_rows[0] for _, group_rows in groups)
            positions = {row.shard: index for index, row in enumerate(rows)}
            self._epoch_rows[epoch] = rows
            self._epoch_positions[epoch] = positions
        return rows, positions

    def schedule(self, plan: SamplePlan) -> None:
        """Schedule the current shard and its configured node frontier."""

        row = self._row(plan)
        if not self.lookahead_shards:
            self._schedule_row(row)
            return
        ordered_rows, positions = self._node_shard_order(plan.epoch)
        try:
            start = positions[plan.shard]
        except KeyError as exc:
            raise DataContractError("sample plan shard is outside its node shard order") from exc
        scheduled_until = self._scheduled_until.get(plan.epoch, start)
        stop = min(len(ordered_rows), start + self.lookahead_shards)
        for lookahead_row in ordered_rows[scheduled_until:stop]:
            self._schedule_row(lookahead_row)
        self._scheduled_until[plan.epoch] = max(scheduled_until, stop)
        if not scheduled_until <= start < stop:
            self._schedule_row(row)

    def wait(self, plan: SamplePlan) -> None:
        """Wait for a scheduled current shard, leaving reads authoritative."""

        row = self._row(plan)
        with self._lock:
            future = self._futures.get(row.shard)
        if future is None:
            return
        try:
            # A resolver's deadline starts only when its executor task starts.
            # Include queue time here: a demanded shard must not wait through
            # the transfer budgets of every speculative shard ahead of it.
            future.result(timeout=float(getattr(self.resolver, "max_elapsed_seconds", 300.0)))
        except FutureTimeout as exc:
            if not future.done():
                # Cancel queued work, but retain a running task so a later
                # prepare cannot create a second download for the same shard.
                if future.cancel():
                    with self._lock:
                        if self._futures.get(row.shard) is future:
                            self._futures.pop(row.shard, None)
                raise GCSDownloadTimeout(
                    f"GCS prefetch wait exceeded its time budget for {row.shard}"
                ) from exc
            with self._lock:
                if self._futures.get(row.shard) is future:
                    self._futures.pop(row.shard, None)
            # An underlying resolver TimeoutError follows the ordinary
            # speculative-error path; the authoritative reader still retries.
        except GCSDownloadTimeout:
            with self._lock:
                if self._futures.get(row.shard) is future:
                    self._futures.pop(row.shard, None)
            raise
        except Exception:
            with self._lock:
                if self._futures.get(row.shard) is future:
                    self._futures.pop(row.shard, None)
            # Prefetch is only an I/O hint. The normal reader immediately
            # resolves this same shard and retains its existing retry/error
            # behavior, so a speculative failure cannot change sample flow.

    def prepare(self, plan: SamplePlan) -> None:
        """Schedule and await the shard needed by an existing sample plan."""

        self.schedule(plan)
        self.wait(plan)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._futures.clear()


def build_shard_prefetcher(
    data: Mapping[str, Any],
    *,
    rows: Sequence[IndexRow],
    sampler: CanonicalSampler,
    resolver: ShardResolver,
    node_leader: bool,
    current_shard_workers: int = 0,
) -> ShardPrefetcher | None:
    """Build the GCS-only prefetch path for one reader process."""

    depth = shard_prefetch_depth(data)
    transport = data.get("transport", {})
    if not isinstance(transport, Mapping) or str(transport.get("kind", "")).lower() != "gcs":
        return None
    if str(data.get("partition_mode", "")) == "node_shard":
        if not node_leader or depth == 0:
            return None
        return ShardPrefetcher(
            rows,
            resolver,
            max_workers=min(4, depth),
            sampler=sampler,
            lookahead_shards=depth,
        )
    workers = int(current_shard_workers)
    if workers < 1:
        return None
    return ShardPrefetcher(rows, resolver, max_workers=workers)


__all__ = [
    "CacheWarmupResult",
    "ShardPrefetcher",
    "build_shard_prefetcher",
    "shard_prefetch_depth",
    "warm_shard_cache",
]
