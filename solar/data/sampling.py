"""Deterministic occurrence ownership, shuffle, and random starts.

The sampler is deliberately transport-free. A local directory and an object
store therefore produce the same plans for the same index and topology.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

import numpy as np

from solar.errors import DataContractError

from .index import IndexRow

PartitionMode = Literal["global_occurrence", "node_shard"]
ShardPartitionScope = Literal["complete", "collection_relative"]


@dataclass(frozen=True)
class SamplingConfig:
    seed: int
    pixel_frames: int
    random_start: bool = True
    fixed_start_from_index: bool = False
    clip_seconds: float | None = None
    output_fps: float | None = None
    shuffle_buffer: int = 4096
    partition_mode: PartitionMode = "global_occurrence"
    shard_partition_scope: ShardPartitionScope = "complete"

    def __post_init__(self) -> None:
        if self.pixel_frames < 1:
            raise DataContractError("pixel_frames must be positive")
        if self.shuffle_buffer < 1:
            raise DataContractError("shuffle_buffer must be positive")
        if self.shard_partition_scope not in {"complete", "collection_relative"}:
            raise DataContractError("unknown shard_partition_scope")
        if self.random_start and self.fixed_start_from_index:
            raise DataContractError(
                "random_start and fixed_start_from_index are mutually exclusive"
            )
        if (self.clip_seconds is None) != (self.output_fps is None):
            raise DataContractError("clip_seconds and output_fps must be set together")
        if self.clip_seconds is not None:
            expected = round(self.clip_seconds * float(self.output_fps)) + 1
            if expected != self.pixel_frames:
                raise DataContractError(
                    f"clip_seconds/output_fps imply {expected} frames, expected {self.pixel_frames}"
                )


@dataclass(frozen=True)
class ReaderIdentity:
    """One logical DP DataLoader worker, never a raw SP process."""

    rank: int
    world_size: int
    worker_id: int = 0
    num_workers: int = 1
    node_id: int = 0
    node_count: int = 1
    local_rank: int = 0
    local_world_size: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.world_size:
            raise DataContractError("reader rank is outside world_size")
        if not 0 <= self.worker_id < self.num_workers:
            raise DataContractError("worker_id is outside num_workers")
        if not 0 <= self.node_id < self.node_count:
            raise DataContractError("node_id is outside node_count")
        if not 0 <= self.local_rank < self.local_world_size:
            raise DataContractError("local_rank is outside local_world_size")

    @classmethod
    def from_topology(
        cls,
        topology: object,
        *,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> ReaderIdentity:
        """Map raw ranks to logical DP readers so all SP peers are identical."""

        required = (
            "dp_rank",
            "dp_world_size",
            "node_id",
            "node_count",
            "local_dp_rank",
            "local_dp_world_size",
        )
        try:
            values = {name: int(getattr(topology, name)) for name in required}
        except (AttributeError, TypeError, ValueError) as exc:
            raise DataContractError("invalid distributed topology") from exc
        return cls(
            rank=values["dp_rank"],
            world_size=values["dp_world_size"],
            worker_id=worker_id,
            num_workers=num_workers,
            node_id=values["node_id"],
            node_count=values["node_count"],
            local_rank=values["local_dp_rank"],
            local_world_size=values["local_dp_world_size"],
        )

    @property
    def global_reader(self) -> int:
        return self.rank * self.num_workers + self.worker_id

    @property
    def total_readers(self) -> int:
        return self.world_size * self.num_workers

    @property
    def local_reader(self) -> int:
        return self.local_rank * self.num_workers + self.worker_id

    @property
    def local_readers(self) -> int:
        return self.local_world_size * self.num_workers


@dataclass(frozen=True)
class Occurrence:
    row: IndexRow
    repeat_ordinal: int


@dataclass(frozen=True)
class SamplePlan:
    sample_id: str
    key: str
    shard: str
    row_ordinal: int
    repeat_ordinal: int
    epoch: int
    start_frame: int
    source_frame_indices: tuple[int, ...]
    reader_rank: int
    worker_id: int


def frame_offsets(config: SamplingConfig, source_fps: float) -> np.ndarray:
    if config.clip_seconds is None:
        return np.arange(config.pixel_frames, dtype=np.int64)
    fps = source_fps if source_fps > 0 else float(config.output_fps)
    offsets = np.rint(
        np.arange(config.pixel_frames, dtype=np.float64) * (fps / float(config.output_fps))
    ).astype(np.int64)
    return np.maximum.accumulate(offsets)


class CanonicalSampler:
    """Produce deterministic healthy-path sample plans.

    Equality is scoped to the same topology and worker count. Topology changes
    intentionally repartition the canonical occurrence stream.
    """

    def __init__(
        self,
        rows: Sequence[IndexRow],
        config: SamplingConfig,
        identity: ReaderIdentity,
    ) -> None:
        if not rows:
            raise DataContractError("sampler needs at least one index row")
        self.rows = tuple(rows)
        self.config = config
        self.identity = identity
        self._rng = random.Random(config.seed + identity.global_reader * 100003)

    def iter_epoch(self, epoch: int) -> Iterator[SamplePlan]:
        if epoch < 0:
            raise DataContractError("epoch must be non-negative")
        if self.config.partition_mode == "node_shard":
            yield from self._iter_node_shard(epoch)
        else:
            yield from self._iter_global(epoch)

    def _plan(self, occurrence: Occurrence, epoch: int) -> SamplePlan:
        values = occurrence.row.values
        try:
            num_frames = int(values.get("num_frames") or values["manifest"]["video"]["num_frames"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataContractError(f"sample {occurrence.row.sample_id} lacks num_frames") from exc
        try:
            source_fps = float(
                values.get("fps") or values.get("manifest", {}).get("video", {}).get("fps") or 0.0
            )
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"sample {occurrence.row.sample_id} has invalid fps") from exc
        offsets = frame_offsets(self.config, source_fps)
        max_offset = int(offsets[-1])
        if num_frames <= max_offset:
            raise DataContractError(
                f"sample {occurrence.row.sample_id} has {num_frames} frames, needs "
                f"an offset through {max_offset}"
            )
        if self.config.random_start:
            start = self._rng.randint(0, num_frames - max_offset - 1)
        elif self.config.fixed_start_from_index:
            try:
                start = int(values["start_frame"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataContractError(
                    f"sample {occurrence.row.sample_id} lacks a valid fixed start_frame"
                ) from exc
        else:
            # A disabled random start selects the first valid frame.
            start = 0
        if start < 0 or start + max_offset >= num_frames:
            raise DataContractError(
                f"sample {occurrence.row.sample_id} fixed start {start} is outside "
                f"{num_frames} source frames"
            )
        indices = tuple(int(item) for item in start + offsets)
        return SamplePlan(
            sample_id=occurrence.row.sample_id,
            key=occurrence.row.key,
            shard=occurrence.row.shard,
            row_ordinal=occurrence.row.ordinal,
            repeat_ordinal=occurrence.repeat_ordinal,
            epoch=epoch,
            start_frame=start,
            source_frame_indices=indices,
            reader_rank=self.identity.rank,
            worker_id=self.identity.worker_id,
        )

    def _iter_global(self, epoch: int) -> Iterator[SamplePlan]:
        selected: list[Occurrence] = []
        virtual_index = 0
        for row in self.rows:
            for repeat in range(row.epoch_repeats):
                if virtual_index % self.identity.total_readers == self.identity.global_reader:
                    selected.append(Occurrence(row, repeat))
                virtual_index += 1
        if not selected:
            raise DataContractError("this reader owns no virtual occurrences")

        # The production path streams through this finite buffer instead of
        # loading and globally shuffling the complete index.
        buffer: list[Occurrence] = []
        for occurrence in selected:
            if self.config.shuffle_buffer == 1:
                # Bypass ``randrange(1)`` to avoid consuming an RNG draw.
                # Even a one-element draw can advance Python's RNG state and
                # would therefore change the subsequent random start.
                yield self._plan(occurrence, epoch)
            else:
                buffer.append(occurrence)
                if len(buffer) >= self.config.shuffle_buffer:
                    index = self._rng.randrange(len(buffer))
                    yield self._plan(buffer.pop(index), epoch)
        while buffer:
            index = self._rng.randrange(len(buffer))
            yield self._plan(buffer.pop(index), epoch)

    def _iter_node_shard(self, epoch: int) -> Iterator[SamplePlan]:
        selected_groups = self.node_shard_groups(epoch)

        reader_cursor = 0
        emitted = 0
        for _, rows in selected_groups:
            occurrences: list[Occurrence] = []
            for row in rows:
                for repeat in range(row.epoch_repeats):
                    if reader_cursor % self.identity.local_readers == self.identity.local_reader:
                        occurrences.append(Occurrence(row, repeat))
                    reader_cursor += 1
            self._rng.shuffle(occurrences)
            for occurrence in occurrences:
                emitted += 1
                yield self._plan(occurrence, epoch)
        if not emitted:
            raise DataContractError("this node-shard reader emitted no occurrences")

    def node_shard_groups(self, epoch: int) -> tuple[tuple[str, tuple[IndexRow, ...]], ...]:
        """Return the deterministic shard order shared by every reader on a node."""

        if self.config.partition_mode != "node_shard":
            raise DataContractError("node shard groups require partition_mode=node_shard")
        if epoch < 0:
            raise DataContractError("epoch must be non-negative")
        groups: list[tuple[str, list[IndexRow]]] = []
        current_shard = ""
        current: list[IndexRow] = []
        for row in self.rows:
            if current and row.shard != current_shard:
                groups.append((current_shard, current))
                current = []
            current_shard = row.shard
            current.append(row)
        if current:
            groups.append((current_shard, current))

        group_rng = random.Random(
            self.config.seed + self.identity.node_id * 1000003 + epoch * 1000000007
        )
        selected_groups: list[tuple[str, tuple[IndexRow, ...]]] = []
        group_buffer: list[tuple[str, list[IndexRow]]] = []
        for shard, rows in groups:
            partition_key = _shard_partition_key(shard, self.config.shard_partition_scope)
            assignment = (
                int.from_bytes(
                    _stable_partition_digest(
                        f"{self.config.seed}|{epoch}|{partition_key}".encode()
                    )[:8],
                    "big",
                )
                % self.identity.node_count
            )
            if assignment != self.identity.node_id:
                continue
            group_buffer.append((shard, rows))
            if len(group_buffer) >= self.config.shuffle_buffer:
                selected_shard, selected_rows = group_buffer.pop(
                    group_rng.randrange(len(group_buffer))
                )
                selected_groups.append((selected_shard, tuple(selected_rows)))
        while group_buffer:
            selected_shard, selected_rows = group_buffer.pop(group_rng.randrange(len(group_buffer)))
            selected_groups.append((selected_shard, tuple(selected_rows)))
        return tuple(selected_groups)


def _shard_partition_key(shard: str, scope: ShardPartitionScope) -> str:
    if scope == "complete":
        return shard
    parts = PurePosixPath(shard).parts
    shard_directory = max(
        (index for index, part in enumerate(parts) if part == "shards"),
        default=-1,
    )
    if shard_directory > 0:
        return "/".join(parts[shard_directory - 1 :])
    return shard


def _stable_partition_digest(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def plan_fingerprint(plans: Iterable[SamplePlan]) -> str:
    digest = hashlib.blake2s()
    digest.update(b"solarwm.sample-plan.v1\n")
    for plan in plans:
        digest.update(
            (
                f"{plan.sample_id}\t{plan.key}\t{plan.shard}\t{plan.row_ordinal}\t"
                f"{plan.repeat_ordinal}\t{plan.epoch}\t{plan.start_frame}\t"
                f"{','.join(map(str, plan.source_frame_indices))}\t"
                f"{plan.reader_rank}\t{plan.worker_id}\n"
            ).encode()
        )
    return digest.hexdigest()
