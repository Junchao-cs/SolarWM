"""Canonical preencoded reader for the embedded LTX-2.5 trainer."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
from safetensors.torch import load as load_safetensors

from solar.data import (
    CanonicalSampler,
    ReaderIdentity,
    SamplingConfig,
    TarShardReader,
    build_shard_prefetcher,
    read_index,
    resolve_index_path,
    resolver_from_config,
)
from solar.data.index import IndexRow
from solar.data.sampling import SamplePlan, plan_fingerprint
from solar.errors import BackendContractError
from solar.runtime import Topology

from .artifact import PREENCODE_SCHEMA, PREENCODE_VERSION, TENSOR_SPECS
from .geometry import STABLE_GEOMETRY


@dataclass(frozen=True)
class TorchBatch:
    sample_id: str
    start_frame: int
    plan_fingerprint: str
    video_latent: torch.Tensor
    first_frame_latent: torch.Tensor
    video_prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    relative_w2c: torch.Tensor
    camera_k: torch.Tensor
    source_indices: torch.Tensor
    camera_source_indices: torch.Tensor


class WorkerStream:
    def __init__(self, sampler: CanonicalSampler) -> None:
        self.sampler = sampler
        self.epoch = 0
        self.epoch_start_rng_state: object = sampler._rng.getstate()
        self.plans = tuple(sampler.iter_epoch(0))
        self.index = 0

    def next(self) -> SamplePlan:
        if self.index >= len(self.plans):
            self.epoch += 1
            self.epoch_start_rng_state = self.sampler._rng.getstate()
            self.plans = tuple(self.sampler.iter_epoch(self.epoch))
            self.index = 0
        result = self.plans[self.index]
        self.index += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "index": self.index,
            "epoch_start_rng_state": self.epoch_start_rng_state,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        epoch = int(value["epoch"])
        index = int(value["index"])
        if epoch < 0 or index < 0:
            raise BackendContractError("LTX reader checkpoint position is invalid")
        self.sampler._rng.setstate(value["epoch_start_rng_state"])
        plans = tuple(self.sampler.iter_epoch(epoch))
        if index > len(plans):
            raise BackendContractError("LTX reader checkpoint exceeds its epoch")
        self.epoch = epoch
        self.epoch_start_rng_state = value["epoch_start_rng_state"]
        self.plans = plans
        self.index = index


class StreamingWorkerStream:
    """Advance one deterministic worker without expanding a full epoch in memory."""

    def __init__(self, sampler: CanonicalSampler) -> None:
        self.sampler = sampler
        self.epoch = 0
        self.index = 0
        self.epoch_start_rng_state: object = sampler._rng.getstate()
        self._plans = iter(sampler.iter_epoch(0))

    def next(self) -> SamplePlan:
        try:
            result = next(self._plans)
        except StopIteration:
            self.epoch += 1
            self.index = 0
            self.epoch_start_rng_state = self.sampler._rng.getstate()
            self._plans = iter(self.sampler.iter_epoch(self.epoch))
            try:
                result = next(self._plans)
            except StopIteration as exc:
                raise BackendContractError("LTX reader worker emitted no sample plans") from exc
        self.index += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "index": self.index,
            "epoch_start_rng_state": self.epoch_start_rng_state,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        epoch = int(value["epoch"])
        index = int(value["index"])
        if epoch < 0 or index < 0:
            raise BackendContractError("LTX reader checkpoint position is invalid")
        self.sampler._rng.setstate(value["epoch_start_rng_state"])
        plans = iter(self.sampler.iter_epoch(epoch))
        try:
            for _ in range(index):
                next(plans)
        except StopIteration as exc:
            raise BackendContractError("LTX reader checkpoint exceeds its epoch") from exc
        self.epoch = epoch
        self.index = index
        self.epoch_start_rng_state = value["epoch_start_rng_state"]
        self._plans = plans


class VerifiedShardResolver:
    """Compatibility adapter over shared transport resolvers.

    Path containment, declared size, and bucket generation handling live in
    :mod:`solar.data.transport` so LTX, Wan, and H3 share one contract.
    """

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def resolve(self, row: IndexRow) -> Any:
        return self.delegate.resolve(row)


def verified_resolver_from_config(
    data: Mapping[str, Any],
) -> VerifiedShardResolver:
    transport = data.get("transport")
    if not isinstance(transport, Mapping):
        raise BackendContractError("LTX data.transport must be a mapping")
    kind = str(transport.get("kind") or "").strip().lower()
    root = str(transport.get("root") or "").strip()
    if kind not in {"local", "gcs"} or not root:
        raise BackendContractError(
            "LTX data.transport requires kind local/gcs and a non-empty root"
        )
    if kind == "local":
        if not root.startswith("/"):
            raise BackendContractError("local LTX data.transport.root must be absolute")
        if any(key in transport for key in ("cache_dir", "cache_max_gib")):
            raise BackendContractError("local LTX data.transport cannot configure a GCS cache")
        cache_dir = None
        max_gib = 256.0
    else:
        if not root.startswith("gs://"):
            raise BackendContractError("gcs LTX data.transport.root must be a gs:// URI")
        cache_dir = str(transport.get("cache_dir") or "")
        if not cache_dir.startswith("/"):
            raise BackendContractError("gcs LTX data.transport.cache_dir must be absolute")
        try:
            max_gib = float(transport["cache_max_gib"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(
                "gcs LTX data.transport.cache_max_gib must be numeric"
            ) from exc
        if max_gib <= 0:
            raise BackendContractError("gcs LTX data.transport.cache_max_gib must be positive")
    return VerifiedShardResolver(
        resolver_from_config(
            root,
            cache_dir=cache_dir,
            max_gib=max_gib,
        )
    )


def _metadata(payload: bytes) -> Mapping[str, str]:
    if len(payload) < 10:
        raise BackendContractError("LTX safetensors payload is truncated")
    header_length = struct.unpack("<Q", payload[:8])[0]
    if header_length < 2 or 8 + header_length > len(payload):
        raise BackendContractError("LTX safetensors header length is invalid")
    try:
        header = json.loads(payload[8 : 8 + header_length])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendContractError("LTX safetensors header is invalid") from exc
    if not isinstance(header, Mapping) or set(header) != set(TENSOR_SPECS) | {"__metadata__"}:
        raise BackendContractError("LTX safetensors tensor names differ")
    metadata = header["__metadata__"]
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise BackendContractError("LTX safetensors metadata is invalid")
    return metadata


def _member(row: IndexRow) -> str:
    direct = row.values.get("preencoded_member") or row.values.get("ltx25_preencoded_member")
    if direct:
        return str(direct)
    members = row.values.get("members", {})
    if isinstance(members, Mapping):
        value = members.get("ltx25.safetensors")
        if value:
            return str(value)
    raise BackendContractError(f"preencoded row {row.sample_id!r} lacks its tensor member")


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    plan: SamplePlan,
) -> None:
    if set(tensors) != set(TENSOR_SPECS):
        raise BackendContractError("preencoded LTX tensor inventory differs")
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}
    for name, spec in TENSOR_SPECS.items():
        value = tensors[name]
        if tuple(value.shape) != spec.shape or value.dtype != dtypes[spec.dtype]:
            raise BackendContractError(f"preencoded LTX tensor {name} layout differs")
    if not torch.equal(
        tensors["first_frame_latent"],
        tensors["video_latent"][:, :1],
    ):
        raise BackendContractError("preencoded first frame is not bit-equal to latent 0")
    mask = tensors["prompt_attention_mask"]
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.any()):
        raise BackendContractError("preencoded caption mask must be nonempty binary")
    if bool((mask[1:] < mask[:-1]).any()):
        raise BackendContractError("preencoded caption mask is not left padded")
    expected_source = torch.tensor(plan.source_frame_indices, dtype=torch.int64)
    if not torch.equal(tensors["source_indices"], expected_source):
        raise BackendContractError("preencoded source indices differ from canonical plan")
    camera_rows = torch.tensor(
        STABLE_GEOMETRY.camera_pixel_indices,
        dtype=torch.int64,
    )
    if not torch.equal(
        tensors["camera_source_indices"],
        expected_source.index_select(0, camera_rows),
    ):
        raise BackendContractError("preencoded camera rows differ from causal selection")
    viewmats = tensors["relative_w2c"]
    intrinsics = tensors["camera_K"]
    if not bool(torch.isfinite(viewmats).all() and torch.isfinite(intrinsics).all()):
        raise BackendContractError("preencoded LTX camera tensors are non-finite")
    if not torch.allclose(
        viewmats[0],
        torch.eye(4, dtype=torch.float32),
        rtol=0.0,
        atol=1e-5,
    ):
        raise BackendContractError("preencoded relative W2C row zero must be identity")
    if not bool((intrinsics[:, 0, 0] > 0).all() and (intrinsics[:, 1, 1] > 0).all()):
        raise BackendContractError("preencoded camera focal lengths must be positive")


class PreencodedBatchSource:
    """Ordered canonical worker merge with asynchronous shard materialization."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        data = config["data"]
        model = config["model"]
        distributed = config["distributed"]
        if not all(isinstance(value, Mapping) for value in (data, model, distributed)):
            raise BackendContractError("LTX model/data/distributed config must be mappings")
        self.checkpoint_identity = f"path:{model['checkpoint_path']}"
        self.index_path = resolve_index_path(data, "index")
        rows = read_index(self.index_path)
        normalized_rows: list[IndexRow] = []
        for row in rows:
            values = dict(row.values)
            manifest = values.get("manifest")
            preencoding = manifest.get("preencoding") if isinstance(manifest, Mapping) else None
            if isinstance(preencoding, Mapping):
                if values.get("start_frame") is None and preencoding.get("start_frame") is not None:
                    values["start_frame"] = preencoding["start_frame"]
                if (
                    values.get("source_frame_indices") is None
                    and preencoding.get("source_frame_indices") is not None
                ):
                    values["source_frame_indices"] = preencoding["source_frame_indices"]
            if values.get("num_frames") is None:
                indices = values.get("source_frame_indices")
                if not isinstance(indices, (list, tuple)):
                    indices = values.get("source_indices")
                if isinstance(indices, (list, tuple)) and indices:
                    try:
                        values["num_frames"] = max(int(value) for value in indices) + 1
                    except (TypeError, ValueError) as exc:
                        raise BackendContractError(
                            f"preencoded row {row.sample_id!r} has invalid source indices"
                        ) from exc
                elif values.get("start_frame") is not None:
                    values["num_frames"] = int(values["start_frame"]) + STABLE_GEOMETRY.pixel_frames
            normalized_rows.append(IndexRow.from_mapping(row.ordinal, values))
        self.rows = tuple(normalized_rows)
        topology = Topology.from_environ(int(distributed["sequence_parallel_size"]))
        self.topology = topology
        worker_count = int(data.get("num_workers", 1) or 1)
        sampling = SamplingConfig(
            seed=int(data.get("seed", 42)),
            pixel_frames=STABLE_GEOMETRY.pixel_frames,
            random_start=False,
            fixed_start_from_index=True,
            shuffle_buffer=int(data.get("shuffle_buffer", 32)),
            partition_mode=str(data.get("partition_mode", "global_occurrence")),
            shard_partition_scope="collection_relative",
        )
        self.streams = tuple(
            StreamingWorkerStream(
                CanonicalSampler(
                    self.rows,
                    sampling,
                    ReaderIdentity.from_topology(
                        topology,
                        worker_id=worker,
                        num_workers=worker_count,
                    ),
                )
            )
            for worker in range(worker_count)
        )
        self.cursor = 0
        self.prefetch_factor = int(data.get("prefetch_factor", 2) or 0)
        if self.prefetch_factor < 1:
            raise BackendContractError("LTX data.prefetch_factor must be positive")
        self.reader_threads = int(data.get("reader_threads", max(32, worker_count * 4)) or 0)
        if self.reader_threads < worker_count:
            raise BackendContractError("LTX data.reader_threads must be at least data.num_workers")
        self.max_rel_translation = _optional_positive_float(
            data.get("max_rel_translation"),
            "data.max_rel_translation",
        )
        self.max_camera_abs = _optional_positive_float(
            data.get("max_camera_abs"),
            "data.max_camera_abs",
        )
        self._data_config = dict(data)
        self._tar_cache_size = int(data.get("tar_cache_size", 4))
        self._resolver = verified_resolver_from_config(self._data_config)
        self._readers = tuple(
            TarShardReader(self._resolver, max_open=self._tar_cache_size) for _ in self.streams
        )
        self._executor: ThreadPoolExecutor | None = None
        self._shard_prefetcher = None
        self._pending: tuple[deque[tuple[SamplePlan, Future[TorchBatch]]], ...] = ()
        self._start_prefetch()

    def close(self) -> None:
        self._stop_prefetch()

    def _start_prefetch(
        self,
        pending_plans: tuple[tuple[SamplePlan, ...], ...] | None = None,
    ) -> None:
        self._shard_prefetcher = build_shard_prefetcher(
            self._data_config,
            rows=self.rows,
            sampler=self.streams[0].sampler,
            resolver=self._resolver,
            node_leader=int(self.topology.local_rank) == 0,
            current_shard_workers=min(4, len(self.streams)),
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.reader_threads,
            thread_name_prefix="solar-ltx-reader",
        )
        self._pending = tuple(deque() for _ in self.streams)
        if pending_plans is not None:
            if len(pending_plans) != len(self.streams):
                raise BackendContractError("LTX reader pending worker topology differs")
            for worker, plans in enumerate(pending_plans):
                if len(plans) > self.prefetch_factor:
                    raise BackendContractError("LTX reader pending queue exceeds prefetch_factor")
                for plan in plans:
                    self._submit(worker, plan)
        for _ in range(self.prefetch_factor):
            for worker in range(len(self.streams)):
                if len(self._pending[worker]) < self.prefetch_factor:
                    self._submit(worker, self.streams[worker].next())

    def _stop_prefetch(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        shard_prefetcher, self._shard_prefetcher = self._shard_prefetcher, None
        if shard_prefetcher is not None:
            shard_prefetcher.close()
        for reader in self._readers:
            reader.close()
        self._pending = ()

    def _load(self, plan: SamplePlan) -> TorchBatch:
        row = self.rows[plan.row_ordinal]
        if self._shard_prefetcher is not None:
            self._shard_prefetcher.wait(plan)
        return load_preencoded_row(
            row,
            plan=plan,
            shards=self._readers[plan.worker_id],
        )

    def _submit(self, worker: int, plan: SamplePlan) -> None:
        if plan.worker_id != worker:
            raise BackendContractError("LTX reader pending plan belongs to another worker")
        if not 0 <= plan.row_ordinal < len(self.rows):
            raise BackendContractError("LTX reader pending plan row is invalid")
        row = self.rows[plan.row_ordinal]
        if (plan.sample_id, plan.key, plan.shard) != (row.sample_id, row.key, row.shard):
            raise BackendContractError("LTX reader pending plan identity differs")
        if self._executor is None:
            raise BackendContractError("LTX reader prefetch executor is closed")
        # Node-level GCS lookahead follows worker zero's ordered shard stream.
        # That stream still crosses groups with no local occurrence, while a
        # single frontier keeps the configured window bounded.
        if self._shard_prefetcher is not None and (
            str(self._data_config.get("partition_mode")) != "node_shard" or worker == 0
        ):
            self._shard_prefetcher.schedule(plan)
        self._pending[worker].append((plan, self._executor.submit(self._load, plan)))

    def _refill(self, worker: int) -> None:
        while len(self._pending[worker]) < self.prefetch_factor:
            self._submit(worker, self.streams[worker].next())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "solarwm.ltx25.reader-state.v3",
            "index_path": str(self.index_path),
            "cursor": self.cursor,
            "workers": [stream.state_dict() for stream in self.streams],
            "prefetch_factor": self.prefetch_factor,
            "pending": [[_plan_state(plan) for plan, _ in queue] for queue in self._pending],
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        schema = value.get("schema")
        if schema not in {
            "solarwm.ltx25.reader-state.v1",
            "solarwm.ltx25.reader-state.v2",
            "solarwm.ltx25.reader-state.v3",
        }:
            raise BackendContractError("unknown LTX reader checkpoint schema")
        if str(value.get("index_path")) != str(self.index_path):
            raise BackendContractError("LTX reader checkpoint index path differs")
        workers = value.get("workers")
        if not isinstance(workers, list) or len(workers) != len(self.streams):
            raise BackendContractError("LTX reader checkpoint worker topology differs")
        cursor = int(value.get("cursor", -1))
        if not 0 <= cursor < len(self.streams):
            raise BackendContractError("LTX reader checkpoint cursor is invalid")
        pending_plans: tuple[tuple[SamplePlan, ...], ...] | None = None
        if schema == "solarwm.ltx25.reader-state.v2":
            if int(value.get("prefetch_factor", 0)) != self.prefetch_factor:
                raise BackendContractError("LTX reader checkpoint prefetch_factor differs")
            raw_pending = value.get("pending")
            if not isinstance(raw_pending, list) or len(raw_pending) != len(self.streams):
                raise BackendContractError("LTX reader checkpoint pending queues differ")
            parsed: list[tuple[SamplePlan, ...]] = []
            for queue in raw_pending:
                if not isinstance(queue, list):
                    raise BackendContractError("LTX reader checkpoint pending queue is invalid")
                parsed.append(tuple(_plan_from_state(plan) for plan in queue))
            pending_plans = tuple(parsed)
        elif schema == "solarwm.ltx25.reader-state.v3":
            if int(value.get("prefetch_factor", 0)) != self.prefetch_factor:
                raise BackendContractError("LTX reader checkpoint prefetch_factor differs")
            raw_pending = value.get("pending")
            if not isinstance(raw_pending, list) or len(raw_pending) != len(self.streams):
                raise BackendContractError("LTX reader checkpoint pending queues differ")
            parsed = []
            for queue in raw_pending:
                if not isinstance(queue, list):
                    raise BackendContractError("LTX reader checkpoint pending queue is invalid")
                parsed.append(tuple(_plan_from_state(plan) for plan in queue))
            pending_plans = tuple(parsed)
        self._stop_prefetch()
        for stream, stream_state in zip(self.streams, workers, strict=True):
            if not isinstance(stream_state, Mapping):
                raise BackendContractError("LTX reader worker state is invalid")
            stream.load_state_dict(stream_state)
        self.cursor = cursor
        self._start_prefetch(pending_plans)

    def next(self) -> TorchBatch:
        worker = self.cursor
        while True:
            plan, future = self._pending[worker].popleft()
            batch = future.result()
            self._refill(worker)
            row = self.rows[plan.row_ordinal]
            if (
                _camera_magnitude_rejection(
                    batch,
                    source_fps=_source_fps(row),
                    max_rel_translation=self.max_rel_translation,
                    max_camera_abs=self.max_camera_abs,
                )
                is None
            ):
                self.cursor = (worker + 1) % len(self.streams)
                return batch

    @property
    def checkpoint_id(self) -> str:
        return self.checkpoint_identity


def _plan_state(plan: SamplePlan) -> dict[str, Any]:
    return {
        "sample_id": plan.sample_id,
        "key": plan.key,
        "shard": plan.shard,
        "row_ordinal": plan.row_ordinal,
        "repeat_ordinal": plan.repeat_ordinal,
        "epoch": plan.epoch,
        "start_frame": plan.start_frame,
        "source_frame_indices": list(plan.source_frame_indices),
        "reader_rank": plan.reader_rank,
        "worker_id": plan.worker_id,
    }


def _plan_from_state(value: Any) -> SamplePlan:
    if not isinstance(value, Mapping):
        raise BackendContractError("LTX reader checkpoint pending plan is invalid")
    try:
        return SamplePlan(
            sample_id=str(value["sample_id"]),
            key=str(value["key"]),
            shard=str(value["shard"]),
            row_ordinal=int(value["row_ordinal"]),
            repeat_ordinal=int(value["repeat_ordinal"]),
            epoch=int(value["epoch"]),
            start_frame=int(value["start_frame"]),
            source_frame_indices=tuple(int(item) for item in value["source_frame_indices"]),
            reader_rank=int(value["reader_rank"]),
            worker_id=int(value["worker_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError("LTX reader checkpoint pending plan is invalid") from exc


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BackendContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise BackendContractError(f"{name} must be finite and positive")
    return result


def _source_fps(row: IndexRow) -> float:
    values = row.values
    candidates = (
        values.get("fps"),
        values.get("source_fps"),
        values.get("metadata", {}).get("source_fps")
        if isinstance(values.get("metadata"), Mapping)
        else None,
        values.get("manifest", {}).get("source_fps")
        if isinstance(values.get("manifest"), Mapping)
        else None,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            fps = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fps) and fps > 0:
            return fps
    return 0.0


def _camera_magnitude_rejection(
    batch: TorchBatch,
    *,
    source_fps: float,
    max_rel_translation: float | None,
    max_camera_abs: float | None,
) -> str | None:
    """Apply the recoverable first-10-second camera guard."""

    if max_rel_translation is None and max_camera_abs is None:
        return None
    audit_pixel_frames = STABLE_GEOMETRY.pixel_frames
    if math.isfinite(source_fps) and source_fps > 0:
        audit_pixel_frames = min(
            audit_pixel_frames,
            max(1, math.ceil(source_fps * 10.0)),
        )
    camera_offsets = batch.camera_source_indices[0] - batch.source_indices[0, 0]
    audit_rows = camera_offsets < audit_pixel_frames
    if not bool(audit_rows.any()):
        raise BackendContractError("LTX camera magnitude audit selected no causal rows")
    viewmats = batch.relative_w2c[0, audit_rows]
    rotation_inverse = viewmats[:, :3, :3].transpose(-1, -2)
    relative_c2w = torch.zeros_like(viewmats)
    relative_c2w[:, :3, :3] = rotation_inverse
    relative_c2w[:, :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        rotation_inverse,
        viewmats[:, :3, 3],
    )
    relative_c2w[:, 3, 3] = 1.0
    translation = float(torch.linalg.vector_norm(relative_c2w[:, :3, 3], dim=-1).amax().item())
    if max_rel_translation is not None and translation > max_rel_translation:
        return (
            "camera exceeds max_rel_translation: "
            f"observed={translation:.9g} limit={max_rel_translation:.9g}"
        )
    absolute = float(relative_c2w.abs().amax().item())
    if max_camera_abs is not None and absolute > max_camera_abs:
        return f"camera exceeds max_camera_abs: observed={absolute:.9g} limit={max_camera_abs:.9g}"
    return None


def _fixed_plan(row: IndexRow) -> SamplePlan:
    try:
        start = int(row.values["start_frame"])
        indices = tuple(int(item) for item in row.values["source_frame_indices"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError(
            f"preencoded row {row.sample_id!r} lacks fixed source indices"
        ) from exc
    if len(indices) != STABLE_GEOMETRY.pixel_frames or indices != tuple(
        range(start, start + STABLE_GEOMETRY.pixel_frames)
    ):
        raise BackendContractError(
            f"preencoded row {row.sample_id!r} source indices are not contiguous"
        )
    return SamplePlan(
        sample_id=row.sample_id,
        key=row.key,
        shard=row.shard,
        row_ordinal=row.ordinal,
        repeat_ordinal=0,
        epoch=0,
        start_frame=start,
        source_frame_indices=indices,
        reader_rank=0,
        worker_id=0,
    )


def load_preencoded_row(
    row: IndexRow,
    *,
    shards: TarShardReader,
    plan: SamplePlan | None = None,
) -> TorchBatch:
    """Strictly materialize one indexed artifact without changing index order."""

    payload = shards.read(row, _member(row))
    metadata = _metadata(payload)
    if metadata.get("schema") != PREENCODE_SCHEMA:
        raise BackendContractError("preencoded LTX schema differs")
    if metadata.get("version") != PREENCODE_VERSION:
        raise BackendContractError("preencoded LTX artifact version differs")
    tensors = load_safetensors(payload)
    if plan is not None:
        selected = plan
    elif "start_frame" in row.values or "source_frame_indices" in row.values:
        selected = _fixed_plan(row)
    else:
        try:
            indices = tuple(int(item) for item in tensors["source_indices"].tolist())
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(
                f"preencoded row {row.sample_id!r} lacks tensor source indices"
            ) from exc
        if not indices:
            raise BackendContractError(
                f"preencoded row {row.sample_id!r} has empty tensor source indices"
            )
        selected = SamplePlan(
            sample_id=row.sample_id,
            key=row.key,
            shard=row.shard,
            row_ordinal=row.ordinal,
            repeat_ordinal=0,
            epoch=0,
            start_frame=indices[0],
            source_frame_indices=indices,
            reader_rank=0,
            worker_id=0,
        )
    if (
        selected.row_ordinal != row.ordinal
        or selected.sample_id != row.sample_id
        or selected.key != row.key
        or selected.shard != row.shard
    ):
        raise BackendContractError("preencoded row and sample plan identities differ")
    if metadata.get("sample_id") != selected.sample_id:
        raise BackendContractError("preencoded payload sample_id differs from index")
    _validate_tensors(tensors, plan=selected)
    return TorchBatch(
        sample_id=selected.sample_id,
        start_frame=selected.start_frame,
        plan_fingerprint=plan_fingerprint((selected,)),
        video_latent=tensors["video_latent"].unsqueeze(0),
        first_frame_latent=tensors["first_frame_latent"].unsqueeze(0),
        video_prompt_embeds=tensors["video_prompt_embeds"].unsqueeze(0),
        prompt_attention_mask=tensors["prompt_attention_mask"].unsqueeze(0),
        relative_w2c=tensors["relative_w2c"].unsqueeze(0),
        camera_k=tensors["camera_K"].unsqueeze(0),
        source_indices=tensors["source_indices"].unsqueeze(0),
        camera_source_indices=tensors["camera_source_indices"].unsqueeze(0),
    )


class IndexedPreencodedSource:
    """Ordered random access used by standalone inference and validation."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        index_field: str = "index",
    ) -> None:
        data = config["data"]
        if not isinstance(data, Mapping):
            raise BackendContractError("LTX data config must be a mapping")
        index_data = dict(data)
        if index_field != "index":
            index_data["index"] = config[index_field]
        self.index_path = resolve_index_path(index_data, "index")
        self.rows = read_index(self.index_path)
        resolver = verified_resolver_from_config(data)
        self.shards = TarShardReader(
            resolver,
            max_open=int(data.get("tar_cache_size", 4)),
        )
        self._by_id = {row.sample_id: row for row in self.rows}

    def get(self, sample_id: str) -> TorchBatch:
        try:
            row = self._by_id[sample_id]
        except KeyError as exc:
            raise BackendContractError(
                f"inference sample {sample_id!r} is outside the fixed index"
            ) from exc
        return load_preencoded_row(row, shards=self.shards)

    def case_fingerprint(self, sample_id: str, batch: TorchBatch | None = None) -> str:
        value = batch or self.get(sample_id)
        digest = hashlib.blake2s()
        digest.update(value.relative_w2c.contiguous().numpy().tobytes())
        digest.update(value.camera_k.contiguous().numpy().tobytes())
        return digest.hexdigest()

    def close(self) -> None:
        self.shards.close()


__all__ = [
    "IndexedPreencodedSource",
    "PreencodedBatchSource",
    "TorchBatch",
    "VerifiedShardResolver",
    "WorkerStream",
    "load_preencoded_row",
    "verified_resolver_from_config",
]
