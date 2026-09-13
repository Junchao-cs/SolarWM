"""Wan2.2 fixed-window tensor-corpus reader."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from solarwm.data.archive import RawSample, RawSampleReader, TarShardReader
from solarwm.data.camera import CameraGuards
from solarwm.data.index import IndexRow, read_index, resolve_index_path
from solarwm.data.prefetch import build_shard_prefetcher
from solarwm.data.sampling import CanonicalSampler, ReaderIdentity, SamplingConfig
from solarwm.data.transport import GCSDownloadTimeout, resolver_from_config
from solarwm.errors import DataContractError
from solarwm.runtime.distributed import DataReadiness

from ..windows import (
    SIX_WINDOW_153F_DATASETS,
    WINDOW_HASH_NAMESPACE_81F,
    WINDOW_HASH_NAMESPACE_153F,
    expected_81f_window_start,
    expected_153f_window_start,
)
from .data import (
    _READER_STATE_KEY,
    WanPlanMultiplexer,
    _advance_worker_epoch,
    _data_loader_generator,
    _is_skippable_camera_guard_error,
    _restore_worker_cursor,
    _worker_progress_payload,
    build_camera_tokens,
)

TI2V_5B_81F_VERSION = "solarwm_wan22_ti2v_5b_480p_81f_v1"
TI2V_5B_153F_VERSION = "solarwm_wan22_ti2v_5b_480p_153f_v1"
TI2V_5B_720P_153F_VERSION = "solarwm_wan22_ti2v_5b_720p_153f_v1"
I2V_A14B_153F_VERSION = "solarwm_wan22_i2v_a14b_480p_153f_v1"
MATERIALIZED_INDEX_WINDOW_ASSIGNMENT = "materialized_index_v1"
DETERMINISTIC_HASH_WINDOW_ASSIGNMENT = "deterministic_hash_v1"
_WINDOW_SAMPLE_ID = re.compile(r"^(?P<source>.+)/latent-(?P<frames>81|153)f-w(?P<index>[0-9]{2})$")


def _training_resolver(config: Mapping[str, Any]) -> Any:
    data = config["data"]
    transport = data["transport"]
    runtime = config.get("runtime", {})
    return resolver_from_config(
        str(transport["root"]),
        cache_dir=runtime.get("data_cache_dir", transport.get("cache_dir")),
        max_gib=float(runtime.get("data_cache_max_gib", transport.get("cache_max_gib", 256))),
    )


@dataclass(frozen=True)
class DecodedWanPreencodedSample:
    sample_id: str
    key: str
    start_frame: int
    source_frame_indices: tuple[int, ...]
    prompt: str
    latents: Any
    prompt_embeds: Any
    i2v_y: Any | None
    camera: Mapping[str, Any]


def _manifest_preencoding(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    value = manifest.get("preencoding", {})
    if not isinstance(value, Mapping):
        raise DataContractError("preencoded Wan manifest.preencoding must be a mapping")
    return value


def _rows_with_fixed_starts(rows: Sequence[IndexRow]) -> tuple[IndexRow, ...]:
    """Expose embedded fixed-window starts to the transport-free sampler."""

    normalized: list[IndexRow] = []
    for row in rows:
        values = dict(row.values)
        raw_start = values.get("start_frame")
        if raw_start is None:
            manifest = values.get("manifest", {})
            if not isinstance(manifest, Mapping):
                raise DataContractError(
                    f"preencoded sample {row.sample_id!r} needs an embedded manifest"
                )
            raw_start = _manifest_preencoding(manifest).get("start_frame")
        try:
            start = int(raw_start)
        except (TypeError, ValueError) as exc:
            raise DataContractError(
                f"preencoded sample {row.sample_id!r} has no valid fixed start"
            ) from exc
        values["start_frame"] = start
        normalized.append(
            IndexRow(
                ordinal=row.ordinal,
                sample_id=row.sample_id,
                key=row.key,
                shard=row.shard,
                epoch_repeats=row.epoch_repeats,
                values=values,
            )
        )
    return tuple(normalized)


def _int_field(values: Mapping[str, Any], name: str) -> int:
    try:
        return int(values[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(f"preencoding.{name} must be an integer") from exc


def _validate_materialized_window_identity(
    sample: RawSample,
    preencoding: Mapping[str, Any],
    *,
    num_frames: int,
    pixel_frames: int,
    start: int,
) -> None:
    """Bind a published fixed-window row to its source without rehashing renamed IDs."""

    match = _WINDOW_SAMPLE_ID.fullmatch(sample.plan.sample_id)
    if match is None:
        raise DataContractError("materialized sample_id must end in /latent-{81,153}f-wNN")
    if int(match.group("frames")) != pixel_frames:
        raise DataContractError("materialized sample_id frame count differs from its contract")
    source_sample_id = str(preencoding.get("source_sample_id", ""))
    source_dataset = str(preencoding.get("source_dataset", ""))
    if source_sample_id != match.group("source"):
        raise DataContractError(
            "preencoding.source_sample_id differs from the materialized sample_id"
        )
    source_parts = source_sample_id.split("/")
    if not source_parts or source_dataset not in source_parts[:2]:
        raise DataContractError(
            "preencoding.source_dataset differs from the materialized source_sample_id"
        )
    window_index = _int_field(preencoding, "window_index")
    window_count = _int_field(preencoding, "window_count")
    if window_index != int(match.group("index")):
        raise DataContractError("preencoding.window_index differs from the materialized sample_id")
    if not 0 <= start <= num_frames - pixel_frames:
        raise DataContractError("materialized window lies outside its source")
    if pixel_frames == 81:
        expected_start = expected_81f_window_start(preencoding, num_frames)
        if start != expected_start:
            raise DataContractError(
                "materialized 81f window differs from its uniform five-second geometry"
            )
    elif source_dataset in SIX_WINDOW_153F_DATASETS:
        expected_start = expected_153f_window_start(preencoding, num_frames)
        if start != expected_start:
            raise DataContractError(
                "materialized long-form 153f window differs from its six-window geometry"
            )
    elif window_count != 1 or window_index != 0:
        raise DataContractError("ordinary materialized 153f source must carry one window")


def _validate_preencoding(
    sample: RawSample,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    data = config["data"]
    preencoding = _manifest_preencoding(sample.manifest)
    version = str(preencoding.get("version", ""))
    family = str(config["model"]["family"])
    if family == "wan22_i2v_a14b":
        expected_version = I2V_A14B_153F_VERSION
    elif str(data.get("preencode_schema", "")) == "solarwm.wan22_ti2v_5b.480p.81f.v1":
        expected_version = TI2V_5B_81F_VERSION
    elif str(data.get("preencode_schema", "")) == "solarwm.wan22_ti2v_5b.720p.153f.v1":
        expected_version = TI2V_5B_720P_153F_VERSION
    else:
        expected_version = TI2V_5B_153F_VERSION
    accepted_versions = {
        expected_version,
        str(data["preencode_schema"]),
    }
    aliases = data.get("preencode_version_aliases", ())
    if isinstance(aliases, str) or not isinstance(aliases, (list, tuple)):
        raise DataContractError("data.preencode_version_aliases must be a list of strings")
    if not all(isinstance(alias, str) and alias for alias in aliases):
        raise DataContractError("data.preencode_version_aliases must contain non-empty strings")
    accepted_versions.update(aliases)
    if version not in accepted_versions:
        raise DataContractError(
            f"preencoding.version {version!r} is not one of {sorted(accepted_versions)}"
        )
    expected_scalars = {
        "pixel_frames": int(data["pixel_frames"]),
        "latent_frames": int(data["latent_frames"]),
        "target_h": int(data["height"]),
        "target_w": int(data["width"]),
    }
    for name, expected in expected_scalars.items():
        actual = _int_field(preencoding, name)
        if actual != expected:
            raise DataContractError(
                f"preencoding.{name}={actual} does not match configured {expected}"
            )
    if str(preencoding.get("dtype", "")) != str(data["latent_dtype"]):
        raise DataContractError("preencoding dtype does not match data.latent_dtype")
    if tuple(preencoding.get("latent_shape", ())) != tuple(data["latent_shape"]):
        raise DataContractError("preencoding latent_shape does not match data.latent_shape")
    if tuple(preencoding.get("prompt_shape", ())) != tuple(data["prompt_shape"]):
        raise DataContractError("preencoding prompt_shape does not match data.prompt_shape")
    pixel_frames = int(data["pixel_frames"])
    expected_namespace = (
        WINDOW_HASH_NAMESPACE_81F if pixel_frames == 81 else WINDOW_HASH_NAMESPACE_153F
    )
    accepted_namespaces = {expected_namespace}
    namespace_aliases = data.get("preencode_window_namespace_aliases", ())
    if isinstance(namespace_aliases, str) or not isinstance(namespace_aliases, (list, tuple)):
        raise DataContractError("data.preencode_window_namespace_aliases must be a list of strings")
    if not all(isinstance(alias, str) and alias for alias in namespace_aliases):
        raise DataContractError(
            "data.preencode_window_namespace_aliases must contain non-empty strings"
        )
    accepted_namespaces.update(namespace_aliases)
    namespace = str(preencoding.get("window_hash_namespace", ""))
    if namespace not in accepted_namespaces:
        raise DataContractError(
            "preencoding window_hash_namespace differs from the configured fixed-window contract"
        )
    video = sample.manifest.get("video", {})
    if not isinstance(video, Mapping):
        raise DataContractError("preencoded Wan manifest.video must be a mapping")
    try:
        num_frames = int(video["num_frames"])
        start = int(preencoding["start_frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError("preencoded Wan source geometry is incomplete") from exc
    window_assignment = str(data.get("preencode_window_assignment", ""))
    if window_assignment == MATERIALIZED_INDEX_WINDOW_ASSIGNMENT:
        _validate_materialized_window_identity(
            sample,
            preencoding,
            num_frames=num_frames,
            pixel_frames=pixel_frames,
            start=start,
        )
    elif window_assignment == DETERMINISTIC_HASH_WINDOW_ASSIGNMENT:
        expected_start = (
            expected_81f_window_start(preencoding, num_frames)
            if pixel_frames == 81
            else expected_153f_window_start(preencoding, num_frames)
        )
        if start != expected_start:
            raise DataContractError("stored window start differs from its deterministic assignment")
    else:
        raise DataContractError("unknown data.preencode_window_assignment")
    if start != sample.plan.start_frame:
        raise DataContractError("stored window start differs from its index assignment")
    if tuple(sample.plan.source_frame_indices) != tuple(range(start, start + pixel_frames)):
        raise DataContractError("preencoded corpus must select one contiguous 16fps window")
    if _int_field(preencoding, "source_frame_first") != start:
        raise DataContractError("preencoding.source_frame_first differs from start_frame")
    if _int_field(preencoding, "source_frame_last") != start + pixel_frames - 1:
        raise DataContractError("preencoding.source_frame_last differs from the fixed window")
    return preencoding


def decode_preencoded_sample(
    sample: RawSample,
    config: Mapping[str, Any],
) -> DecodedWanPreencodedSample:
    """Validate one frozen artifact, deserialize its tensors, and retain camera math."""

    try:
        import torch
        from safetensors.torch import load as load_safetensors
    except ImportError as exc:
        raise DataContractError("preencoded Wan training requires torch and safetensors") from exc

    _validate_preencoding(sample, config)
    tensor_payload = sample.members["preencoded_member"]
    try:
        tensors = load_safetensors(tensor_payload)
    except Exception as exc:
        raise DataContractError(f"cannot decode preencoded Wan tensors: {exc}") from exc
    latents = tensors.get("latents")
    prompt_embeds = tensors.get("prompt_embeds")
    if latents is None or prompt_embeds is None:
        raise DataContractError("preencoded Wan safetensors must contain latents and prompt_embeds")
    data = config["data"]
    family = str(config["model"]["family"])
    i2v_y = tensors.get("i2v_y")
    if family == "wan22_i2v_a14b":
        if i2v_y is None:
            raise DataContractError("I2V-A14B preencoded artifacts require i2v_y")
        if tuple(i2v_y.shape) != tuple(data["i2v_y_shape"]):
            raise DataContractError(
                f"preencoded i2v_y shape {tuple(i2v_y.shape)} != {tuple(data['i2v_y_shape'])}"
            )
        if i2v_y.dtype != torch.bfloat16:
            raise DataContractError("preencoded Wan i2v_y must be bfloat16")
    elif i2v_y is not None:
        raise DataContractError("TI2V-5B preencoded artifacts may not contain i2v_y")
    if tuple(latents.shape) != tuple(data["latent_shape"]):
        raise DataContractError(
            f"preencoded latent shape {tuple(latents.shape)} != {tuple(data['latent_shape'])}"
        )
    if tuple(prompt_embeds.shape) != tuple(data["prompt_shape"]):
        raise DataContractError(
            f"preencoded prompt shape {tuple(prompt_embeds.shape)} != {tuple(data['prompt_shape'])}"
        )
    if latents.dtype != torch.bfloat16 or prompt_embeds.dtype != torch.bfloat16:
        raise DataContractError(
            "preencoded Wan latents and prompt embeddings must both be bfloat16"
        )
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all().item()):
            raise DataContractError(f"preencoded Wan tensor {name!r} contains non-finite values")
    video = sample.manifest["video"]
    try:
        source_fps = float(video.get("fps") or 0.0)
    except (TypeError, ValueError) as exc:
        raise DataContractError("preencoded Wan source fps must be numeric") from exc
    if source_fps != float(data.get("fps", 16.0)):
        raise DataContractError("preencoded corpus source fps differs from configured output fps")
    camera = build_camera_tokens(
        sample.members["camera_member"],
        sample.plan.source_frame_indices,
        sample.manifest,
        source_fps=source_fps,
        output_fps=float(data.get("fps", 16.0)),
        frame_sequence_length=int(config["model"]["frame_sequence_length"]),
        guards=CameraGuards(
            max_rel_translation=float(data["max_rel_translation"]),
            max_camera_abs=float(data["max_camera_abs"]),
        ),
        configured_array_key=str(data["camera_array_key"]),
    )
    return DecodedWanPreencodedSample(
        sample_id=sample.plan.sample_id,
        key=sample.plan.key,
        start_frame=sample.plan.start_frame,
        source_frame_indices=sample.plan.source_frame_indices,
        prompt=sample.caption,
        latents=latents,
        prompt_embeds=prompt_embeds,
        i2v_y=i2v_y,
        camera=camera,
    )


def collate_preencoded_samples(
    samples: Sequence[DecodedWanPreencodedSample],
) -> Mapping[str, Any]:
    if not samples:
        raise DataContractError("cannot collate an empty preencoded Wan batch")
    import torch

    result = {
        "sample_ids": tuple(sample.sample_id for sample in samples),
        "keys": tuple(sample.key for sample in samples),
        "start_frames": tuple(sample.start_frame for sample in samples),
        "source_frame_indices": tuple(sample.source_frame_indices for sample in samples),
        "prompts": tuple(sample.prompt for sample in samples),
        "latents": torch.stack([sample.latents for sample in samples]),
        "prompt_embeds": torch.stack([sample.prompt_embeds for sample in samples]),
        "camera": {
            "viewmats": torch.stack([sample.camera["viewmats"] for sample in samples]),
            "K": torch.stack([sample.camera["K"] for sample in samples]),
        },
        "preencoded": True,
    }
    has_i2v_y = [sample.i2v_y is not None for sample in samples]
    if any(has_i2v_y) and not all(has_i2v_y):
        raise DataContractError("cannot collate mixed Wan samples with and without i2v_y")
    if all(has_i2v_y):
        result["i2v_y"] = torch.stack([sample.i2v_y for sample in samples])
    return result


def iter_preencoded_batches(
    config: Mapping[str, Any],
    topology: Any,
    *,
    worker_id: int = 0,
    num_workers: int = 1,
    rows: Sequence[IndexRow] | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Yield the deterministic fixed-window occurrence stream."""

    data = config["data"]
    materialized_rows = (
        _rows_with_fixed_starts(read_index(resolve_index_path(data, "train_index")))
        if rows is None
        else tuple(rows)
    )
    identity = ReaderIdentity.from_topology(topology, worker_id=worker_id, num_workers=num_workers)
    pixel_frames = int(data["pixel_frames"])
    output_fps = float(data.get("fps", 16.0))
    sampling = SamplingConfig(
        seed=int(data["seed"]),
        pixel_frames=pixel_frames,
        random_start=False,
        fixed_start_from_index=True,
        clip_seconds=(pixel_frames - 1) / output_fps,
        output_fps=output_fps,
        shuffle_buffer=int(data.get("shuffle_buffer", 32)),
        partition_mode=str(data["partition_mode"]),
    )
    resolver = _training_resolver(config)
    micro_batch = int(config["train"]["micro_batch_size"])
    pending: list[DecodedWanPreencodedSample] = []
    sampler = CanonicalSampler(materialized_rows, sampling, identity)
    prefetcher = build_shard_prefetcher(
        data,
        rows=materialized_rows,
        sampler=sampler,
        resolver=resolver,
        node_leader=int(getattr(topology, "local_rank", identity.local_rank)) == 0
        and identity.worker_id == 0,
    )
    try:
        with TarShardReader(resolver, max_open=int(data.get("tar_cache_size", 4))) as shards:
            reader = RawSampleReader(
                materialized_rows,
                shards,
                member_fields=("preencoded_member", "camera_member"),
            )
            epoch = 0
            while True:
                for plan in sampler.iter_epoch(epoch):
                    try:
                        if prefetcher is not None:
                            prefetcher.prepare(plan)
                        decoded = decode_preencoded_sample(reader.materialize(plan), config)
                    except DataContractError as exc:
                        # The WDS reader treats camera-magnitude guard
                        # failures as filtered samples and advances this worker's
                        # occurrence stream. All other integrity/contract failures
                        # remain fatal.
                        if not _is_skippable_camera_guard_error(exc):
                            raise
                        continue
                    pending.append(decoded)
                    if len(pending) == micro_batch:
                        yield collate_preencoded_samples(pending)
                        pending = []
                epoch += 1
    finally:
        if prefetcher is not None:
            prefetcher.close()


def _iter_stateful_preencoded_batches(
    config: Mapping[str, Any],
    topology: Any,
    *,
    rows: Sequence[IndexRow],
    worker_id: int,
    num_workers: int,
    worker_state: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    """Run one original preencoded DataLoader worker and report consumed state."""

    data = config["data"]
    pixel_frames = int(data["pixel_frames"])
    output_fps = float(data.get("fps", 16.0))
    sampling = SamplingConfig(
        seed=int(data["seed"]),
        pixel_frames=pixel_frames,
        random_start=False,
        fixed_start_from_index=True,
        clip_seconds=(pixel_frames - 1) / output_fps,
        output_fps=output_fps,
        shuffle_buffer=int(data.get("shuffle_buffer", 32)),
        partition_mode=str(data["partition_mode"]),
    )
    cursor = _restore_worker_cursor(
        rows,
        sampling,
        topology,
        worker_id=worker_id,
        num_workers=num_workers,
        state=worker_state,
    )
    resolver = _training_resolver(config)
    micro_batch = int(config["train"]["micro_batch_size"])
    pending: list[DecodedWanPreencodedSample] = []
    prefetcher = build_shard_prefetcher(
        data,
        rows=rows,
        sampler=cursor.sampler,
        resolver=resolver,
        node_leader=int(getattr(topology, "local_rank", cursor.sampler.identity.local_rank)) == 0
        and worker_id == 0,
    )
    try:
        with TarShardReader(resolver, max_open=int(data.get("tar_cache_size", 4))) as shards:
            reader = RawSampleReader(
                rows,
                shards,
                member_fields=("preencoded_member", "camera_member"),
            )
            while True:
                while cursor.cursor < len(cursor.plans):
                    plan = cursor.plans[cursor.cursor]
                    cursor.cursor += 1
                    try:
                        if prefetcher is not None:
                            prefetcher.prepare(plan)
                        decoded = decode_preencoded_sample(reader.materialize(plan), config)
                    except GCSDownloadTimeout:
                        continue
                    except DataContractError as exc:
                        if not _is_skippable_camera_guard_error(exc):
                            raise
                        continue
                    cursor.healthy_samples += 1
                    pending.append(decoded)
                    if len(pending) == micro_batch:
                        batch = dict(collate_preencoded_samples(pending))
                        batch[_READER_STATE_KEY] = _worker_progress_payload(cursor)
                        yield batch
                        pending = []
                if cursor.healthy_samples == 0:
                    raise RuntimeError(
                        f"preencoded Wan reader rank={cursor.sampler.identity.rank} "
                        f"worker={worker_id} emitted no samples in epoch {cursor.epoch} "
                        f"from {data['train_index']}"
                    )
                _advance_worker_epoch(cursor)
    finally:
        if prefetcher is not None:
            prefetcher.close()


class StatefulPreencodedWanReader:
    """Original multi-process preencoded loader with consumed-cursor checkpoints."""

    def __init__(self, config: Mapping[str, Any], topology: Any) -> None:
        data = config["data"]
        self.config = config
        self.topology = topology
        self.rows = _rows_with_fixed_starts(read_index(resolve_index_path(data, "train_index")))
        pixel_frames = int(data["pixel_frames"])
        output_fps = float(data.get("fps", 16.0))
        self.sampling = SamplingConfig(
            seed=int(data["seed"]),
            pixel_frames=pixel_frames,
            random_start=False,
            fixed_start_from_index=True,
            clip_seconds=(pixel_frames - 1) / output_fps,
            output_fps=output_fps,
            shuffle_buffer=int(data.get("shuffle_buffer", 32)),
            partition_mode=str(data["partition_mode"]),
        )
        self.loader_workers = int(data.get("num_workers", 0))
        self.num_workers = max(1, self.loader_workers)
        self.micro_batch_size = int(config["train"]["micro_batch_size"])
        if self.micro_batch_size < 1:
            raise DataContractError("Wan train.micro_batch_size must be positive")
        self.plan = WanPlanMultiplexer(
            self.rows,
            self.sampling,
            topology,
            num_workers=self.num_workers,
        )
        self._loader: Any | None = None
        self._iterator: Any | None = None
        self._closed = False
        self._readiness = DataReadiness(enabled=str(data["transport"]["root"]).startswith("gs://"))

    def __iter__(self) -> StatefulPreencodedWanReader:
        return self

    def __next__(self) -> Mapping[str, Any]:
        return self.next()

    def next(self) -> Mapping[str, Any]:
        return self._readiness.read(self._next_batch)

    def _next_batch(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("Wan preencoded reader is closed")
        if self._iterator is None:
            self._start_loader()
        assert self._iterator is not None
        batch = next(self._iterator)
        if not isinstance(batch, dict):
            raise DataContractError("Wan preencoded DataLoader returned a non-mapping batch")
        worker_state = batch.pop(_READER_STATE_KEY, None)
        if not isinstance(worker_state, Mapping):
            raise DataContractError("Wan preencoded DataLoader batch lacks reader state")
        self.plan.record_worker_progress(worker_state)
        self.plan.finish_batch(int(worker_state["worker_id"]))
        return batch

    def _start_loader(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise DataContractError("preencoded Wan training requires torch") from exc

        initial_state = self.plan.state_dict()
        rows = self.rows
        config = self.config
        topology = self.topology
        first_worker = int(initial_state["next_worker"])
        worker_states = tuple(initial_state["workers"])

        class _LogicalDPPreencodedDataset(torch.utils.data.IterableDataset):
            def __iter__(self) -> Iterator[Mapping[str, Any]]:
                info = torch.utils.data.get_worker_info()
                physical_worker = 0 if info is None else int(info.id)
                worker_count = 1 if info is None else int(info.num_workers)
                logical_worker = (physical_worker + first_worker) % worker_count
                yield from _iter_stateful_preencoded_batches(
                    config,
                    topology,
                    rows=rows,
                    worker_id=logical_worker,
                    num_workers=worker_count,
                    worker_state=worker_states[logical_worker],
                )

        options: dict[str, Any] = {
            "dataset": _LogicalDPPreencodedDataset(),
            "batch_size": None,
            "num_workers": self.loader_workers,
            "pin_memory": False,
            "generator": _data_loader_generator(torch, self.config, self.topology),
        }
        if self.loader_workers:
            options.update(
                prefetch_factor=int(self.config["data"].get("prefetch_factor", 2)),
                persistent_workers=True,
            )
        self._loader = torch.utils.data.DataLoader(**options)
        self._iterator = iter(self._loader)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "solarwm.wan22.preencoded-reader-state.v1",
            "micro_batch_size": self.micro_batch_size,
            "plan": self.plan.state_dict(),
        }

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Quiesce speculative workers so restart and continuation share one frontier."""

        self._shutdown_loader()
        return self.state_dict()

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if self._iterator is not None:
            raise DataContractError("cannot restore Wan preencoded reader after iteration started")
        if value.get("schema") != "solarwm.wan22.preencoded-reader-state.v1":
            raise DataContractError("unsupported Wan preencoded reader checkpoint schema")
        if int(value.get("micro_batch_size", 0)) != self.micro_batch_size:
            raise DataContractError("Wan preencoded reader microbatch changed at resume")
        plan = value.get("plan")
        if not isinstance(plan, Mapping):
            raise DataContractError("Wan preencoded reader checkpoint lacks plan state")
        self.plan.load_state_dict(plan)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shutdown_loader()

    def _shutdown_loader(self) -> None:
        shutdown = getattr(self._iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        self._iterator = None
        self._loader = None


def build_preencoded_dataloader(config: Mapping[str, Any], topology: Any) -> Any:
    return StatefulPreencodedWanReader(config, topology)


__all__ = [
    "I2V_A14B_153F_VERSION",
    "SIX_WINDOW_153F_DATASETS",
    "TI2V_5B_81F_VERSION",
    "TI2V_5B_153F_VERSION",
    "TI2V_5B_720P_153F_VERSION",
    "WINDOW_HASH_NAMESPACE_81F",
    "WINDOW_HASH_NAMESPACE_153F",
    "DecodedWanPreencodedSample",
    "StatefulPreencodedWanReader",
    "build_preencoded_dataloader",
    "collate_preencoded_samples",
    "decode_preencoded_sample",
    "expected_81f_window_start",
    "expected_153f_window_start",
    "iter_preencoded_batches",
]
