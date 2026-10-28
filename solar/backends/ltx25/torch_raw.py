"""Index-authoritative raw WDS decoding for online training and preencoding."""

from __future__ import annotations

import io
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from solar.data import (
    CanonicalSampler,
    RawSampleReader,
    ReaderIdentity,
    SamplingConfig,
    ShardPrefetcher,
    TarShardReader,
    build_shard_prefetcher,
    read_index,
    resolve_index_path,
)
from solar.data.index import IndexRow
from solar.data.sampling import SamplePlan
from solar.errors import BackendContractError
from solar.inference import InferenceCase
from solar.runtime import Topology
from solar.runtime.serialization import canonical_json_bytes

from .artifact import TensorArtifact
from .codec import LTX25OnlineCodec, RawSample, encode_online
from .geometry import STABLE_GEOMETRY, cover_resize
from .inference import InferencePlan
from .torch_data import (
    TorchBatch,
    WorkerStream,
    verified_resolver_from_config,
)


def _source_video(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    value = manifest.get("video", {})
    return value if isinstance(value, Mapping) else {}


def _source_integer(values: Mapping[str, Any], manifest: Mapping[str, Any], name: str) -> int:
    video = _source_video(manifest)
    try:
        result = int(values.get(name) or video[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError(f"raw LTX row lacks source {name}") from exc
    if result < 1:
        raise BackendContractError(f"raw LTX source {name} must be positive")
    return result


def _source_fps(values: Mapping[str, Any], manifest: Mapping[str, Any]) -> float:
    video = _source_video(manifest)
    try:
        result = float(values.get("fps") or video["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError("raw LTX row lacks source FPS provenance") from exc
    if not np.isfinite(result) or result <= 0:
        raise BackendContractError("raw LTX source FPS must be finite and positive")
    return result


def _decode_video(
    payload: bytes,
    indices: tuple[int, ...],
    *,
    source_height: int,
    source_width: int,
) -> torch.Tensor:
    import av
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tv_functional

    selected = set(indices)
    arrays = []
    with av.open(io.BytesIO(payload), mode="r") as container:
        if not container.streams.video:
            raise BackendContractError("raw LTX member has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 2
        for ordinal, frame in enumerate(container.decode(video=0)):
            if ordinal in selected:
                array = frame.to_ndarray(format="rgb24")
                if tuple(array.shape) != (source_height, source_width, 3):
                    raise BackendContractError(
                        "decoded LTX frame geometry differs from the frozen index"
                    )
                arrays.append(array)
            if ordinal >= indices[-1]:
                break
    if len(arrays) != STABLE_GEOMETRY.pixel_frames:
        raise BackendContractError(
            f"raw LTX decode returned {len(arrays)} selected frames, expected 153"
        )
    transform = cover_resize(source_height, source_width)
    resized = []
    for offset in range(0, len(arrays), 8):
        value = (
            torch.from_numpy(np.stack(arrays[offset : offset + 8]))
            .permute(0, 3, 1, 2)
            .to(torch.float32)
            .div_(255.0)
        )
        if (transform.resized_height, transform.resized_width) != (
            source_height,
            source_width,
        ):
            value = tv_functional.resize(
                value,
                [transform.resized_height, transform.resized_width],
                interpolation=InterpolationMode.BICUBIC,
            )
        value.clamp_(0.0, 1.0)
        resized.append(
            value[
                :,
                :,
                transform.crop_top : transform.crop_top + STABLE_GEOMETRY.height,
                transform.crop_left : transform.crop_left + STABLE_GEOMETRY.width,
            ].contiguous()
        )
    result = torch.cat(resized, dim=0)
    if tuple(result.shape) != (153, 3, 512, 768):
        raise BackendContractError("resized LTX video geometry differs")
    return result


def _invert_se3(value: np.ndarray) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float32)
    if matrices.ndim != 3 or tuple(matrices.shape[1:]) != (4, 4):
        raise BackendContractError("raw LTX camera poses must be [N,4,4]")
    rotation = np.swapaxes(matrices[:, :3, :3], -1, -2)
    output = np.zeros_like(matrices)
    output[:, :3, :3] = rotation
    output[:, :3, 3] = -np.einsum("...ij,...j->...i", rotation, matrices[:, :3, 3])
    output[:, 3, 3] = 1.0
    return output


def _manifest_convention(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("camera", {})
    if not isinstance(value, Mapping):
        return ""
    return str(value.get("convention") or "").strip().lower()


def _camera_poses(payload: bytes, manifest: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        supported = set(archive.files) & {
            "c2w",
            "camera_c2w",
            "w2c",
            "camera_w2c",
            "relative_w2c",
            "viewmats",
            "extrinsic",
            "extrinsics",
        }
        if len(supported) != 1:
            raise BackendContractError("raw LTX camera sidecar must expose exactly one pose array")
        key = next(iter(supported))
        poses = np.asarray(archive[key], dtype=np.float32).copy()
    if key in {"c2w", "camera_c2w"}:
        return poses, "absolute_c2w"
    if key in {"w2c", "camera_w2c"}:
        return _invert_se3(poses), "absolute_c2w"
    if key in {"relative_w2c", "viewmats"}:
        return poses, "relative_w2c"
    if "c2w" not in _manifest_convention(manifest):
        raise BackendContractError(f"ambiguous raw LTX camera key {key!r} lacks a C2W declaration")
    return poses, "absolute_c2w"


def _intrinsics(payload: bytes) -> np.ndarray:
    loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            keys = [key for key in ("camera_K", "K", "intrinsics") if key in loaded.files]
            if len(keys) != 1:
                raise BackendContractError("raw LTX intrinsics NPZ must expose exactly one K array")
            value = np.asarray(loaded[keys[0]])
        finally:
            loaded.close()
    else:
        value = np.asarray(loaded)
    if value.shape == (4,):
        value = value.reshape(1, 4)
    if value.ndim == 2 and value.shape[1] == 4:
        matrices = np.zeros((value.shape[0], 3, 3), dtype=np.float64)
        matrices[:, 0, 0] = value[:, 0]
        matrices[:, 1, 1] = value[:, 1]
        matrices[:, 0, 2] = value[:, 2]
        matrices[:, 1, 2] = value[:, 3]
        matrices[:, 2, 2] = 1.0
        return matrices
    if value.shape == (3, 3) or (value.ndim == 3 and tuple(value.shape[1:]) == (3, 3)):
        return np.asarray(value, dtype=np.float64)
    raise BackendContractError(f"unsupported raw LTX intrinsics shape {value.shape}")


def _select_rows(
    value: np.ndarray,
    indices: tuple[int, ...],
    *,
    trailing_shape: tuple[int, ...],
    name: str,
    allow_static: bool,
) -> np.ndarray:
    array = np.asarray(value)
    if allow_static and tuple(array.shape) == trailing_shape:
        return np.broadcast_to(array, (STABLE_GEOMETRY.pixel_frames, *trailing_shape)).copy()
    if array.ndim != len(trailing_shape) + 1 or tuple(array.shape[1:]) != trailing_shape:
        raise BackendContractError(f"raw LTX {name} has invalid shape {array.shape}")
    if array.shape[0] > indices[-1]:
        return array[np.asarray(indices, dtype=np.int64)].copy()
    if array.shape[0] == STABLE_GEOMETRY.pixel_frames and indices == tuple(range(153)):
        return array.copy()
    if allow_static and array.shape[0] == 1:
        return np.broadcast_to(array, (153, *trailing_shape)).copy()
    raise BackendContractError(
        f"raw LTX {name} rows cannot bind to source ordinals {indices[0]}..{indices[-1]}"
    )


@dataclass(frozen=True)
class RawWindow:
    sample: RawSample
    plan_fingerprint: str


def normalize_training_window(row: IndexRow) -> IndexRow:
    """Normalize a generic raw recipe to the configured first-frame policy."""

    values = dict(row.values)
    has_start = "start_frame" in values
    has_indices = "source_frame_indices" in values
    if has_start != has_indices:
        raise BackendContractError(f"raw LTX row {row.sample_id!r} has a partial source window")
    if not has_start:
        values["start_frame"] = 0
        values["source_frame_indices"] = list(range(STABLE_GEOMETRY.pixel_frames))
    return IndexRow.from_mapping(row.ordinal, values)


class RawIndexedStream:
    """Synchronous worker merge whose plans are independent of shard transport."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        logical_dp: bool,
        initialize_readers: bool = True,
        physical_once: bool = False,
    ) -> None:
        from solar.data.sampling import plan_fingerprint

        self._plan_fingerprint = plan_fingerprint
        data = config["data"]
        if not isinstance(data, Mapping):
            raise BackendContractError("raw LTX data config must be a mapping")
        self.index_path = resolve_index_path(data, "index")
        self.rows = read_index(self.index_path)
        self.rows = tuple(normalize_training_window(row) for row in self.rows)
        topology: Topology | None = None
        if initialize_readers:
            worker_count = int(data.get("num_workers", 1) or 1)
            if logical_dp:
                distributed = config["distributed"]
                topology = Topology.from_environ(int(distributed["sequence_parallel_size"]))
                identities = tuple(
                    ReaderIdentity.from_topology(
                        topology,
                        worker_id=worker,
                        num_workers=worker_count,
                    )
                    for worker in range(worker_count)
                )
            else:
                import os

                rank = int(os.environ.get("RANK", "0"))
                world = int(os.environ.get("WORLD_SIZE", "1"))
                identities = tuple(
                    ReaderIdentity(
                        rank=rank,
                        world_size=world,
                        worker_id=worker,
                        num_workers=worker_count,
                        local_rank=int(os.environ.get("LOCAL_RANK", str(rank))),
                        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", str(world))),
                    )
                    for worker in range(worker_count)
                )
            sampling = SamplingConfig(
                seed=int(data.get("seed", 42)),
                pixel_frames=STABLE_GEOMETRY.pixel_frames,
                random_start=False,
                fixed_start_from_index=True,
                shuffle_buffer=int(data.get("shuffle_buffer", 32)),
                partition_mode=str(data.get("partition_mode", "global_occurrence")),
            )
            sampling_rows = (
                tuple(
                    IndexRow.from_mapping(
                        row.ordinal,
                        {**row.values, "epoch_repeats": 1},
                    )
                    for row in self.rows
                )
                if physical_once
                else self.rows
            )
            self.streams = tuple(
                WorkerStream(CanonicalSampler(sampling_rows, sampling, identity))
                for identity in identities
            )
        else:
            self.streams = ()
        self.cursor = 0
        self._data_config = dict(data)
        self._topology = topology
        self._resolver = verified_resolver_from_config(data)
        self.shards = TarShardReader(
            self._resolver,
            max_open=int(data.get("tar_cache_size", 2)),
        )
        self.reader = RawSampleReader(
            self.rows,
            self.shards,
            member_fields=("video_member", "camera_member", "intrinsics_member"),
        )
        self._shard_prefetcher = self._build_shard_prefetcher()

    def _build_shard_prefetcher(self) -> ShardPrefetcher | None:
        if self._topology is None or not self.streams:
            return None
        return build_shard_prefetcher(
            self._data_config,
            rows=self.rows,
            sampler=self.streams[0].sampler,
            resolver=self._resolver,
            node_leader=int(self._topology.local_rank) == 0,
        )

    def _materialize(self, plan: Any) -> RawWindow:
        if self._shard_prefetcher is not None:
            self._shard_prefetcher.prepare(plan)
        raw = self.reader.materialize(plan)
        values = raw.index_values
        height = _source_integer(values, raw.manifest, "height")
        width = _source_integer(values, raw.manifest, "width")
        indices = tuple(int(item) for item in plan.source_frame_indices)
        frozen_indices = tuple(int(item) for item in values["source_frame_indices"])
        if indices != frozen_indices:
            raise BackendContractError("raw LTX sampler differs from frozen source indices")
        frames = _decode_video(
            raw.members["video_member"],
            indices,
            source_height=height,
            source_width=width,
        )
        poses, convention = _camera_poses(raw.members["camera_member"], raw.manifest)
        selected_poses = _select_rows(
            poses,
            indices,
            trailing_shape=(4, 4),
            name="camera poses",
            allow_static=False,
        )
        selected_intrinsics = _select_rows(
            _intrinsics(raw.members["intrinsics_member"]),
            indices,
            trailing_shape=(3, 3),
            name="camera intrinsics",
            allow_static=True,
        )
        return RawWindow(
            RawSample(
                sample_id=plan.sample_id,
                key=plan.key,
                frames=frames,
                caption=raw.caption,
                source_indices=np.asarray(indices, dtype=np.int64),
                camera_poses=selected_poses,
                camera_intrinsics=selected_intrinsics,
                camera_convention=convention,
                source_height=height,
                source_width=width,
                source_fps=_source_fps(values, raw.manifest),
            ),
            self._plan_fingerprint((plan,)),
        )

    def materialize_row(self, row: IndexRow) -> RawWindow:
        """Materialize one frozen index row without advancing a training reader."""

        try:
            start = int(row.values["start_frame"])
            indices = tuple(int(item) for item in row.values["source_frame_indices"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(
                f"raw LTX row {row.sample_id!r} lacks fixed source indices"
            ) from exc
        if len(indices) != STABLE_GEOMETRY.pixel_frames or indices != tuple(
            range(start, start + STABLE_GEOMETRY.pixel_frames)
        ):
            raise BackendContractError(
                f"raw LTX row {row.sample_id!r} source indices are not contiguous"
            )
        return self._materialize(
            SamplePlan(
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
        )

    def next(self) -> RawWindow:
        if not self.streams:
            raise BackendContractError("random-access LTX stream has no sequential readers")
        stream = self.streams[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.streams)
        return self._materialize(stream.next())

    def iter_epoch_zero(self) -> Iterator[RawSample]:
        if not self.streams:
            raise BackendContractError("random-access LTX stream has no epoch reader")
        while True:
            emitted = False
            for stream in self.streams:
                if stream.epoch == 0 and stream.index < len(stream.plans):
                    emitted = True
                    yield self._materialize(stream.next()).sample
            if not emitted:
                return

    def state_dict(self) -> dict[str, Any]:
        if not self.streams:
            raise BackendContractError("random-access LTX stream has no reader state")
        return {
            "schema": "solarwm.ltx25.raw-reader-state.v1",
            "index_path": str(self.index_path),
            "cursor": self.cursor,
            "workers": [stream.state_dict() for stream in self.streams],
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not self.streams:
            raise BackendContractError("random-access LTX stream has no reader state")
        if value.get("schema") != "solarwm.ltx25.raw-reader-state.v1" or str(
            value.get("index_path")
        ) != str(self.index_path):
            raise BackendContractError("raw LTX reader checkpoint identity differs")
        workers = value.get("workers")
        if not isinstance(workers, list) or len(workers) != len(self.streams):
            raise BackendContractError("raw LTX reader worker topology differs")
        cursor = int(value.get("cursor", -1))
        if not 0 <= cursor < len(self.streams):
            raise BackendContractError("raw LTX reader cursor is invalid")
        for stream, state in zip(self.streams, workers, strict=True):
            if not isinstance(state, Mapping):
                raise BackendContractError("raw LTX reader worker state is invalid")
            stream.load_state_dict(state)
        self.cursor = cursor
        if self._shard_prefetcher is not None:
            self._shard_prefetcher.close()
        self._shard_prefetcher = self._build_shard_prefetcher()

    def close(self) -> None:
        if self._shard_prefetcher is not None:
            self._shard_prefetcher.close()
        self.shards.close()


def _artifact_tensor(value: TensorArtifact) -> torch.Tensor:
    dtype = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "I64": torch.int64,
    }[value.dtype]
    return torch.frombuffer(bytearray(value.data), dtype=dtype).reshape(value.shape).clone()


def _encoded_batch(encoded: Any, *, plan_fingerprint: str) -> TorchBatch:
    tensors = encoded.tensors
    return TorchBatch(
        sample_id=encoded.sample_id,
        start_frame=encoded.start_frame,
        plan_fingerprint=plan_fingerprint,
        video_latent=_artifact_tensor(tensors["video_latent"]).unsqueeze(0),
        first_frame_latent=_artifact_tensor(tensors["first_frame_latent"]).unsqueeze(0),
        video_prompt_embeds=_artifact_tensor(tensors["video_prompt_embeds"]).unsqueeze(0),
        prompt_attention_mask=_artifact_tensor(tensors["prompt_attention_mask"]).unsqueeze(0),
        relative_w2c=_artifact_tensor(tensors["relative_w2c"]).unsqueeze(0),
        camera_k=_artifact_tensor(tensors["camera_K"]).unsqueeze(0),
        source_indices=_artifact_tensor(tensors["source_indices"]).unsqueeze(0),
        camera_source_indices=_artifact_tensor(tensors["camera_source_indices"]).unsqueeze(0),
    )


def _row_control_fingerprint(row: IndexRow) -> str:
    """Bind a case to immutable index controls and its source shard generation."""

    import hashlib

    values = row.values
    control = {
        "schema": "solarwm.ltx25.raw-case.v1",
        "sample_id": row.sample_id,
        "key": row.key,
        "shard": row.shard,
        "shard_generation": str(values.get("shard_generation") or ""),
        "shard_size": int(values.get("shard_size") or 0),
        "shard_md5_b64": str(values.get("shard_md5_b64") or ""),
        "shard_digest": str(values.get("shard_digest") or ""),
        "video_member": str(values.get("video_member") or ""),
        "camera_member": str(values.get("camera_member") or ""),
        "intrinsics_member": str(values.get("intrinsics_member") or ""),
        "start_frame": int(values["start_frame"]),
        "source_frame_indices": [int(item) for item in values["source_frame_indices"]],
    }
    return hashlib.blake2s(canonical_json_bytes(control)).hexdigest()


class RawOnlineBatchSource:
    def __init__(self, config: Mapping[str, Any], codec: LTX25OnlineCodec) -> None:
        self.stream = RawIndexedStream(config, logical_dp=True)
        self.codec = codec
        model = config["model"]
        if not isinstance(model, Mapping):
            raise BackendContractError("LTX model config must be a mapping")
        self.checkpoint_identity = f"path:{model['checkpoint_path']}"

    def next(self) -> TorchBatch:
        window = self.stream.next()
        encoded = encode_online(window.sample, self.codec)
        return _encoded_batch(
            encoded,
            plan_fingerprint=window.plan_fingerprint,
        )

    @property
    def checkpoint_id(self) -> str:
        return self.checkpoint_identity

    def state_dict(self) -> dict[str, Any]:
        return self.stream.state_dict()

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        self.stream.load_state_dict(value)

    def close(self) -> None:
        self.stream.close()


class RawInferenceSource:
    """Fixed random access over the raw index using a shared official codec."""

    def __init__(self, config: Mapping[str, Any], codec: LTX25OnlineCodec) -> None:
        self.stream = RawIndexedStream(
            config,
            logical_dp=True,
            initialize_readers=False,
        )
        self.codec = codec
        self.rows = self.stream.rows
        self._by_id = {row.sample_id: row for row in self.rows}
        self._fingerprints = {row.sample_id: _row_control_fingerprint(row) for row in self.rows}
        self._encoded: dict[str, TorchBatch] = {}

    def get(self, sample_id: str) -> TorchBatch:
        cached = self._encoded.get(sample_id)
        if cached is not None:
            return cached
        try:
            row = self._by_id[sample_id]
        except KeyError as exc:
            raise BackendContractError(
                f"inference sample {sample_id!r} is outside the fixed raw index"
            ) from exc
        window = self.stream.materialize_row(row)
        encoded = encode_online(window.sample, self.codec)
        batch = _encoded_batch(
            encoded,
            plan_fingerprint=window.plan_fingerprint,
        )
        self._encoded[sample_id] = batch
        return batch

    def case_fingerprint(self, sample_id: str, batch: TorchBatch | None = None) -> str:
        del batch
        try:
            return self._fingerprints[sample_id]
        except KeyError as exc:
            raise BackendContractError(
                f"inference sample {sample_id!r} is outside the fixed raw index"
            ) from exc

    def case_for_row(
        self,
        row: IndexRow,
        *,
        slot: int,
        plan: InferencePlan,
        camera_translation_transform: str,
    ) -> InferenceCase:
        try:
            start = int(row.values["start_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(
                f"raw inference row {row.sample_id!r} lacks start_frame"
            ) from exc
        return InferenceCase(
            slot=slot,
            sample_id=row.sample_id,
            prompt=str(row.values.get("caption") or ""),
            start_frame=start,
            noise_seed=plan.spec.seed + slot,
            camera_fingerprint=self.case_fingerprint(row.sample_id),
            metadata={
                "key": row.key,
                "raw_control_fingerprint": self.case_fingerprint(row.sample_id),
                "shard": row.shard,
                "source_pixel_frames": STABLE_GEOMETRY.pixel_frames,
                "output_pixel_frames": STABLE_GEOMETRY.pixel_frames,
                "train_latent_frames": STABLE_GEOMETRY.latent_frames,
                "rollout_latent_frames": STABLE_GEOMETRY.latent_frames,
                "generation_mode": "bidirectional",
                "sample_solver": "stg-euler",
                "num_inference_steps": plan.spec.num_inference_steps,
                "camera_translation_transform": camera_translation_transform,
                "artifact_valid": True,
            },
        )

    def close(self) -> None:
        self._encoded.clear()
        self.stream.close()


__all__ = [
    "RawIndexedStream",
    "RawInferenceSource",
    "RawOnlineBatchSource",
    "RawWindow",
    "normalize_training_window",
]
