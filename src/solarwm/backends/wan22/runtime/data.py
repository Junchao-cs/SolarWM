"""Transport-neutral raw Wan sample decoding and logical-DP iteration."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from solarwm.data.archive import RawSample, RawSampleReader, TarShardReader
from solarwm.data.camera import (
    CameraGuards,
    absolute_c2w,
    camera_audit_prefix_frames,
    invert_se3,
    load_camera_npz,
)
from solarwm.data.index import IndexRow, read_index, resolve_index_path
from solarwm.data.prefetch import build_shard_prefetcher
from solarwm.data.sampling import (
    CanonicalSampler,
    ReaderIdentity,
    SamplePlan,
    SamplingConfig,
    plan_fingerprint,
)
from solarwm.data.transport import resolver_from_config
from solarwm.errors import DataContractError
from solarwm.runtime.distributed import DataReadiness
from solarwm.runtime.safe_state import decode_python_rng_state, encode_python_rng_state

FX_NORM = 969.6969696969696 / (960.0 * 2)
FY_NORM = 969.6969696969696 / (540.0 * 2)
CAMERA_CONVENTION = "authoritative_source_c2w_no_axis_flip"
_CAMERA_ARRAY_KEYS = frozenset({"c2w", "w2c", "vipe_c2w", "vipe_w2c", "relative_w2c"})
_SKIPPABLE_CAMERA_GUARD_ERRORS = frozenset(
    {
        "relative translation exceeds configured guard",
        "camera matrix exceeds configured absolute guard",
    }
)


def _is_skippable_camera_guard_error(exc: DataContractError) -> bool:
    return str(exc) in _SKIPPABLE_CAMERA_GUARD_ERRORS


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
class DecodedWanSample:
    sample_id: str
    key: str
    start_frame: int
    source_frame_indices: tuple[int, ...]
    caption: str
    pixels: Any
    camera: Mapping[str, Any]


@dataclass
class _WanWorkerCursor:
    sampler: CanonicalSampler
    epoch: int
    cursor: int
    healthy_samples: int
    epoch_start_rng_state: object
    plans: tuple[SamplePlan, ...]
    plan_digest: str


_READER_STATE_KEY = "__solarwm_reader_state__"


def _data_loader_generator(torch: Any, config: Mapping[str, Any], topology: Any) -> Any:
    """Return a deterministic loader-only generator without touching training RNG."""

    raw_rank = int(getattr(topology, "raw_rank", getattr(topology, "rank", 0)))
    seed = (int(config["data"]["seed"]) + 1_000_003 * raw_rank) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _worker_state_payload(worker: _WanWorkerCursor) -> dict[str, Any]:
    return {
        "worker_id": worker.sampler.identity.worker_id,
        "epoch": worker.epoch,
        "cursor": worker.cursor,
        "healthy_samples": worker.healthy_samples,
        "epoch_start_rng_state": encode_python_rng_state(worker.epoch_start_rng_state),
        "plan_fingerprint": worker.plan_digest,
    }


def _worker_progress_payload(worker: _WanWorkerCursor) -> dict[str, Any]:
    return {
        "worker_id": worker.sampler.identity.worker_id,
        "epoch": worker.epoch,
        "cursor": worker.cursor,
        "healthy_samples": worker.healthy_samples,
        "plan_fingerprint": worker.plan_digest,
    }


def _restore_worker_cursor(
    rows: Sequence[IndexRow],
    sampling: SamplingConfig,
    topology: Any,
    *,
    worker_id: int,
    num_workers: int,
    state: Mapping[str, Any] | None = None,
) -> _WanWorkerCursor:
    identity = ReaderIdentity.from_topology(
        topology,
        worker_id=worker_id,
        num_workers=num_workers,
    )
    sampler = CanonicalSampler(rows, sampling, identity)
    if state is None:
        epoch = 0
        cursor = 0
        healthy_samples = 0
        epoch_start = sampler._rng.getstate()
    else:
        if int(state.get("worker_id", -1)) != worker_id:
            raise DataContractError("Wan reader checkpoint worker order differs")
        epoch = int(state.get("epoch", -1))
        cursor = int(state.get("cursor", -1))
        healthy_samples = int(state.get("healthy_samples", -1))
        if epoch < 0 or cursor < 0 or healthy_samples < 0:
            raise DataContractError("Wan reader checkpoint cursor is invalid")
        epoch_start = decode_python_rng_state(state.get("epoch_start_rng_state"))
        sampler._rng.setstate(epoch_start)
    plans = tuple(sampler.iter_epoch(epoch))
    if not plans:
        raise DataContractError(f"Wan logical reader worker {worker_id} owns no sample plans")
    plans_digest = plan_fingerprint(plans)
    if state is not None and state.get("plan_fingerprint") != plans_digest:
        raise DataContractError("Wan reader sample plan changed at resume")
    if cursor > len(plans):
        raise DataContractError("Wan reader checkpoint exceeds its epoch")
    if healthy_samples > cursor:
        raise DataContractError("Wan reader healthy sample count exceeds its cursor")
    return _WanWorkerCursor(
        sampler,
        epoch,
        cursor,
        healthy_samples,
        epoch_start,
        plans,
        plans_digest,
    )


def _advance_worker_epoch(worker: _WanWorkerCursor) -> None:
    worker.epoch += 1
    worker.cursor = 0
    worker.healthy_samples = 0
    worker.epoch_start_rng_state = worker.sampler._rng.getstate()
    worker.plans = tuple(worker.sampler.iter_epoch(worker.epoch))
    if not worker.plans:
        raise DataContractError(
            f"Wan logical reader worker {worker.sampler.identity.worker_id} "
            f"owns no sample plans in epoch {worker.epoch}"
        )
    worker.plan_digest = plan_fingerprint(worker.plans)


class WanPlanMultiplexer:
    """Replay DataLoader's ordered logical-worker stream with checkpointable cursors."""

    def __init__(
        self,
        rows: Sequence[IndexRow],
        sampling: SamplingConfig,
        topology: Any,
        *,
        num_workers: int,
    ) -> None:
        self.rows = tuple(rows)
        self.sampling = sampling
        self.topology = topology
        self.num_workers = int(num_workers)
        if self.num_workers < 1:
            raise DataContractError("Wan reader num_workers must be positive")
        workers: list[_WanWorkerCursor] = []
        for worker_id in range(self.num_workers):
            workers.append(
                _restore_worker_cursor(
                    self.rows,
                    self.sampling,
                    topology,
                    worker_id=worker_id,
                    num_workers=self.num_workers,
                )
            )
        self.workers = workers
        self.next_worker = 0

    def _advance_epoch(self, worker: _WanWorkerCursor) -> None:
        _advance_worker_epoch(worker)

    def next_plan(self, worker_id: int) -> SamplePlan:
        worker = self.workers[int(worker_id)]
        if worker.cursor == len(worker.plans):
            self._advance_epoch(worker)
        plan = worker.plans[worker.cursor]
        worker.cursor += 1
        return plan

    def finish_batch(self, worker_id: int) -> None:
        if int(worker_id) != self.next_worker:
            raise DataContractError("Wan reader completed a batch out of worker order")
        self.next_worker = (self.next_worker + 1) % self.num_workers

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "solarwm.wan22.plan-multiplexer.v1",
            "num_workers": self.num_workers,
            "next_worker": self.next_worker,
            "workers": [_worker_state_payload(worker) for worker in self.workers],
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema") != "solarwm.wan22.plan-multiplexer.v1":
            raise DataContractError("unsupported Wan reader plan checkpoint schema")
        if int(value.get("num_workers", 0)) != self.num_workers:
            raise DataContractError("Wan reader worker count changed at resume")
        raw_workers = value.get("workers")
        if not isinstance(raw_workers, list) or len(raw_workers) != self.num_workers:
            raise DataContractError("Wan reader checkpoint worker states differ")
        restored: list[_WanWorkerCursor] = []
        for worker_id, raw in enumerate(raw_workers):
            if not isinstance(raw, Mapping) or int(raw.get("worker_id", -1)) != worker_id:
                raise DataContractError("Wan reader checkpoint worker order differs")
            restored.append(
                _restore_worker_cursor(
                    self.rows,
                    self.sampling,
                    self.topology,
                    worker_id=worker_id,
                    num_workers=self.num_workers,
                    state=raw,
                )
            )
        next_worker = int(value.get("next_worker", -1))
        if not 0 <= next_worker < self.num_workers:
            raise DataContractError("Wan reader next worker is outside its worker count")
        self.workers = restored
        self.next_worker = next_worker

    def record_worker_progress(self, value: Mapping[str, Any]) -> None:
        worker_id = int(value.get("worker_id", -1))
        if worker_id != self.next_worker:
            raise DataContractError("Wan reader returned a batch out of worker order")
        worker = self.workers[worker_id]
        epoch = int(value.get("epoch", -1))
        cursor = int(value.get("cursor", -1))
        healthy_samples = int(value.get("healthy_samples", -1))
        if epoch < worker.epoch:
            raise DataContractError("Wan reader worker checkpoint moved backwards")
        while worker.epoch < epoch:
            _advance_worker_epoch(worker)
        if value.get("plan_fingerprint") != worker.plan_digest:
            raise DataContractError("Wan reader worker sample plan changed")
        if not worker.cursor <= cursor <= len(worker.plans):
            raise DataContractError("Wan reader worker cursor is not monotonic")
        if not worker.healthy_samples <= healthy_samples <= cursor:
            raise DataContractError("Wan reader worker healthy sample count is not monotonic")
        worker.cursor = cursor
        worker.healthy_samples = healthy_samples


def latent_aligned_pixel_indices(pixel_frames: int) -> np.ndarray:
    """Return Wan's causal VAE alignment: 0, 1, 5, 9, ..."""

    if pixel_frames < 1 or (pixel_frames - 1) % 4:
        raise DataContractError("Wan pixel frames must be 1 + 4*(latent_frames-1)")
    latent_frames = 1 + (pixel_frames - 1) // 4
    return np.asarray([0, *(1 + 4 * index for index in range(latent_frames - 1))])


def _relative_c2w_fp32(
    matrices: np.ndarray,
    storage: str,
) -> np.ndarray:
    """Build first-frame-relative C2W with stable FP32 operation order."""

    if storage == "relative_w2c":
        relative = invert_se3(matrices)
        relative[0] = np.eye(4, dtype=np.float32)
        return relative
    c2w = absolute_c2w(matrices, storage)
    w2c = invert_se3(c2w)
    target_c2w = np.eye(4, dtype=np.float32)
    absolute_to_relative = target_c2w @ w2c[0]
    relative = np.empty_like(c2w)
    relative[0] = target_c2w
    for index in range(1, c2w.shape[0]):
        relative[index] = absolute_to_relative @ c2w[index]
    return relative


def _torch_invert_se3_fp32(matrices: Any) -> Any:
    """Invert SE3 matrices with the camera-token FP32 operation order."""

    import torch

    rotation = matrices[..., :3, :3]
    rotation_t = rotation.transpose(-1, -2)
    result = torch.zeros_like(matrices)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        rotation_t,
        matrices[..., :3, 3],
    )
    result[..., 3, 3] = 1.0
    return result


def decode_video(
    payload: bytes,
    frame_indices: Sequence[int],
    *,
    height: int,
    width: int,
) -> Any:
    """Decode selected frames and apply the canonical bilinear-center-crop path."""

    try:
        import decord
        import torch
    except ImportError as exc:
        raise DataContractError("raw Wan decoding requires torch and decord") from exc

    decord.bridge.set_bridge("torch")
    try:
        reader = decord.VideoReader(io.BytesIO(payload), num_threads=1)
        frames = reader.get_batch([int(value) for value in frame_indices])
    except Exception as exc:
        raise DataContractError(f"cannot decode selected video frames: {exc}") from exc
    if frames.dtype != torch.uint8:
        frames = frames.to(torch.uint8)
    return _preprocess_video_frames(frames, height=height, width=width)


def _preprocess_video_frames(frames: Any, *, height: int, width: int) -> Any:
    """Bound full-resolution FP32 temporaries without changing pixel arithmetic."""

    import torch
    import torch.nn.functional as functional

    _, source_height, source_width, _ = frames.shape
    resized_height, resized_width = source_height, source_width
    if (source_height, source_width) != (height, width):
        scale = max(height / source_height, width / source_width)
        resized_height = round(source_height * scale)
        resized_width = round(source_width * scale)
    top = (resized_height - height) // 2
    left = (resized_width - width) // 2
    processed = []
    # Resize is independent per frame. Keep the same layout, FP32 operation
    # order, interpolation and crop, but do not expand a whole 4K clip at once.
    for chunk in frames.split(8, dim=0):
        chunk = chunk.permute(0, 3, 1, 2).contiguous().float() / 255.0
        if (resized_height, resized_width) != (source_height, source_width):
            chunk = functional.interpolate(
                chunk,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            )
        chunk = chunk[:, :, top : top + height, left : left + width]
        processed.append((chunk * 2.0 - 1.0).to(torch.bfloat16))
    return torch.cat(processed, dim=0)


def _camera_array_key(manifest: Mapping[str, Any], configured: str) -> str:
    configured = str(configured).strip()
    if configured not in _CAMERA_ARRAY_KEYS:
        raise DataContractError(
            f"data.camera_array_key must be one of {sorted(_CAMERA_ARRAY_KEYS)}, got {configured!r}"
        )
    camera = manifest.get("camera", {})
    declared = ""
    if isinstance(camera, Mapping):
        declared = str(camera.get("array_key", "")).strip()
    declared = declared or str(manifest.get("camera_array_key", "")).strip()
    if declared and declared not in _CAMERA_ARRAY_KEYS:
        raise DataContractError(f"unsupported manifest camera array key {declared!r}")
    if declared and declared != configured:
        raise DataContractError(
            "manifest camera array key conflicts with data.camera_array_key: "
            f"{declared!r} != {configured!r}"
        )
    return configured


def _validate_camera_manifest(
    manifest: Mapping[str, Any],
    *,
    guards: CameraGuards,
) -> None:
    camera = manifest.get("camera", {})
    if not isinstance(camera, Mapping):
        raise DataContractError("raw Wan manifest.camera must be a mapping")
    if str(camera.get("convention", "")) != CAMERA_CONVENTION:
        raise DataContractError(f"raw Wan camera convention must be {CAMERA_CONVENTION!r}")
    if str(camera.get("dtype", "")) != "float32":
        raise DataContractError("raw Wan manifest camera dtype must be float32")
    if camera.get("finite") is not True:
        raise DataContractError("raw Wan manifest must attest camera finite=true")
    expected_floats = {
        "magnitude_audit_seconds": 10.0,
        "max_camera_abs": guards.max_camera_abs,
        "max_rel_translation": guards.max_rel_translation,
    }
    for field, expected in expected_floats.items():
        try:
            actual = float(camera.get(field))
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"raw Wan manifest camera {field} must be numeric") from exc
        if expected is None or actual != float(expected):
            raise DataContractError(
                f"raw Wan manifest camera {field}={actual} does not match {expected}"
            )
    shape = camera.get("shape")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != 3
        or list(shape[1:]) != [4, 4]
    ):
        raise DataContractError("raw Wan manifest camera shape must be [T,4,4]")
    try:
        if int(shape[0]) < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise DataContractError("raw Wan manifest camera shape[0] must be positive") from exc


def _source_fps(
    index_values: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> float:
    video = manifest.get("video", {})
    raw_manifest = video.get("fps") if isinstance(video, Mapping) else None
    raw_index = index_values.get("fps")
    try:
        index_fps = float(raw_index or 0.0)
        manifest_fps = float(raw_manifest or 0.0)
    except (TypeError, ValueError) as exc:
        raise DataContractError("raw Wan index/manifest fps must be numeric") from exc
    if not np.isfinite(index_fps) or not np.isfinite(manifest_fps):
        raise DataContractError("raw Wan index/manifest fps must be finite")
    if index_fps > 0 and manifest_fps > 0 and index_fps != manifest_fps:
        raise DataContractError(
            f"raw Wan index fps={index_fps} conflicts with manifest fps={manifest_fps}"
        )
    return index_fps or manifest_fps


def build_camera_tokens(
    payload: bytes,
    source_frame_indices: Sequence[int],
    manifest: Mapping[str, Any],
    *,
    source_fps: float,
    output_fps: float,
    frame_sequence_length: int,
    guards: CameraGuards,
    configured_array_key: str,
    manifest_guards: CameraGuards | None = None,
) -> Mapping[str, Any]:
    """Select, rebase, guard, and spatially expand authoritative cameras."""

    import torch

    # The manifest records the guards used when the sample was published.
    # Validation may override runtime guards without changing that record.
    _validate_camera_manifest(manifest, guards=manifest_guards or guards)
    matrices, storage = load_camera_npz(
        payload,
        _camera_array_key(manifest, configured_array_key),
    )
    matrices = np.asarray(matrices, dtype=np.float32)
    declared_shape = tuple(int(value) for value in manifest["camera"]["shape"])
    if matrices.shape != declared_shape:
        raise DataContractError(
            "camera NPZ shape does not match manifest.camera.shape: "
            f"{matrices.shape} != {declared_shape}"
        )
    source_indices = np.asarray(source_frame_indices, dtype=np.int64)
    if source_indices.min(initial=0) < 0 or source_indices.max(initial=0) >= len(matrices):
        raise DataContractError("selected camera frames are outside the source trajectory")
    latent_pixel = latent_aligned_pixel_indices(len(source_indices))
    selected = matrices[source_indices[latent_pixel]]
    # Preserve the camera-token FP32 operation order. This
    # deliberately differs from the shared canonical relative-W2C shortcut.
    relative_c2w = _relative_c2w_fp32(selected, storage)
    audit_pixels = camera_audit_prefix_frames(
        len(source_indices), output_fps if output_fps > 0 else source_fps
    )
    audit_latents = int(np.searchsorted(latent_pixel, audit_pixels, side="left"))
    audit_latents = max(1, min(audit_latents, len(relative_c2w)))
    relative_c2w = guards.apply(relative_c2w, audit_rows=audit_latents)

    viewmats = _torch_invert_se3_fp32(torch.as_tensor(relative_c2w, dtype=torch.float32))
    viewmats = (
        viewmats.unsqueeze(1)
        .expand(-1, int(frame_sequence_length), -1, -1)
        .reshape(-1, 4, 4)
        .contiguous()
    )
    intrinsics = torch.zeros((1, 3, 3), dtype=torch.float32)
    intrinsics[:, 0, 0] = FX_NORM
    intrinsics[:, 1, 1] = FY_NORM
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    intrinsics[:, 2, 2] = 1.0
    intrinsics = intrinsics.expand(viewmats.shape[0], -1, -1).contiguous()
    return {"viewmats": viewmats, "K": intrinsics}


def decode_raw_sample(sample: RawSample, config: Mapping[str, Any]) -> DecodedWanSample:
    data = config["data"]
    model = config["model"]
    source_fps = _source_fps(sample.index_values, sample.manifest)
    pixels = decode_video(
        sample.members["video_member"],
        sample.plan.source_frame_indices,
        height=int(data["height"]),
        width=int(data["width"]),
    )
    camera = build_camera_tokens(
        sample.members["camera_member"],
        sample.plan.source_frame_indices,
        sample.manifest,
        source_fps=source_fps,
        output_fps=float(data.get("fps", 16.0)),
        frame_sequence_length=int(model["frame_sequence_length"]),
        guards=CameraGuards(
            max_rel_translation=float(data["max_rel_translation"]),
            max_camera_abs=float(data["max_camera_abs"]),
        ),
        configured_array_key=str(data["camera_array_key"]),
    )
    return DecodedWanSample(
        sample_id=sample.plan.sample_id,
        key=sample.plan.key,
        start_frame=sample.plan.start_frame,
        source_frame_indices=sample.plan.source_frame_indices,
        caption=sample.caption,
        pixels=pixels,
        camera=camera,
    )


def _materialize_raw_sample(
    reader: RawSampleReader,
    plan: SamplePlan,
    config: Mapping[str, Any],
    *,
    prepare: Callable[[SamplePlan], None] | None = None,
) -> DecodedWanSample | None:
    """Materialize one legacy-WDS sample, skipping sample-local failures."""

    try:
        if prepare is not None:
            prepare(plan)
        return decode_raw_sample(reader.materialize(plan), config)
    except Exception as exc:
        # Match the pre-refactor WDS reader: shard/member, manifest, video,
        # and camera failures reject only this sample and advance the stream.
        print(
            f"[wds][rank{plan.reader_rank}] skip {plan.key}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def collate_raw_samples(samples: Sequence[DecodedWanSample]) -> Mapping[str, Any]:
    if not samples:
        raise DataContractError("cannot collate an empty Wan batch")
    import torch

    return {
        "sample_ids": tuple(sample.sample_id for sample in samples),
        "keys": tuple(sample.key for sample in samples),
        "start_frames": tuple(sample.start_frame for sample in samples),
        "source_frame_indices": tuple(sample.source_frame_indices for sample in samples),
        "prompts": tuple(sample.caption for sample in samples),
        "pixels": torch.stack([sample.pixels for sample in samples]),
        "camera": {
            "viewmats": torch.stack([sample.camera["viewmats"] for sample in samples]),
            "K": torch.stack([sample.camera["K"] for sample in samples]),
        },
    }


def iter_raw_batches(
    config: Mapping[str, Any],
    topology: Any,
    *,
    worker_id: int = 0,
    num_workers: int = 1,
) -> Iterator[Mapping[str, Any]]:
    """Yield infinite healthy-path microbatches for one logical-DP worker."""

    data = config["data"]
    rows = read_index(resolve_index_path(data, "train_index"))
    identity = ReaderIdentity.from_topology(topology, worker_id=worker_id, num_workers=num_workers)
    sampling = SamplingConfig(
        seed=int(data["seed"]),
        pixel_frames=int(data["pixel_frames"]),
        random_start=bool(data["random_start"]),
        fixed_start_from_index=not bool(data["random_start"]),
        clip_seconds=(int(data["pixel_frames"]) - 1) / float(data.get("fps", 16.0)),
        output_fps=float(data.get("fps", 16.0)),
        shuffle_buffer=int(data.get("shuffle_buffer", 64)),
        partition_mode=str(data["partition_mode"]),
    )
    resolver = _training_resolver(config)
    micro_batch = int(config["train"]["micro_batch_size"])
    pending: list[DecodedWanSample] = []
    sampler = CanonicalSampler(rows, sampling, identity)
    prefetcher = build_shard_prefetcher(
        data,
        rows=rows,
        sampler=sampler,
        resolver=resolver,
        node_leader=int(getattr(topology, "local_rank", identity.local_rank)) == 0
        and identity.worker_id == 0,
    )
    try:
        with TarShardReader(resolver, max_open=int(data.get("tar_cache_size", 4))) as shards:
            reader = RawSampleReader(rows, shards)
            epoch = 0
            while True:
                healthy_samples = 0
                for plan in sampler.iter_epoch(epoch):
                    decoded = _materialize_raw_sample(
                        reader,
                        plan,
                        config,
                        prepare=None if prefetcher is None else prefetcher.prepare,
                    )
                    if decoded is None:
                        continue
                    healthy_samples += 1
                    pending.append(decoded)
                    if len(pending) == micro_batch:
                        yield collate_raw_samples(pending)
                        pending = []
                if healthy_samples == 0:
                    raise RuntimeError(
                        f"raw Wan reader rank={identity.rank} worker={worker_id} "
                        f"emitted no samples in epoch {epoch} from {data['train_index']}"
                    )
                epoch += 1
    finally:
        if prefetcher is not None:
            prefetcher.close()


def _iter_stateful_raw_batches(
    config: Mapping[str, Any],
    topology: Any,
    *,
    rows: Sequence[IndexRow],
    worker_id: int,
    num_workers: int,
    worker_state: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    """Run one original DataLoader worker and report only consumed cursor state."""

    data = config["data"]
    sampling = SamplingConfig(
        seed=int(data["seed"]),
        pixel_frames=int(data["pixel_frames"]),
        random_start=bool(data["random_start"]),
        fixed_start_from_index=not bool(data["random_start"]),
        clip_seconds=(int(data["pixel_frames"]) - 1) / float(data.get("fps", 16.0)),
        output_fps=float(data.get("fps", 16.0)),
        shuffle_buffer=int(data.get("shuffle_buffer", 64)),
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
    pending: list[DecodedWanSample] = []
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
            reader = RawSampleReader(rows, shards)
            while True:
                while cursor.cursor < len(cursor.plans):
                    plan = cursor.plans[cursor.cursor]
                    cursor.cursor += 1
                    decoded = _materialize_raw_sample(
                        reader,
                        plan,
                        config,
                        prepare=None if prefetcher is None else prefetcher.prepare,
                    )
                    if decoded is None:
                        continue
                    cursor.healthy_samples += 1
                    pending.append(decoded)
                    if len(pending) == micro_batch:
                        batch = dict(collate_raw_samples(pending))
                        batch[_READER_STATE_KEY] = _worker_progress_payload(cursor)
                        yield batch
                        pending = []
                if cursor.healthy_samples == 0:
                    raise RuntimeError(
                        f"raw Wan reader rank={cursor.sampler.identity.rank} "
                        f"worker={worker_id} emitted no samples in epoch {cursor.epoch} "
                        f"from {data['train_index']}"
                    )
                _advance_worker_epoch(cursor)
    finally:
        if prefetcher is not None:
            prefetcher.close()


class StatefulRawWanReader:
    """Original multi-process Wan loader with checkpointed consumed cursors."""

    def __init__(self, config: Mapping[str, Any], topology: Any) -> None:
        data = config["data"]
        self.config = config
        self.topology = topology
        self.rows = read_index(resolve_index_path(data, "train_index"))
        self.sampling = SamplingConfig(
            seed=int(data["seed"]),
            pixel_frames=int(data["pixel_frames"]),
            random_start=bool(data["random_start"]),
            fixed_start_from_index=not bool(data["random_start"]),
            clip_seconds=(int(data["pixel_frames"]) - 1) / float(data.get("fps", 16.0)),
            output_fps=float(data.get("fps", 16.0)),
            shuffle_buffer=int(data.get("shuffle_buffer", 64)),
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

    def __iter__(self) -> StatefulRawWanReader:
        return self

    def __next__(self) -> Mapping[str, Any]:
        return self.next()

    def next(self) -> Mapping[str, Any]:
        return self._readiness.read(self._next_batch)

    def _next_batch(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("Wan raw reader is closed")
        if self._iterator is None:
            self._start_loader()
        assert self._iterator is not None
        batch = next(self._iterator)
        if not isinstance(batch, dict):
            raise DataContractError("Wan raw DataLoader returned a non-mapping batch")
        worker_state = batch.pop(_READER_STATE_KEY, None)
        if not isinstance(worker_state, Mapping):
            raise DataContractError("Wan raw DataLoader batch lacks reader state")
        self.plan.record_worker_progress(worker_state)
        self.plan.finish_batch(int(worker_state["worker_id"]))
        return batch

    def _start_loader(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise DataContractError("raw Wan training requires torch") from exc

        initial_state = self.plan.state_dict()
        rows = self.rows
        config = self.config
        topology = self.topology
        first_worker = int(initial_state["next_worker"])
        worker_states = tuple(initial_state["workers"])

        class _LogicalDPDataset(torch.utils.data.IterableDataset):
            def __iter__(self) -> Iterator[Mapping[str, Any]]:
                info = torch.utils.data.get_worker_info()
                physical_worker = 0 if info is None else int(info.id)
                worker_count = 1 if info is None else int(info.num_workers)
                logical_worker = (physical_worker + first_worker) % worker_count
                yield from _iter_stateful_raw_batches(
                    config,
                    topology,
                    rows=rows,
                    worker_id=logical_worker,
                    num_workers=worker_count,
                    worker_state=worker_states[logical_worker],
                )

        options: dict[str, Any] = {
            "dataset": _LogicalDPDataset(),
            "batch_size": None,
            "num_workers": self.loader_workers,
            "pin_memory": False,
            # DataLoader otherwise draws its worker base seed from the training
            # process' global CPU RNG when the iterator is created. A resumed
            # process creates a fresh iterator, while an uninterrupted process
            # keeps its persistent iterator, so using a dedicated generator is
            # required to preserve the checkpointed training RNG stream.
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
            "schema": "solarwm.wan22.raw-reader-state.v1",
            "micro_batch_size": self.micro_batch_size,
            "plan": self.plan.state_dict(),
        }

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Quiesce speculative workers so restart and continuation share one frontier."""

        self._shutdown_loader()
        return self.state_dict()

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if self._iterator is not None:
            raise DataContractError("cannot restore Wan raw reader after iteration started")
        if value.get("schema") != "solarwm.wan22.raw-reader-state.v1":
            raise DataContractError("unsupported Wan raw reader checkpoint schema")
        if int(value.get("micro_batch_size", 0)) != self.micro_batch_size:
            raise DataContractError("Wan raw reader microbatch changed at resume")
        plan = value.get("plan")
        if not isinstance(plan, Mapping):
            raise DataContractError("Wan raw reader checkpoint lacks plan state")
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


def build_raw_dataloader(config: Mapping[str, Any], topology: Any) -> Any:
    """Build a checkpointable logical-worker stream for online Wan training."""

    return StatefulRawWanReader(config, topology)


__all__ = [
    "DecodedWanSample",
    "StatefulRawWanReader",
    "WanPlanMultiplexer",
    "build_camera_tokens",
    "build_raw_dataloader",
    "collate_raw_samples",
    "decode_raw_sample",
    "decode_video",
    "iter_raw_batches",
    "latent_aligned_pixel_indices",
]
