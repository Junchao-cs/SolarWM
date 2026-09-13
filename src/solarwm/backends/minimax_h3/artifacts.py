"""Concrete H3 preencoded WebDataset reader and immutable artifact contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from solarwm.data import (
    CanonicalSampler,
    IndexRow,
    ReaderIdentity,
    SamplingConfig,
    TarShardReader,
    build_shard_prefetcher,
    read_index,
    resolver_from_config,
    select_index_rows,
)
from solarwm.data.sampling import SamplePlan, plan_fingerprint
from solarwm.errors import DataContractError
from solarwm.preencode import EncoderContract, TensorSpec
from solarwm.runtime import Topology

from .camera import validate_absolute_c2w, validate_normalized_intrinsics
from .codec import H3_PREENCODE_VERSION
from .geometry import STABLE_STAGE0P5_GEOMETRY, latent_aligned_pixel_indices


class H3CameraFilterError(DataContractError):
    """A valid H3 sample rejected by the camera magnitude filter."""


def h3_encoder_contract(
    *,
    encoder_identity: str = "official-minimax-h3",
    silence_artifact_profile: Mapping[str, Any] | None = None,
) -> EncoderContract:
    geometry = STABLE_STAGE0P5_GEOMETRY
    extras = {
        "encoder_identity": encoder_identity,
        "qwen_hidden_state": 50,
        "qwen_presentation": "<Picture 1> joint image+caption",
        "video_vae_posterior": "sample(seed=42)-fp16-round",
        "video_vae_normalization": "config.latents_mean_std",
        "source_fps_policy": "audit_only",
        "frame_sampling": "exact_contiguous",
    }
    if silence_artifact_profile is not None:
        silence_profile = json.loads(json.dumps(dict(silence_artifact_profile), sort_keys=True))
        if silence_profile != h3_silence_profile():
            raise DataContractError("H3 silence artifact profile is unsupported")
        extras["silence_artifact_profile"] = silence_profile
    return EncoderContract(
        schema="solarwm.encoder.v1",
        family="minimax_h3",
        format_version=H3_PREENCODE_VERSION,
        pixel_frames=geometry.pixel_frames,
        latent_frames=geometry.encoded_latents,
        height=geometry.height,
        width=geometry.width,
        camera_convention="absolute_c2w+normalized_K",
        tensors=(
            TensorSpec("target_latents", (24, 47, 48, 84), "bfloat16"),
            TensorSpec("anchor_latents", (24, 1, 48, 84), "bfloat16"),
            TensorSpec("prompt_embeds", (None, 5120), "bfloat16"),
            TensorSpec("text_token_tags", (None,), "int64"),
            TensorSpec("source_frame_indices", (158,), "int64"),
            TensorSpec("camera_c2w", (158, 4, 4), "float32"),
            TensorSpec("camera_K", (158, 3, 3), "float32"),
        ),
        extras=extras,
    )


def h3_silence_profile(*, pixel_frames: int = 158) -> dict[str, Any]:
    """Readable identity for the supported AudioVAE-encoded silence condition."""

    lengths = {153: 255, 158: 263, 170: 283}
    if pixel_frames not in lengths:
        raise DataContractError("unsupported H3 silence duration")
    return {
        "schema": "solarwm.minimax-h3-silence-profile.v1",
        "artifact": "official-audiovae-encoded-stereo-zero-waveform",
        "format": H3_PREENCODE_VERSION,
        "waveform": "real_stereo_zero_waveform",
        "audio_sampling_rate": 32_000,
        "audio_hop_length": 800,
        "normalization": "audio_vae.config.latents_mean_std",
        "posterior": "mode",
        "tensor": {
            "key": f"silence_{pixel_frames}f",
            "shape": [2, 32, lengths[pixel_frames]],
            "dtype": "bfloat16",
        },
    }


def _encoder_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = str(payload.get("schema") or "")
    if schema == "solarwm.encoder.v1":
        try:
            contract = EncoderContract(
                schema=schema,
                family=str(payload["family"]),
                format_version=str(payload["format_version"]),
                pixel_frames=int(payload["pixel_frames"]),
                latent_frames=int(payload["latent_frames"]),
                height=int(payload["height"]),
                width=int(payload["width"]),
                camera_convention=str(payload["camera_convention"]),
                tensors=tuple(TensorSpec(**value) for value in payload["tensors"]),
                extras=dict(payload.get("extras", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataContractError("invalid H3 encoder contract") from exc
        if contract.family != "minimax_h3":
            raise DataContractError("H3 encoder contract selected a different family")
        return json.loads(json.dumps(contract.as_dict(), sort_keys=True))
    if schema == "solarwm_minimax_h3_encoder_contract_v1":
        if not isinstance(payload.get("silence"), Mapping):
            raise DataContractError("H3 provider encoder contract lacks its silence receipt")
        # This provider control contains transaction metadata only. The
        # backend's supported, readable encoder semantics are authoritative.
        return json.loads(
            json.dumps(
                h3_encoder_contract(encoder_identity="official-minimax-h3-codec").as_dict(),
                sort_keys=True,
            )
        )
    raise DataContractError(f"unsupported H3 encoder contract schema {schema!r}")


def read_encoder_contract(path: str | Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Read provider metadata and return the normalized semantic encoder profile."""

    source = Path(path)
    try:
        payload = json.loads(source.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"cannot read H3 encoder contract {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DataContractError("H3 encoder contract JSON must contain an object")
    return dict(payload), _encoder_profile(payload)


@dataclass(frozen=True)
class H3ArtifactBatch:
    sample_id: str
    start_frame: int
    plan_fingerprint: str
    target_latents: Any
    anchor_latents: Any
    prompt_embeds: Any
    text_token_tags: Any
    source_frame_indices: Any
    camera_viewmats: Any
    camera_K: Any
    source_fps: float | None
    validation_slot: int | None = None
    validation_noise_seed: int | None = None
    rollout_camera_viewmats: Any = None
    rollout_camera_K: Any = None
    dataset_source: str = ""


@dataclass
class _H3WorkerCursor:
    identity: ReaderIdentity
    epoch: int
    cursor: int
    plans: tuple[SamplePlan, ...]


class H3PlanMultiplexer:
    """Replay torch DataLoader's ordered round-robin iterable-worker schedule."""

    def __init__(
        self,
        rows: Sequence[IndexRow],
        sampling: SamplingConfig,
        topology: Topology,
        *,
        num_workers: int,
        state_schema: str,
    ) -> None:
        self.rows = tuple(rows)
        self.sampling = sampling
        self.topology = topology
        self.num_workers = int(num_workers)
        self.state_schema = str(state_schema)
        if self.num_workers < 1:
            raise DataContractError("H3 reader num_workers must be positive")
        self.workers: list[_H3WorkerCursor] = []
        for worker_id in range(self.num_workers):
            identity = ReaderIdentity.from_topology(
                topology,
                worker_id=worker_id,
                num_workers=self.num_workers,
            )
            self.workers.append(
                _H3WorkerCursor(
                    identity=identity,
                    epoch=0,
                    cursor=0,
                    plans=self._plans_for_epoch(identity, 0),
                )
            )
        self.next_worker = 0

    def _plans_for_epoch(
        self,
        identity: ReaderIdentity,
        epoch: int,
    ) -> tuple[SamplePlan, ...]:
        # Replay prior finite epochs from the worker-specific fixed seed. This
        # keeps compact epoch/cursor state exactly resumable.
        sampler = CanonicalSampler(self.rows, self.sampling, identity)
        plans: tuple[SamplePlan, ...] = ()
        for current in range(epoch + 1):
            plans = tuple(sampler.iter_epoch(current))
        if not plans:
            raise DataContractError(
                f"H3 logical reader worker {identity.worker_id} owns no samples"
            )
        return plans

    @property
    def current_plan_fingerprint(self) -> str:
        if self.num_workers == 1:
            return plan_fingerprint(self.workers[0].plans)
        digest = hashlib.blake2s(b"solarwm.minimax-h3-worker-plan.v1\n")
        for worker in self.workers:
            digest.update(
                f"{worker.identity.worker_id}\t{plan_fingerprint(worker.plans)}\n".encode()
            )
        return digest.hexdigest()

    def state_dict(self, *, encoder_profile: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": self.state_schema,
            "num_workers": self.num_workers,
            "next_worker": self.next_worker,
            "workers": [
                {
                    "worker_id": worker.identity.worker_id,
                    "epoch": worker.epoch,
                    "cursor": worker.cursor,
                    "plan_fingerprint": plan_fingerprint(worker.plans),
                }
                for worker in self.workers
            ],
            "plan_fingerprint": self.current_plan_fingerprint,
            "encoder_profile": json.loads(json.dumps(dict(encoder_profile), sort_keys=True)),
        }

    def load_state_dict(
        self,
        values: Mapping[str, Any],
        *,
        encoder_profile: Mapping[str, Any],
    ) -> None:
        if values.get("schema") != self.state_schema:
            raise DataContractError("unsupported H3 reader checkpoint schema")
        expected_encoder = json.loads(json.dumps(dict(encoder_profile), sort_keys=True))
        if values.get("encoder_profile") != expected_encoder:
            raise DataContractError("H3 reader encoder profile changed at resume")
        if int(values.get("num_workers", 0)) != self.num_workers:
            raise DataContractError("H3 reader worker count changed at resume")
        raw_workers = values.get("workers")
        if not isinstance(raw_workers, list) or len(raw_workers) != self.num_workers:
            raise DataContractError("H3 reader checkpoint worker states differ")
        restored: list[_H3WorkerCursor] = []
        for expected_id, raw in enumerate(raw_workers):
            if not isinstance(raw, Mapping) or int(raw.get("worker_id", -1)) != expected_id:
                raise DataContractError("H3 reader checkpoint worker order differs")
            epoch, cursor = int(raw["epoch"]), int(raw["cursor"])
            if epoch < 0:
                raise DataContractError("H3 reader resume epoch is negative")
            identity = self.workers[expected_id].identity
            plans = self._plans_for_epoch(identity, epoch)
            if raw.get("plan_fingerprint") != plan_fingerprint(plans):
                raise DataContractError("H3 reader worker plan changed at resume")
            if not 0 <= cursor <= len(plans):
                raise DataContractError("H3 reader resume cursor is outside its epoch")
            restored.append(_H3WorkerCursor(identity, epoch, cursor, plans))
        next_worker = int(values.get("next_worker", -1))
        if not 0 <= next_worker < self.num_workers:
            raise DataContractError("H3 reader next worker is outside its worker count")
        self.workers = restored
        self.next_worker = next_worker
        if values.get("plan_fingerprint") != self.current_plan_fingerprint:
            raise DataContractError("H3 reader aggregate plan changed at resume")

    def next_plan(self) -> SamplePlan:
        worker = self.workers[self.next_worker]
        if worker.cursor == len(worker.plans):
            worker.epoch += 1
            worker.cursor = 0
            worker.plans = self._plans_for_epoch(worker.identity, worker.epoch)
        plan = worker.plans[worker.cursor]
        worker.cursor += 1
        self.next_worker = (self.next_worker + 1) % self.num_workers
        return plan


def validate_fixed_validation_rows(
    rows: tuple[IndexRow, ...],
    *,
    logical_world_size: int,
    sample_count: int | None = None,
    selection_seed: int = 0,
    noise_seed: int = 0,
    require_preencoded_identity: bool = False,
) -> tuple[IndexRow, ...]:
    """Select deterministic complete waves from a recipe test index."""

    logical_world = int(logical_world_size)
    slots = logical_world if sample_count is None else int(sample_count)
    if logical_world <= 0 or slots <= 0 or slots % logical_world:
        raise DataContractError(
            "H3 fixed validation sample count must form complete logical-DP waves: "
            f"sample_count={slots} logical_world_size={logical_world_size}"
        )
    if len(rows) < slots:
        raise DataContractError(
            "recipe test index has fewer rows than validation.sample_count: "
            f"rows={len(rows)} sample_count={slots}"
        )
    # Camera guards require inspecting the encoded artifact. Build one seeded
    # candidate order from the recipe test index; the stream skips rejected
    # candidates and assigns slots only to successfully materialized cases.
    selected = select_index_rows(rows, sample_count=len(rows), seed=selection_seed)
    if int(noise_seed) < 0 or int(noise_seed) + slots > 2**63:
        raise DataContractError("H3 validation noise seed is outside [0, 2**63)")
    normalized = []
    for candidate, row in enumerate(selected):
        values = dict(row.values)
        try:
            start = int(values["start_frame"])
            shard_size = int(values["shard_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataContractError(
                f"H3 recipe test row {row.sample_id!r} lacks start/shard identity"
            ) from exc
        if start < 0 or row.epoch_repeats != 1 or shard_size <= 0:
            raise DataContractError(f"H3 recipe test row {row.sample_id!r} is invalid")
        if require_preencoded_identity:
            required = {
                "h3_preencoded_member",
                "h3_provenance_member",
                "manifest_member",
                "shard_generation",
            }
            missing = sorted(name for name in required if values.get(name) in (None, ""))
            if missing:
                raise DataContractError(f"H3 recipe test row {row.sample_id!r} lacks {missing}")
            generation = str(values["shard_generation"]).strip()
            if not generation:
                raise DataContractError(
                    f"H3 recipe test row {row.sample_id!r} lacks provider generation"
                )
        normalized.append(IndexRow.from_mapping(candidate, values))
    return tuple(normalized)


def _member_for(row: Any) -> str:
    values = row.values
    members = values.get("members")
    if isinstance(members, Mapping):
        for suffix in ("tensors.safetensors", "h3.safetensors", "preencoded.safetensors"):
            if suffix in members:
                return str(members[suffix])
        candidates = [
            str(value) for key, value in members.items() if str(key).endswith(".safetensors")
        ]
        if len(candidates) == 1:
            return candidates[0]
    for key in ("h3_preencoded_member", "preencoded_member"):
        if values.get(key):
            return str(values[key])
    raise DataContractError(f"H3 index row {row.sample_id!r} lacks a tensor member")


def align_h3_camera(tensors: Mapping[str, Any], *, audit_latents: int = 47) -> tuple[Any, Any]:
    """Validate and convert encoded absolute cameras to H3 model rows."""
    import torch

    if audit_latents not in {45, 47}:
        raise DataContractError("H3 camera audit must cover 45 Stage1 or 47 physical latents")

    def invert_se3_torch(matrices: Any) -> Any:
        rotation = matrices[..., :3, :3]
        rotation_inverse = rotation.transpose(-1, -2)
        output = torch.zeros_like(matrices)
        output[..., :3, :3] = rotation_inverse
        output[..., :3, 3] = -torch.einsum(
            "...ij,...j->...i",
            rotation_inverse,
            matrices[..., :3, 3],
        )
        output[..., 3, 3] = 1.0
        return output

    if "camera_token_viewmats" in tensors or "camera_token_K" in tensors:
        if "camera_token_viewmats" not in tensors or "camera_token_K" not in tensors:
            raise DataContractError("token camera artifact is incomplete")
        raw_views = tensors["camera_token_viewmats"]
        raw_intrinsics = tensors["camera_token_K"]
        if raw_views.dtype != torch.float32 or raw_intrinsics.dtype != torch.float32:
            raise DataContractError("token camera artifacts must be FP32")
        views = raw_views.contiguous()
        intrinsics = raw_intrinsics.contiguous()
        expected = 47 * 1008
        if tuple(views.shape) != (expected, 4, 4) or tuple(intrinsics.shape) != (
            expected,
            3,
            3,
        ):
            raise DataContractError("token camera artifact has the wrong packed shape")
        if not bool(torch.isfinite(views).all()) or not bool(torch.isfinite(intrinsics).all()):
            raise DataContractError("token camera artifact contains non-finite values")
        validate_normalized_intrinsics(intrinsics.cpu().numpy())
        if float(views[: audit_latents * 1008].abs().max().item()) > 20.0:
            raise H3CameraFilterError("token camera artifact exceeds the magnitude guard")
        return views, intrinsics
    if "camera_c2w" not in tensors or "camera_K" not in tensors:
        raise DataContractError("H3 tensors require camera_c2w and camera_K")
    if tensors["camera_c2w"].dtype != torch.float32 or tensors["camera_K"].dtype != torch.float32:
        raise DataContractError("H3 structured camera artifacts must be FP32")
    c2w = tensors["camera_c2w"].detach().cpu()
    K = tensors["camera_K"].detach().cpu()
    indices = torch.from_numpy(latent_aligned_pixel_indices(158)).long()
    if c2w.shape[0] == 158:
        c2w = c2w.index_select(0, indices)
    elif c2w.shape[0] != 47:
        raise DataContractError("camera_c2w must have 158 pixel or 47 latent rows")
    if K.shape == (3, 3):
        K = K.unsqueeze(0).expand(47, -1, -1).clone()
    elif K.shape[0] == 158:
        K = K.index_select(0, indices)
    elif K.shape[0] != 47:
        raise DataContractError("camera_K must have 158 pixel, 47 latent, or one static row")
    validate_absolute_c2w(c2w.numpy())
    validate_normalized_intrinsics(K.numpy())
    # Preserve the Torch SE(3) arithmetic operation order. The
    # NumPy equivalent differs by a few FP32 ULPs and changes fused-PRoPE loss.
    first_w2c = invert_se3_torch(c2w[:1])[0]
    relative_c2w = torch.matmul(first_w2c.unsqueeze(0), c2w)
    relative_c2w[0] = torch.eye(4, dtype=relative_c2w.dtype)
    views = invert_se3_torch(relative_c2w).contiguous()
    translation = torch.linalg.vector_norm(views[:audit_latents, :3, 3], dim=-1)
    if float(translation.max()) > 20.0 or float(views[:audit_latents].abs().max()) > 20.0:
        raise H3CameraFilterError("H3 camera exceeds the magnitude guards")
    return views, K.contiguous()


class H3PreencodedStream:
    """Deterministic logical-DP stream over H3 WDS artifacts."""

    def __init__(
        self,
        *,
        root: str,
        index: str,
        topology: Topology,
        seed: int,
        encoder_contract_path: str,
        cache_dir: str | None = None,
        cache_max_gib: float = 256.0,
        gcs_prefetch_shards: int = 0,
        shuffle_buffer: int = 4096,
        num_workers: int = 1,
        fixed_validation: bool = False,
        fixed_validation_sample_count: int | None = None,
        fixed_validation_selection_seed: int = 0,
        fixed_validation_noise_seed: int = 0,
        fixed_validation_sample_ids: Sequence[str] | None = None,
        camera_audit_latents: int = 47,
    ) -> None:
        self.camera_audit_latents = int(camera_audit_latents)
        rows = read_index(index)
        normalized: list[IndexRow] = []
        for row in rows:
            values = dict(row.values)
            indices = values.get("source_frame_indices")
            if values.get("num_frames") is None and isinstance(indices, (list, tuple)):
                try:
                    values["num_frames"] = max(int(value) for value in indices) + 1
                except (TypeError, ValueError) as exc:
                    raise DataContractError(
                        f"H3 sample {row.sample_id!r} has invalid source_frame_indices"
                    ) from exc
            if values.get("num_frames") is None and values.get("start_frame") is not None:
                # Finalized H3 indexes keep only the
                # frozen occurrence start and immutable member identities.
                # The encoded tensor carries the authoritative 158 contiguous
                # source indices, which are checked against this reconstructed
                # minimum span after materialization.
                try:
                    start_frame = int(values["start_frame"])
                except (TypeError, ValueError) as exc:
                    raise DataContractError(
                        f"H3 sample {row.sample_id!r} has invalid start_frame"
                    ) from exc
                if start_frame < 0:
                    raise DataContractError(f"H3 sample {row.sample_id!r} has negative start_frame")
                values["num_frames"] = start_frame + 158
            metadata = values.get("metadata", {})
            if (
                values.get("fps") is None
                and isinstance(metadata, Mapping)
                and metadata.get("source_fps") is not None
            ):
                values["fps"] = metadata["source_fps"]
            normalized.append(IndexRow.from_mapping(row.ordinal, values))
        self.rows = tuple(normalized)
        self.topology = topology
        self.fixed_validation = bool(fixed_validation)
        self.fixed_validation_sample_count = (
            int(fixed_validation_sample_count) if self.fixed_validation else 0
        )
        self.fixed_validation_noise_seed = int(fixed_validation_noise_seed)
        self.fixed_validation_successes = 0
        if self.fixed_validation and int(num_workers) != 1:
            raise DataContractError("H3 fixed validation requires one deterministic worker")
        if self.fixed_validation:
            frozen_ids = (
                tuple(str(value) for value in fixed_validation_sample_ids)
                if fixed_validation_sample_ids is not None
                else None
            )
            if frozen_ids is None:
                self.rows = validate_fixed_validation_rows(
                    self.rows,
                    logical_world_size=topology.dp_world_size,
                    sample_count=fixed_validation_sample_count,
                    selection_seed=fixed_validation_selection_seed,
                    noise_seed=fixed_validation_noise_seed,
                    require_preencoded_identity=True,
                )
            else:
                expected = int(fixed_validation_sample_count or 0)
                if len(frozen_ids) != expected or expected % int(topology.dp_world_size):
                    raise DataContractError(
                        "H3 frozen validation sample IDs do not form complete waves"
                    )
                by_id = {row.sample_id: row for row in self.rows}
                if len(by_id) != len(self.rows):
                    raise DataContractError("H3 test index contains duplicate sample IDs")
                restored = []
                for slot, sample_id in enumerate(frozen_ids):
                    try:
                        row = by_id[sample_id]
                    except KeyError as exc:
                        raise DataContractError(
                            f"H3 frozen validation sample {sample_id!r} left the test index"
                        ) from exc
                    validated = validate_fixed_validation_rows(
                        (row,),
                        logical_world_size=1,
                        sample_count=1,
                        noise_seed=0,
                        require_preencoded_identity=True,
                    )[0]
                    restored.append(IndexRow.from_mapping(slot, dict(validated.values)))
                self.rows = tuple(restored)
        self.sampling = SamplingConfig(
            seed=int(seed),
            pixel_frames=158,
            random_start=False,
            fixed_start_from_index=True,
            shuffle_buffer=1 if self.fixed_validation else int(shuffle_buffer),
            partition_mode="global_occurrence" if self.fixed_validation else "node_shard",
        )
        self.encoder_contract, self.encoder_profile = read_encoder_contract(encoder_contract_path)
        resolver = resolver_from_config(root, cache_dir=cache_dir, max_gib=cache_max_gib)
        self.shards = TarShardReader(resolver, max_open=4)
        self.plan = H3PlanMultiplexer(
            self.rows,
            self.sampling,
            topology,
            num_workers=int(num_workers),
            state_schema="solarwm.minimax-h3-reader.v3",
        )
        self.identity = self.plan.workers[0].identity
        prefetch_data = {
            "transport": {"kind": "gcs" if root.startswith("gs://") else "local"},
            "partition_mode": self.sampling.partition_mode,
            "gcs_prefetch_shards": gcs_prefetch_shards,
        }
        prefetch_sampler = CanonicalSampler(self.rows, self.sampling, self.identity)
        self._shard_prefetcher = build_shard_prefetcher(
            prefetch_data,
            rows=self.rows,
            sampler=prefetch_sampler,
            resolver=resolver,
            node_leader=int(topology.local_rank) == 0,
        )

    @property
    def epoch(self) -> int:
        return self.plan.workers[0].epoch

    @property
    def cursor(self) -> int:
        return self.plan.workers[0].cursor

    @property
    def _plans(self) -> tuple[SamplePlan, ...]:
        return self.plan.workers[0].plans

    @property
    def current_plan_fingerprint(self) -> str:
        return self.plan.current_plan_fingerprint

    def state_dict(self) -> dict[str, Any]:
        return self.plan.state_dict(
            encoder_profile=self.encoder_profile,
        )

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        self.plan.load_state_dict(
            values,
            encoder_profile=self.encoder_profile,
        )

    def close(self) -> None:
        prefetcher = getattr(self, "_shard_prefetcher", None)
        if prefetcher is not None:
            prefetcher.close()
        self.shards.close()

    def __iter__(self) -> Iterator[H3ArtifactBatch]:
        while True:
            yield self.next()

    def next(self) -> H3ArtifactBatch:
        rejected = 0
        while True:
            # One DataLoader worker owns its pending output slot until it yields.
            current_worker = self.plan.next_worker
            try:
                batch = self._next_once()
            except H3CameraFilterError as exc:
                self.plan.next_worker = current_worker
                rejected += 1
                if self.fixed_validation and rejected >= len(self.rows):
                    raise H3CameraFilterError(
                        "H3 recipe test index has no camera-safe validation candidate"
                    ) from exc
                continue
            except Exception:
                self.plan.next_worker = current_worker
                raise
            if not self.fixed_validation:
                return batch
            slot = self.fixed_validation_successes * int(self.topology.dp_world_size) + int(
                self.topology.dp_rank
            )
            if slot >= self.fixed_validation_sample_count:
                raise DataContractError("H3 fixed validation requested more cases than configured")
            self.fixed_validation_successes += 1
            return replace(
                batch,
                validation_slot=slot,
                validation_noise_seed=self.fixed_validation_noise_seed + slot,
            )

    def _next_once(self) -> H3ArtifactBatch:
        plan = self.plan.next_plan()
        row = self.rows[plan.row_ordinal]
        prefetcher = getattr(self, "_shard_prefetcher", None)
        if prefetcher is not None:
            prefetcher.prepare(plan)
        return self.read_artifact(row, plan)

    def read_artifact(self, row: IndexRow, plan: SamplePlan) -> H3ArtifactBatch:
        """Materialize a frozen occurrence without advancing the training reader."""

        row_profile = row.values.get("encoder_profile")
        if row_profile is not None and row_profile != self.encoder_profile:
            raise DataContractError(f"H3 sample {row.sample_id!r} encoder profile differs")
        payload = self.shards.read(row, _member_for(row))
        try:
            import torch
            from safetensors.torch import load

            tensors = load(payload)
        except (ImportError, ValueError) as exc:
            raise DataContractError("H3 preencoded member is not valid safetensors") from exc
        required = {
            "target_latents",
            "anchor_latents",
            "prompt_embeds",
            "text_token_tags",
            "source_frame_indices",
        }
        if not required <= set(tensors):
            raise DataContractError(f"H3 preencoded member lacks {sorted(required - set(tensors))}")
        target = tensors["target_latents"]
        anchor = tensors["anchor_latents"]
        prompt = tensors["prompt_embeds"]
        tags = tensors["text_token_tags"]
        indices = tensors["source_frame_indices"]
        if tuple(target.shape) != (24, 47, 48, 84) or tuple(anchor.shape) != (
            24,
            1,
            48,
            84,
        ):
            raise DataContractError("H3 VisualVAE tensor geometry differs")
        target_dtype = str(target.dtype).removeprefix("torch.")
        anchor_dtype = str(anchor.dtype).removeprefix("torch.")
        if target_dtype != "bfloat16" or anchor_dtype != "bfloat16":
            raise DataContractError("H3 VisualVAE artifacts must be BF16")
        if prompt.dtype != torch.bfloat16:
            raise DataContractError("H3 Qwen prompt artifacts must be BF16")
        if tags.dtype != torch.int64:
            raise DataContractError("H3 text token tags must be INT64")
        if indices.dtype != torch.int64:
            raise DataContractError("H3 source-frame indices must be INT64")
        if prompt.ndim != 2 or prompt.shape[-1] != 5120 or tags.shape != prompt.shape[:1]:
            raise DataContractError("H3 Qwen prompt artifacts differ")
        for name, tensor in (
            ("target_latents", target),
            ("anchor_latents", anchor),
            ("prompt_embeds", prompt),
        ):
            if not bool(torch.isfinite(tensor.float()).all()):
                raise DataContractError(f"H3 {name} contains non-finite values")
        if tuple(indices.shape) != (158,) or not np.all(np.diff(indices.numpy()) == 1):
            raise DataContractError("H3 source indices must be 158 exact contiguous frames")
        if tuple(int(value) for value in indices.tolist()) != tuple(plan.source_frame_indices):
            raise DataContractError("H3 tensor source indices differ from the sample plan")
        views, K = align_h3_camera(tensors, audit_latents=self.camera_audit_latents)
        metadata = row.values.get("metadata", {})
        source_fps = row.values.get("fps")
        if source_fps is None and isinstance(metadata, Mapping):
            source_fps = metadata.get("source_fps")
        if source_fps is not None:
            source_fps = float(source_fps)
            if not np.isfinite(source_fps) or source_fps <= 0:
                raise DataContractError("H3 source FPS must be positive when present")
        return H3ArtifactBatch(
            sample_id=row.sample_id,
            start_frame=plan.start_frame,
            plan_fingerprint=self.current_plan_fingerprint,
            target_latents=target.contiguous(),
            anchor_latents=anchor.contiguous(),
            prompt_embeds=prompt.contiguous(),
            text_token_tags=tags.contiguous(),
            source_frame_indices=indices.contiguous(),
            camera_viewmats=views,
            camera_K=K,
            source_fps=source_fps,
            validation_slot=None,
            validation_noise_seed=None,
            dataset_source=row.sample_id.split("/")[0],
        )


def load_silence_latents(
    path: str | Path, *, pixel_frames: int = 158
) -> tuple[Any, Mapping[str, Any]]:
    """Load the official AudioVAE-encoded 158f silence tensor ``[2,32,263]``."""

    source = Path(path)
    if not source.is_file():
        raise DataContractError(f"H3 silence artifact is missing: {source}")
    try:
        from safetensors.torch import load_file

        tensors = load_file(str(source), device="cpu")
    except (ImportError, ValueError) as exc:
        raise DataContractError("H3 silence artifact is not valid safetensors") from exc
    lengths = {153: 255, 158: 263, 170: 283}
    if pixel_frames not in lengths:
        raise DataContractError("H3 silence supports the encoded 153/158/170-frame profiles")
    key = f"silence_{pixel_frames}f"
    if key not in tensors:
        raise DataContractError(f"H3 silence artifact lacks the required {key!r} tensor")
    value = tensors[key].contiguous()
    if (
        tuple(value.shape) != (2, 32, lengths[pixel_frames])
        or str(value.dtype).removeprefix("torch.") != "bfloat16"
    ):
        raise DataContractError(f"H3 silence must be BF16 [2,32,{lengths[pixel_frames]}]")
    if not bool(value.isfinite().all()):
        raise DataContractError("H3 silence contains non-finite values")
    return value, h3_silence_profile(pixel_frames=pixel_frames)


__all__ = [
    "H3ArtifactBatch",
    "H3PlanMultiplexer",
    "H3PreencodedStream",
    "align_h3_camera",
    "h3_encoder_contract",
    "h3_silence_profile",
    "load_silence_latents",
    "read_encoder_contract",
    "validate_fixed_validation_rows",
]
