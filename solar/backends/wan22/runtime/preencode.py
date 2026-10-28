"""Wan 153f offline encoding and commit-last corpus assembly.

The CUDA provider deliberately uses the same ``Wan*OnlineCodec`` objects as
raw training.  The orchestration layer is provider-injectable so its ordering,
sharding, and publication transactions can be exercised without a GPU.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from solar.config.loader import canonical_json
from solar.data.archive import RawSample, RawSampleReader, TarShardReader
from solar.data.camera import CameraGuards
from solar.data.index import IndexRow, read_index, resolve_index_path
from solar.data.sampling import SamplePlan
from solar.data.transport import resolver_from_config
from solar.errors import BackendContractError, DataContractError
from solar.preencode import EncodedPayload, EncoderContract, write_shard
from solar.runtime.create_only import publish_directory_no_replace

from ..codec import PreencodedProfile, validate_preencode_config
from .data import build_camera_tokens, decode_video


@dataclass(frozen=True)
class WanPreencodeSummary:
    physical_root: Path
    logical_root: Path
    samples: int
    shards: int
    train_samples: int
    test_samples: int
    encoder_contract_digest: str
    physical_complete_digest: str
    logical_complete_digest: str


class WanPreencodeProvider(Protocol):
    """One model-family encoder below the common transaction layer."""

    family: str
    contract: EncoderContract

    def encode(self, sample: RawSample, config: Mapping[str, Any]) -> EncodedPayload: ...


def _digest_bytes(value: bytes) -> str:
    return hashlib.blake2s(value).hexdigest()


def _atomic_write(path: Path, value: bytes) -> str:
    if path.exists():
        raise DataContractError(f"preencode control already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _digest_bytes(value)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise DataContractError("cannot publish an empty preencode index")
    return b"".join(canonical_json(dict(row)) for row in rows)


def _write_gzip_index(path: Path, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    raw = _jsonl_bytes(rows)
    encoded = gzip.compress(raw, compresslevel=9, mtime=0)
    compressed_digest = _atomic_write(path, encoded)
    return {
        "path": path.name,
        "rows": len(rows),
        "decompressed_digest": _digest_bytes(raw),
        "compressed_digest": compressed_digest,
        "compressed_bytes": len(encoded),
    }


def _safe_component(value: Any, *, field: str) -> str:
    text = str(value or "").strip().lower()
    path = PurePosixPath(text)
    if (
        not text
        or "/" in text
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DataContractError(f"{field} must be one portable path component")
    return text


def _row_value(
    sample: RawSample,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if name in sample.index_values:
            return sample.index_values[name]
    metadata = sample.manifest.get("metadata", {})
    if isinstance(metadata, Mapping):
        for name in names:
            if name in metadata:
                return metadata[name]
    preencoding = sample.manifest.get("preencoding", {})
    if isinstance(preencoding, Mapping):
        for name in names:
            if name in preencoding:
                return preencoding[name]
    return default


def _split_for_sample(sample: RawSample) -> str:
    split = _safe_component(
        _row_value(sample, "split", "role", "recipe_role"),
        field="preencode split",
    )
    if split not in {"train", "test"}:
        raise DataContractError("preencode split must be train or test")
    return split


def _source_fps(sample: RawSample) -> float:
    video = sample.manifest.get("video", {})
    raw_manifest = video.get("fps") if isinstance(video, Mapping) else None
    raw_index = sample.index_values.get("fps")
    try:
        index_fps = float(raw_index or 0.0)
        manifest_fps = float(raw_manifest or 0.0)
    except (TypeError, ValueError) as exc:
        raise DataContractError("raw preencode source fps must be numeric") from exc
    if index_fps > 0 and manifest_fps > 0 and index_fps != manifest_fps:
        raise DataContractError("raw preencode index and manifest fps differ")
    value = index_fps or manifest_fps
    if value <= 0:
        raise DataContractError("raw preencode source fps must be positive")
    return value


def _source_plan(row: IndexRow, *, pixel_frames: int, rank: int) -> SamplePlan:
    values = row.values
    raw_indices = values.get("source_frame_indices")
    if raw_indices is not None:
        if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
            raise DataContractError("source_frame_indices must be an integer sequence")
        try:
            indices = tuple(int(value) for value in raw_indices)
        except (TypeError, ValueError) as exc:
            raise DataContractError("source_frame_indices contains a non-integer") from exc
        if not indices:
            raise DataContractError("source_frame_indices may not be empty")
        start = indices[0]
    else:
        try:
            start = int(values["start_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataContractError(
                f"preencode row {row.sample_id!r} lacks a fixed start_frame"
            ) from exc
        indices = tuple(range(start, start + int(pixel_frames)))
    expected = tuple(range(start, start + int(pixel_frames)))
    if indices != expected:
        raise DataContractError(
            f"preencode row {row.sample_id!r} is not one contiguous {pixel_frames}f window"
        )
    try:
        manifest = values.get("manifest", {})
        video = manifest.get("video", {}) if isinstance(manifest, Mapping) else {}
        num_frames = int(values.get("num_frames") or video["num_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(f"preencode row {row.sample_id!r} lacks source num_frames") from exc
    if start < 0 or indices[-1] >= num_frames:
        raise DataContractError(
            f"preencode window for {row.sample_id!r} is outside {num_frames} source frames"
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
        reader_rank=int(rank),
        worker_id=0,
    )


def _manifest_for_payload(
    sample: RawSample,
    config: Mapping[str, Any],
    profile: PreencodedProfile,
    contract: EncoderContract,
    tensors_digest: str,
) -> dict[str, Any]:
    from ..windows import (
        SIX_WINDOW_153F_DATASETS,
        WINDOW_HASH_NAMESPACE_153F,
        expected_153f_window_start,
    )

    manifest = copy.deepcopy(dict(sample.manifest))
    video = manifest.get("video", {})
    if not isinstance(video, Mapping):
        raise DataContractError("raw preencode manifest.video must be a mapping")
    try:
        num_frames = int(video["num_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError("raw preencode manifest lacks video.num_frames") from exc
    source_dataset = str(
        _row_value(sample, "source_dataset", "dataset", default=sample.scene)
    ).strip()
    source_sample_id = str(
        _row_value(sample, "source_sample_id", default=sample.plan.sample_id)
    ).strip()
    try:
        window_index = int(_row_value(sample, "window_index", default=0))
        window_count = int(_row_value(sample, "window_count", default=1))
    except (TypeError, ValueError) as exc:
        raise DataContractError("preencode window ordinal/count must be integers") from exc
    preencoding = {
        "version": profile.schema,
        "source_sample_id": source_sample_id,
        "source_dataset": source_dataset,
        "window_index": window_index,
        "window_count": window_count,
        "window_hash_namespace": WINDOW_HASH_NAMESPACE_153F,
        "start_frame": sample.plan.start_frame,
        "source_frame_first": sample.plan.source_frame_indices[0],
        "source_frame_last": sample.plan.source_frame_indices[-1],
        "pixel_frames": profile.pixel_frames,
        "latent_frames": contract.latent_frames,
        "target_h": contract.height,
        "target_w": contract.width,
        "dtype": profile.dtype,
        "latent_shape": list(profile.latent_shape),
        "prompt_shape": list(profile.prompt_shape),
        "encoder_contract_digest": contract.digest,
        "tensors_digest": tensors_digest,
        "caption_source": "frozen",
        "camera_source": "preserve",
    }
    if profile.i2v_y_shape is not None:
        preencoding["i2v_y_shape"] = list(profile.i2v_y_shape)
    if source_dataset in SIX_WINDOW_153F_DATASETS:
        expected_start = expected_153f_window_start(preencoding, num_frames)
        if expected_start != sample.plan.start_frame:
            raise DataContractError(
                "fixed long-form preencode start differs from the frozen 153f "
                f"window assignment: {sample.plan.start_frame} != {expected_start}"
            )
    elif window_count != 1 or window_index != 0:
        raise DataContractError("ordinary materialized 153f source must carry one window")
    manifest["preencoding"] = preencoding
    return manifest


class CudaWanPreencodeProvider:
    """Heavy Wan VAE/UMT5 provider; transformer weights are never allocated."""

    def __init__(self, config: Mapping[str, Any], profile: PreencodedProfile) -> None:
        try:
            import torch
        except ImportError as exc:
            raise BackendContractError("Wan preencoding requires torch") from exc
        if not torch.cuda.is_available():
            raise BackendContractError(
                "Wan preencoding requires CUDA; use an injected provider only for tests"
            )
        from .assets import WanAssetLayout
        from .codec import Wan5BOnlineCodec, WanA14BOnlineCodec
        from .components import Wan5BVAE, WanA14BVAE, WanTextEncoder

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        self.family = str(config["model"]["family"])
        layout = WanAssetLayout.from_config(config)
        text = WanTextEncoder(layout.text_encoder, layout.tokenizer).to(self.device)
        vae = (
            Wan5BVAE(layout.vae) if self.family == "wan22_ti2v_5b" else WanA14BVAE(layout.vae)
        ).to(self.device)
        codec_type = Wan5BOnlineCodec if self.family == "wan22_ti2v_5b" else WanA14BOnlineCodec
        default_sequence = 405 if self.family == "wan22_ti2v_5b" else 1560
        self.frame_sequence_length = int(
            config["model"].get("frame_sequence_length", default_sequence)
        )
        self.codec = codec_type(
            vae,
            text,
            pixel_frames=profile.pixel_frames,
            height=profile.latent_shape[2] * (16 if self.family == "wan22_ti2v_5b" else 8),
            width=profile.latent_shape[3] * (16 if self.family == "wan22_ti2v_5b" else 8),
            frame_sequence_length=self.frame_sequence_length,
        )
        self.contract = self.codec.contract
        self.profile = profile

    def encode(self, sample: RawSample, config: Mapping[str, Any]) -> EncodedPayload:
        from safetensors.torch import save as save_safetensors

        from solar.preencode.contracts import validate_encoded_tensors

        data = config["data"]
        source_fps = _source_fps(sample)
        if source_fps != float(data.get("fps", 16.0)):
            raise DataContractError(
                "Wan 153f preencoding requires source fps to equal configured output fps"
            )
        pixels = decode_video(
            sample.members["video_member"],
            sample.plan.source_frame_indices,
            height=int(data["height"]),
            width=int(data["width"]),
        ).to(self.device)
        camera = build_camera_tokens(
            sample.members["camera_member"],
            sample.plan.source_frame_indices,
            sample.manifest,
            source_fps=source_fps,
            output_fps=float(data.get("fps", 16.0)),
            frame_sequence_length=self.frame_sequence_length,
            guards=CameraGuards(
                max_rel_translation=float(data.get("max_rel_translation", 20.0)),
                max_camera_abs=float(data.get("max_camera_abs", 20.0)),
            ),
            configured_array_key=str(data["camera_array_key"]),
        )
        batch = self.codec.encode_batch(
            sample_ids=(sample.plan.sample_id,),
            pixels=pixels.unsqueeze(0),
            captions=(sample.caption,),
            camera={
                "viewmats": camera["viewmats"].unsqueeze(0).to(self.device),
                "K": camera["K"].unsqueeze(0).to(self.device),
            },
        )
        values: dict[str, Any] = {
            "latents": batch["latents"][0].detach().cpu(),
            "prompt_embeds": batch["prompt_embeds"][0].detach().cpu(),
            "camera_viewmats": batch["camera"]["viewmats"][0].detach().cpu(),
            "camera_K": batch["camera"]["K"][0].detach().cpu(),
        }
        if "i2v_y" in batch:
            values["i2v_y"] = batch["i2v_y"][0].detach().cpu()
        validate_encoded_tensors(values, self.contract)
        serialized = save_safetensors(
            {
                key: values[key].contiguous()
                for key in sorted(values)
                if key not in {"camera_viewmats", "camera_K"}
            }
        )
        manifest = _manifest_for_payload(
            sample,
            config,
            self.profile,
            self.contract,
            _digest_bytes(serialized),
        )
        split = _split_for_sample(sample)
        tier = _safe_component(_row_value(sample, "kept_tier", "tier"), field="kept_tier")
        if tier not in {"high", "xhigh"}:
            raise DataContractError("preencode kept_tier must be high or xhigh")
        dataset = _safe_component(
            _row_value(sample, "dataset", "source_dataset", default=sample.scene),
            field="dataset",
        )
        source_sample_id = str(manifest["preencoding"]["source_sample_id"])
        source_video = manifest["video"]
        source_identity = {
            name: sample.index_values.get(name)
            for name in (
                "shard",
                "shard_generation",
                "shard_size",
                "shard_md5_b64",
                "shard_digest",
            )
            if sample.index_values.get(name) not in {None, ""}
        }
        return EncodedPayload(
            sample_id=sample.plan.sample_id,
            key=sample.plan.key,
            source_sample_id=source_sample_id,
            start_frame=sample.plan.start_frame,
            source_frame_indices=sample.plan.source_frame_indices,
            encoder_contract_digest=self.contract.digest,
            members={
                "preencoded.safetensors": serialized,
                "camera.npz": sample.members["camera_member"],
                "manifest.json": canonical_json(manifest),
            },
            metadata={
                "split": split,
                "kept_tier": tier,
                "dataset": dataset,
                "caption": sample.caption,
                "num_frames": int(source_video["num_frames"]),
                "fps": source_fps,
                "source_identity": source_identity,
                "camera_sidecar_digest": _digest_bytes(sample.members["camera_member"]),
            },
        )


@dataclass
class _Coordinator:
    rank: int = 0
    world_size: int = 1
    initialized_here: bool = False

    @classmethod
    def create(cls) -> _Coordinator:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world == 1:
            return cls()
        try:
            import torch
            import torch.distributed as dist
        except ImportError as exc:
            raise BackendContractError("distributed preencoding requires torch") from exc
        if not torch.cuda.is_available():
            raise BackendContractError("distributed Wan preencoding requires CUDA")
        initialized_here = False
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            initialized_here = True
        return cls(dist.get_rank(), dist.get_world_size(), initialized_here)

    def broadcast(self, value: Any) -> Any:
        if self.world_size == 1:
            return value
        import torch.distributed as dist

        payload = [value]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def barrier(self) -> None:
        if self.world_size > 1:
            import torch.distributed as dist

            dist.barrier()

    def collective_error(self, local_error: str | None, *, phase: str) -> None:
        if self.world_size == 1:
            failures = [(self.rank, local_error)] if local_error else []
        else:
            import torch.distributed as dist

            gathered: list[Any] = [None] * self.world_size
            dist.all_gather_object(gathered, (self.rank, local_error))
            failures = [item for item in gathered if item[1]]
        if failures:
            detail = "; ".join(f"rank={rank}: {error}" for rank, error in failures)
            raise BackendContractError(f"Wan preencode {phase} failed collectively: {detail}")

    def close(self) -> None:
        if self.initialized_here:
            import torch.distributed as dist

            dist.destroy_process_group()


def _estimated_tar_bytes(payload: EncodedPayload) -> int:
    members = sum(len(value) for value in payload.members.values())
    # USTAR rounds every member to 512 bytes and adds provenance + two end blocks.
    return members + (len(payload.members) + 3) * 1024


def _validate_provider_contract(
    provider: WanPreencodeProvider,
    profile: PreencodedProfile,
) -> None:
    contract = provider.contract
    expected_sequence = 405 if provider.family == "wan22_ti2v_5b" else 1560
    expected: dict[str, tuple[tuple[int, ...], str]] = {
        "latents": (profile.latent_shape, profile.dtype),
        "prompt_embeds": (profile.prompt_shape, profile.dtype),
        "camera_viewmats": (
            (contract.latent_frames * expected_sequence, 4, 4),
            "float32",
        ),
        "camera_K": (
            (contract.latent_frames * expected_sequence, 3, 3),
            "float32",
        ),
    }
    if profile.i2v_y_shape is not None:
        expected["i2v_y"] = (profile.i2v_y_shape, profile.dtype)
    actual = {spec.name: (tuple(spec.shape), spec.dtype.lower()) for spec in contract.tensors}
    if actual != expected:
        raise BackendContractError(
            f"preencode provider tensor contract differs: got {actual}, expected {expected}"
        )
    if (
        contract.format_version
        != (
            "solarwm.wan22-ti2v-5b.online.v1"
            if provider.family == "wan22_ti2v_5b"
            else "solarwm.wan22-i2v-a14b.online.v1"
        )
        or contract.camera_convention != "first-frame-relative-w2c-fp32"
    ):
        raise BackendContractError(
            "preencode provider version/camera convention differs from online training"
        )


def _promote_row(
    row: Mapping[str, Any],
    *,
    ordinal: int,
    epoch_repeats: int,
) -> dict[str, Any]:
    result = dict(row)
    members = result.get("members", {})
    if not isinstance(members, Mapping):
        raise DataContractError("preencoded row members must be a mapping")
    aliases = {
        "preencoded.safetensors": "preencoded_member",
        "camera.npz": "camera_member",
        "manifest.json": "manifest_member",
    }
    for suffix, field in aliases.items():
        name = str(members.get(suffix) or "")
        if not name:
            raise DataContractError(f"preencoded row lacks {suffix}")
        result[field] = name
    metadata = result.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise DataContractError("preencoded row metadata must be a mapping")
    for field in (
        "split",
        "kept_tier",
        "dataset",
        "caption",
        "num_frames",
        "fps",
    ):
        result[field] = metadata[field]
    # Local trees are content-addressed but are not evidence of a cloud upload.
    result["shard_generation"] = f"local-digest:{result['shard_digest']}"
    # Preencoding materializes each physical source row exactly once. Preserve
    # its logical sampling weight in the published index instead of encoding
    # duplicate tensor payloads.
    result["epoch_repeats"] = int(epoch_repeats)
    result["_source_ordinal"] = int(ordinal)
    return result


def _write_rank_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(path, _jsonl_bytes(rows))


def _read_rank_rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataContractError(f"rank control row is not an object: {path}")
                result.append(value)
    return result


def _create_staging(target: Path) -> Path:
    if target.exists():
        raise DataContractError(f"preencode output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    staging.mkdir()
    return staging


def _assemble_physical(
    staging: Path,
    *,
    config: Mapping[str, Any],
    provider: WanPreencodeProvider,
    rows: Sequence[Mapping[str, Any]],
    shard_records: Sequence[Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["_source_ordinal"]))
    clean_rows = []
    for raw in ordered:
        row = dict(raw)
        row.pop("_source_ordinal", None)
        clean_rows.append(row)
    physical_index = _write_gzip_index(staging / "control/physical-index.jsonl.gz", clean_rows)
    _atomic_write(
        staging / "control/encoder-contract.json",
        canonical_json(provider.contract.as_dict()),
    )
    complete = {
        "schema": "solarwm.wan22-preencode-physical-complete.v1",
        "family": provider.family,
        "generation_id": str(config["preencode"]["generation_id"]),
        "encoder_contract_digest": provider.contract.digest,
        "samples": len(clean_rows),
        "shards": len(shard_records),
        "index": physical_index,
        "ordered_sample_id_digest": _digest_bytes(
            b"".join(f"{row['sample_id']}\n".encode() for row in clean_rows)
        ),
        "shard_records": list(sorted(shard_records, key=lambda item: item["path"])),
    }
    complete_bytes = canonical_json(complete)
    return _atomic_write(staging / "COMPLETE.json", complete_bytes), complete


def _assemble_logical(
    staging: Path,
    *,
    config: Mapping[str, Any],
    provider: WanPreencodeProvider,
    rows: Sequence[Mapping[str, Any]],
    physical_complete_digest: str,
) -> tuple[str, Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["_source_ordinal"]))
    clean: list[dict[str, Any]] = []
    for raw in ordered:
        row = dict(raw)
        row.pop("_source_ordinal", None)
        clean.append(row)
    train = [row for row in clean if row["split"] == "train"]
    test = [row for row in clean if row["split"] == "test"]
    if not train or not test:
        raise DataContractError("logical preencode publication requires train and test rows")
    overlap = {row["sample_id"] for row in train} & {row["sample_id"] for row in test}
    if overlap:
        raise DataContractError(f"preencoded train/test overlap: {sorted(overlap)[:8]}")
    source_overlap = {row["source_sample_id"] for row in train} & {
        row["source_sample_id"] for row in test
    }
    if source_overlap:
        raise DataContractError(
            f"preencoded train/test source overlap: {sorted(source_overlap)[:8]}"
        )
    train_index = _write_gzip_index(staging / "train-index.jsonl.gz", train)
    test_index = _write_gzip_index(staging / "test-index.jsonl.gz", test)
    recipe = {
        "schema": "solarwm.wan22-preencode-recipe.v1",
        "family": provider.family,
        "generation_id": str(config["preencode"]["generation_id"]),
        "encoder_contract_digest": provider.contract.digest,
        "train_index": train_index,
        "test_index": test_index,
        "camera_convention": provider.contract.camera_convention,
        "physical_complete_digest": physical_complete_digest,
    }
    _atomic_write(staging / "recipe.json", canonical_json(recipe))
    stats = {
        "schema": "solarwm.wan22-preencode-stats.v1",
        "physical_samples": len(clean),
        "train_samples": len(train),
        "test_samples": len(test),
        "train_test_overlap": 0,
        "train_test_source_overlap": 0,
        "datasets": sorted({str(row["dataset"]) for row in clean}),
        "tiers": sorted({str(row["kept_tier"]) for row in clean}),
    }
    _atomic_write(staging / "stats.json", canonical_json(stats))
    complete = {
        "schema": "solarwm.wan22-preencode-logical-complete.v1",
        "family": provider.family,
        "generation_id": str(config["preencode"]["generation_id"]),
        "physical_complete_digest": physical_complete_digest,
        "encoder_contract_digest": provider.contract.digest,
        "train_index": train_index,
        "test_index": test_index,
        "train_samples": len(train),
        "test_samples": len(test),
    }
    return _atomic_write(staging / "COMPLETE.json", canonical_json(complete)), complete


def run_wan_preencode(
    config: Mapping[str, Any],
    *,
    provider: WanPreencodeProvider | None = None,
) -> WanPreencodeSummary:
    """Encode one fixed plan and commit physical before logical controls.

    Existing targets are never overwritten.  A failed physical phase has no
    COMPLETE marker; a failed logical phase may leave an independently valid
    physical generation, but never a logical COMPLETE.
    """

    family = str(config.get("model", {}).get("family", ""))
    profile = validate_preencode_config(config, expected_family=family)
    coordinator = _Coordinator.create()
    try:
        preflight_error: str | None = None
        physical_target: Path | None = None
        logical_target: Path | None = None
        try:
            if provider is None:
                provider = CudaWanPreencodeProvider(config, profile)
            if provider.family != family:
                raise BackendContractError(
                    f"preencode provider family {provider.family!r} != {family!r}"
                )
            if provider.contract.family != family:
                raise BackendContractError("preencode provider encoder contract has wrong family")
            if provider.contract.pixel_frames != profile.pixel_frames:
                raise BackendContractError("preencode provider pixel-frame contract differs")
            _validate_provider_contract(provider, profile)

            physical_target = Path(str(config["preencode"]["output_root"])).resolve()
            logical_target = Path(str(config["preencode"]["logical_output_root"])).resolve()
            if (
                physical_target == logical_target
                or physical_target in logical_target.parents
                or logical_target in physical_target.parents
            ):
                raise DataContractError("physical and logical preencode roots must not overlap")
        except Exception as exc:
            preflight_error = f"{type(exc).__name__}: {exc}"
        coordinator.collective_error(preflight_error, phase="provider preflight")
        if provider is None or physical_target is None or logical_target is None:
            raise BackendContractError("Wan preencode provider preflight returned incomplete state")
        setup_error: str | None = None
        physical_staging: Path | None = None
        if coordinator.rank == 0:
            try:
                if logical_target.exists():
                    raise DataContractError(f"preencode output already exists: {logical_target}")
                physical_staging = _create_staging(physical_target)
            except Exception as exc:
                setup_error = f"{type(exc).__name__}: {exc}"
        setup_error = coordinator.broadcast(setup_error)
        staging_text = coordinator.broadcast(
            str(physical_staging) if physical_staging is not None else ""
        )
        if setup_error:
            raise BackendContractError(f"Wan preencode setup failed: {setup_error}")
        physical_staging = Path(staging_text)

        data = config["data"]
        transport = data["transport"]
        rows = None
        resolver = None
        source_error: str | None = None
        max_samples = 0
        max_bytes = 0
        try:
            rows = read_index(resolve_index_path(data, "index"))
            if str(transport["kind"]) == "gcs":
                for row in rows:
                    required = (
                        row.values.get("shard_generation"),
                        row.values.get("shard_size"),
                        row.values.get("shard_md5_b64"),
                    )
                    if not all(required):
                        raise DataContractError(
                            f"GCS preencode source row {row.sample_id!r} lacks immutable identity"
                        )
            resolver = resolver_from_config(
                str(transport["root"]),
                cache_dir=transport.get("cache_dir"),
                max_gib=float(transport.get("cache_max_gib", 256)),
            )
            max_samples = int(config["preencode"].get("shard_max_samples", 64))
            max_bytes = int(config["preencode"].get("shard_max_bytes", 2_147_483_648))
            if max_samples < 1 or max_bytes < 1:
                raise DataContractError("preencode shard limits must be positive")
        except Exception as exc:
            source_error = f"{type(exc).__name__}: {exc}"
        coordinator.collective_error(source_error, phase="source preflight")
        if rows is None or resolver is None:
            raise BackendContractError("Wan preencode source preflight returned incomplete state")
        source_repeats = {row.ordinal: row.epoch_repeats for row in rows}

        buffers: dict[tuple[str, str], list[tuple[int, EncodedPayload]]] = defaultdict(list)
        buffer_bytes: dict[tuple[str, str], int] = defaultdict(int)
        counters: dict[tuple[str, str], int] = defaultdict(int)
        local_rows: list[dict[str, Any]] = []
        local_shards: list[dict[str, Any]] = []

        def flush(group: tuple[str, str]) -> None:
            pending = buffers[group]
            if not pending:
                return
            dataset, tier = group
            sequence = counters[group]
            relative = f"{dataset}/shards/kept-{tier}-r{coordinator.rank:04d}-{sequence:06d}.tar"
            receipt = write_shard(
                physical_staging,
                relative,
                [payload for _, payload in pending],
            )
            if receipt.size > max_bytes:
                raise DataContractError(
                    f"encoded shard {receipt.relative_path!r} has {receipt.size} bytes, "
                    f"above shard_max_bytes={max_bytes}"
                )
            promoted = [
                _promote_row(
                    row,
                    ordinal=ordinal,
                    epoch_repeats=source_repeats[ordinal],
                )
                for row, (ordinal, _) in zip(receipt.rows, pending, strict=True)
            ]
            local_rows.extend(promoted)
            receipt_key = f"{dataset}/shards/_parts/tasks/{PurePosixPath(relative).stem}.json"
            receipt_value = {
                "schema": "solarwm.wan22-preencode-shard-receipt.v1",
                "path": receipt.relative_path,
                "samples": receipt.samples,
                "bytes": receipt.size,
                "digest": receipt.digest,
                "md5_b64": receipt.md5_b64,
                "identity": f"local-digest:{receipt.digest}",
            }
            _atomic_write(
                physical_staging / receipt_key,
                canonical_json(receipt_value),
            )
            local_shards.append({**receipt_value, "receipt": receipt_key})
            counters[group] += 1
            buffers[group] = []
            buffer_bytes[group] = 0

        local_error: str | None = None
        try:
            with TarShardReader(resolver, max_open=int(data.get("tar_cache_size", 4))) as shards:
                reader = RawSampleReader(rows, shards)
                for row in rows:
                    if row.ordinal % coordinator.world_size != coordinator.rank:
                        continue
                    plan = _source_plan(
                        row,
                        pixel_frames=profile.pixel_frames,
                        rank=coordinator.rank,
                    )
                    payload = provider.encode(reader.materialize(plan), config)
                    metadata = payload.metadata
                    group = (
                        _safe_component(metadata.get("dataset"), field="dataset"),
                        _safe_component(metadata.get("kept_tier"), field="kept_tier"),
                    )
                    estimate = _estimated_tar_bytes(payload)
                    if estimate > max_bytes:
                        raise DataContractError(
                            f"sample {payload.sample_id!r} exceeds shard_max_bytes"
                        )
                    if buffers[group] and (
                        len(buffers[group]) >= max_samples
                        or buffer_bytes[group] + estimate > max_bytes
                    ):
                        flush(group)
                    buffers[group].append((row.ordinal, payload))
                    buffer_bytes[group] += estimate
            for group in sorted(buffers):
                flush(group)
            if not local_rows:
                raise DataContractError(f"preencode rank {coordinator.rank} owns no source rows")
            _write_rank_rows(
                physical_staging / f"control/ranks/rank-{coordinator.rank:04d}-rows.jsonl",
                local_rows,
            )
            _atomic_write(
                physical_staging / f"control/ranks/rank-{coordinator.rank:04d}-shards.json",
                canonical_json({"shards": local_shards}),
            )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        coordinator.collective_error(local_error, phase="encoding")
        coordinator.barrier()

        physical_digest = ""
        logical_digest = ""
        all_rows: list[dict[str, Any]] = []
        all_shards: list[dict[str, Any]] = []
        commit_error: str | None = None
        if coordinator.rank == 0:
            try:
                for rank in range(coordinator.world_size):
                    all_rows.extend(
                        _read_rank_rows(
                            physical_staging / f"control/ranks/rank-{rank:04d}-rows.jsonl"
                        )
                    )
                    shard_control = json.loads(
                        (physical_staging / f"control/ranks/rank-{rank:04d}-shards.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    all_shards.extend(shard_control["shards"])
                physical_digest, _ = _assemble_physical(
                    physical_staging,
                    config=config,
                    provider=provider,
                    rows=all_rows,
                    shard_records=all_shards,
                )
                publish_directory_no_replace(
                    physical_staging,
                    physical_target,
                    error_type=DataContractError,
                    label="Wan physical preencode publication",
                )

                logical_staging = _create_staging(logical_target)
                try:
                    logical_digest, _ = _assemble_logical(
                        logical_staging,
                        config=config,
                        provider=provider,
                        rows=all_rows,
                        physical_complete_digest=physical_digest,
                    )
                    publish_directory_no_replace(
                        logical_staging,
                        logical_target,
                        error_type=DataContractError,
                        label="Wan logical preencode publication",
                    )
                except Exception:
                    # Retain the partial logical tree for diagnosis.  It has no
                    # COMPLETE unless assembly reached its final operation.
                    raise
            except Exception as exc:
                commit_error = f"{type(exc).__name__}: {exc}"
        commit_error = coordinator.broadcast(commit_error)
        physical_digest = str(coordinator.broadcast(physical_digest))
        logical_digest = str(coordinator.broadcast(logical_digest))
        if commit_error:
            raise BackendContractError(f"Wan preencode commit failed: {commit_error}")
        coordinator.barrier()

        if coordinator.rank != 0:
            # Counts are read from rank-0's committed marker so every process
            # returns the same summary without gathering a million-row index.
            complete = json.loads((logical_target / "COMPLETE.json").read_text(encoding="utf-8"))
            physical = json.loads((physical_target / "COMPLETE.json").read_text(encoding="utf-8"))
            samples = int(physical["samples"])
            shards_count = int(physical["shards"])
            train_count = int(complete["train_samples"])
            test_count = int(complete["test_samples"])
        else:
            samples = len(all_rows)
            shards_count = len(all_shards)
            train_count = sum(row["split"] == "train" for row in all_rows)
            test_count = sum(row["split"] == "test" for row in all_rows)
        return WanPreencodeSummary(
            physical_root=physical_target,
            logical_root=logical_target,
            samples=samples,
            shards=shards_count,
            train_samples=train_count,
            test_samples=test_count,
            encoder_contract_digest=provider.contract.digest,
            physical_complete_digest=physical_digest,
            logical_complete_digest=logical_digest,
        )
    finally:
        coordinator.close()


__all__ = [
    "CudaWanPreencodeProvider",
    "WanPreencodeProvider",
    "WanPreencodeSummary",
    "run_wan_preencode",
]
