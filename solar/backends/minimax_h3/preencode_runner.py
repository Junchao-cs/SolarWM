"""Concrete raw-WDS to H3 preencoded-WDS production runner."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from solar.config.loader import canonical_json
from solar.data import (
    CanonicalSampler,
    IndexRow,
    RawSampleReader,
    ReaderIdentity,
    SamplingConfig,
    TarShardReader,
    read_index,
    resolve_index_path,
    resolver_from_config,
)
from solar.errors import DataContractError
from solar.preencode import EncodedPayload, write_index, write_shard
from solar.runtime.create_only import publish_directory_no_replace
from solar.runtime.distributed import collective_call, collective_rank_zero_call

from .official_codec import H3_VIDEO_VAE_SEED, OfficialH3Codec
from .optional import load_conditioners, require_h3_runtime
from .raw_data import decode_camera, decode_resize_video, normalize_raw_source_windows

H3_ENCODER_CONTRACT_PATH = "encoder-contract.json"
H3_SILENCE_PATH = "silence/h3-silence-158f.safetensors"
H3_INDEX_PATH = "index.jsonl"
H3_COMPLETE_PATH = "COMPLETE.json"


def _publish_staging(output_root: Path, requested_root: Path) -> None:
    publish_directory_no_replace(
        output_root,
        requested_root,
        error_type=DataContractError,
        label="H3 preencode output",
    )


def _published_rows(receipts: Sequence[Any]) -> list[dict[str, Any]]:
    """Promote sampler controls required to replay every encoded window."""

    result: list[dict[str, Any]] = []
    for receipt in receipts:
        for source in receipt.rows:
            row = dict(source)
            provenance_metadata = dict(row.get("metadata", {}))
            member_digest = provenance_metadata.get("member_digest")
            if not isinstance(member_digest, Mapping):
                raise DataContractError("published H3 row lacks member content digest provenance")
            members = row.get("members")
            if not isinstance(members, Mapping):
                raise DataContractError("published H3 row lacks canonical member names")
            try:
                tensor_member = str(members["tensors.safetensors"])
                manifest_member = str(members["manifest.json"])
                tensor_digest = str(member_digest["tensors.safetensors"])
                manifest_digest = str(member_digest["manifest.json"])
                provenance_member = str(row["provenance_member"])
            except KeyError as exc:
                raise DataContractError("published H3 row has incomplete member identity") from exc
            provenance = {
                "schema": "solarwm.preencoded-sample.v1",
                "sample_id": row["sample_id"],
                "key": row["key"],
                "source_sample_id": row["source_sample_id"],
                "start_frame": row["start_frame"],
                "source_frame_indices": row["source_frame_indices"],
                "encoder_contract_digest": row["encoder_contract_digest"],
                "members": dict(members),
                "metadata": provenance_metadata,
            }
            source_indices = tuple(int(value) for value in row["source_frame_indices"])
            if len(source_indices) != 158 or any(
                right != left + 1 for left, right in pairwise(source_indices)
            ):
                raise DataContractError("published H3 source indices must be 158 contiguous rows")
            metadata = dict(provenance_metadata)
            minimum_frames = max(source_indices) + 1
            raw_source_frames = metadata.pop("source_num_frames", None)
            source_frames = minimum_frames if raw_source_frames is None else int(raw_source_frames)
            if source_frames < minimum_frames:
                raise DataContractError("published H3 source length is shorter than its window")
            row["num_frames"] = source_frames
            source_repeats = int(metadata.pop("source_epoch_repeats", 1))
            if source_repeats < 1:
                raise DataContractError("published H3 epoch_repeats must be positive")
            row["epoch_repeats"] = source_repeats
            if metadata.get("source_fps") is not None:
                row["fps"] = float(metadata["source_fps"])
            row["metadata"] = metadata
            row.update(
                {
                    "h3_preencoded_member": tensor_member,
                    "h3_provenance_member": provenance_member,
                    "manifest_member": manifest_member,
                    "tensor_digest": tensor_digest,
                    "manifest_digest": manifest_digest,
                    "provenance_digest": hashlib.blake2s(canonical_json(provenance)).hexdigest(),
                }
            )
            if (
                row.get("shard_size")
                and row.get("shard_md5_b64")
                and not row.get("shard_generation")
            ):
                # The freshly committed tree is local. Preserve its complete
                # immutable identity without pretending it already has a GCS
                # provider generation; upload tooling must replace this value.
                row["shard_generation"] = f"local-digest:{row['shard_digest']}"
            result.append(row)
    return result


def _payload(
    *,
    sample: Any,
    plan: Any,
    values: Mapping[str, Any],
    contract_digest: str,
) -> EncodedPayload:
    from safetensors.torch import save

    tensor_bytes = save(dict(values), metadata={"format": "h3.158f.v1"})
    manifest = {
        "schema": "solarwm.minimax-h3-preencoded.v1",
        "sample_id": plan.sample_id,
        "caption": sample.caption,
        "start_frame": plan.start_frame,
        "source_frame_indices": list(plan.source_frame_indices),
        "encoder_contract_digest": contract_digest,
        "source_manifest": dict(sample.manifest),
    }
    video_manifest = sample.manifest.get("video", {})
    source_num_frames = sample.index_values.get("num_frames") or (
        video_manifest.get("num_frames") if isinstance(video_manifest, Mapping) else None
    )
    manifest_bytes = canonical_json(manifest)
    return EncodedPayload(
        sample_id=plan.sample_id,
        key=plan.key,
        source_sample_id=str(sample.manifest.get("source_sample_id") or plan.sample_id),
        start_frame=plan.start_frame,
        source_frame_indices=tuple(plan.source_frame_indices),
        encoder_contract_digest=contract_digest,
        members={
            "tensors.safetensors": tensor_bytes,
            "manifest.json": manifest_bytes,
        },
        metadata={
            "format_version": "h3.158f.v1",
            "source_fps": video_manifest.get("fps")
            if isinstance(video_manifest, Mapping)
            else None,
            "source_num_frames": source_num_frames,
            "source_epoch_repeats": int(sample.index_values.get("epoch_repeats", 1)),
            "member_digest": {
                "tensors.safetensors": hashlib.blake2s(tensor_bytes).hexdigest(),
                "manifest.json": hashlib.blake2s(manifest_bytes).hexdigest(),
            },
        },
    )


def run_preencode(config: Mapping[str, Any]) -> int:
    """Encode one finite raw-index epoch and publish deterministic local WDS."""

    torch, _diffusers, _transformers = require_h3_runtime()
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    model_cfg = config["model"]
    data_cfg = config["data"]
    output_cfg = config["preencode"]

    def phase(call: Any, label: str) -> Any:
        return collective_call(
            call,
            dist=dist,
            rank=rank,
            world_size=world,
            label=f"H3 preencode {label}",
            error_type=DataContractError,
        )

    def rank_zero_phase(call: Any, label: str) -> Any:
        return collective_rank_zero_call(
            call,
            dist=dist,
            rank=rank,
            world_size=world,
            label=f"H3 preencode {label}",
            error_type=DataContractError,
        )

    def load_codec() -> OfficialH3Codec:
        modules = load_conditioners(model_cfg, device=device, schedulers=False)
        return OfficialH3Codec(
            text_encoder=modules.text_encoder,
            tokenizer=modules.tokenizer,
            processor=modules.processor,
            video_vae=modules.video_vae,
            audio_vae=modules.audio_vae,
            device=device,
            encoder_identity=str(model_cfg["codec_identity"]),
        )

    codec = phase(load_codec, "codec setup")
    requested_root = Path(str(output_cfg["output_root"])).resolve()

    def setup_staging() -> str:
        if requested_root.exists() or requested_root.is_symlink():
            raise DataContractError(f"H3 preencode output already exists: {requested_root}")
        requested_root.parent.mkdir(parents=True, exist_ok=True)
        staging = requested_root.with_name(f".{requested_root.name}.{uuid.uuid4().hex}.partial")
        staging.mkdir()
        return str(staging)

    staging_value = str(rank_zero_phase(setup_staging, "staging setup"))
    output_root = Path(staging_value)

    def validate_staging_visibility() -> None:
        if not output_root.is_dir() or output_root.is_symlink():
            raise DataContractError("H3 preencode staging directory was not created")

    phase(validate_staging_visibility, "staging visibility")

    def write_silence() -> str:
        silence_path = output_root / H3_SILENCE_PATH
        silence_path.parent.mkdir(parents=True, exist_ok=True)
        silence_bytes = codec.silence_artifact_bytes()
        silence_path.write_bytes(silence_bytes)
        return hashlib.blake2s(silence_bytes).hexdigest()

    silence_digest = str(rank_zero_phase(write_silence, "silence artifact"))
    phase(codec.bind_silence_profile, "silence binding")

    def write_encoder_contract() -> None:
        contract_path = output_root / H3_ENCODER_CONTRACT_PATH
        contract_path.write_text(
            json.dumps(codec.contract.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    rank_zero_phase(write_encoder_contract, "encoder-contract publication")

    def source_setup() -> tuple[Any, ...]:
        rows = normalize_raw_source_windows(read_index(resolve_index_path(data_cfg, "index")))
        source_order = {row.sample_id: row.ordinal for row in rows}
        if len(source_order) != len(rows):
            raise DataContractError("H3 preencode source index contains duplicate sample IDs")
        # Preencoding materializes each physical row once. The source weighting is
        # copied into the published row rather than redundantly encoding repeats.
        sampling_rows = tuple(
            IndexRow.from_mapping(
                row.ordinal,
                {**row.values, "epoch_repeats": 1},
            )
            for row in rows
        )
        sampler = CanonicalSampler(
            sampling_rows,
            SamplingConfig(
                seed=int(data_cfg.get("seed", 42)),
                pixel_frames=158,
                random_start=bool(data_cfg.get("random_start", False)),
                fixed_start_from_index=bool(data_cfg.get("fixed_start_from_index", False)),
                shuffle_buffer=int(data_cfg.get("shuffle_buffer", 4096)),
                partition_mode="global_occurrence",
            ),
            ReaderIdentity(rank=rank, world_size=world),
        )
        transport = data_cfg["transport"]
        resolver = resolver_from_config(
            str(transport["root"]),
            cache_dir=transport.get("cache_dir"),
            max_gib=float(transport.get("cache_max_gib", 256)),
        )
        return rows, source_order, sampler, resolver

    rows, source_order, sampler, resolver = phase(source_setup, "source preflight")

    def encode_rank() -> list[dict[str, Any]]:
        receipts = []
        pending: list[EncodedPayload] = []
        per_shard = int(output_cfg.get("samples_per_shard", 8))
        with TarShardReader(resolver, max_open=2) as shards:
            reader = RawSampleReader(rows, shards)
            shard_index = 0
            for plan in sampler.iter_epoch(0):
                raw = reader.materialize(plan)
                frames, transform = decode_resize_video(
                    raw.members["video_member"],
                    plan.source_frame_indices,
                    decord_num_threads=int(data_cfg.get("decord_num_threads", 2)),
                )
                intrinsics_bytes = None
                intrinsics_member = str(raw.index_values.get("intrinsics_member") or "")
                if intrinsics_member:
                    intrinsics_bytes = shards.read(rows[plan.row_ordinal], intrinsics_member)
                c2w, K = decode_camera(
                    raw.members["camera_member"],
                    plan.source_frame_indices,
                    transform,
                    intrinsics_bytes=intrinsics_bytes,
                )
                values = codec.encode(
                    sample_id=plan.sample_id,
                    pixels=frames,
                    caption=raw.caption,
                    camera={
                        "camera_c2w": c2w,
                        "camera_K": K,
                        "source_frame_indices": plan.source_frame_indices,
                    },
                    seed=H3_VIDEO_VAE_SEED,
                )
                pending.append(
                    _payload(
                        sample=raw,
                        plan=plan,
                        values=values,
                        contract_digest=codec.contract.digest,
                    )
                )
                if len(pending) == per_shard:
                    receipts.append(
                        write_shard(
                            output_root,
                            f"shards/rank-{rank:05d}-part-{shard_index:06d}.tar",
                            pending,
                        )
                    )
                    shard_index += 1
                    pending = []
            if pending:
                receipts.append(
                    write_shard(
                        output_root,
                        f"shards/rank-{rank:05d}-part-{shard_index:06d}.tar",
                        pending,
                    )
                )
        return _published_rows(receipts)

    local_rows = phase(encode_rank, "rank encoding")
    if world > 1:
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, local_rows)
        all_rows = [row for rank_rows in gathered for row in rank_rows]
    else:
        all_rows = local_rows

    def finalize() -> bool:
        try:
            all_rows.sort(key=lambda row: source_order[str(row["sample_id"])])
        except KeyError as exc:
            raise DataContractError("published H3 row is absent from the source index") from exc
        index_digest = write_index(output_root / H3_INDEX_PATH, all_rows)
        complete = {
            "schema": "solarwm.minimax-h3-preencode-complete.v1",
            "samples": len(all_rows),
            "encoder_contract_digest": codec.contract.digest,
            "silence_digest": silence_digest,
            "index_digest": index_digest,
            "shards": sorted({str(row["shard"]) for row in all_rows}),
        }
        (output_root / H3_COMPLETE_PATH).write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_staging(output_root, requested_root)
        return True

    rank_zero_phase(finalize, "corpus finalization")
    return 0


__all__ = [
    "H3_COMPLETE_PATH",
    "H3_ENCODER_CONTRACT_PATH",
    "H3_INDEX_PATH",
    "H3_SILENCE_PATH",
    "run_preencode",
]
