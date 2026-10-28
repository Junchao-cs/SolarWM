"""Canonical serialization from the official online codec into shared shards."""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from typing import Any

from solar.errors import BackendContractError
from solar.preencode import EncodedPayload, EncoderContract, TensorSpec
from solar.runtime.serialization import canonical_json_bytes

from .artifact import PreencodedSample, TensorArtifact, artifact_contract
from .codec import ONLINE_BEHAVIOR_STATUS, ONLINE_CODEC_PROTOCOL
from .geometry import STABLE_GEOMETRY


def encoder_contract() -> EncoderContract:
    return EncoderContract(
        schema="solarwm.encoder.v1",
        family="ltx25_video",
        format_version="solarwm_ltx25_video_153f_h512_w768_v1",
        pixel_frames=STABLE_GEOMETRY.pixel_frames,
        latent_frames=STABLE_GEOMETRY.latent_frames,
        height=STABLE_GEOMETRY.height,
        width=STABLE_GEOMETRY.width,
        camera_convention="relative_w2c+normalized_K",
        tensors=(
            TensorSpec("video_latent", (128, 20, 16, 24), "bfloat16"),
            TensorSpec("first_frame_latent", (128, 1, 16, 24), "bfloat16"),
            TensorSpec("video_prompt_embeds", (1024, 4096), "bfloat16"),
            TensorSpec("prompt_attention_mask", (1024,), "int64"),
            TensorSpec("relative_w2c", (20, 4, 4), "float32"),
            TensorSpec("camera_K", (20, 3, 3), "float32"),
            TensorSpec("source_indices", (153,), "int64"),
            TensorSpec("camera_source_indices", (20,), "int64"),
        ),
        extras={
            "codec_protocol": ONLINE_CODEC_PROTOCOL,
            "behavior_status": ONLINE_BEHAVIOR_STATUS,
            "artifact_contract": artifact_contract(),
        },
    )


def serialize_safetensors(
    tensors: Mapping[str, TensorArtifact],
    *,
    metadata: Mapping[str, str],
) -> bytes:
    """Serialize validated tensor bytes without importing Torch/safetensors."""

    if not tensors:
        raise BackendContractError("cannot serialize an empty LTX tensor mapping")
    if any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise BackendContractError("safetensors metadata must contain only strings")
    header: dict[str, Any] = {"__metadata__": dict(metadata)}
    payload = bytearray()
    offset = 0
    for name, tensor in sorted(tensors.items()):
        if not name or name == "__metadata__" or not isinstance(tensor, TensorArtifact):
            raise BackendContractError(f"invalid LTX safetensors member {name!r}")
        end = offset + len(tensor.data)
        header[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        payload.extend(tensor.data)
        offset = end
    raw_header = json.dumps(
        header,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    raw_header += b" " * (-len(raw_header) % 8)
    return struct.pack("<Q", len(raw_header)) + raw_header + bytes(payload)


def encoded_payload(sample: PreencodedSample, *, codec_identity: str) -> EncodedPayload:
    """Bind one strict sample to shared deterministic shard provenance."""

    if not codec_identity.strip():
        raise BackendContractError("preencoding requires a nonempty codec identity")
    contract = encoder_contract()
    tensor_bytes = serialize_safetensors(
        sample.tensors,
        metadata={
            "schema": sample.schema,
            "version": sample.version,
            "reader_contract": sample.reader_contract,
            "sample_id": sample.sample_id,
            "codec_identity": codec_identity,
        },
    )
    source_indices = tuple(int(item) for item in sample.tensors["source_indices"].numpy().tolist())
    manifest = {
        "schema": "solarwm.ltx25.preencoded-sample.v1",
        "sample_id": sample.sample_id,
        "key": sample.key,
        "start_frame": sample.start_frame,
        "source_fps": sample.source_fps,
        "source_frame_indices": list(source_indices),
        "encoder_contract_digest": contract.digest,
        "codec_identity": codec_identity,
        "artifact": artifact_contract(),
        "members": ["ltx25.safetensors", "manifest.json"],
    }
    return EncodedPayload(
        sample_id=sample.sample_id,
        key=sample.key,
        source_sample_id=sample.sample_id,
        start_frame=sample.start_frame,
        source_frame_indices=source_indices,
        encoder_contract_digest=contract.digest,
        members={
            "ltx25.safetensors": tensor_bytes,
            "manifest.json": canonical_json_bytes(manifest),
        },
        metadata={
            "codec_identity": codec_identity,
            "behavior_status": ONLINE_BEHAVIOR_STATUS,
            "source_fps": sample.source_fps,
        },
    )


__all__ = ["encoded_payload", "encoder_contract", "serialize_safetensors"]
