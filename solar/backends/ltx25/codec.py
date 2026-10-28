"""Injectable online LTX codec boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from solar.errors import BackendContractError

from .artifact import PreencodedSample, TensorArtifact
from .camera import prepare_latent_camera
from .geometry import STABLE_GEOMETRY, cover_resize, validate_contiguous_source_indices

ONLINE_CODEC_PROTOCOL = "solarwm.ltx25.online_codec.v1"
ONLINE_BEHAVIOR_STATUS = "unpaired"


@dataclass(frozen=True)
class RawSample:
    """One exact source window before model-family encoding."""

    sample_id: str
    key: str
    frames: object
    caption: str
    source_indices: object
    camera_poses: object
    camera_intrinsics: object
    camera_convention: str
    source_height: int
    source_width: int
    source_fps: float


@dataclass(frozen=True)
class CodecPayload:
    """Only the tensors produced by official LTX VAE/Gemma computation."""

    video_latent: TensorArtifact
    first_frame_latent: TensorArtifact
    video_prompt_embeds: TensorArtifact
    prompt_attention_mask: TensorArtifact


@runtime_checkable
class LTX25OnlineCodec(Protocol):
    """Concrete implementation supplied by the selected LTX runtime."""

    identity: str

    def encode_video(self, frames: object) -> tuple[TensorArtifact, TensorArtifact]: ...

    def encode_prompt(self, caption: str) -> tuple[TensorArtifact, TensorArtifact]: ...


def validate_raw_sample(sample: RawSample) -> np.ndarray:
    if not isinstance(sample, RawSample):
        raise BackendContractError("sample must be an LTX RawSample")
    if (
        not isinstance(sample.sample_id, str)
        or not sample.sample_id.strip()
        or not isinstance(sample.key, str)
        or not sample.key.strip()
    ):
        raise BackendContractError("raw sample identity must be non-empty")
    if not isinstance(sample.caption, str):
        raise BackendContractError("caption must be a string")
    try:
        frame_count = len(sample.frames)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BackendContractError("frames must expose temporal length") from exc
    if frame_count != STABLE_GEOMETRY.pixel_frames:
        raise BackendContractError("raw LTX sample must contain exactly 153 frames")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (sample.source_height, sample.source_width)
    ):
        raise BackendContractError("source dimensions must be positive")
    if (
        isinstance(sample.source_fps, bool)
        or not isinstance(sample.source_fps, (int, float))
        or not np.isfinite(sample.source_fps)
        or sample.source_fps <= 0
    ):
        raise BackendContractError("source_fps must be positive provenance")
    return validate_contiguous_source_indices(sample.source_indices)


def encode_online(sample: RawSample, codec: LTX25OnlineCodec) -> PreencodedSample:
    """Encode through an injected official codec and reuse the artifact validator.

    The caller must bind codec weights, preprocessing, and identity in run
    provenance before changing the status.
    """

    source_indices = validate_raw_sample(sample)
    if not isinstance(codec, LTX25OnlineCodec):
        raise BackendContractError("codec does not implement the complete online protocol")
    if not isinstance(codec.identity, str) or not codec.identity.strip():
        raise BackendContractError("codec.identity must be a non-empty provenance value")
    video, first = codec.encode_video(sample.frames)
    prompt, mask = codec.encode_prompt(sample.caption)
    resize = cover_resize(sample.source_height, sample.source_width)
    relative, K = prepare_latent_camera(
        sample.camera_poses,
        sample.camera_intrinsics,
        convention=sample.camera_convention,
        resize=resize,
    )
    camera_source = source_indices[np.asarray(STABLE_GEOMETRY.camera_pixel_indices)]
    tensors = {
        "video_latent": video,
        "first_frame_latent": first,
        "video_prompt_embeds": prompt,
        "prompt_attention_mask": mask,
        "relative_w2c": TensorArtifact.from_numpy(relative, dtype="F32"),
        "camera_K": TensorArtifact.from_numpy(K, dtype="F32"),
        "source_indices": TensorArtifact.from_numpy(source_indices, dtype="I64"),
        "camera_source_indices": TensorArtifact.from_numpy(camera_source, dtype="I64"),
    }
    return PreencodedSample(
        sample_id=sample.sample_id,
        key=sample.key,
        start_frame=int(source_indices[0]),
        source_fps=float(sample.source_fps),
        tensors=tensors,
    )


def online_behavior_contract() -> dict[str, str]:
    return {
        "protocol": ONLINE_CODEC_PROTOCOL,
        "status": ONLINE_BEHAVIOR_STATUS,
        "required_implementation": "official_ltx_video_vae_plus_gemma4_feature_extractor",
    }


__all__ = [
    "ONLINE_BEHAVIOR_STATUS",
    "ONLINE_CODEC_PROTOCOL",
    "CodecPayload",
    "LTX25OnlineCodec",
    "RawSample",
    "encode_online",
    "online_behavior_contract",
    "validate_raw_sample",
]
