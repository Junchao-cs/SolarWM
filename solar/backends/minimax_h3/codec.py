"""Dependency-light contracts for MiniMax-H3 raw and encoded model inputs.

The concrete lazy runtime lives in :mod:`official_codec`; keeping its protocol
and geometry checks here lets CPU-only installations audit the data boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .camera import (
    first_frame_relative_w2c,
    validate_absolute_c2w,
    validate_normalized_intrinsics,
)
from .geometry import (
    STABLE_STAGE0P5_GEOMETRY,
    audio_latents_for_video,
    latent_aligned_pixel_indices,
)
from .layout import H3PackedLayout, build_stage0p5_layout

H3_PREENCODE_VERSION = "h3.158f.v1"
H3_TEXT_HIDDEN_DIM = 5120
H3_AUDIO_HIDDEN_DIM = 32


@dataclass(frozen=True)
class H3RawSample:
    """One exact contiguous source sample before model-specific encoding."""

    sample_id: str
    frames: object
    caption: str
    source_frame_indices: object
    camera_c2w: object
    camera_intrinsics: object
    height: int = STABLE_STAGE0P5_GEOMETRY.height
    width: int = STABLE_STAGE0P5_GEOMETRY.width
    source_fps: float | None = None


@dataclass(frozen=True)
class H3EncodedPayload:
    """Artifacts the official H3 preencoder must produce."""

    target_latents: object
    anchor_latent: object
    prompt_embeddings: object
    text_token_tags: object
    audio_latents: object


@dataclass(frozen=True)
class H3PreencodedSample:
    """Per-sample ``h3.158f.v1`` artifact; silence is stored separately."""

    sample_id: str
    encoder_identity: str
    target_latents: object
    anchor_latent: object
    prompt_embeddings: object
    text_token_tags: object
    source_frame_indices: object
    camera_c2w: object
    camera_intrinsics: object
    preencode_version: str = H3_PREENCODE_VERSION
    source_fps: float | None = None


@dataclass(frozen=True)
class H3ModelInputs:
    """Validated single-sample inputs ready for the H3 transformer adapter."""

    sample_id: str
    codec_identity: str
    preencode_version: str
    target_latents: object
    anchor_latent: object
    prompt_embeddings: object
    text_token_tags: object
    audio_latents: object
    layout: H3PackedLayout
    camera_viewmats: np.ndarray
    camera_intrinsics: np.ndarray
    source_frame_indices: np.ndarray
    source_fps: float | None


@runtime_checkable
class H3Codec(Protocol):
    """Required codec implementation used by offline preencoding.

    The methods must invoke the matching official H3 encoders.  Replacing the
    prompt, image anchor, or encoded-silence branches with zeros is not a valid
    implementation of this protocol.
    """

    identity: str

    def encode_target_video(self, sample: H3RawSample) -> object: ...

    def encode_visual_anchor(self, sample: H3RawSample) -> object: ...

    def encode_joint_prompt(self, sample: H3RawSample) -> tuple[object, object]: ...

    def encode_silence_audio(self, sample: H3RawSample) -> object: ...


def _shape(value: object, *, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{name} must expose a tensor-like shape")
    try:
        return tuple(int(size) for size in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} has an invalid shape {shape!r}") from exc


def _dtype_name(value: object, *, name: str) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise TypeError(f"{name} must expose a tensor-like dtype")
    result = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    return result


def _require_shape(value: object, expected: tuple[int, ...], *, name: str) -> None:
    observed = _shape(value, name=name)
    if observed != expected:
        raise ValueError(f"{name} shape {observed} != {expected}")


def _require_dtype(value: object, expected: str, *, name: str) -> None:
    observed = _dtype_name(value, name=name)
    aliases = {
        "bfloat16": {"bfloat16", "bf16"},
        "float32": {"float32", "float"},
        "int64": {"int64", "long"},
    }
    if observed not in aliases[expected]:
        raise TypeError(f"{name} dtype {observed!r} is not {expected}")


def validate_raw_sample(sample: H3RawSample) -> None:
    """Validate exact 158-frame selection and authoritative camera tracks."""

    if not isinstance(sample, H3RawSample):
        raise TypeError("sample must be H3RawSample")
    if not sample.sample_id.strip():
        raise ValueError("sample_id must be non-empty")
    if not isinstance(sample.caption, str):
        raise TypeError("caption must be a string")
    profile = STABLE_STAGE0P5_GEOMETRY
    if (sample.height, sample.width) != (profile.height, profile.width):
        raise ValueError(
            f"raw sample canvas must be {(profile.height, profile.width)}, "
            f"got {(sample.height, sample.width)}"
        )
    try:
        frame_count = len(sample.frames)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("frames must expose its temporal length") from exc
    if frame_count != profile.pixel_frames:
        raise ValueError(
            f"raw sample must contain {profile.pixel_frames} frames, got {frame_count}"
        )
    source_indices = np.asarray(sample.source_frame_indices)
    if source_indices.dtype != np.int64 or source_indices.shape != (profile.pixel_frames,):
        raise TypeError(f"source_frame_indices must be int64 [{profile.pixel_frames}]")
    if not np.all(np.diff(source_indices) == 1):
        raise ValueError("source_frame_indices must be exact contiguous frames")
    c2w = validate_absolute_c2w(sample.camera_c2w)
    K = validate_normalized_intrinsics(sample.camera_intrinsics)
    if c2w.shape != (profile.pixel_frames, 4, 4):
        raise ValueError(f"camera_c2w must be [{profile.pixel_frames},4,4]")
    if K.shape != (profile.pixel_frames, 3, 3):
        raise ValueError(f"camera_intrinsics must be [{profile.pixel_frames},3,3]")
    if c2w.dtype != np.float32 or K.dtype != np.float32:
        raise TypeError("camera_c2w and camera_intrinsics must be float32 artifacts")
    if sample.source_fps is not None and (
        not np.isfinite(sample.source_fps) or sample.source_fps <= 0
    ):
        raise ValueError("source_fps, when present for audit, must be positive")


def _validate_video_text_tensors(
    target_latents: object,
    anchor_latent: object,
    prompt_embeddings: object,
    text_token_tags: object,
) -> None:
    profile = STABLE_STAGE0P5_GEOMETRY
    _require_shape(
        target_latents,
        (
            profile.latent_channels,
            profile.encoded_latents,
            profile.latent_height,
            profile.latent_width,
        ),
        name="target_latents",
    )
    _require_shape(
        anchor_latent,
        (
            profile.latent_channels,
            1,
            profile.latent_height,
            profile.latent_width,
        ),
        name="anchor_latent",
    )
    prompt_shape = _shape(prompt_embeddings, name="prompt_embeddings")
    if len(prompt_shape) != 2 or prompt_shape[0] < 1 or prompt_shape[1] != H3_TEXT_HIDDEN_DIM:
        raise ValueError("prompt_embeddings must have shape [L,5120] with L >= 1")
    _require_shape(
        text_token_tags,
        (prompt_shape[0],),
        name="text_token_tags",
    )
    for name, value in (
        ("target_latents", target_latents),
        ("anchor_latent", anchor_latent),
        ("prompt_embeddings", prompt_embeddings),
    ):
        _require_dtype(value, "bfloat16", name=name)
    _require_dtype(text_token_tags, "int64", name="text_token_tags")


def validate_encoded_payload(payload: H3EncodedPayload) -> None:
    """Validate all tensor outputs of an injected preencoding codec."""

    if not isinstance(payload, H3EncodedPayload):
        raise TypeError("codec output must be H3EncodedPayload")
    _validate_video_text_tensors(
        payload.target_latents,
        payload.anchor_latent,
        payload.prompt_embeddings,
        payload.text_token_tags,
    )
    validate_silence_latents(payload.audio_latents)


def validate_preencoded_sample(sample: H3PreencodedSample) -> None:
    """Validate one immutable per-sample ``h3.158f.v1`` artifact."""

    if not isinstance(sample, H3PreencodedSample):
        raise TypeError("sample must be H3PreencodedSample")
    if not sample.sample_id.strip() or not sample.encoder_identity.strip():
        raise ValueError("sample_id and encoder_identity must be non-empty")
    if sample.preencode_version != H3_PREENCODE_VERSION:
        raise ValueError(
            f"preencode_version must be {H3_PREENCODE_VERSION!r}, got {sample.preencode_version!r}"
        )
    profile = STABLE_STAGE0P5_GEOMETRY
    _validate_video_text_tensors(
        sample.target_latents,
        sample.anchor_latent,
        sample.prompt_embeddings,
        sample.text_token_tags,
    )
    source_indices = np.asarray(sample.source_frame_indices)
    if source_indices.dtype != np.int64 or source_indices.shape != (profile.pixel_frames,):
        raise TypeError(f"source_frame_indices must be int64 [{profile.pixel_frames}]")
    if not np.all(np.diff(source_indices) == 1):
        raise ValueError("source_frame_indices must be exact contiguous frames")
    c2w = validate_absolute_c2w(sample.camera_c2w)
    intrinsics = validate_normalized_intrinsics(sample.camera_intrinsics)
    if c2w.shape != (profile.pixel_frames, 4, 4):
        raise ValueError(f"camera_c2w must be [{profile.pixel_frames},4,4]")
    if intrinsics.shape != (profile.pixel_frames, 3, 3):
        raise ValueError(f"camera_intrinsics must be [{profile.pixel_frames},3,3]")
    if c2w.dtype != np.float32 or intrinsics.dtype != np.float32:
        raise TypeError("camera_c2w and camera_intrinsics must be float32 artifacts")
    if sample.source_fps is not None and (
        not np.isfinite(sample.source_fps) or sample.source_fps <= 0
    ):
        raise ValueError("source_fps, when present for audit, must be positive")


def validate_silence_latents(silence_latents: object) -> None:
    """Validate the separately versioned official encoded-silence artifact."""

    profile = STABLE_STAGE0P5_GEOMETRY
    _require_shape(
        silence_latents,
        (2, H3_AUDIO_HIDDEN_DIM, audio_latents_for_video(profile.pixel_frames)),
        name="audio_latents",
    )
    _require_dtype(silence_latents, "bfloat16", name="audio_latents")


def assemble_model_inputs(
    sample: H3PreencodedSample,
    silence_latents: object,
) -> H3ModelInputs:
    """Convert validated sample/global artifacts into packed model inputs."""

    validate_preencoded_sample(sample)
    validate_silence_latents(silence_latents)
    profile = STABLE_STAGE0P5_GEOMETRY
    alignment = latent_aligned_pixel_indices(profile.pixel_frames)
    c2w = np.asarray(sample.camera_c2w, dtype=np.float32)[alignment]
    latent_K = np.asarray(sample.camera_intrinsics, dtype=np.float32)[alignment]
    relative_w2c = first_frame_relative_w2c(c2w)
    tags_array = np.asarray(sample.text_token_tags, dtype=np.int64)
    layout = build_stage0p5_layout(
        tags_array,
        profile.encoded_latents,
        profile.latent_height,
        profile.latent_width,
        audio_latents_for_video(profile.pixel_frames),
    )
    return H3ModelInputs(
        sample_id=sample.sample_id,
        codec_identity=sample.encoder_identity,
        preencode_version=sample.preencode_version,
        target_latents=sample.target_latents,
        anchor_latent=sample.anchor_latent,
        prompt_embeddings=sample.prompt_embeddings,
        text_token_tags=sample.text_token_tags,
        audio_latents=silence_latents,
        layout=layout,
        camera_viewmats=relative_w2c[layout.camera_frame_ids],
        camera_intrinsics=latent_K[layout.camera_frame_ids],
        source_frame_indices=np.asarray(sample.source_frame_indices, dtype=np.int64),
        source_fps=sample.source_fps,
    )


def encode_raw_sample(sample: H3RawSample, codec: H3Codec) -> H3ModelInputs:
    """Run an official-compatible codec and assemble validated model inputs."""

    validate_raw_sample(sample)
    if not isinstance(codec, H3Codec):
        raise TypeError("codec does not implement the complete H3Codec protocol")
    identity = str(codec.identity).strip()
    if not identity:
        raise ValueError("codec.identity must be a non-empty provenance identifier")
    prompt, tags = codec.encode_joint_prompt(sample)
    payload = H3EncodedPayload(
        target_latents=codec.encode_target_video(sample),
        anchor_latent=codec.encode_visual_anchor(sample),
        prompt_embeddings=prompt,
        text_token_tags=tags,
        audio_latents=codec.encode_silence_audio(sample),
    )
    validate_encoded_payload(payload)

    preencoded = H3PreencodedSample(
        sample_id=sample.sample_id,
        encoder_identity=identity,
        target_latents=payload.target_latents,
        anchor_latent=payload.anchor_latent,
        prompt_embeddings=payload.prompt_embeddings,
        text_token_tags=payload.text_token_tags,
        source_frame_indices=sample.source_frame_indices,
        camera_c2w=sample.camera_c2w,
        camera_intrinsics=sample.camera_intrinsics,
        source_fps=sample.source_fps,
    )
    return assemble_model_inputs(preencoded, payload.audio_latents)


__all__ = [
    "H3_AUDIO_HIDDEN_DIM",
    "H3_PREENCODE_VERSION",
    "H3_TEXT_HIDDEN_DIM",
    "H3Codec",
    "H3EncodedPayload",
    "H3ModelInputs",
    "H3PreencodedSample",
    "H3RawSample",
    "assemble_model_inputs",
    "encode_raw_sample",
    "validate_encoded_payload",
    "validate_preencoded_sample",
    "validate_raw_sample",
    "validate_silence_latents",
]
