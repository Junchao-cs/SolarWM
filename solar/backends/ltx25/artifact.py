"""Exact trainer-facing schema for LTX-2.5 artifacts."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

from solar.errors import BackendContractError

from .camera import validate_latent_camera
from .geometry import STABLE_GEOMETRY, validate_contiguous_source_indices

PREENCODE_SCHEMA: Final = "solarwm_ltx25_video_preencoded_v1"
PREENCODE_VERSION: Final = "solarwm_ltx25_video_153f_h512_w768_v1"
READER_CONTRACT: Final = "ltx25.v1"
CAPTION_CACHE_STAGE: Final = "gemma4_feature_extractor_preconnector"
CAMERA_CONVENTION: Final = "relative_w2c+normalized_K"

_DTYPE_BYTES = MappingProxyType({"BF16": 2, "F32": 4, "I64": 8})


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        count = int(np.prod(self.shape, dtype=np.int64))
        return count * _DTYPE_BYTES[self.dtype]


TENSOR_SPECS: Final = MappingProxyType(
    {
        "video_latent": TensorSpec("BF16", STABLE_GEOMETRY.latent_shape),
        "first_frame_latent": TensorSpec("BF16", STABLE_GEOMETRY.first_frame_latent_shape),
        "video_prompt_embeds": TensorSpec("BF16", (1024, 4096)),
        "prompt_attention_mask": TensorSpec("I64", (1024,)),
        "relative_w2c": TensorSpec("F32", (20, 4, 4)),
        "camera_K": TensorSpec("F32", (20, 3, 3)),
        "source_indices": TensorSpec("I64", (153,)),
        "camera_source_indices": TensorSpec("I64", (20,)),
    }
)


@dataclass(frozen=True)
class TensorArtifact:
    """One contiguous little-endian tensor payload."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_BYTES:
            raise BackendContractError(f"unsupported artifact dtype {self.dtype!r}")
        shape = tuple(self.shape)
        if not shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape
        ):
            raise BackendContractError(f"invalid artifact shape {shape!r}")
        payload = bytes(self.data)
        expected = int(np.prod(shape, dtype=np.int64)) * _DTYPE_BYTES[self.dtype]
        if len(payload) != expected:
            raise BackendContractError(
                f"artifact byte length {len(payload)} != {expected} for {self.dtype}{shape}"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "data", payload)

    @classmethod
    def from_numpy(cls, value: object, *, dtype: str) -> TensorArtifact:
        """Build F32/I64 artifacts; BF16 must come from the official codec."""

        array = np.asarray(value)
        if dtype == "F32":
            encoded = np.ascontiguousarray(array, dtype="<f4")
        elif dtype == "I64":
            encoded = np.ascontiguousarray(array, dtype="<i8")
        else:
            raise BackendContractError(
                "NumPy cannot portably encode BF16; use official codec bytes"
            )
        return cls(dtype=dtype, shape=tuple(encoded.shape), data=encoded.tobytes(order="C"))

    def numpy(self) -> np.ndarray:
        if self.dtype == "F32":
            dtype = np.dtype("<f4")
        elif self.dtype == "I64":
            dtype = np.dtype("<i8")
        else:
            raise BackendContractError("BF16 artifact values require a BF16-aware runtime")
        return np.frombuffer(self.data, dtype=dtype).reshape(self.shape)


def _first_frame_bytes(video: TensorArtifact) -> bytes:
    channels, frames, height, width = video.shape
    plane_bytes = height * width * _DTYPE_BYTES[video.dtype]
    channel_bytes = frames * plane_bytes
    return b"".join(
        video.data[channel * channel_bytes : channel * channel_bytes + plane_bytes]
        for channel in range(channels)
    )


@dataclass(frozen=True)
class PreencodedSample:
    """One fully validated LTX training sample."""

    sample_id: str
    key: str
    start_frame: int
    source_fps: float
    tensors: Mapping[str, TensorArtifact]
    schema: str = PREENCODE_SCHEMA
    version: str = PREENCODE_VERSION
    reader_contract: str = READER_CONTRACT
    camera_convention: str = CAMERA_CONVENTION
    caption_cache_stage: str = CAPTION_CACHE_STAGE
    vae_encode_mode: str = "direct"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id.strip()
            or not isinstance(self.key, str)
            or not self.key.strip()
        ):
            raise BackendContractError("sample_id and key must be non-empty")
        if (
            isinstance(self.start_frame, bool)
            or not isinstance(self.start_frame, int)
            or self.start_frame < 0
        ):
            raise BackendContractError("start_frame must be a nonnegative integer")
        if (
            isinstance(self.source_fps, bool)
            or not isinstance(self.source_fps, (int, float))
            or not np.isfinite(self.source_fps)
            or self.source_fps <= 0
        ):
            raise BackendContractError("source_fps must be positive audit metadata")
        if (
            self.schema != PREENCODE_SCHEMA
            or self.version != PREENCODE_VERSION
            or self.reader_contract != READER_CONTRACT
        ):
            raise BackendContractError("LTX preencode schema/version contract drifted")
        if self.camera_convention != CAMERA_CONVENTION:
            raise BackendContractError("LTX camera convention must be relative_w2c+normalized_K")
        if self.caption_cache_stage != CAPTION_CACHE_STAGE:
            raise BackendContractError("caption cache must contain Gemma4 preconnector features")
        if self.vae_encode_mode != "direct":
            raise BackendContractError("the 153-frame profile requires direct VAE encoding")
        if set(self.tensors) != set(TENSOR_SPECS):
            raise BackendContractError(
                "artifact tensor keys differ; "
                f"missing={sorted(set(TENSOR_SPECS) - set(self.tensors))}, "
                f"extra={sorted(set(self.tensors) - set(TENSOR_SPECS))}"
            )
        for name, spec in TENSOR_SPECS.items():
            tensor = self.tensors[name]
            if not isinstance(tensor, TensorArtifact):
                raise BackendContractError(f"{name} must be a TensorArtifact")
            if (tensor.dtype, tensor.shape) != (spec.dtype, spec.shape):
                raise BackendContractError(
                    f"{name} must be {spec.dtype}{spec.shape}, got {tensor.dtype}{tensor.shape}"
                )
            if tensor.dtype == "BF16":
                bits = np.frombuffer(tensor.data, dtype="<u2")
                if np.any((bits & np.uint16(0x7F80)) == np.uint16(0x7F80)):
                    raise BackendContractError(f"{name} contains BF16 NaN or Inf")
        if self.tensors["first_frame_latent"].data != _first_frame_bytes(
            self.tensors["video_latent"]
        ):
            raise BackendContractError(
                "first_frame_latent must be bit-equal to video_latent[:,0:1]"
            )

        source = validate_contiguous_source_indices(
            self.tensors["source_indices"].numpy(), start=self.start_frame
        )
        camera_indices = self.tensors["camera_source_indices"].numpy()
        expected_camera = source[np.asarray(STABLE_GEOMETRY.camera_pixel_indices)]
        if not np.array_equal(camera_indices, expected_camera):
            raise BackendContractError(
                "camera_source_indices must select causal rows from source_indices"
            )
        mask = self.tensors["prompt_attention_mask"].numpy()
        if not np.any(mask) or np.any((mask != 0) & (mask != 1)):
            raise BackendContractError("prompt_attention_mask must be binary and non-empty")
        if any(left > right for left, right in itertools.pairwise(mask.tolist())):
            raise BackendContractError("prompt_attention_mask must use left padding")
        validate_latent_camera(
            self.tensors["relative_w2c"].numpy(),
            self.tensors["camera_K"].numpy(),
        )
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))


def artifact_contract() -> dict[str, object]:
    """Return a JSON-compatible public artifact schema."""

    return {
        "schema": PREENCODE_SCHEMA,
        "version": PREENCODE_VERSION,
        "reader_contract": READER_CONTRACT,
        "tensor_layout": "C,T,H,W",
        "camera_convention": CAMERA_CONVENTION,
        "caption_cache_stage": CAPTION_CACHE_STAGE,
        "source_fps_policy": "provenance_only_not_used_for_frame_selection",
        "model_fps": 24.0,
        "model_fps_usage": "ltx_rope_temporal_positions_only",
        "vae_encode_mode": "direct",
        "tensors": {
            name: {"dtype": spec.dtype, "shape": list(spec.shape)}
            for name, spec in TENSOR_SPECS.items()
        },
    }


__all__ = [
    "CAMERA_CONVENTION",
    "CAPTION_CACHE_STAGE",
    "PREENCODE_SCHEMA",
    "PREENCODE_VERSION",
    "READER_CONTRACT",
    "TENSOR_SPECS",
    "PreencodedSample",
    "TensorArtifact",
    "TensorSpec",
    "artifact_contract",
]
