"""The same codec contract is used by offline production and online training."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from solar.config.loader import canonical_json
from solar.errors import DataContractError


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | None, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name or not self.shape or not self.dtype:
            raise DataContractError("tensor spec requires name, shape, and dtype")
        if any(value is not None and value < 0 for value in self.shape):
            raise DataContractError(f"tensor spec {self.name!r} has a negative dimension")


@dataclass(frozen=True)
class EncoderContract:
    schema: str
    family: str
    format_version: str
    pixel_frames: int
    latent_frames: int
    height: int
    width: int
    camera_convention: str
    tensors: tuple[TensorSpec, ...]
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != "solarwm.encoder.v1":
            raise DataContractError("encoder schema must be solarwm.encoder.v1")
        if not self.family or not self.format_version or not self.camera_convention:
            raise DataContractError("encoder family/format/camera convention are required")
        if min(self.pixel_frames, self.latent_frames, self.height, self.width) < 1:
            raise DataContractError("encoder dimensions must be positive")
        names = [spec.name for spec in self.tensors]
        if not names or len(names) != len(set(names)):
            raise DataContractError("encoder tensor names must be non-empty and unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "tensors": [asdict(spec) for spec in self.tensors],
        }

    @property
    def digest(self) -> str:
        return hashlib.blake2s(canonical_json(self.as_dict())).hexdigest()


class OnlineCodec(Protocol):
    """Model-owned raw-to-tensor implementation shared with offline encoding."""

    @property
    def contract(self) -> EncoderContract: ...

    def encode(
        self,
        *,
        sample_id: str,
        pixels: Any,
        caption: str,
        camera: Any,
        seed: int,
    ) -> Mapping[str, Any]: ...


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise DataContractError(f"tensor {type(value).__name__} has no dtype")
    rendered = str(dtype).lower()
    for prefix in ("torch.", "numpy."):
        if rendered.startswith(prefix):
            rendered = rendered[len(prefix) :]
    return rendered


def validate_encoded_tensors(values: Mapping[str, Any], contract: EncoderContract) -> None:
    expected = {spec.name: spec for spec in contract.tensors}
    if set(values) != set(expected):
        raise DataContractError(
            f"encoded tensor fields differ: got {sorted(values)}, expected {sorted(expected)}"
        )
    for name, spec in expected.items():
        value = values[name]
        try:
            shape = tuple(int(dimension) for dimension in value.shape)
        except (AttributeError, TypeError, ValueError) as exc:
            raise DataContractError(f"encoded field {name!r} has no valid shape") from exc
        if len(shape) != len(spec.shape) or any(
            expected_dimension is not None and actual != expected_dimension
            for actual, expected_dimension in zip(shape, spec.shape, strict=True)
        ):
            raise DataContractError(
                f"encoded field {name!r} shape {shape} != contract {spec.shape}"
            )
        dtype = _dtype_name(value)
        if dtype != spec.dtype.lower():
            raise DataContractError(f"encoded field {name!r} dtype {dtype!r} != {spec.dtype!r}")
        tensor_isfinite = getattr(value, "isfinite", None)
        if callable(tensor_isfinite):
            finite = tensor_isfinite()
        else:
            try:
                import numpy as np

                finite = np.isfinite(value)
            except (TypeError, ValueError) as exc:
                raise DataContractError(
                    f"encoded field {name!r} cannot be checked for finiteness"
                ) from exc
        finite_all = finite.all()
        if hasattr(finite_all, "item"):
            finite_all = finite_all.item()
        if not bool(finite_all):
            raise DataContractError(f"encoded field {name!r} contains non-finite values")


@dataclass(frozen=True)
class EncodedPayload:
    """Serialized sample ready for deterministic tar packing."""

    sample_id: str
    key: str
    source_sample_id: str
    start_frame: int
    source_frame_indices: tuple[int, ...]
    encoder_contract_digest: str
    members: Mapping[str, bytes]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.key or not self.source_sample_id:
            raise DataContractError("encoded payload lacks sample/key/source identity")
        if self.start_frame < 0 or not self.source_frame_indices:
            raise DataContractError("encoded payload has an invalid frame window")
        if self.source_frame_indices[0] != self.start_frame:
            raise DataContractError("encoded payload start differs from first source frame")
        if len(self.encoder_contract_digest) != 64:
            raise DataContractError(
                "encoded payload has an invalid encoder contract content digest"
            )
        if not self.members:
            raise DataContractError("encoded payload has no serialized members")
        for suffix, value in self.members.items():
            if not suffix or "/" in suffix or suffix in {".", ".."} or not isinstance(value, bytes):
                raise DataContractError(f"invalid encoded member suffix {suffix!r}")
