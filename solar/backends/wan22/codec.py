"""Wan preencoded tensor schemas and online-codec extension protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from solar.errors import BackendContractError

from .contracts import PROFILES, WAN_FAMILIES, WAN_TI2V_5B_720P_153F_SCHEMA


@dataclass(frozen=True)
class PreencodedProfile:
    schema: str
    family: str
    pixel_frames: int
    latent_shape: tuple[int, int, int, int]
    prompt_shape: tuple[int, int]
    dtype: str = "bfloat16"
    i2v_y_shape: tuple[int, int, int, int] | None = None


PREENCODED_PROFILES = {
    "solarwm.wan22_ti2v_5b.480p.81f.v1": PreencodedProfile(
        schema="solarwm.wan22_ti2v_5b.480p.81f.v1",
        family="wan22_ti2v_5b",
        pixel_frames=81,
        latent_shape=(21, 48, 30, 54),
        prompt_shape=(512, 4096),
    ),
    "solarwm.wan22_ti2v_5b.480p.153f.v1": PreencodedProfile(
        schema="solarwm.wan22_ti2v_5b.480p.153f.v1",
        family="wan22_ti2v_5b",
        pixel_frames=153,
        latent_shape=(39, 48, 30, 54),
        prompt_shape=(512, 4096),
    ),
    "solarwm.wan22_i2v_a14b.480p.153f.v1": PreencodedProfile(
        schema="solarwm.wan22_i2v_a14b.480p.153f.v1",
        family="wan22_i2v_a14b",
        pixel_frames=153,
        latent_shape=(39, 16, 60, 104),
        prompt_shape=(512, 4096),
        i2v_y_shape=(39, 20, 60, 104),
    ),
    WAN_TI2V_5B_720P_153F_SCHEMA: PreencodedProfile(
        schema=WAN_TI2V_5B_720P_153F_SCHEMA,
        family="wan22_ti2v_5b",
        pixel_frames=153,
        latent_shape=(39, 48, 44, 80),
        prompt_shape=(512, 4096),
    ),
}

PREENCODE_WINDOW_ASSIGNMENTS = frozenset({"deterministic_hash_v1", "materialized_index_v1"})


def expected_tensor_shapes(
    *, family: str, pixel_frames: int
) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int, int, int] | None]:
    if family not in WAN_FAMILIES:
        raise BackendContractError(f"unknown Wan family {family!r}")
    if pixel_frames not in {81, 153}:
        raise BackendContractError("Wan tensor profiles support 81 or 153 pixel frames")
    latent_frames = 21 if pixel_frames == 81 else 39
    if family == "wan22_ti2v_5b":
        return (latent_frames, 48, 30, 54), (512, 4096), None
    return (
        (latent_frames, 16, 60, 104),
        (512, 4096),
        (latent_frames, 20, 60, 104),
    )


def validate_tensor_data_config(data: Mapping[str, Any], *, family: str) -> None:
    pixel_frames = int(data.get("pixel_frames", 0))
    if str(data.get("encoding", "")).strip().lower() == "preencoded":
        profile = profile_for(data.get("preencode_schema"), family=family)
        latent, prompt, i2v_y = (
            profile.latent_shape,
            profile.prompt_shape,
            profile.i2v_y_shape,
        )
    else:
        latent, prompt, i2v_y = expected_tensor_shapes(family=family, pixel_frames=pixel_frames)
    if str(data.get("latent_dtype", "")) != "bfloat16":
        raise BackendContractError("data.latent_dtype must be bfloat16")
    if tuple(data.get("latent_shape", ())) != latent:
        raise BackendContractError(f"data.latent_shape must be {list(latent)}")
    if tuple(data.get("prompt_shape", ())) != prompt:
        raise BackendContractError(f"data.prompt_shape must be {list(prompt)}")
    if i2v_y is not None and tuple(data.get("i2v_y_shape", ())) != i2v_y:
        raise BackendContractError(f"data.i2v_y_shape must be {list(i2v_y)}")
    if str(data.get("caption_source", "")) != "frozen":
        raise BackendContractError("data.caption_source must be frozen")
    if str(data.get("camera_source", "")) != "preserve":
        raise BackendContractError("data.camera_source must be preserve")


@runtime_checkable
class OnlineCodec(Protocol):
    """Model-specific raw sample encoder used below the common data plan."""

    family: str

    def encode_video(self, frames: Any, *, first_frame_conditioned: bool) -> Mapping[str, Any]: ...

    def encode_text(self, caption: str) -> Any: ...


def profile_for(schema: Any, *, family: str) -> PreencodedProfile:
    key = str(schema or "")
    try:
        profile = PREENCODED_PROFILES[key]
    except KeyError as exc:
        raise BackendContractError(f"unknown Wan preencode schema {key!r}") from exc
    if profile.family != family:
        raise BackendContractError(
            f"preencode schema {key!r} belongs to {profile.family}, not {family}"
        )
    return profile


def validate_preencoded_data_config(data: Mapping[str, Any], *, family: str) -> PreencodedProfile:
    profile = profile_for(data.get("preencode_schema"), family=family)
    if int(data.get("pixel_frames", 0)) != profile.pixel_frames:
        raise BackendContractError(f"{profile.schema} requires {profile.pixel_frames} pixel frames")
    if str(data.get("latent_dtype", "")) != profile.dtype:
        raise BackendContractError(f"{profile.schema} requires latent_dtype={profile.dtype}")
    if tuple(data.get("latent_shape", ())) != profile.latent_shape:
        raise BackendContractError(f"latent_shape must be {list(profile.latent_shape)}")
    if tuple(data.get("prompt_shape", ())) != profile.prompt_shape:
        raise BackendContractError(f"prompt_shape must be {list(profile.prompt_shape)}")
    if (
        profile.i2v_y_shape is not None
        and tuple(data.get("i2v_y_shape", ())) != profile.i2v_y_shape
    ):
        raise BackendContractError(f"i2v_y_shape must be {list(profile.i2v_y_shape)}")
    window_assignment = str(data.get("preencode_window_assignment", ""))
    if window_assignment not in PREENCODE_WINDOW_ASSIGNMENTS:
        raise BackendContractError(
            "data.preencode_window_assignment must be deterministic_hash_v1 "
            "or materialized_index_v1"
        )
    return profile


def validate_preencode_config(
    config: Mapping[str, Any], *, expected_family: str
) -> PreencodedProfile:
    if expected_family not in WAN_FAMILIES:
        raise BackendContractError(f"unknown Wan family {expected_family!r}")
    model = config.get("model", {})
    data = config.get("data", {})
    preencode = config.get("preencode", {})
    if not all(isinstance(value, Mapping) for value in (model, data, preencode)):
        raise BackendContractError("model, data, and preencode must be mappings")
    if str(model.get("family", "")) != expected_family:
        raise BackendContractError(f"model.family must be {expected_family}")
    from .contracts import _validate_model_assets

    _validate_model_assets(model)
    profile = profile_for(preencode.get("schema"), family=expected_family)
    family_profile = PROFILES[expected_family]
    required = {
        "data.pixel_frames": int(data.get("pixel_frames", 0)) == profile.pixel_frames,
        "data.height": int(data.get("height", 0)) == family_profile.height,
        "data.width": int(data.get("width", 0)) == family_profile.width,
        "data.fps": float(data.get("fps", 0.0)) == 16.0,
        "data.index": _is_relative(data.get("index")),
        "data.camera_array_key": str(data.get("camera_array_key", "")) == "c2w",
        "data.max_rel_translation": float(data.get("max_rel_translation", 0.0)) == 20.0,
        "data.max_camera_abs": float(data.get("max_camera_abs", 0.0)) == 20.0,
        "preencode.dtype": str(preencode.get("dtype", "")) == "bfloat16",
        "preencode.caption_source": str(preencode.get("caption_source", "")) == "frozen",
        "preencode.camera_source": str(preencode.get("camera_source", "")) == "preserve",
        "preencode.output_root": str(preencode.get("output_root", "")).startswith("/"),
        "preencode.logical_output_root": str(preencode.get("logical_output_root", "")).startswith(
            "/"
        ),
        "preencode.generation_id": _is_component(preencode.get("generation_id")),
        "preencode.shard_max_samples": int(preencode.get("shard_max_samples", 0)) > 0,
        "preencode.shard_max_bytes": int(preencode.get("shard_max_bytes", 0)) > 0,
    }
    physical_root = Path(str(preencode.get("output_root", ""))).resolve()
    logical_root = Path(str(preencode.get("logical_output_root", ""))).resolve()
    if (
        physical_root == logical_root
        or physical_root in logical_root.parents
        or logical_root in physical_root.parents
    ):
        required["preencode.output_roots_nonoverlapping"] = False
    failed = [field for field, accepted in required.items() if not accepted]
    if failed:
        raise BackendContractError(f"Wan preencode contract mismatch: {failed}")
    return profile


def _is_relative(value: Any) -> bool:
    text = str(value or "")
    return (
        bool(text)
        and not text.startswith("/")
        and "://" not in text
        and ".." not in text.split("/")
    )


def _is_component(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and "/" not in text and "\\" not in text and text not in {".", ".."}


__all__ = [
    "PREENCODED_PROFILES",
    "PREENCODE_WINDOW_ASSIGNMENTS",
    "OnlineCodec",
    "PreencodedProfile",
    "expected_tensor_shapes",
    "profile_for",
    "validate_preencode_config",
    "validate_preencoded_data_config",
    "validate_tensor_data_config",
]
