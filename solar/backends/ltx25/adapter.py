"""Pure packed-document and parameter-free adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solar.errors import BackendContractError

from .camera import TokenCamera, expand_token_camera
from .geometry import STABLE_GEOMETRY

FUSED_PROPE_ORDER = (
    "native_ltx_3d_rope",
    "fused_camera_prope_qkv",
    "one_video_self_attention_call",
    "prope_output_transform",
    "native_head_gate_and_output_projection",
)
LORA_SUFFIXES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def lora_target_modules() -> tuple[str, ...]:
    """Return the exact 48-block q/k/v/o + FFN target set (480 modules)."""

    targets = tuple(
        f"transformer_blocks.{block}.{suffix}" for block in range(48) for suffix in LORA_SUFFIXES
    )
    if len(targets) != 480 or len(set(targets)) != 480:
        raise BackendContractError("internal LTX LoRA target construction drifted")
    return targets


@dataclass(frozen=True)
class FusedPropeContract:
    """Parameter-free modification restricted to video ``attn1``."""

    attention_modules: int = 48
    parameter_keys_added: int = 0
    state_dict_keys_changed: bool = False
    applies_to: str = "video_self_attention_attn1_only"
    forbidden: tuple[str, ...] = ("caption_cross_attention_attn2", "gemma_connector")
    operation_order: tuple[str, ...] = FUSED_PROPE_ORDER

    def __post_init__(self) -> None:
        expected = (48, 0, False, "video_self_attention_attn1_only", FUSED_PROPE_ORDER)
        observed = (
            self.attention_modules,
            self.parameter_keys_added,
            self.state_dict_keys_changed,
            self.applies_to,
            self.operation_order,
        )
        if observed != expected:
            raise BackendContractError("fused PRoPE adapter contract drifted")


FUSED_PROPE_CONTRACT = FusedPropeContract()


@dataclass(frozen=True)
class PackedStage0p5:
    """Stage0.5 video document after native patch-size-one packing."""

    video_tokens: np.ndarray
    token_timesteps: np.ndarray
    first_frame_mask: np.ndarray
    loss_mask: np.ndarray
    camera: TokenCamera | None

    def __post_init__(self) -> None:
        if self.video_tokens.ndim != 3 or self.video_tokens.shape[1:] != (
            STABLE_GEOMETRY.video_tokens,
            STABLE_GEOMETRY.latent_channels,
        ):
            raise BackendContractError("packed video must be [B,7680,128]")
        batch = self.video_tokens.shape[0]
        expected = (batch, STABLE_GEOMETRY.video_tokens)
        for name, value in (
            ("token_timesteps", self.token_timesteps),
            ("first_frame_mask", self.first_frame_mask),
            ("loss_mask", self.loss_mask),
        ):
            if value.shape != expected:
                raise BackendContractError(f"{name} must be {expected}")
        if not np.array_equal(self.loss_mask, ~self.first_frame_mask):
            raise BackendContractError("loss mask must exclude exactly the clean first latent")
        if self.camera is not None and self.camera.viewmats.shape[0] != batch:
            raise BackendContractError("camera and video batch sizes differ")


def pack_stage0p5(
    video_latent: object,
    sigma: object,
    *,
    relative_w2c: object | None = None,
    camera_K: object | None = None,
) -> PackedStage0p5:
    """Pack ``[B,C,T,H,W]`` in exact F-H-W-major token order."""

    latent = np.asarray(video_latent)
    expected = (
        STABLE_GEOMETRY.latent_channels,
        STABLE_GEOMETRY.latent_frames,
        STABLE_GEOMETRY.latent_height,
        STABLE_GEOMETRY.latent_width,
    )
    if latent.ndim != 5 or latent.shape[1:] != expected:
        raise BackendContractError("video_latent must be [B,128,20,16,24]")
    if not np.issubdtype(latent.dtype, np.floating):
        raise BackendContractError("video_latent must use a floating dtype")
    batch = latent.shape[0]
    sigma_array = np.asarray(sigma, dtype=np.float32)
    if sigma_array.ndim == 0:
        sigma_array = np.full((batch,), sigma_array, dtype=np.float32)
    if (
        sigma_array.shape != (batch,)
        or not np.isfinite(sigma_array).all()
        or np.any(sigma_array < 0)
        or np.any(sigma_array > 1)
    ):
        raise BackendContractError("sigma must be scalar or finite [B] in [0,1]")
    tokens = latent.transpose(0, 2, 3, 4, 1).reshape(batch, STABLE_GEOMETRY.video_tokens, -1)
    first = np.zeros((batch, STABLE_GEOMETRY.video_tokens), dtype=np.bool_)
    first[:, : STABLE_GEOMETRY.tokens_per_latent] = True
    timesteps = np.broadcast_to(sigma_array[:, None], first.shape).copy()
    timesteps[first] = 0.0
    if (relative_w2c is None) != (camera_K is None):
        raise BackendContractError("camera requires both relative_w2c and camera_K")
    camera = None if relative_w2c is None else expand_token_camera(relative_w2c, camera_K)
    return PackedStage0p5(tokens, timesteps, first, ~first, camera)


def unpack_video_tokens(tokens: object) -> np.ndarray:
    """Invert patch-size-one F-H-W-major packing."""

    value = np.asarray(tokens)
    if value.ndim != 3 or value.shape[1:] != (
        STABLE_GEOMETRY.video_tokens,
        STABLE_GEOMETRY.latent_channels,
    ):
        raise BackendContractError("video tokens must be [B,7680,128]")
    batch = value.shape[0]
    return value.reshape(
        batch,
        STABLE_GEOMETRY.latent_frames,
        STABLE_GEOMETRY.latent_height,
        STABLE_GEOMETRY.latent_width,
        STABLE_GEOMETRY.latent_channels,
    ).transpose(0, 4, 1, 2, 3)


def verify_parameter_free_state_keys(before: object, after: object) -> None:
    if tuple(before) != tuple(after):
        raise BackendContractError("parameter-free LTX adapter changed checkpoint state keys")


__all__ = [
    "FUSED_PROPE_CONTRACT",
    "FUSED_PROPE_ORDER",
    "FusedPropeContract",
    "PackedStage0p5",
    "lora_target_modules",
    "pack_stage0p5",
    "unpack_video_tokens",
    "verify_parameter_free_state_keys",
]
