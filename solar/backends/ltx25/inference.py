"""One shared inference plan used directly by standalone and validation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from solar.errors import BackendContractError

from .flow import restore_clean_first_latent
from .geometry import STABLE_GEOMETRY

NATIVE_BASE_SHIFT = 0.95
NATIVE_MAX_SHIFT = 2.05
NATIVE_TERMINAL = 0.1
NATIVE_TOKEN_COUNT = 4096
NATIVE_BASE_ANCHOR = 1024
NATIVE_MAX_ANCHOR = 4096

# Bit pattern produced by the Torch 2.9.1 CPU implementation of the
# official one-stage LTX2Scheduler expression.  NumPy linspace/division differs
# by up to two FP32 ULPs, which is enough to change a fixed-seed trajectory.
_NATIVE_SIGMA_BITS_30 = (
    0x3F7FFFFF,
    0x3F7EB581,
    0x3F7D56A0,
    0x3F7BE16C,
    0x3F7A53B2,
    0x3F78AAEF,
    0x3F76E44F,
    0x3F74FC85,
    0x3F72EFD5,
    0x3F70B9E3,
    0x3F6E559C,
    0x3F6BBD13,
    0x3F68E94C,
    0x3F65D1FA,
    0x3F626D32,
    0x3F5EAF04,
    0x3F5A88DB,
    0x3F55E8D7,
    0x3F50B8B8,
    0x3F4ADC7A,
    0x3F44305B,
    0x3F3C85E4,
    0x3F339FBE,
    0x3F292B10,
    0x3F1CB540,
    0x3F0D9B19,
    0x3EF5D804,
    0x3EC66BC0,
    0x3E8840F0,
    0x3DCCCCD0,
    0x00000000,
)


@dataclass(frozen=True)
class GuidanceSpec:
    cfg_scale: float = 3.0
    stg_scale: float = 1.0
    rescale_scale: float = 0.7
    stg_blocks: tuple[int, ...] = (28,)

    def __post_init__(self) -> None:
        values = (self.cfg_scale, self.stg_scale, self.rescale_scale)
        if any(not np.isfinite(value) for value in values):
            raise BackendContractError("guidance scales must be finite")
        if not 0 <= self.rescale_scale <= 1:
            raise BackendContractError("guidance rescale must be in [0,1]")
        if self.stg_scale != 0 and not self.stg_blocks:
            raise BackendContractError("nonzero STG requires a block list")
        if any(
            isinstance(block, bool) or not isinstance(block, int) or block < 0
            for block in self.stg_blocks
        ):
            raise BackendContractError("STG blocks must be nonnegative integers")
        object.__setattr__(self, "stg_blocks", tuple(self.stg_blocks))


@dataclass(frozen=True)
class InferenceSpec:
    num_inference_steps: int = 30
    seed: int = 10
    fps: int = 24
    decoder_mode: str = "diffvae_chunked_eager"

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_inference_steps, bool)
            or not isinstance(self.num_inference_steps, int)
            or self.num_inference_steps != 30
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise BackendContractError(
                "LTX inference requires exactly 30 steps and a nonnegative seed"
            )
        if self.fps != 24 or self.decoder_mode != "diffvae_chunked_eager":
            raise BackendContractError("LTX inference requires 24 FPS chunked eager DiffVAE")


def native_sigma_schedule(num_inference_steps: int = 30) -> np.ndarray:
    """Return the bit-exact 4,096-token shifted/stretched FP32 grid."""

    if isinstance(num_inference_steps, bool) or num_inference_steps != 30:
        raise BackendContractError("LTX inference requires exactly 30 steps")
    result = np.asarray(_NATIVE_SIGMA_BITS_30, dtype="<u4").view("<f4").copy()
    if np.any(result[:-1] <= result[1:]):
        raise BackendContractError("native sigma schedule must be strictly descending")
    return result


@dataclass(frozen=True)
class InferencePlan:
    spec: InferenceSpec
    guidance: GuidanceSpec
    sigmas: np.ndarray
    latent_shape: tuple[int, int, int, int] = STABLE_GEOMETRY.latent_shape
    clean_first_after_every_step: bool = True

    def __post_init__(self) -> None:
        if not np.array_equal(self.sigmas, native_sigma_schedule(self.spec.num_inference_steps)):
            raise BackendContractError("inference plan does not use the native LTX schedule")
        if (
            self.latent_shape != STABLE_GEOMETRY.latent_shape
            or not self.clean_first_after_every_step
        ):
            raise BackendContractError("inference geometry/first-frame policy drifted")
        sigmas = np.asarray(self.sigmas, dtype=np.float32).copy()
        sigmas.setflags(write=False)
        object.__setattr__(self, "sigmas", sigmas)


def build_inference_plan(
    spec: InferenceSpec | None = None,
    guidance: GuidanceSpec | None = None,
) -> InferencePlan:
    selected_spec = spec or InferenceSpec()
    return InferencePlan(
        spec=selected_spec,
        guidance=guidance or GuidanceSpec(),
        sigmas=native_sigma_schedule(selected_spec.num_inference_steps),
    )


def euler_velocity_step(
    sample: object,
    velocity: object,
    sigma: float,
    sigma_next: float,
) -> np.ndarray:
    value = np.asarray(sample)
    prediction = np.asarray(velocity)
    if value.shape != prediction.shape:
        raise BackendContractError("sample and velocity shapes differ")
    if not 0 <= sigma_next < sigma <= 1:
        raise BackendContractError("Euler requires 0 <= sigma_next < sigma <= 1")
    return value.astype(np.float32) + prediction.astype(np.float32) * np.float32(sigma_next - sigma)


def guided_clean_prediction(
    conditioned: object,
    unconditioned: object,
    perturbed: object,
    guidance: GuidanceSpec,
) -> np.ndarray:
    conditioned_array = np.asarray(conditioned)
    if not np.issubdtype(conditioned_array.dtype, np.floating):
        raise BackendContractError("guidance predictions must use floating dtypes")
    output_dtype = conditioned_array.dtype
    cond = conditioned_array.astype(np.float32)
    uncond = np.asarray(unconditioned, dtype=np.float32)
    stg = np.asarray(perturbed, dtype=np.float32)
    if cond.shape != uncond.shape or cond.shape != stg.shape:
        raise BackendContractError("guidance prediction shapes differ")
    prediction = (
        cond + (guidance.cfg_scale - 1.0) * (cond - uncond) + guidance.stg_scale * (cond - stg)
    )
    if guidance.rescale_scale:
        std = float(np.std(prediction, ddof=1))
        cond_std = float(np.std(cond, ddof=1))
        if not np.isfinite(std) or not np.isfinite(cond_std) or std == 0:
            raise BackendContractError("cannot rescale a zero-variance guidance prediction")
        factor = guidance.rescale_scale * cond_std / std + 1 - guidance.rescale_scale
        prediction *= factor
    return prediction.astype(output_dtype)


@runtime_checkable
class InferenceRunner(Protocol):
    def run(self, plan: InferencePlan, request: Any) -> Any: ...


@dataclass(frozen=True)
class ValidationInferenceAdapter:
    """Validation delegates to the same runner and exact same plan object."""

    runner: InferenceRunner
    plan: InferencePlan

    def __post_init__(self) -> None:
        if not isinstance(self.runner, InferenceRunner):
            raise BackendContractError("runner does not implement the inference protocol")

    def infer(self, request: Any) -> Any:
        return self.runner.run(self.plan, request)

    def validate(self, request: Any) -> Any:
        return self.runner.run(self.plan, request)


def restore_after_step(sample: object, first_frame: object) -> np.ndarray:
    return restore_clean_first_latent(sample, first_frame)


__all__ = [
    "GuidanceSpec",
    "InferencePlan",
    "InferenceRunner",
    "InferenceSpec",
    "ValidationInferenceAdapter",
    "build_inference_plan",
    "euler_velocity_step",
    "guided_clean_prediction",
    "native_sigma_schedule",
    "restore_after_step",
]
