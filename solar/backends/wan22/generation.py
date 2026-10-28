"""One generation-plan parser shared by validation and standalone inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from solar.errors import BackendContractError
from solar.runtime.output_layout import (
    camera_inference_output_layout,
    validation_pass_component,
)


@dataclass(frozen=True)
class GenerationPass:
    name: str
    weights: str
    mode: str
    solver: str
    num_inference_steps: int
    rollout_latent_frames: int
    min_rollout_latent_frames: int
    fixed_plan_pixel_frames: int
    variable_rollout_by_source: bool


@dataclass(frozen=True)
class GenerationPlan:
    index: str
    selection_seed: int
    noise_seed: int
    sample_count: int
    passes: tuple[GenerationPass, ...]


def _relative_test_index(value: Any) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text
        or text.startswith("/")
        or "://" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BackendContractError("data.test_index must be a relative POSIX path")
    return path.as_posix()


def _component(value: Any) -> str:
    return validation_pass_component(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendContractError(f"{name} must be a non-negative integer")
    return value


def resolve_generation_plan(config: Mapping[str, Any]) -> GenerationPlan:
    """Resolve one generation contract for training validation and inference.

    ``inference.source`` must be ``validation``; alternate inference-only
    sampler blocks are rejected so standalone inference cannot drift from the
    checkpoint's validation behavior.
    """

    validation = config.get("validation", {})
    inference = config.get("inference", {})
    if not isinstance(validation, Mapping) or not isinstance(inference, Mapping):
        raise BackendContractError("validation and inference must be mappings")
    train = config.get("train", {})
    data = config.get("data", {})
    if not isinstance(train, Mapping) or not isinstance(data, Mapping):
        raise BackendContractError("train and data must be mappings")
    stage = str(train.get("stage", "")).strip().lower()
    objective = str(train.get("objective", "")).strip().lower()
    action = str(config.get("action", "")).strip().lower()
    length_mode = str(inference.get("length", "fixed")).strip().lower()
    if length_mode not in {"fixed", "camera"}:
        raise BackendContractError("inference.length must be fixed or camera")
    camera_length = length_mode == "camera"
    if camera_length and (action != "infer" or stage != "stage2"):
        raise BackendContractError(
            "camera-length inference is supported only for standalone Stage2 inference"
        )
    if camera_length:
        camera_inference_output_layout(config)
    if action == "infer" and str(inference.get("source", "")) != "validation":
        raise BackendContractError("inference.source must be validation")
    raw_passes = validation.get("passes", [])
    if not isinstance(raw_passes, list) or not raw_passes:
        raise BackendContractError("validation.passes must be a non-empty list")
    passes: list[GenerationPass] = []
    seen: set[str] = set()
    block = int(config.get("model", {}).get("num_frame_per_block", 0))
    for raw in raw_passes:
        if not isinstance(raw, Mapping):
            raise BackendContractError("every validation pass must be a mapping")
        rollout_latent_frames = int(raw.get("rollout_latent_frames", block if camera_length else 0))
        item = GenerationPass(
            name=_component(raw.get("name")),
            weights=str(raw.get("weights", "")).strip().lower(),
            mode=str(raw.get("mode", "")).strip().lower(),
            solver=str(raw.get("solver", "")).strip().lower(),
            num_inference_steps=int(raw.get("num_inference_steps", 0)),
            rollout_latent_frames=rollout_latent_frames,
            min_rollout_latent_frames=int(
                raw.get(
                    "min_rollout_latent_frames",
                    validation.get(
                        "min_rollout_latent_frames",
                        rollout_latent_frames,
                    ),
                )
            ),
            fixed_plan_pixel_frames=int(
                raw.get(
                    "fixed_plan_pixel_frames",
                    validation.get(
                        "fixed_plan_pixel_frames",
                        1 + 4 * (rollout_latent_frames - 1),
                    ),
                )
            ),
            variable_rollout_by_source=bool(
                raw.get(
                    "variable_rollout_by_source",
                    validation.get("variable_rollout_by_source", False),
                )
            ),
        )
        if item.name in seen:
            raise BackendContractError("validation pass names must be unique")
        allowed_weights = {"model"} if camera_length else {"live", "ema"}
        if item.weights not in allowed_weights:
            expected_weights = "model" if camera_length else "live or ema"
            raise BackendContractError(
                f"validation pass {item.name} weights must be {expected_weights}"
            )
        if item.mode not in {"bidirectional", "autoregressive"}:
            raise BackendContractError(f"validation pass {item.name} has invalid mode")
        if item.solver not in {"unipc", "flowmap", "self_forcing"}:
            raise BackendContractError(f"validation pass {item.name} has invalid solver")
        if item.num_inference_steps <= 0 or item.rollout_latent_frames <= 0:
            raise BackendContractError(
                f"validation pass {item.name} step/frame counts must be positive"
            )
        if (
            item.min_rollout_latent_frames <= 0
            or item.min_rollout_latent_frames > item.rollout_latent_frames
            or item.fixed_plan_pixel_frames != 1 + 4 * (item.rollout_latent_frames - 1)
        ):
            raise BackendContractError(
                f"validation pass {item.name} variable rollout bounds are invalid"
            )
        if item.variable_rollout_by_source:
            if stage not in {"stage1", "stage2"}:
                raise BackendContractError(
                    "source-variable validation requires an autoregressive stage"
                )
            if block < 1 or any(
                value % block
                for value in (
                    item.min_rollout_latent_frames,
                    item.rollout_latent_frames,
                )
            ):
                raise BackendContractError(
                    f"validation pass {item.name} variable rollout must align "
                    "to num_frame_per_block"
                )
        elif item.min_rollout_latent_frames != item.rollout_latent_frames:
            raise BackendContractError(
                f"validation pass {item.name} sets a minimum without variable_rollout_by_source"
            )
        expected = {
            "stage0p5": ("bidirectional", "unipc"),
            "stage1": (
                "autoregressive",
                "flowmap" if objective == "anyflow_forward_map" else "unipc",
            ),
            "stage2": ("autoregressive", "self_forcing"),
        }.get(stage)
        if expected is None or (item.mode, item.solver) != expected:
            raise BackendContractError(
                f"validation pass {item.name} mode/solver {(item.mode, item.solver)} "
                f"does not match {stage or 'unknown-stage'} {expected}"
            )
        if stage == "stage0p5" and item.rollout_latent_frames != int(data.get("latent_frames", 0)):
            raise BackendContractError(
                f"validation pass {item.name} must use the configured Stage0.5 length"
            )
        seen.add(item.name)
        passes.append(item)
    if camera_length and len(passes) != 1:
        raise BackendContractError("camera-length inference requires exactly one model pass")
    sample_count = _nonnegative_integer(validation.get("sample_count"), "validation.sample_count")
    if sample_count == 0:
        raise BackendContractError("validation.sample_count must be positive")
    return GenerationPlan(
        index=_relative_test_index(data.get("test_index")),
        selection_seed=_nonnegative_integer(
            validation.get("selection_seed"), "validation.selection_seed"
        ),
        noise_seed=_nonnegative_integer(validation.get("noise_seed"), "validation.noise_seed"),
        sample_count=sample_count,
        passes=tuple(passes),
    )


__all__ = ["GenerationPass", "GenerationPlan", "resolve_generation_plan"]
