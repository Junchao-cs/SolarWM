"""Training validation through the standalone LTX inference implementation."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from solar.errors import BackendContractError
from solar.inference import (
    InferenceEngine,
    publish_comparison_complete,
    publish_comparison_partition,
)
from solar.inference.validation_plan import (
    load_validation_plan,
    publish_validation_plan,
    validation_plan_key,
    validation_plan_payload,
)
from solar.runtime.distributed import collective_call, propagate_collective_error
from solar.runtime.output_layout import (
    cleanup_validation_staging,
    public_validation_dir,
    validation_pass_component,
    validation_staging_root,
)

from .checkpoint import StrictModelLoadReceipt
from .codec import LTX25OnlineCodec
from .inference import InferencePlan
from .official_codec import OfficialDiffVAEDecoder
from .torch_data import IndexedPreencodedSource
from .torch_inference import (
    LTX25InferenceAdapter,
    LTX25Sampler,
    inference_cases,
    load_negative_caption,
)


@dataclass(frozen=True)
class ValidationPass:
    name: str
    weights: str


def _validation_passes(value: object) -> tuple[ValidationPass, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise BackendContractError("LTX validation.passes must be a non-empty list")
    result = []
    seen = set()
    for raw in value:
        if isinstance(raw, Mapping):
            name = str(raw.get("name", "")).strip()
            weights = str(raw.get("weights", "")).strip().lower()
        else:
            name = weights = str(raw).strip().lower()
        name = validation_pass_component(name)
        if name in seen:
            raise BackendContractError("LTX validation pass names must be unique path components")
        if weights not in {"live", "ema"}:
            raise BackendContractError("LTX validation pass weights must be live or ema")
        seen.add(name)
        result.append(ValidationPass(name=name, weights=weights))
    return tuple(result)


def _partition_validation_cases(
    cases: tuple[Any, ...],
    *,
    dp_rank: int,
    dp_world_size: int,
) -> tuple[Any, ...]:
    """Assign every fixed case to one logical-DP group in deterministic waves."""

    if dp_world_size < 1 or not 0 <= dp_rank < dp_world_size:
        raise BackendContractError("LTX validation has invalid logical-DP topology")
    if not cases or len(cases) % dp_world_size:
        raise BackendContractError("LTX validation fixed cases must form complete logical-DP waves")
    return cases[dp_rank::dp_world_size]


class TrainingValidation:
    """Fixed cases and official decoder shared by every validation step."""

    def __init__(
        self,
        config: Mapping[str, Any],
        plan: InferencePlan,
        *,
        device: torch.device,
        model_receipt: StrictModelLoadReceipt,
        online_codec: LTX25OnlineCodec | None = None,
    ) -> None:
        validation = config["validation"]
        if not isinstance(validation, Mapping):
            raise BackendContractError("LTX validation config must be a mapping")
        inference = validation["inference"]
        if not isinstance(inference, Mapping):
            raise BackendContractError("LTX validation.inference must be a mapping")
        source_config = dict(config)
        source_data = dict(config["data"])
        source_data["index"] = source_data["test_index"]
        source_config["data"] = source_data
        if str(source_data["input_mode"]) == "raw_online":
            if online_codec is None:
                raise BackendContractError(
                    "raw-online LTX validation requires the shared official codec"
                )
            from .torch_raw import RawInferenceSource

            self.source = RawInferenceSource(source_config, online_codec)
        else:
            self.source = IndexedPreencodedSource(source_config)
        self.config = config
        self.sample_count = int(validation["sample_count"])
        self.plan_path = (
            Path(str(config["runtime"]["output_dir"])).expanduser().resolve()
            / "validation"
            / "frozen-plan.json"
        )
        self.plan_key = validation_plan_key("ltx25", config)
        self.negative_caption, self.negative_mask = load_negative_caption(
            str(inference["negative_caption_cache"]),
            device=device,
        )
        self.decoder = OfficialDiffVAEDecoder(
            str(config["model"]["codec"]["video_vae_path"]),
            device=device,
        )
        self.plan = plan
        self.device = device
        self.model_receipt = model_receipt
        self.output_dir = Path(str(config["runtime"]["output_dir"])).expanduser().resolve()
        self.sp_size = int(config["distributed"]["sequence_parallel_size"])
        self.passes = _validation_passes(validation.get("passes", ("live",)))

    def _resolve_cases(self) -> tuple[tuple[Any, ...], str, str]:
        cases = load_validation_plan(
            self.plan_path,
            backend="ltx25",
            plan_key=self.plan_key,
            expected_count=self.sample_count,
        )
        local_digest = (
            hashlib.sha256(self.plan_path.read_bytes()).hexdigest() if cases is not None else None
        )
        if dist.is_initialized():
            states: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(states, local_digest)
            present = [value for value in states if value is not None]
            if present and len(present) != len(states):
                raise BackendContractError(
                    "frozen LTX validation plan exists on only part of the distributed world"
                )
            if len(set(present)) > 1:
                raise BackendContractError("frozen LTX validation plan differs between nodes")
        if cases is not None:
            return cases, "loaded", str(local_digest)

        validation = self.config["validation"]
        created = inference_cases(
            self.source,
            self.plan,
            camera_translation_transform=str(self.config["model"]["camera_translation_transform"]),
            sample_count=self.sample_count,
            selection_seed=int(validation["selection_seed"]),
        )
        payload = validation_plan_payload(
            backend="ltx25",
            plan_key=self.plan_key,
            cases=created,
        )
        payload_digest = hashlib.sha256(payload).hexdigest()
        if dist.is_initialized():
            digests: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(digests, payload_digest)
            if len(set(digests)) != 1:
                raise BackendContractError("LTX ranks disagree before freezing validation plan")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_error: str | None = None
        if local_rank == 0:
            try:
                publish_validation_plan(
                    self.plan_path,
                    backend="ltx25",
                    plan_key=self.plan_key,
                    cases=created,
                )
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            failures: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(failures, local_error)
            errors = [value for value in failures if value]
            if errors:
                raise BackendContractError(
                    "LTX validation plan publication failed collectively: " + " | ".join(errors)
                )
            dist.barrier()
        elif local_error is not None:
            raise BackendContractError(f"LTX validation plan publication failed: {local_error}")
        return created, "created", payload_digest

    def __call__(self, runtime: Any, step: int) -> Mapping[str, Any]:
        self.cases, plan_source, plan_digest = self._resolve_cases()
        reports: dict[str, Any] = {}
        rank = int(runtime.distributed.rank)
        world_size = int(runtime.distributed.world_size)
        sp_rank = int(runtime.distributed.sp_rank)
        dp_rank = int(runtime.distributed.dp_rank)
        dp_world_size = int(runtime.distributed.dp_world_size)
        local_rank = int(runtime.distributed.local_rank)
        if world_size != dp_world_size * self.sp_size:
            raise BackendContractError(
                "LTX validation logical-DP topology does not cover the distributed world"
            )
        local_cases = _partition_validation_cases(
            self.cases,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        for validation_pass in self.passes:
            pass_name = validation_pass.name
            staging_pass = validation_staging_root(self.output_dir) / (
                f"step-{step:08d}-{pass_name}"
            )
            comparison_destination = public_validation_dir(
                self.output_dir,
                step=int(step),
                pass_name=pass_name,
            )
            pass_cases = tuple(
                replace(
                    case,
                    metadata={
                        **dict(case.metadata),
                        "generation_pass": {
                            "name": pass_name,
                            "weights": validation_pass.weights,
                            "mode": "bidirectional",
                            "solver": "stg-euler",
                            "num_inference_steps": self.plan.spec.num_inference_steps,
                            "rollout_latent_frames": case.metadata["rollout_latent_frames"],
                            "output_rollout_latent_frames": case.metadata["rollout_latent_frames"],
                        },
                    },
                )
                for case in local_cases
            )
            context = (
                runtime.ema.swapped_into(runtime.generator)
                if validation_pass.weights == "ema"
                else nullcontext()
            )
            adapter_source = self.model_receipt.adapter_checkpoint_manifest_digest or "initialized"
            weights_id = (
                f"{self.model_receipt.provider_identity}:"
                f"{self.model_receipt.ltx_core_version}:{adapter_source}:"
                f"step-{step}:{validation_pass.weights}:{pass_name}"
            )

            def exchange(
                local_error: str | None,
                phase: str,
                pass_name: str = pass_name,
            ) -> None:
                propagate_collective_error(
                    local_error,
                    dist=dist,
                    rank=rank,
                    world_size=world_size,
                    label=f"LTX validation {pass_name} {phase}",
                    error_type=BackendContractError,
                )

            entered = False

            def enter_context(context: Any = context) -> None:
                nonlocal entered
                context.__enter__()
                entered = True

            try:
                collective_call(
                    enter_context,
                    dist=dist,
                    rank=rank,
                    world_size=world_size,
                    label=f"LTX validation {pass_name} weights setup",
                    error_type=BackendContractError,
                )
            except Exception:
                if entered:
                    context.__exit__(None, None, None)
                raise
            try:
                sampler = LTX25Sampler(
                    runtime.generator,
                    self.source,
                    self.negative_caption,
                    self.negative_mask,
                    self.plan,
                    self.device,
                )
                adapter = LTX25InferenceAdapter(
                    sampler,
                    self.decoder,
                    model_receipt=self.model_receipt,
                )
                comparison_error: str | None = None
                if sp_rank == 0:
                    summary = InferenceEngine(adapter).run(
                        pass_cases,
                        weights_id=weights_id,
                        output_dir=(staging_pass / f"dp-rank-{dp_rank:05d}"),
                        collective_error=exchange,
                    )
                    reports[pass_name] = {
                        "cases": summary.cases,
                        "ordered_manifest_digest": summary.ordered_manifest_digest,
                        "output_dir": str(comparison_destination),
                    }
                    try:
                        publish_comparison_partition(
                            summary.output_dir,
                            comparison_destination,
                            step=int(step),
                            pass_name=pass_name,
                            cases=len(self.cases),
                            dp_world_size=dp_world_size,
                            sp_size=self.sp_size,
                            run_root=self.output_dir,
                            logical_world_size_per_round=dp_world_size,
                        )
                    except Exception as exc:
                        comparison_error = f"{type(exc).__name__}: {exc}"
                else:
                    exchange(None, "output setup")
                    for case_index, local_case in enumerate(local_cases):
                        case_error: str | None = None
                        try:
                            sampler.sample(local_case)
                        except Exception as exc:
                            case_error = f"{type(exc).__name__}: {exc}"
                        exchange(case_error, f"case wave {case_index}")
                    reports[pass_name] = {
                        "cases": len(local_cases),
                        "writer_rank": dp_rank * self.sp_size,
                    }
                    exchange(None, "partition commit")
                exchange(comparison_error, "comparison publication")
                complete_error: str | None = None
                if local_rank == 0:
                    try:
                        publish_comparison_complete(
                            comparison_destination,
                            step=int(step),
                            pass_name=pass_name,
                            local_slots=len(
                                tuple((comparison_destination / "compare").glob("rank*.mp4"))
                            ),
                            global_slots=len(self.cases),
                        )
                    except Exception as exc:
                        complete_error = f"{type(exc).__name__}: {exc}"
                exchange(complete_error, "comparison commit")
            finally:
                if entered:
                    collective_call(
                        lambda context=context: context.__exit__(None, None, None),
                        dist=dist,
                        rank=rank,
                        world_size=world_size,
                        label=f"LTX validation {pass_name} weights restore",
                        error_type=BackendContractError,
                    )
            cleanup_error: str | None = None
            if local_rank == 0:
                try:
                    cleanup_validation_staging(
                        staging_pass,
                        output_dir=self.output_dir,
                    )
                except Exception as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"
            exchange(cleanup_error, "staging cleanup")
        return {
            "schema": "solarwm.ltx25.validation.v1",
            "step": step,
            "passes": reports,
            "shared_inference_implementation": True,
            "validation_plan": {
                "path": str(self.plan_path),
                "source": plan_source,
                "digest": plan_digest,
            },
        }

    def close(self) -> None:
        self.source.close()
        self.decoder.close()


__all__ = ["TrainingValidation"]
