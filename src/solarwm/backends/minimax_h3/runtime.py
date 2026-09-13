"""MiniMax-H3 training, transactional checkpoints and shared validation lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from solarwm.checkpoint import (
    CheckpointContract,
    CheckpointTransaction,
    assert_resume_compatible,
    verify_checkpoint,
)
from solarwm.data import resolve_index_path
from solarwm.errors import BackendContractError
from solarwm.inference import (
    InferenceCase,
    InferenceEngine,
    publish_comparison_complete,
    publish_comparison_partition,
    run_validation,
)
from solarwm.inference.validation_plan import (
    load_validation_plan,
    publish_validation_plan,
    validation_plan_key,
    validation_plan_payload,
)
from solarwm.runtime import Topology, rng_identity
from solarwm.runtime.distributed import gather_and_assert_sp_identity
from solarwm.runtime.output_layout import (
    checkpoint_model_dir,
    cleanup_validation_staging,
    public_validation_dir,
    validation_staging_root,
)
from solarwm.runtime.randomness import seed_process
from solarwm.runtime.safe_state import (
    decode_numpy_rng_state,
    decode_python_rng_state,
    encode_numpy_rng_state,
    encode_python_rng_state,
)
from solarwm.training import (
    BatchIdentity,
    GradientStatus,
    JsonlEventSink,
    MicrobatchResult,
    StepPolicy,
    TrainingEngine,
)

from .artifacts import H3PreencodedStream, h3_silence_profile, load_silence_latents
from .distributed import get_sp_group, get_sp_rank, get_sp_size, sync_lora_gradients
from .inference import camera_fingerprint, package_generated
from .optional import load_conditioners, load_transformer, require_h3_runtime
from .stage0p5 import H3Stage0p5Core


def _base_model_profile(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Portable, readable identity for a compatible MiniMax-H3 base model."""

    profile = {
        "schema": "solarwm.minimax-h3-base-profile.v1",
        "family": "minimax_h3",
        "architecture": str(model_cfg.get("architecture", "minimax-h3-33b")),
        "component": str(model_cfg.get("transformer_subfolder", "transformer")),
        "parameter_dtype_profile": "bfloat16-blocks+float32-io-timestep-v1",
    }
    revision = str(model_cfg.get("revision") or "").strip()
    if revision:
        profile["revision"] = revision
    return profile


def _base_model_load_receipt(model_cfg: Mapping[str, Any], model: Any) -> dict[str, Any]:
    """Describe the strictly loaded model with portable semantic fields."""

    parameters = tuple(model.named_parameters())
    if not parameters or len({name for name, _ in parameters}) != len(parameters):
        raise BackendContractError("H3 base model parameter inventory is empty or duplicated")
    dtype_inventory: dict[str, dict[str, int]] = {}
    for _name, parameter in parameters:
        dtype = str(parameter.dtype).removeprefix("torch.")
        entry = dtype_inventory.setdefault(dtype, {"tensors": 0, "parameters": 0})
        entry["tensors"] += 1
        entry["parameters"] += int(parameter.numel())
    return {
        "schema": "solarwm.minimax-h3-base-load.v1",
        "profile": _base_model_profile(model_cfg),
        "strict_state_load": {
            "missing_keys": 0,
            "unexpected_keys": 0,
            "mismatched_shapes": 0,
            "load_errors": 0,
        },
        "parameter_inventory": {
            "tensors": len(parameters),
            "parameters": sum(int(parameter.numel()) for _, parameter in parameters),
            "dtypes": dict(sorted(dtype_inventory.items())),
        },
    }


def _base_weights_label(model_cfg: Mapping[str, Any]) -> str:
    profile = _base_model_profile(model_cfg)
    revision = f"@{profile['revision']}" if profile.get("revision") else ""
    return f"{profile['architecture']}/{profile['component']}{revision}"


def _topology(*, sp_size: int, require_torchrun: bool) -> Topology:
    required = {"WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK"}
    if required <= set(os.environ):
        return Topology.from_environ(sp_size)
    if require_torchrun:
        raise BackendContractError("H3 SP2 training must be launched with torchrun rank variables")
    return Topology(1, 0, 1, 0, sp_size=1)


def _reader(
    config: Mapping[str, Any],
    topology: Topology,
    *,
    index_field: str = "train_index",
    fixed_validation: bool = False,
    fixed_validation_sample_count: int | None = None,
    fixed_validation_sample_ids: Sequence[str] | None = None,
) -> Any:
    if fixed_validation and config["validation"].get("prepared_plan"):
        from .validation_inputs import H3PreparedValidationStream

        return H3PreparedValidationStream(config, topology)
    if fixed_validation and config["train"]["stage"] in {"stage1", "stage2"}:
        from .validation_stream import H3IndexedValidationStream

        return H3IndexedValidationStream(config, topology, frozen_ids=fixed_validation_sample_ids)
    data = config["data"]
    transport = data["transport"]
    common = {
        "root": str(transport["root"]),
        "index": str(resolve_index_path(data, index_field)),
        "topology": topology,
        "seed": int(data.get("seed", 42)),
        "cache_dir": transport.get("cache_dir"),
        "cache_max_gib": float(transport.get("cache_max_gib", 256)),
        "gcs_prefetch_shards": (0 if fixed_validation else data.get("gcs_prefetch_shards", 0)),
        "shuffle_buffer": int(data.get("shuffle_buffer", 4096)),
        "num_workers": 1 if fixed_validation else int(data.get("num_workers", 1)),
    }
    if fixed_validation:
        validation = config["validation"]
        common.update(
            fixed_validation_selection_seed=int(validation["selection_seed"]),
            fixed_validation_noise_seed=int(validation["noise_seed"]),
        )
    return H3PreencodedStream(
        **common,
        encoder_contract_path=str(data["encoder_contract_path"]),
        fixed_validation=fixed_validation,
        fixed_validation_sample_count=fixed_validation_sample_count,
        fixed_validation_sample_ids=fixed_validation_sample_ids,
        camera_audit_latents=47 if fixed_validation else int(data["train_target_latents"]),
    )


def _validation_schedule(
    validation: Mapping[str, Any],
    topology: Topology,
) -> tuple[int, int]:
    raw_count = validation.get("sample_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
        raise BackendContractError("H3 validation.sample_count must be a positive integer")
    dp_world_size = int(topology.dp_world_size)
    if raw_count % dp_world_size:
        raise BackendContractError(
            "H3 validation.sample_count must form complete logical-DP waves: "
            f"sample_count={raw_count} dp_world_size={dp_world_size}"
        )
    return raw_count, raw_count // dp_world_size


def _publish_h3_validation_complete(
    destination: Path,
    *,
    step: int,
    pass_name: str,
    expected_local_slots: Sequence[int],
    global_slots: int,
) -> None:
    compare_paths = tuple(sorted((destination / "compare").glob("rank*.mp4")))
    manifest_paths = tuple(sorted((destination / "manifests").glob("rank_*.json")))
    local_slots = tuple(sorted(int(slot) for slot in expected_local_slots))
    if len(compare_paths) != len(local_slots) or len(manifest_paths) != len(local_slots):
        raise BackendContractError(
            "H3 validation output count differs before completion: "
            f"compare={len(compare_paths)} manifests={len(manifest_paths)} "
            f"expected_local={len(local_slots)} global={global_slots}"
        )
    slots: list[int] = []
    for path in manifest_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            slots.append(int(payload["logical_validation_rank"]))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise BackendContractError(
                f"H3 validation manifest has no usable logical slot: {path}"
            ) from exc
    if sorted(slots) != list(local_slots):
        raise BackendContractError(
            "H3 validation completion requires each node-local logical slot exactly once"
        )
    publish_comparison_complete(
        destination,
        step=int(step),
        pass_name=pass_name,
        local_slots=len(local_slots),
        global_slots=int(global_slots),
    )


def _checkpoint_contract(
    *,
    encoder_profile: Mapping[str, Any],
    silence_profile: Mapping[str, Any],
    base_model: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> CheckpointContract:
    train = config["train"] if config is not None else {}
    stage = str(train.get("stage", "stage0p5"))
    stage_extras = {}
    if stage != "stage0p5":
        stage_extras = {
            "chunk_latents": 5,
            "window_chunks": 6,
            "target_latents": 45 if stage == "stage1" else 47,
            "rollout_latents": 50,
            "student_rope_mode": "sliding_local" if stage == "stage2" else "native_absolute",
            "training_profile": {
                key: value
                for key, value in train.items()
                if key not in {"max_steps", "global_batch_size"}
            },
        }
    return CheckpointContract(
        family="minimax_h3",
        stage=stage,
        causal_mode=str(train.get("causal_mode", "bidirectional")),
        objective=str(train.get("objective", "flow_matching")),
        objective_variant="v1_5" if stage == "stage1" else "data_ward_velocity",
        camera_translation_transform="logd4",
        parameterization="peft-lora-r384-alpha384",
        sp_size=4 if stage == "stage2" else 2,
        data_generation="h3.158f.v1",
        extras={
            "encoder_profile": json.loads(json.dumps(dict(encoder_profile), sort_keys=True)),
            "silence_profile": json.loads(json.dumps(dict(silence_profile), sort_keys=True)),
            "base_model": json.loads(json.dumps(dict(base_model), sort_keys=True)),
            "lora_target_count": 312,
            "lora_trainable_parameters": 2_075_394_048,
            "optimizer": "fp32_master_adamw",
            "ema": "rank_local_fp32_student_start39"
            if stage == "stage2"
            else "rank_local_fp32_start0",
            **stage_extras,
        },
    )


def _assert_encoder_silence_identity(
    encoder_values: Mapping[str, Any], silence_profile: Mapping[str, Any]
) -> None:
    """Reject readable silence semantics outside the selected encoder profile."""

    encoder_extras = encoder_values.get("extras", {})
    expected_silence = (
        encoder_extras.get("silence_artifact_profile")
        if isinstance(encoder_extras, Mapping)
        else None
    )
    compatible_duration = expected_silence == h3_silence_profile() and any(
        dict(silence_profile) == h3_silence_profile(pixel_frames=frames)
        for frames in (153, 158, 170)
    )
    if (
        expected_silence is not None
        and expected_silence != dict(silence_profile)
        and not compatible_duration
    ):
        raise BackendContractError(
            "H3 encoder contract was produced with a different silence artifact"
        )


def _collective_failures(
    dist: Any,
    topology: Topology,
    local_error: str,
) -> list[str]:
    errors = [local_error]
    if dist.is_initialized():
        errors = [""] * topology.raw_world_size
        dist.all_gather_object(errors, local_error)
    return [error for error in errors if error]


def _collective_call(
    call: Any,
    *,
    dist: Any,
    topology: Topology,
    label: str,
) -> Any:
    """Run rank-local work and report failures before the next collective phase."""

    result = None
    local_error = ""
    try:
        result = call()
    except Exception as exc:
        local_error = f"rank {topology.raw_rank}: {type(exc).__name__}: {exc}"
    failures = _collective_failures(dist, topology, local_error)
    if failures:
        raise BackendContractError(f"H3 distributed {label} failed: " + " | ".join(failures))
    if result is None:
        raise BackendContractError(f"H3 distributed {label} produced no result")
    return result


def _load_h3_validation_plan(
    config: Mapping[str, Any],
    *,
    dist: Any,
    topology: Topology,
    sample_count: int,
) -> tuple[tuple[InferenceCase, ...] | None, Path, str, str]:
    path = (
        Path(str(config["runtime"]["output_dir"])).expanduser().resolve()
        / "validation"
        / "frozen-plan.json"
    )
    key = validation_plan_key("minimax_h3", config)

    def read_local() -> tuple[tuple[InferenceCase, ...] | None, str | None]:
        cases = load_validation_plan(
            path,
            backend="minimax_h3",
            plan_key=key,
            expected_count=sample_count,
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if cases is not None else None
        return cases, digest

    cases, digest = _collective_call(
        read_local,
        dist=dist,
        topology=topology,
        label="validation plan read",
    )
    states = [digest]
    if dist.is_initialized():
        states = [None] * topology.raw_world_size
        dist.all_gather_object(states, digest)
    present = [value for value in states if value is not None]
    if present and len(present) != len(states):
        raise BackendContractError(
            "frozen H3 validation plan exists on only part of the distributed world"
        )
    if len(set(present)) > 1:
        raise BackendContractError("frozen H3 validation plan differs between nodes")
    return cases, path, key, str(digest or "")


def _gather_h3_validation_cases(
    cases: Sequence[InferenceCase],
    *,
    dist: Any,
    topology: Topology,
) -> tuple[InferenceCase, ...]:
    partitions: list[Any] = [tuple(cases)]
    if dist.is_initialized():
        partitions = [None] * topology.raw_world_size
        dist.all_gather_object(partitions, tuple(cases))
    by_slot: dict[int, InferenceCase] = {}
    for partition in partitions:
        if not isinstance(partition, tuple):
            raise BackendContractError("H3 gathered an invalid validation case partition")
        for case in partition:
            if not isinstance(case, InferenceCase):
                raise BackendContractError("H3 gathered a non-InferenceCase validation value")
            previous = by_slot.setdefault(case.slot, case)
            if previous != case:
                raise BackendContractError(f"H3 SP peers disagree on validation slot {case.slot}")
    ordered = tuple(by_slot[slot] for slot in sorted(by_slot))
    if [case.slot for case in ordered] != list(range(len(ordered))):
        raise BackendContractError("H3 gathered validation slots are not complete")
    return ordered


def _publish_h3_validation_plan(
    path: Path,
    *,
    key: str,
    cases: Sequence[InferenceCase],
    dist: Any,
    topology: Topology,
) -> str:
    payload = validation_plan_payload(backend="minimax_h3", plan_key=key, cases=cases)
    digest = hashlib.sha256(payload).hexdigest()
    digests = [digest]
    if dist.is_initialized():
        digests = [None] * topology.raw_world_size
        dist.all_gather_object(digests, digest)
    if len(set(digests)) != 1:
        raise BackendContractError("H3 ranks disagree before freezing validation plan")
    local_error = ""
    if int(topology.local_rank) == 0:
        try:
            publish_validation_plan(
                path,
                backend="minimax_h3",
                plan_key=key,
                cases=cases,
            )
        except Exception as exc:
            local_error = f"rank {topology.raw_rank}: {type(exc).__name__}: {exc}"
    failures = _collective_failures(dist, topology, local_error)
    if failures:
        raise BackendContractError(
            "H3 validation plan publication failed collectively: " + " | ".join(failures)
        )
    if dist.is_initialized():
        dist.barrier()
    return digest


class H3TrainingRuntime:
    """Heavy implementation of the shared :class:`TrainingRuntime` protocol."""

    def __init__(self, config: Mapping[str, Any], *, defer_resume: bool = False) -> None:
        torch, _diffusers, _transformers = require_h3_runtime()
        import torch.distributed as dist

        from .ema import H3ShardedEMA
        from .fsdp import finite_clip_norm, initialize_distributed, wrap_h3_fsdp
        from .lora import inject_h3_lora
        from .optimizer import FP32MasterAdamW

        self.torch = torch
        self.dist = dist
        self.config = config
        self.model_cfg = config["model"]
        self.train_cfg = config["train"]
        self.stage = str(self.train_cfg["stage"])
        self.extra_roles: dict[str, Any] = {}
        self.student_step = 0
        self.checkpoint_cfg = config["checkpoint"]
        self.runtime_cfg = config["runtime"]
        sp_size = int(config["distributed"]["sequence_parallel_size"])
        provisional = _topology(sp_size=sp_size, require_torchrun=True)
        configured_world = int(config["distributed"]["world_size"])
        if provisional.raw_world_size != configured_world:
            raise BackendContractError(
                f"torchrun WORLD_SIZE={provisional.raw_world_size} differs from "
                f"configured H3 world_size={configured_world}"
            )
        initialize_distributed(sp_size=sp_size, local_rank=provisional.local_rank)
        self.topology = Topology.from_environ(sp_size)
        self.rank = self.topology.raw_rank
        self.device = torch.device("cuda", self.topology.local_rank)
        self.is_main = self.rank == 0
        identity = rng_identity("minimax_h3", int(config["data"].get("seed", 42)), self.topology)
        seed_process(identity.model_init_seed)
        modules = load_transformer(self.model_cfg, device=self.device)
        if self.stage == "stage1":
            modules.transformer.enable_anyflow(float(self.train_cfg["anyflow_gate"]))
            modules.fp32_fsdp_units = (
                *modules.fp32_fsdp_units,
                modules.transformer.delta_embedding.linear_1,
                modules.transformer.delta_embedding.linear_2,
            )
        modules.transformer.train().requires_grad_(False)
        base_model = _base_model_load_receipt(self.model_cfg, modules.transformer)
        wrapped, self.lora = inject_h3_lora(
            modules.transformer,
            self.model_cfg["adapter"],
            base_identity=base_model,
        )
        self.model = wrap_h3_fsdp(
            wrapped,
            local_rank=self.topology.local_rank,
            transformer_block_cls=modules.transformer_block_cls,
            fp32_units=modules.fp32_fsdp_units,
            ignored_parameters=self.lora.parameters,
            activation_checkpointing=bool(self.train_cfg["fsdp"]["activation_checkpointing"]),
            frozen_base_shard_size=self.train_cfg["fsdp"].get("frozen_base_shard_size"),
        )
        self._weights_id = _base_weights_label(self.model_cfg)
        if self.stage != "stage0p5":
            from .weights import load_initial_weights

            self._weights_id = load_initial_weights(
                self.checkpoint_cfg["initialization"]["student"],
                self.lora,
            )
        self.objective_seed = int(identity.objective_seed)
        optimizer_cfg = self.train_cfg["optimizer"]
        self.optimizer = FP32MasterAdamW(
            self.lora.parameters,
            lr=float(optimizer_cfg["learning_rate"]),
            betas=tuple(float(value) for value in optimizer_cfg.get("betas", (0.9, 0.95))),
            eps=float(optimizer_cfg.get("epsilon", 1.0e-8)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        )
        warmup = int(optimizer_cfg["warmup_steps"])
        total = int(self.train_cfg["max_steps"])

        def lr_scale(step: int) -> float:
            if step < warmup:
                return float(step) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            minimum = float(optimizer_cfg.get("min_lr_ratio", 0.1))
            return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_scale)
        self.ema = (
            None
            if self.stage == "stage2"
            else H3ShardedEMA(
                self.model,
                decay=float(self.checkpoint_cfg["ema"]["decay"]),
                device=self.device,
                trainable_only=True,
            )
        )
        self.reader = _reader(config, self.topology)
        self.silence, silence_profile = load_silence_latents(
            str(config["data"]["silence_latents_path"]),
            pixel_frames=153 if self.stage == "stage1" else 158,
        )
        _assert_encoder_silence_identity(self.reader.encoder_profile, silence_profile)
        # Objective RNG advances continuously across microbatches and is
        # restored exactly by checkpoints.
        torch.manual_seed(self.objective_seed)
        torch.cuda.manual_seed_all(self.objective_seed)
        self.core = self._make_core()
        self.contract = _checkpoint_contract(
            encoder_profile=self.reader.encoder_profile,
            silence_profile=silence_profile,
            base_model=base_model,
            config=config,
        )
        self._global_step = 0
        self._finite_clip_norm = finite_clip_norm
        resume = str(self.checkpoint_cfg.get("resume_from") or "").strip()
        if resume and not defer_resume:
            self.load_checkpoint(resume)

    def _make_core(self) -> Any:
        if self.stage == "stage1":
            from .stage1 import H3Stage1Core

            validation_silence, _ = load_silence_latents(
                str(self.config["data"]["silence_latents_path"]),
                pixel_frames=170,
            )
            return H3Stage1Core(
                self.model,
                self.silence,
                self.device,
                self.train_cfg,
                validation_silence=validation_silence,
            )
        return H3Stage0p5Core(self.model, self.silence, self.device)

    @property
    def global_step(self) -> int:
        return self._global_step

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _next_batch(self, stream: Any | None = None) -> Any:
        source = self.reader if stream is None else stream
        return source.next()

    def _next_batch_collective(self, stream: Any | None = None) -> Any:
        """Fail all FSDP/SP ranks before forward if any reader/codec fails."""

        return _collective_call(
            lambda: self._next_batch(stream),
            dist=self.dist,
            topology=self.topology,
            label="data preflight",
        )

    def _noise_identity(self) -> tuple[int, str]:
        state = self.torch.cuda.get_rng_state(self.device).cpu().numpy().tobytes()
        digest = hashlib.blake2s(state).digest()
        return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF, digest.hex()

    def train_microbatch(self, micro_index: int, grad_accum: int) -> MicrobatchResult:
        if micro_index == 0:
            self._step_started = time.monotonic()
            self.torch.cuda.reset_peak_memory_stats(self.device)
        batch = self._next_batch()
        noise_seed, _ = self._noise_identity()
        identity = BatchIdentity(
            sample_ids=(batch.sample_id,),
            start_frames=(batch.start_frame,),
            noise_seeds=(noise_seed,),
            checkpoint_id=self._weights_id,
            plan_fingerprint=batch.plan_fingerprint,
        )
        should_sync = micro_index == grad_accum - 1
        context = (
            nullcontext()
            if should_sync or not hasattr(self.model, "no_sync")
            else self.model.no_sync()
        )
        with context:
            loss = self.core.forward_loss(batch, noise_seed=None)
            if not bool(self.torch.isfinite(loss).item()):
                raise FloatingPointError(f"H3 {self.stage} loss is non-finite")
            (loss / (int(grad_accum) * get_sp_size())).backward()
        loss_value = float(loss.item())
        return MicrobatchResult(
            identity=identity, losses={str(self.train_cfg["objective"]): loss_value}
        )

    def assert_sp_peer_identity(self, identity: BatchIdentity) -> None:
        del identity

    def prepare_optimizer_step(self) -> GradientStatus:
        sync_lora_gradients(self.lora.parameters)
        finite, norm = self._finite_clip_norm(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            float(self.train_cfg["optimizer"].get("gradient_clip", 1.0)),
        )
        return GradientStatus(finite=finite, norm=norm)

    def optimizer_step(self) -> None:
        self.optimizer.step()

    def scheduler_step(self) -> None:
        self.scheduler.step()

    def ema_update(self, step: int) -> None:
        ema_cfg = self.checkpoint_cfg["ema"]
        if step >= int(ema_cfg["start_step"]) and step % int(ema_cfg["update_every_steps"]) == 0:
            self.ema.update(self.model)

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    def _rank_state(self) -> dict[str, Any]:
        return {
            "reader": self.reader.state_dict(),
            "python_rng": encode_python_rng_state(random.getstate()),
            "numpy_rng": encode_numpy_rng_state(np.random.get_state()),
            "torch_rng": self.torch.get_rng_state(),
            "cuda_rng": self.torch.cuda.get_rng_state(self.device).cpu(),
        }

    def save_checkpoint(self, step: int) -> str:
        torch, dist = self.torch, self.dist
        local_state = self._rank_state()
        if dist.is_initialized():
            rank_states: list[Any] = [None] * self.topology.raw_world_size
            dist.all_gather_object(rank_states, local_state)
        else:
            rank_states = [local_state]
        checkpoint_id = ""
        save_error = ""
        if self.is_main:
            try:
                target = checkpoint_model_dir(
                    str(self.runtime_cfg["output_dir"]),
                    step=int(step),
                    width=6,
                )
                with CheckpointTransaction(target) as transaction:
                    torch.save(
                        {
                            "metadata": self.lora.metadata(),
                            "state": {
                                key: value.detach().cpu()
                                for key, value in self.lora.state_dict().items()
                            },
                        },
                        transaction.path / "adapter.pt",
                    )
                    torch.save(self.optimizer.state_dict(), transaction.path / "optimizer.pt")
                    torch.save(self.scheduler.state_dict(), transaction.path / "scheduler.pt")
                    torch.save(
                        self.ema.state_dict() if self.ema is not None else None,
                        transaction.path / "ema.pt",
                    )
                    for role, state in self.extra_roles.items():
                        torch.save(
                            {
                                "adapter": {
                                    key: value.detach().cpu()
                                    for key, value in state["lora"].state_dict().items()
                                },
                                "optimizer": state["optimizer"].state_dict(),
                                "scheduler": state["scheduler"].state_dict(),
                                "parameter_keys": list(state["lora"].parameter_by_key),
                                "global_step": int(step),
                                "student_step": self.student_step,
                            },
                            transaction.path / f"{role}.pt",
                        )
                    torch.save(rank_states, transaction.path / "rank-state.pt")
                    torch.save(
                        {
                            "global_step": int(step),
                            "weights_id": self._weights_id,
                            "student_step": self.student_step,
                        },
                        transaction.path / "runtime.pt",
                    )
                    verified = transaction.commit(
                        step=int(step),
                        contract=self.contract,
                        required_components=(
                            "adapter.pt",
                            "optimizer.pt",
                            "scheduler.pt",
                            "ema.pt",
                            "rank-state.pt",
                            "runtime.pt",
                            *(f"{role}.pt" for role in self.extra_roles),
                        ),
                        metadata={
                            "roles": {
                                "adapter.pt": "trainable_model",
                                "optimizer.pt": "fp32_master_optimizer",
                                "scheduler.pt": "lr_scheduler",
                                "ema.pt": "fp32_rank_local_ema",
                                "rank-state.pt": "per_raw_rank_rng_and_reader",
                                "runtime.pt": "optimizer_step_and_weight_parent",
                            },
                            "base_model_saved": False,
                            "base_model_reason": "immutable official checkpoint identity",
                        },
                    )
                    checkpoint_id = verified.manifest_digest
            except Exception as exc:
                save_error = f"rank 0: {type(exc).__name__}: {exc}"
        if dist.is_initialized():
            values = [save_error, checkpoint_id]
            dist.broadcast_object_list(values, src=0)
            save_error, checkpoint_id = str(values[0]), str(values[1])
            if save_error:
                raise BackendContractError(f"H3 distributed checkpoint commit failed: {save_error}")
            dist.barrier()
        elif save_error:
            raise BackendContractError(f"H3 checkpoint commit failed: {save_error}")
        self._weights_id = checkpoint_id
        return checkpoint_id

    def load_checkpoint(self, path: str) -> None:
        torch = self.torch
        verified = _collective_call(
            lambda: self._verify_resume_checkpoint(path),
            dist=self.dist,
            topology=self.topology,
            label="checkpoint verification",
        )

        def restore_local_state() -> Mapping[str, Any]:
            root = verified.path
            runtime = torch.load(root / "runtime.pt", map_location="cpu", weights_only=True)
            if int(runtime["global_step"]) != verified.step:
                raise BackendContractError("H3 checkpoint runtime step differs from manifest")
            if self.stage == "stage2" and int(runtime.get("student_step", -1)) != max(
                0, (verified.step - 1) // 5
            ):
                raise BackendContractError("H3 SGF checkpoint student update count differs")
            adapter = torch.load(
                root / "adapter.pt",
                map_location="cpu",
                weights_only=True,
            )
            if adapter.get("metadata") != self.lora.metadata():
                raise BackendContractError("H3 LoRA metadata differs at resume")
            # The manifest byte-verifies one shared adapter file on every rank,
            # so no parameter broadcast is needed inside this failure-collecting phase.
            self.lora.load_state_dict(adapter["state"], broadcast=False)
            self.optimizer.load_state_dict(
                torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True)
            )
            self.scheduler.load_state_dict(
                torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True)
            )
            ema_state = torch.load(root / "ema.pt", map_location="cpu", weights_only=True)
            if ema_state is None:
                if self.stage != "stage2":
                    raise BackendContractError("H3 checkpoint is missing EMA state")
                self.ema = None
            else:
                if self.ema is None:
                    from .ema import H3ShardedEMA

                    self.ema = H3ShardedEMA(
                        self.model, decay=float(ema_state["decay"]), device=self.device
                    )
                self.ema.load_state_dict(ema_state, self.model)
            for role, state in self.extra_roles.items():
                saved = torch.load(root / f"{role}.pt", map_location="cpu", weights_only=True)
                if (
                    saved["parameter_keys"] != list(state["lora"].parameter_by_key)
                    or saved["global_step"] != verified.step
                    or saved["student_step"] != runtime["student_step"]
                ):
                    raise BackendContractError(f"H3 {role} checkpoint identity differs")
                state["lora"].load_state_dict(saved["adapter"], broadcast=False)
                state["optimizer"].load_state_dict(saved["optimizer"])
                state["scheduler"].load_state_dict(saved["scheduler"])
            rank_states = torch.load(
                root / "rank-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            if len(rank_states) != self.topology.raw_world_size:
                raise BackendContractError("H3 checkpoint raw-rank state count differs")
            rank_state = rank_states[self.rank]
            self.reader.load_state_dict(rank_state["reader"])
            random.setstate(decode_python_rng_state(rank_state["python_rng"]))
            np.random.set_state(decode_numpy_rng_state(rank_state["numpy_rng"]))
            torch.set_rng_state(rank_state["torch_rng"])
            torch.cuda.set_rng_state(rank_state["cuda_rng"], self.device)
            return runtime

        runtime = _collective_call(
            restore_local_state,
            dist=self.dist,
            topology=self.topology,
            label="checkpoint restore",
        )
        if int(runtime["global_step"]) != verified.step:
            raise BackendContractError("H3 checkpoint runtime step differs from manifest")
        self._global_step = verified.step
        self.student_step = int(runtime.get("student_step", 0))
        self._weights_id = verified.manifest_digest

    def _verify_resume_checkpoint(self, path: str) -> Any:
        verified = verify_checkpoint(path)
        assert_resume_compatible(self.contract, verified.contract)
        return verified

    def validate(self, step: int) -> Mapping[str, Any]:
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cpu_rng = self.torch.get_rng_state()
        cuda_rng = self.torch.cuda.get_rng_state(self.device)
        try:
            return self._validate(step)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            self.torch.set_rng_state(cpu_rng)
            self.torch.cuda.set_rng_state(cuda_rng, self.device)

    def _validate(self, step: int) -> Mapping[str, Any]:
        generation = {
            "generation_mode": "bidirectional" if self.stage == "stage0p5" else "autoregressive",
            "sample_solver": {
                "stage0p5": "shifted-euler-data-ward",
                "stage1": "flowmap",
                "stage2": "self_forcing",
            }[self.stage],
            "train_latent_frames": 45 if self.stage == "stage1" else 47,
            "rollout_latent_frames": 47 if self.stage == "stage0p5" else 50,
            "decode_latent_frames": 47,
        }
        sample_count, num_waves = _validation_schedule(
            self.config["validation"],
            self.topology,
        )
        frozen_plan, plan_path, plan_key, plan_digest = _load_h3_validation_plan(
            self.config,
            dist=self.dist,
            topology=self.topology,
            sample_count=sample_count,
        )
        plan_source = "loaded" if frozen_plan is not None else "created"
        stream = _collective_call(
            lambda: _reader(
                self.config,
                self.topology,
                index_field="test_index",
                fixed_validation=True,
                fixed_validation_sample_count=sample_count,
                fixed_validation_sample_ids=(
                    tuple(case.sample_id for case in frozen_plan)
                    if frozen_plan is not None
                    else None
                ),
            ),
            dist=self.dist,
            topology=self.topology,
            label="validation reader setup",
        )
        fixed_cases: list[tuple[int, Any, InferenceCase]] = []
        try:
            if hasattr(stream, "prepare"):
                stream.prepare(self.dist)
            for wave_index in range(num_waves):
                batch = self._next_batch_collective(stream)
                expected_slot = wave_index * int(self.topology.dp_world_size) + int(
                    self.topology.dp_rank
                )
                identity_error = ""
                if batch.validation_noise_seed is None or batch.validation_slot is None:
                    identity_error = (
                        f"rank {self.rank}: H3 validation batch lacks its selected slot/noise"
                    )
                elif batch.validation_slot != expected_slot:
                    identity_error = (
                        f"rank {self.rank}: H3 validation wave {wave_index} expected slot "
                        f"{expected_slot}, got {batch.validation_slot}"
                    )
                identity_failures = _collective_failures(
                    self.dist,
                    self.topology,
                    identity_error,
                )
                if identity_failures:
                    raise BackendContractError(
                        "H3 validation fixed-plan identity failed: " + " | ".join(identity_failures)
                    )
                seed = int(batch.validation_noise_seed)
                case = InferenceCase(
                    slot=int(batch.validation_slot),
                    sample_id=batch.sample_id,
                    prompt="",
                    start_frame=batch.start_frame,
                    noise_seed=seed,
                    camera_fingerprint=camera_fingerprint(batch),
                    metadata={
                        "key": batch.sample_id,
                        "plan_fingerprint": batch.plan_fingerprint,
                        "source_pixel_frames": 158,
                        "output_pixel_frames": 158,
                        **generation,
                        "dataset_source": batch.dataset_source,
                        "dataset": batch.dataset_source or batch.sample_id.split("/")[0],
                        "student_step": self.student_step,
                        "camera_translation_transform": "logd4",
                        "artifact_valid": True,
                    },
                )
                if frozen_plan is not None:
                    expected_case = frozen_plan[expected_slot]
                    if (
                        case.slot != expected_case.slot
                        or case.sample_id != expected_case.sample_id
                        or case.start_frame != expected_case.start_frame
                        or case.noise_seed != expected_case.noise_seed
                        or case.camera_fingerprint != expected_case.camera_fingerprint
                    ):
                        raise BackendContractError(
                            f"H3 frozen validation slot {expected_slot} materialization drifted"
                        )
                    case = expected_case
                self.assert_sp_peer_identity(
                    BatchIdentity(
                        (batch.sample_id,),
                        (batch.start_frame,),
                        (seed,),
                        self._weights_id,
                        batch.plan_fingerprint,
                    )
                )
                fixed_cases.append((wave_index, batch, case))
        finally:
            stream.close()
        if frozen_plan is None:
            gathered_cases = _gather_h3_validation_cases(
                tuple(case for _, _, case in fixed_cases),
                dist=self.dist,
                topology=self.topology,
            )
            if len(gathered_cases) != sample_count:
                raise BackendContractError(
                    "H3 frozen validation plan has the wrong global case count"
                )
            plan_digest = _publish_h3_validation_plan(
                plan_path,
                key=plan_key,
                cases=gathered_cases,
                dist=self.dist,
                topology=self.topology,
            )
        was_training = self.model.training
        self.model.eval()
        reports: dict[str, Any] = {}
        try:
            for pass_name in self.config["validation"]["passes"]:
                name = str(pass_name)
                if name == "ema" and self.ema is None:
                    continue
                comparison_destination = public_validation_dir(
                    str(self.runtime_cfg["output_dir"]),
                    step=int(step),
                    pass_name=name,
                )
                context = self.ema.swapped_into(self.model) if name == "ema" else nullcontext()
                entered = False
                enter_error = ""
                try:
                    context.__enter__()
                    entered = True
                except Exception as exc:
                    enter_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
                enter_failures = _collective_failures(
                    self.dist,
                    self.topology,
                    enter_error,
                )
                if enter_failures:
                    if entered:
                        context.__exit__(None, None, None)
                    raise BackendContractError(
                        "H3 distributed validation weight swap failed: "
                        + " | ".join(enter_failures)
                    )
                output_directories: list[str] = []
                ordered_manifest_digests: list[str] = []
                conditioners = None
                try:
                    for wave_index, batch, case in fixed_cases:
                        generated_latents = _collective_call(
                            lambda batch=batch, case=case: self.core.generate(
                                batch,
                                noise_seed=case.noise_seed,
                                num_inference_steps=int(
                                    self.config["validation"]["num_inference_steps"]
                                ),
                            ),
                            dist=self.dist,
                            topology=self.topology,
                            label=f"validation {name} wave {wave_index} generation",
                        )

                        output_error = ""
                        if get_sp_rank() == 0:
                            try:
                                active_case = replace(
                                    case,
                                    metadata={
                                        **dict(case.metadata),
                                        "student_step": self.student_step,
                                        "num_inference_steps": int(
                                            self.config["validation"]["num_inference_steps"]
                                        ),
                                        "generation_pass": {
                                            "name": name,
                                            "weights": name,
                                            "mode": generation["generation_mode"],
                                            "solver": generation["sample_solver"],
                                            "num_inference_steps": int(
                                                self.config["validation"]["num_inference_steps"]
                                            ),
                                            "rollout_latent_frames": int(
                                                case.metadata["rollout_latent_frames"]
                                            ),
                                            "output_rollout_latent_frames": int(
                                                case.metadata["rollout_latent_frames"]
                                            ),
                                        },
                                    },
                                )
                                if conditioners is None:
                                    import gc

                                    gc.collect()
                                    self.torch.cuda.empty_cache()
                                    conditioners = load_conditioners(
                                        self.model_cfg,
                                        device=self.device,
                                        qwen=False,
                                        video_vae=True,
                                        audio_vae=False,
                                        schedulers=False,
                                    )
                                packaged = package_generated(
                                    generated_latents,
                                    video_vae=conditioners.video_vae,
                                    device=self.device,
                                    weights_id=f"{self._weights_id}:{name}:{step}",
                                    num_inference_steps=int(
                                        self.config["validation"]["num_inference_steps"]
                                    ),
                                    reference_latents=batch.target_latents,
                                    sample_solver=generation["sample_solver"],
                                )

                                class CachedAdapter:
                                    family = "minimax_h3"

                                    def __init__(self, sample: Any) -> None:
                                        self.sample = sample

                                    def generate(self, _case: Any, *, weights_id: str) -> Any:
                                        del _case, weights_id
                                        return self.sample

                                destination = (
                                    validation_staging_root(str(self.runtime_cfg["output_dir"]))
                                    / f"step-{int(step):08d}"
                                    / name
                                    / f"wave-{wave_index:03d}"
                                    / f"dp-rank-{self.topology.dp_rank:05d}"
                                )
                                summary = run_validation(
                                    CachedAdapter(packaged),
                                    [active_case],
                                    weights_id=f"{self._weights_id}:{name}:{step}",
                                    output_dir=destination,
                                )
                                publish_comparison_partition(
                                    summary.output_dir,
                                    comparison_destination,
                                    step=int(step),
                                    pass_name=name,
                                    cases=sample_count,
                                    dp_world_size=int(self.topology.dp_world_size),
                                    sp_size=int(self.topology.sp_size),
                                    run_root=Path(str(self.runtime_cfg["output_dir"])),
                                    logical_world_size_per_round=int(self.topology.dp_world_size),
                                )
                                ordered_manifest_digests.append(summary.ordered_manifest_digest)
                            except Exception as exc:
                                output_error = (
                                    f"rank {self.rank}/dp {self.topology.dp_rank} wave "
                                    f"{wave_index}: {type(exc).__name__}: {exc}"
                                )
                        output_failures = _collective_failures(
                            self.dist,
                            self.topology,
                            output_error,
                        )
                        if output_failures:
                            raise BackendContractError(
                                "H3 distributed validation output failed: "
                                + " | ".join(output_failures)
                            )
                finally:
                    conditioners = None
                    self.torch.cuda.empty_cache()
                    restore_error = ""
                    if entered:
                        try:
                            context.__exit__(None, None, None)
                        except Exception as exc:
                            restore_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
                    restore_failures = _collective_failures(
                        self.dist,
                        self.topology,
                        restore_error,
                    )
                    if restore_failures:
                        raise BackendContractError(
                            "H3 distributed validation weight restore failed: "
                            + " | ".join(restore_failures)
                        )
                complete_error = ""
                if int(self.topology.local_rank) == 0:
                    try:
                        expected_local_slots = tuple(
                            wave_index * int(self.topology.dp_world_size)
                            + int(self.topology.node_id) * int(self.topology.local_dp_world_size)
                            + local_dp_rank
                            for wave_index in range(num_waves)
                            for local_dp_rank in range(int(self.topology.local_dp_world_size))
                        )
                        _publish_h3_validation_complete(
                            comparison_destination,
                            step=int(step),
                            pass_name=name,
                            expected_local_slots=expected_local_slots,
                            global_slots=sample_count,
                        )
                    except Exception as exc:
                        complete_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
                complete_failures = _collective_failures(
                    self.dist,
                    self.topology,
                    complete_error,
                )
                if complete_failures:
                    raise BackendContractError(
                        "H3 validation commit failed: " + " | ".join(complete_failures)
                    )
                cleanup_error = ""
                if int(self.topology.local_rank) == 0:
                    try:
                        cleanup_validation_staging(
                            validation_staging_root(str(self.runtime_cfg["output_dir"]))
                            / f"step-{int(step):08d}"
                            / name,
                            output_dir=str(self.runtime_cfg["output_dir"]),
                        )
                    except Exception as exc:
                        cleanup_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
                cleanup_failures = _collective_failures(
                    self.dist,
                    self.topology,
                    cleanup_error,
                )
                if cleanup_failures:
                    raise BackendContractError(
                        "H3 validation staging cleanup failed: " + " | ".join(cleanup_failures)
                    )
                output_directories.append(str(comparison_destination))
                reports[name] = {
                    "writer": get_sp_rank() == 0,
                    "sample_count": sample_count,
                    "local_cases": len(fixed_cases),
                    "slots": [case.slot for _, _, case in fixed_cases],
                    "sample_ids": [case.sample_id for _, _, case in fixed_cases],
                    "ordered_manifest_digests": ordered_manifest_digests,
                    "output_dirs": output_directories,
                }
        finally:
            self.model.train(was_training)
        reports["validation_plan"] = {
            "path": str(plan_path),
            "source": plan_source,
            "digest": plan_digest,
        }
        return reports


def run_training(config: Mapping[str, Any]) -> int:
    if config["train"]["stage"] == "stage2":
        from .stage2_runtime import run_sgf_training

        return run_sgf_training(config)
    runtime = H3TrainingRuntime(config)
    train = config["train"]
    checkpoint = config["checkpoint"]
    validation = config["validation"]
    policy = StepPolicy(
        max_steps=int(train["max_steps"]),
        grad_accum=int(train["gradient_accumulation_steps"]),
        save_every=int(checkpoint.get("save_every_steps", 0)),
        save_steps=tuple(int(step) for step in checkpoint.get("save_steps", ())),
        validate_every=int(validation.get("validate_every_steps", 0)),
        validation_steps=(
            (int(validation["smoke_step"]),) if int(validation["smoke_step"]) > 0 else ()
        ),
    )
    writer = (
        JsonlEventSink(Path(str(config["runtime"]["output_dir"])) / "training-events.jsonl")
        if runtime.is_main
        else None
    )

    def sink(event: Mapping[str, Any]) -> None:
        if writer is None:
            return
        if event["event"] == "optimizer_step":
            runtime.torch.cuda.synchronize(runtime.device)
            event = {
                **event,
                "compute_time_s": time.monotonic() - runtime._step_started,
                "lr": float(runtime.optimizer.param_groups[0]["lr"]),
                "peak_allocated_gib": runtime.torch.cuda.max_memory_allocated(runtime.device)
                / (1024**3),
            }
            loss = event["losses"][str(train["objective"])]
            print(
                f"[h3-{runtime.stage}] step={event['step']} loss={loss:.8f} "
                f"lr={event['lr']:.3e} compute_time_s={event['compute_time_s']:.2f}",
                flush=True,
            )
        writer(event)

    try:
        TrainingEngine(runtime, policy, event_sink=sink).run()
    finally:
        runtime.reader.close()
    return 0


def _load_adapter_checkpoint(
    path: str,
    lora: Any,
    *,
    torch: Any,
    expected_contract: CheckpointContract,
    broadcast: bool = True,
) -> str:
    verified = verify_checkpoint(path)
    assert_resume_compatible(expected_contract, verified.contract)
    adapter = torch.load(verified.path / "adapter.pt", map_location="cpu", weights_only=True)
    if adapter.get("metadata") != lora.metadata():
        raise BackendContractError("inference LoRA metadata differs")
    lora.load_state_dict(adapter["state"], broadcast=broadcast)
    return verified.manifest_digest


def _load_lora_checkpoint(
    path: str,
    lora: Any,
    *,
    weight_source: str,
    broadcast: bool = False,
) -> str:
    """Load a Stage0.5 LoRA package without inheriting training state."""

    source = str(weight_source).strip().lower()
    if source not in {"live", "ema"}:
        raise BackendContractError("H3 inference weight_source must be live or ema")
    checkpoint = Path(path)
    root = checkpoint if checkpoint.is_dir() else checkpoint.parent
    if (root / "checkpoint-manifest.json").is_file():
        from .weights import load_initial_weights

        identity = load_initial_weights(
            {"path": str(root), "stage": "stage0p5", "weight_source": source}, lora
        )
        if broadcast:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                for parameter in lora.parameter_by_key.values():
                    dist.broadcast(parameter.detach(), src=0)
        return identity
    sidecar = root / (
        "adapter_model.safetensors" if source == "live" else "adapter_model_ema.safetensors"
    )
    if not sidecar.is_file():
        raise BackendContractError(f"H3 {source} sidecar is missing: {sidecar}")
    size = sidecar.stat().st_size
    if size <= 0:
        raise BackendContractError(f"H3 {source} sidecar is empty: {sidecar}")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional H3 dependency
        raise BackendContractError("H3 inference requires safetensors") from exc
    values = load_file(str(sidecar), device="cpu")
    lora.load_state_dict(values, broadcast=broadcast)
    if sidecar.stat().st_size != size:
        raise BackendContractError(f"H3 {source} sidecar changed while it was being loaded")
    return f"inventory:file={sidecar.name}:bytes={size}:role={source}"


def run_inference(config: Mapping[str, Any]) -> int:
    if config.get("inference", {}).get("length_policy") == "source":
        from .full_inference import run_source_length_inference

        return run_source_length_inference(config)
    torch, _diffusers, _transformers = require_h3_runtime()
    import torch.distributed as dist

    from .fsdp import initialize_distributed, wrap_h3_fsdp
    from .lora import inject_h3_lora

    stage = str(config["train"]["stage"])
    distributed = config.get("distributed", {})
    configured_sp = int(distributed.get("sequence_parallel_size", 1))
    topology = _topology(sp_size=configured_sp, require_torchrun=True)
    configured_world = int(distributed["world_size"])
    if topology.raw_world_size != configured_world:
        raise BackendContractError(
            f"torchrun WORLD_SIZE={topology.raw_world_size} differs from configured "
            f"H3 inference world_size={configured_world}"
        )
    initialize_distributed(sp_size=topology.sp_size, local_rank=topology.local_rank)
    topology = (
        Topology.from_environ(topology.sp_size)
        if {"WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK"} <= set(os.environ)
        else topology
    )
    device = torch.device("cuda", topology.local_rank)
    identity = rng_identity("minimax_h3", int(config["data"].get("seed", 42)), topology)
    seed_process(identity.model_init_seed)
    modules = load_transformer(config["model"], device=device)
    if stage == "stage1":
        modules.transformer.enable_anyflow(float(config["train"]["anyflow_gate"]))
        modules.fp32_fsdp_units = (
            *modules.fp32_fsdp_units,
            modules.transformer.delta_embedding.linear_1,
            modules.transformer.delta_embedding.linear_2,
        )
    modules.transformer.eval().requires_grad_(False)
    base_model = _base_model_load_receipt(config["model"], modules.transformer)
    model, lora = inject_h3_lora(
        modules.transformer,
        config["model"]["adapter"],
        base_identity=base_model,
    )
    model = wrap_h3_fsdp(
        model,
        local_rank=topology.local_rank,
        transformer_block_cls=modules.transformer_block_cls,
        fp32_units=modules.fp32_fsdp_units,
        ignored_parameters=lora.parameters,
        activation_checkpointing=False,
    )
    sample_count, num_waves = _validation_schedule(config["validation"], topology)
    stream = _reader(
        config,
        topology,
        index_field="test_index",
        fixed_validation=True,
        fixed_validation_sample_count=sample_count,
    )
    fixed_cases: list[tuple[int, Any, InferenceCase]] = []
    try:
        silence, silence_profile = load_silence_latents(
            str(config["data"]["silence_latents_path"]),
            pixel_frames=153 if stage == "stage1" else 158,
        )
        _assert_encoder_silence_identity(stream.encoder_profile, silence_profile)
        expected_contract = _checkpoint_contract(
            encoder_profile=stream.encoder_profile,
            silence_profile=silence_profile,
            base_model=base_model,
        )
        checkpoint_path = str(
            config.get("checkpoint", {}).get("resume_from")
            or config["runtime"].get("weights_checkpoint")
            or ""
        ).strip()
        checkpoint_cfg = config.get("checkpoint", {})
        checkpoint_format = str(checkpoint_cfg.get("format", "solarwm")).strip().lower()
        if checkpoint_path and stage != "stage0p5":
            from .weights import load_initial_weights

            weights_id = _collective_call(
                lambda: load_initial_weights(
                    {
                        "path": checkpoint_path,
                        "stage": stage,
                        "weight_source": checkpoint_cfg.get("weight_source", "live"),
                    },
                    lora,
                ),
                dist=dist,
                topology=topology,
                label="inference checkpoint restore",
            )
        elif checkpoint_path and checkpoint_format == "solarwm_lora":
            weights_id = _collective_call(
                lambda: _load_lora_checkpoint(
                    checkpoint_path,
                    lora,
                    weight_source=str(checkpoint_cfg.get("weight_source", "")),
                    broadcast=False,
                ),
                dist=dist,
                topology=topology,
                label="inference checkpoint restore",
            )
        elif checkpoint_path:
            weights_id = _collective_call(
                lambda: _load_adapter_checkpoint(
                    checkpoint_path,
                    lora,
                    torch=torch,
                    expected_contract=expected_contract,
                    broadcast=False,
                ),
                dist=dist,
                topology=topology,
                label="inference checkpoint restore",
            )
        else:
            weights_id = _base_weights_label(config["model"])
        if hasattr(stream, "prepare"):
            stream.prepare(dist)
        for wave_index in range(num_waves):
            batch = _collective_call(
                stream.next,
                dist=dist,
                topology=topology,
                label=f"inference data wave {wave_index}",
            )
            expected_slot = wave_index * int(topology.dp_world_size) + int(topology.dp_rank)
            identity_error = ""
            if batch.validation_noise_seed is None or batch.validation_slot is None:
                identity_error = (
                    f"rank {topology.raw_rank}: H3 inference batch lacks selected slot/noise"
                )
            elif int(batch.validation_slot) != expected_slot:
                identity_error = (
                    f"rank {topology.raw_rank}: H3 inference wave {wave_index} "
                    f"expected slot {expected_slot}, got {batch.validation_slot}"
                )
            identity_failures = _collective_failures(dist, topology, identity_error)
            if identity_failures:
                raise BackendContractError(
                    "H3 inference selection failed: " + " | ".join(identity_failures)
                )
            seed = int(batch.validation_noise_seed)
            case = InferenceCase(
                slot=expected_slot,
                sample_id=batch.sample_id,
                prompt="",
                start_frame=batch.start_frame,
                noise_seed=seed,
                camera_fingerprint=camera_fingerprint(batch),
                metadata={
                    "key": batch.sample_id,
                    "dataset": batch.dataset_source or batch.sample_id.split("/")[0],
                    "plan_fingerprint": batch.plan_fingerprint,
                    "source_pixel_frames": 158,
                    "output_pixel_frames": 158,
                    "train_latent_frames": 45 if stage == "stage1" else 47,
                    "rollout_latent_frames": 47 if stage == "stage0p5" else 50,
                    "generation_mode": "bidirectional" if stage == "stage0p5" else "autoregressive",
                    "sample_solver": {
                        "stage0p5": "shifted-euler-data-ward",
                        "stage1": "flowmap",
                        "stage2": "self_forcing",
                    }[stage],
                    "camera_translation_transform": "logd4",
                    "artifact_valid": True,
                },
            )
            gather_and_assert_sp_identity(
                {
                    "sample_id": batch.sample_id,
                    "start_frame": batch.start_frame,
                    "noise_seed": seed,
                    "plan_fingerprint": batch.plan_fingerprint,
                },
                sp_size=get_sp_size(),
                group=get_sp_group(),
            )
            fixed_cases.append((wave_index, batch, case))
    finally:
        stream.close()
    if stage == "stage1":
        from .stage1 import H3Stage1Core

        validation_silence, _ = load_silence_latents(
            str(config["data"]["silence_latents_path"]), pixel_frames=170
        )
        core = H3Stage1Core(model, silence, device, config["train"], validation_silence)
    elif stage == "stage2":
        from .stage2 import H3SGFCore

        core = H3SGFCore(model, silence, device)
    else:
        core = H3Stage0p5Core(model, silence, device)
    conditioners = None
    comparison_destination = Path(str(config["runtime"]["output_dir"])) / "inference"
    for wave_index, batch, case in fixed_cases:
        generated_latents = _collective_call(
            lambda batch=batch, case=case: core.generate(
                batch,
                noise_seed=case.noise_seed,
                num_inference_steps=int(config["validation"]["num_inference_steps"]),
            ),
            dist=dist,
            topology=topology,
            label=f"inference wave {wave_index} generation",
        )
        output_error = ""
        if get_sp_rank() == 0:
            try:
                if conditioners is None:
                    conditioners = load_conditioners(
                        config["model"],
                        device=device,
                        qwen=False,
                        video_vae=True,
                        audio_vae=False,
                        schedulers=False,
                    )
                packaged = package_generated(
                    generated_latents,
                    video_vae=conditioners.video_vae,
                    device=device,
                    weights_id=weights_id,
                    num_inference_steps=int(config["validation"]["num_inference_steps"]),
                    reference_latents=batch.target_latents,
                    sample_solver=str(case.metadata["sample_solver"]),
                )

                class CachedAdapter:
                    family = "minimax_h3"

                    def __init__(self, sample: Any) -> None:
                        self.sample = sample

                    def generate(self, _case: Any, *, weights_id: str) -> Any:
                        del _case, weights_id
                        return self.sample

                destination = (
                    Path(str(config["runtime"]["output_dir"]))
                    / "inference-parts"
                    / f"wave-{wave_index:03d}"
                    / f"dp-rank-{topology.dp_rank:05d}"
                )
                summary = InferenceEngine(CachedAdapter(packaged)).run(
                    [case],
                    weights_id=weights_id,
                    output_dir=destination,
                )
                publish_comparison_partition(
                    summary.output_dir,
                    comparison_destination,
                    step=0,
                    pass_name="inference",
                    cases=sample_count,
                    dp_world_size=int(topology.dp_world_size),
                    sp_size=int(topology.sp_size),
                    run_root=Path(str(config["runtime"]["output_dir"])),
                    logical_world_size_per_round=int(topology.dp_world_size),
                )
            except Exception as exc:
                output_error = (
                    f"rank {topology.raw_rank}/dp {topology.dp_rank} wave {wave_index}: "
                    f"{type(exc).__name__}: {exc}"
                )
        output_failures = _collective_failures(dist, topology, output_error)
        if output_failures:
            raise BackendContractError(
                "H3 distributed inference output failed: " + " | ".join(output_failures)
            )
    complete_error = ""
    if int(topology.local_rank) == 0:
        try:
            expected_local_slots = tuple(
                wave_index * int(topology.dp_world_size)
                + int(topology.node_id) * int(topology.local_dp_world_size)
                + local_dp_rank
                for wave_index in range(num_waves)
                for local_dp_rank in range(int(topology.local_dp_world_size))
            )
            _publish_h3_validation_complete(
                comparison_destination,
                step=0,
                pass_name="inference",
                expected_local_slots=expected_local_slots,
                global_slots=sample_count,
            )
        except Exception as exc:
            complete_error = f"rank {topology.raw_rank}: {type(exc).__name__}: {exc}"
    complete_failures = _collective_failures(dist, topology, complete_error)
    if complete_failures:
        raise BackendContractError(
            "H3 distributed inference commit failed: " + " | ".join(complete_failures)
        )
    return 0


__all__ = ["H3TrainingRuntime", "run_inference", "run_training"]
