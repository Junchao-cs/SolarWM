"""Executable Wan2.2 TI2V-5B Stage0.5 flow-matching runtime."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solarwm.errors import BackendContractError
from solarwm.runtime.distributed import gather_and_assert_sp_identity
from solarwm.runtime.output_layout import validation_staging_root
from solarwm.runtime.randomness import model_init_seed
from solarwm.training.ema import ShardedEMA, ema_decay_for_step
from solarwm.training.engine import (
    BatchIdentity,
    GradientStatus,
    JsonlEventSink,
    MicrobatchResult,
    StepPolicy,
    TrainingEngine,
)
from solarwm.training.schedule import make_warmup_cosine

from ..objectives import first_frame_loss_mask, weighted_masked_mse
from .codec import Wan5BOnlineCodec, WanA14BOnlineCodec
from .components import (
    build_diffusion_architecture,
    build_online_codec_components,
    build_online_components,
)
from .data import build_raw_dataloader
from .distributed import cleanup_torchrun, initialize_torchrun, wrap_transformer_fsdp
from .preencoded import build_preencoded_dataloader
from .readiness import probe_runtime
from .sequence_parallel import get_sp_group

# This public namespace is part of the deterministic sampling contract: changing
# it changes which samples drop the first-frame image condition.
_I2V_DROP_NAMESPACE = b"ti2v-image-drop-v1"
_TORCHRUN_OWNER_ENV = "SOLARWM_TORCHRUN_LIFECYCLE_OWNER"


def deterministic_i2v_drop_mask(
    *,
    probability: float,
    batch_size: int,
    seed: int,
    global_step: int,
    logical_rank: int,
    micro_index: int,
    device: Any,
) -> Any:
    """Sample a deterministic per-row image-condition mask without consuming RNG."""

    import torch

    probability = float(probability)
    if not 0 <= probability <= 1:
        raise BackendContractError("i2v image-condition dropout must be in [0,1]")
    threshold = int(probability * (1 << 64))
    values: list[bool] = []
    for sample_index in range(int(batch_size)):
        payload = b"|".join(
            (
                _I2V_DROP_NAMESPACE,
                str(int(seed)).encode(),
                str(int(global_step)).encode(),
                str(int(logical_rank)).encode(),
                str(int(micro_index)).encode(),
                str(sample_index).encode(),
            )
        )
        draw = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
        values.append(draw < threshold)
    return torch.tensor(values, dtype=torch.bool, device=device)


def expand_timesteps_to_tokens(timestep: Any, frame_sequence_length: int) -> Any:
    batch, frames = timestep.shape
    return (
        timestep.unsqueeze(-1)
        .expand(batch, frames, int(frame_sequence_length))
        .reshape(batch, frames * int(frame_sequence_length))
    )


def _digest(*values: bytes) -> str:
    digest = hashlib.blake2s()
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _rng_identity(device: Any) -> str:
    import torch

    cpu_state = torch.random.get_rng_state().numpy().tobytes()
    cuda_state = torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
    return _digest(cpu_state, cuda_state)


class Wan5BStage0p5Runtime:
    """One real optimizer runtime over online-decoded Wan batches."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        diffusion: Any,
        codec: Wan5BOnlineCodec | WanA14BOnlineCodec | None,
        batches: Any,
        optimizer: Any,
        lr_scheduler: Any,
        ema: ShardedEMA,
        topology: Any,
        initialization_receipt: Mapping[str, Any],
    ) -> None:
        import torch

        self.config = config
        self.diffusion = diffusion
        self.codec = codec
        self.data = batches
        self.batches = iter(batches)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema = ema
        self.topology = topology
        self.initialization_receipt = dict(initialization_receipt)
        self.initialization_id = str(self.initialization_receipt.get("initialization_id", ""))
        if not self.initialization_id:
            raise BackendContractError("Wan initialization receipt lacks an identity")
        self.device = torch.device("cuda", topology.local_rank)
        self._global_step = 0
        self.family = str(config["model"]["family"])
        self.checkpoint_id = self.initialization_id

    @property
    def global_step(self) -> int:
        return self._global_step

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def train_microbatch(self, micro_index: int, grad_accum: int) -> MicrobatchResult:
        import torch

        rng_identity = _rng_identity(self.device)
        internal_noise_seed = int(rng_identity[:16], 16)
        batch = next(self.batches)
        camera = {
            key: value.to(self.device, non_blocking=True) for key, value in batch["camera"].items()
        }
        if bool(batch.get("preencoded", False)):
            clean = batch["latents"].to(self.device, non_blocking=True)
            prompt = {"prompt_embeds": batch["prompt_embeds"].to(self.device, non_blocking=True)}
            model_y = batch.get("i2v_y")
            if model_y is not None:
                model_y = model_y.to(self.device, non_blocking=True)
        else:
            if self.codec is None:
                raise BackendContractError("online Wan batch reached a preencoded-only runtime")
            pixels = batch["pixels"].to(self.device, non_blocking=True)
            encoded = self.codec.encode_batch(
                sample_ids=batch["sample_ids"],
                pixels=pixels,
                captions=batch["prompts"],
                camera=camera,
            )
            clean = encoded["latents"]
            prompt = {"prompt_embeds": encoded["prompt_embeds"]}
            camera = encoded["camera"]
            model_y = encoded.get("i2v_y")
        batch_size, latent_frames = clean.shape[:2]
        train = self.config["train"]
        data = self.config["data"]
        model = self.config["model"]
        dropped = deterministic_i2v_drop_mask(
            probability=float(train["i2v_image_condition_dropout"]),
            batch_size=batch_size,
            seed=int(data["seed"]),
            global_step=self.global_step,
            logical_rank=int(self.topology.dp_rank),
            micro_index=int(micro_index),
            device=self.device,
        )
        if int(self.topology.sp_size) > 1:
            from .sequence_parallel import broadcast_sequence_parallel_tensor

            broadcast_sequence_parallel_tensor(dropped)
        official_i2v = self.family == "wan22_i2v_a14b"
        if official_i2v:
            if model_y is None:
                raise BackendContractError("Wan2.2 I2V-A14B training requires official i2v_y")
            if bool(dropped.any().item()):
                model_y = model_y.clone()
                model_y[dropped] = 0
        elif model_y is not None:
            raise BackendContractError("TI2V-5B training may not receive i2v_y")
        timestep_index = (
            torch.randint(
                0,
                int(train["num_train_timesteps"]),
                (batch_size, 1),
                device=self.device,
                dtype=torch.long,
            )
            .expand(batch_size, latent_frames)
            .contiguous()
        )
        if int(self.topology.sp_size) > 1:
            broadcast_sequence_parallel_tensor(timestep_index)
        timestep = self.diffusion.scheduler.timesteps.to(device=self.device, dtype=torch.float32)[
            timestep_index
        ]
        if not official_i2v:
            timestep[~dropped, 0] = 0.0
        noise = torch.randn_like(clean)
        if int(self.topology.sp_size) > 1:
            broadcast_sequence_parallel_tensor(noise)
        noisy = (
            self.diffusion.scheduler.add_noise(
                clean.flatten(0, 1).float(),
                noise.flatten(0, 1).float(),
                timestep.flatten(0, 1),
            )
            .unflatten(0, (batch_size, latent_frames))
            .to(clean.dtype)
        )
        if not official_i2v:
            noisy[~dropped, 0] = clean[~dropped, 0]
        timestep_tokens = expand_timesteps_to_tokens(timestep, int(model["frame_sequence_length"]))
        prediction = self.diffusion(
            noisy,
            prompt,
            camera,
            timestep_tokens,
            i2v_y=model_y,
            sequence_length=latent_frames * int(model["frame_sequence_length"]),
        )
        target = self.diffusion.scheduler.training_target(clean, noise, timestep)
        weight = self.diffusion.scheduler.training_weight(timestep).view(
            batch_size, latent_frames, 1, 1, 1
        )
        mask = (
            torch.ones(
                (batch_size, latent_frames),
                dtype=torch.bool,
                device=self.device,
            )
            if official_i2v
            else first_frame_loss_mask(batch_size, latent_frames, dropped)
        )
        loss = weighted_masked_mse(prediction, target, mask, weight)
        if not bool(torch.isfinite(loss).item()):
            raise BackendContractError(f"non-finite Wan Stage0.5 loss: {loss.item()}")
        # FSDP reduces over logical DP only. Each SP rank contributes a
        # complementary token/head shard, so scale before backward and sum
        # the replicas exactly once in prepare_optimizer_step().
        loss_scale = int(grad_accum) * int(self.topology.sp_size)
        (loss / loss_scale).backward()
        plan_payload = {
            "sample_ids": list(batch["sample_ids"]),
            "start_frames": list(batch["start_frames"]),
            "source_frame_indices": [list(value) for value in batch["source_frame_indices"]],
        }
        fingerprint = hashlib.blake2s(repr(plan_payload).encode()).hexdigest()
        identity = BatchIdentity(
            sample_ids=tuple(str(value) for value in batch["sample_ids"]),
            start_frames=tuple(int(value) for value in batch["start_frames"]),
            noise_seeds=tuple(internal_noise_seed + index for index in range(batch_size)),
            checkpoint_id=self.checkpoint_id,
            plan_fingerprint=fingerprint,
        )
        return MicrobatchResult(identity=identity, losses={"flow_matching": loss.item()})

    def assert_sp_peer_identity(self, identity: BatchIdentity) -> None:
        gather_and_assert_sp_identity(
            {
                "sample_ids": identity.sample_ids,
                "start_frames": identity.start_frames,
                "noise_seeds": identity.noise_seeds,
                "plan_fingerprint": identity.plan_fingerprint,
            },
            sp_size=int(self.topology.sp_size),
            group=get_sp_group(),
        )

    def prepare_optimizer_step(self) -> GradientStatus:
        import torch

        from .sequence_parallel import sync_sequence_parallel_gradients

        sync_sequence_parallel_gradients(self.diffusion.module)

        clip = float(self.config["train"]["optimizer"]["grad_clip"])
        module = self.diffusion.module
        if hasattr(module, "clip_grad_norm_"):
            norm = module.clip_grad_norm_(clip)
        else:
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), clip)
        value = float(norm.detach().float().item())
        return GradientStatus(finite=bool(torch.isfinite(norm).item()), norm=value)

    def optimizer_step(self) -> None:
        self.optimizer.step()

    def scheduler_step(self) -> None:
        self.lr_scheduler.step()

    def ema_update(self, step: int) -> None:
        ema_config = self.config["train"]["ema"]
        if step < int(ema_config["start_step"]):
            return
        if step % int(ema_config.get("update_every", 1)):
            return
        decay = ema_decay_for_step(
            target_decay=float(ema_config["decay"]),
            global_step=step,
            warmup_steps=int(ema_config.get("warmup_steps", 0)),
        )
        self.ema.update(self.diffusion.module, decay=decay)

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    def save_checkpoint(self, step: int) -> str:
        from .checkpoint import save_full_checkpoint

        if int(step) != self.global_step:
            raise BackendContractError(
                "Wan checkpoint step differs from the completed optimizer step"
            )
        identity = save_full_checkpoint(
            config=self.config,
            step=step,
            diffusion=self.diffusion,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            ema=self.ema,
            reader=self.data,
            device=self.device,
        )
        self.checkpoint_id = f"digest:{identity}"
        return identity

    def validate(self, step: int) -> Mapping[str, Any]:
        if int(step) != self.global_step:
            raise BackendContractError("Wan validation step differs from the live optimizer step")
        if self.codec is None:
            raise BackendContractError(
                "inline Wan validation requires a raw online VAE/text codec; "
                "preencoded-only training must configure a separate raw "
                "validation runtime"
            )
        from dataclasses import asdict

        from .inference import TrainingWanGenerationAdapter, run_wan_validation

        target = validation_staging_root(str(self.config["runtime"]["output_dir"])) / (
            f"step-{int(step):06d}"
        )
        summary = run_wan_validation(
            self.config,
            provider=TrainingWanGenerationAdapter(self),
            output_dir=target,
        )
        generation = asdict(summary)
        generation["output_dir"] = str(summary.output_dir)
        return {
            "schema": "solarwm.wan22-training-validation.v1",
            "stage": str(self.config["train"]["stage"]),
            "step": int(step),
            "generation": generation,
        }

    def close(self) -> None:
        close = getattr(self.data, "close", None)
        if callable(close):
            close()


def _bind_full_resume_initialization(runtime: Any, restored: Any) -> None:
    """Bind the runtime identity after full checkpoint restoration succeeds."""
    runtime.initialization_receipt = {
        "schema": "solarwm.wan22-full-resume-initialization.v1",
        "initialization_id": restored.identity,
        "source_step": restored.step,
        "source_path": str(restored.path),
        "standalone": restored.standalone,
        "weights": ["live", "ema", "optimizer", "scheduler", "reader", "rng"],
    }
    runtime.initialization_id = restored.identity
    runtime.set_global_step(restored.step)
    runtime.checkpoint_id = restored.identity


def build_stage0p5_runtime(config: Mapping[str, Any]) -> Wan5BStage0p5Runtime:
    """Construct assets, FSDP, the selected codec/reader, optimizer, and EMA."""

    import torch

    topology = initialize_torchrun(int(config["distributed"]["sequence_parallel_size"]))
    declared_world = int(config["distributed"]["world_size"])
    if int(topology.raw_world_size) != declared_world:
        raise BackendContractError(
            "Wan torchrun world size differs from distributed.world_size: "
            f"{topology.raw_world_size} != {declared_world}"
        )
    family = str(config["model"]["family"])
    checkpoint = config.get("checkpoint", {})
    full_resume = isinstance(checkpoint, Mapping) and str(checkpoint.get("mode")) == "full_resume"
    readiness = probe_runtime(
        config,
        family=family,
        require_cuda=True,
        require_transformer_weights=not full_resume,
        validate_index_contents=False,
    )
    readiness.require_ready()

    initialization_seed = model_init_seed(family, int(config["data"]["seed"]))
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    preencoded = str(config["data"]["encoding"]) == "preencoded"
    # Preencoded training still needs the VAE/T5 path for online validation,
    # so both input modes construct the same codec assets.
    if full_resume:
        diffusion = build_diffusion_architecture(config)
        text_encoder, vae = build_online_codec_components(config)
        initialization_receipt = {
            "schema": "solarwm.wan22-full-resume-initialization.v1",
            "initialization_id": "pending:full-resume",
        }
    else:
        diffusion, text_encoder, vae, weights = build_online_components(config)
        initialization_receipt = weights.initialization_receipt()
    device = torch.device("cuda", topology.local_rank)
    text_encoder.to(device)
    vae.to(device)
    diffusion.module.train().requires_grad_(True)
    diffusion.module = wrap_transformer_fsdp(diffusion.module, config, topology)

    stream_seed = int(config["data"]["seed"]) * 100003 + int(topology.dp_rank) * 1024
    torch.manual_seed(stream_seed)
    torch.cuda.manual_seed_all(stream_seed)
    optimizer_config = config["train"]["optimizer"]
    optimizer = torch.optim.AdamW(
        diffusion.module.parameters(),
        lr=float(optimizer_config["lr"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = make_warmup_cosine(
        optimizer,
        warmup_steps=int(config["train"]["warmup_steps"]),
        total_steps=int(config["train"]["max_steps"]),
        min_lr_ratio=float(optimizer_config.get("min_lr_ratio", 0.1)),
    )
    ema_config = config["train"]["ema"]
    ema = ShardedEMA(
        diffusion.module,
        decay=float(ema_config["decay"]),
        device=device,
        dtype=torch.float32,
    )
    if isinstance(checkpoint, Mapping) and str(checkpoint.get("mode")) == "weights_only":
        if list(checkpoint.get("weights", [])) not in (
            ["live", "ema"],
            ["ema", "ema"],
        ):
            raise BackendContractError(
                "Wan Stage0.5 weights-only runtime requires LIVE/EMA or EMA/EMA inheritance"
            )
        from .checkpoint import load_live_and_ema_weights_checkpoint

        restored_weights = load_live_and_ema_weights_checkpoint(
            config=config,
            path=str(checkpoint["path"]),
            diffusion=diffusion,
            ema=ema,
        )
        initialization_receipt = {
            "schema": "solarwm.wan22-dual-weights-initialization.v1",
            "initialization_id": restored_weights.identity,
            "source_step": restored_weights.source_step,
            "source_path": str(restored_weights.path),
            "standalone": restored_weights.standalone,
            "weights": ["live", "ema"],
            "ema_num_updates": 0,
        }
    codec_type = WanA14BOnlineCodec if family == "wan22_i2v_a14b" else Wan5BOnlineCodec
    codec = codec_type(
        vae,
        text_encoder,
        pixel_frames=int(config["data"]["pixel_frames"]),
        height=int(config["data"]["height"]),
        width=int(config["data"]["width"]),
        frame_sequence_length=int(config["model"]["frame_sequence_length"]),
    )
    loader = (
        build_preencoded_dataloader(config, topology)
        if preencoded
        else build_raw_dataloader(config, topology)
    )
    runtime = Wan5BStage0p5Runtime(
        config,
        diffusion=diffusion,
        codec=codec,
        batches=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        ema=ema,
        topology=topology,
        initialization_receipt=initialization_receipt,
    )
    if isinstance(checkpoint, Mapping) and str(checkpoint.get("mode")) == "full_resume":
        from .checkpoint import load_full_checkpoint

        restored = load_full_checkpoint(
            config=config,
            path=str(checkpoint["path"]),
            diffusion=runtime.diffusion,
            optimizer=runtime.optimizer,
            scheduler=runtime.lr_scheduler,
            ema=runtime.ema,
            reader=runtime.data,
            device=runtime.device,
        )
        _bind_full_resume_initialization(runtime, restored)
    return runtime


def run_stage0p5_training(config: Mapping[str, Any]) -> int:
    """Run the configured finite Stage0.5 job through the shared engine."""

    owner = os.environ.get(_TORCHRUN_OWNER_ENV, "backend").strip().lower()
    if owner not in {"backend", "caller"}:
        raise BackendContractError(
            f"{_TORCHRUN_OWNER_ENV} must be backend or caller, got {owner!r}"
        )
    runtime = None
    try:
        runtime = build_stage0p5_runtime(config)
        train = config["train"]
        runtime_config = config.get("runtime", {})
        max_steps = int(runtime_config.get("max_steps_override", train["max_steps"]))
        output = Path(str(runtime_config["output_dir"]))
        sink = JsonlEventSink(output / "events" / f"rank-{runtime.topology.raw_rank:05d}.jsonl")
        policy = StepPolicy(
            max_steps=max_steps,
            grad_accum=int(train["grad_accum"]),
            save_every=int(runtime_config.get("save_every", 0)),
            validate_every=int(runtime_config.get("validate_every", 0)),
            validation_steps=tuple(
                int(value) for value in runtime_config.get("validation_steps", ())
            ),
        )
        completed_step = TrainingEngine(runtime, policy, event_sink=sink).run()
        if completed_step != max_steps:
            raise BackendContractError(
                f"Wan training stopped at step {completed_step}, expected {max_steps}"
            )
        # Backend return values are process exit codes, not optimizer steps.
        return 0
    finally:
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()
        # Ordinary CLI calls are owned here. Wrappers needing post-training
        # collectives explicitly retain ownership until their final barrier.
        if owner == "backend":
            cleanup_torchrun()


__all__ = [
    "Wan5BStage0p5Runtime",
    "build_stage0p5_runtime",
    "deterministic_i2v_drop_mask",
    "expand_timesteps_to_tokens",
    "run_stage0p5_training",
]
