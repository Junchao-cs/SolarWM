"""Executable Wan2.2 TI2V-5B Stage1 teacher-forcing runtime."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solarwm.errors import BackendContractError
from solarwm.runtime.randomness import model_init_seed
from solarwm.training.ema import ShardedEMA
from solarwm.training.engine import (
    BatchIdentity,
    JsonlEventSink,
    MicrobatchResult,
    StepPolicy,
    TrainingEngine,
)
from solarwm.training.schedule import make_warmup_cosine

from ..objectives import first_frame_loss_mask, weighted_masked_mse
from .checkpoint import load_full_checkpoint, load_weights_only_checkpoint
from .codec import Wan5BOnlineCodec
from .components import (
    build_diffusion_architecture,
    build_online_codec_components,
    build_online_components,
)
from .data import build_raw_dataloader
from .distributed import cleanup_torchrun, initialize_torchrun, wrap_transformer_fsdp
from .readiness import probe_runtime
from .stage0p5 import (
    Wan5BStage0p5Runtime,
    _bind_full_resume_initialization,
    _rng_identity,
    deterministic_i2v_drop_mask,
    expand_timesteps_to_tokens,
)

_TORCHRUN_OWNER_ENV = "SOLARWM_TORCHRUN_LIFECYCLE_OWNER"


def sample_per_block_timestep_indices(
    batch_size: int,
    latent_frames: int,
    num_frame_per_block: int,
    *,
    num_train_timesteps: int,
    device: Any,
    generator: Any | None = None,
) -> Any:
    """Draw B*F timesteps and broadcast within each temporal block."""

    import torch

    batch_size = int(batch_size)
    latent_frames = int(latent_frames)
    block_frames = int(num_frame_per_block)
    if batch_size < 1 or latent_frames < 1 or block_frames < 1:
        raise BackendContractError("Wan Stage1 timestep dimensions must be positive")
    if latent_frames % block_frames:
        raise BackendContractError(
            "Wan Stage1 latent frames must divide evenly into timestep blocks"
        )
    values = torch.randint(
        0,
        int(num_train_timesteps),
        (batch_size, latent_frames),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    blocks = values.reshape(batch_size, latent_frames // block_frames, block_frames)
    blocks[:, :, 1:] = blocks[:, :, :1]
    return blocks.reshape(batch_size, latent_frames)


class Wan5BStage1Runtime(Wan5BStage0p5Runtime):
    """Independent teacher forcing with one [clean|noisy] model forward."""

    def train_microbatch(self, micro_index: int, grad_accum: int) -> MicrobatchResult:
        import torch

        rng_identity = _rng_identity(self.device)
        batch = next(self.batches)
        camera = {
            key: value.to(self.device, non_blocking=True) for key, value in batch["camera"].items()
        }
        if self.codec is None:
            raise BackendContractError("Wan Stage1 requires the online T5/VAE codec")
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
        batch_size, latent_frames = clean.shape[:2]
        train = self.config["train"]
        data = self.config["data"]
        model = self.config["model"]
        block_frames = int(model["num_frame_per_block"])

        dropped = deterministic_i2v_drop_mask(
            probability=float(train["i2v_image_condition_dropout"]),
            batch_size=batch_size,
            seed=int(data["seed"]),
            global_step=self.global_step,
            logical_rank=int(self.topology.dp_rank),
            micro_index=int(micro_index),
            device=self.device,
        )
        timestep_index = sample_per_block_timestep_indices(
            batch_size,
            latent_frames,
            block_frames,
            num_train_timesteps=int(train["num_train_timesteps"]),
            device=self.device,
        )
        timestep = self.diffusion.scheduler.timesteps.to(
            device=self.device,
            dtype=torch.float32,
        )[timestep_index]
        timestep[~dropped, 0] = 0.0

        noise = torch.randn_like(clean)
        noisy = (
            self.diffusion.scheduler.add_noise(
                clean.flatten(0, 1).float(),
                noise.flatten(0, 1).float(),
                timestep.flatten(0, 1),
            )
            .unflatten(0, (batch_size, latent_frames))
            .to(clean.dtype)
        )
        noisy[~dropped, 0] = clean[~dropped, 0]
        if int(train["noise_augmentation_max_timestep"]) != 0:
            raise BackendContractError("Wan Stage1 supports clean context only")
        augmentation_timestep = torch.zeros_like(timestep)
        timestep_tokens = expand_timesteps_to_tokens(
            timestep,
            int(model["frame_sequence_length"]),
        )
        augmentation_tokens = expand_timesteps_to_tokens(
            augmentation_timestep,
            int(model["frame_sequence_length"]),
        )
        prediction = self.diffusion.forward_train_tf(
            noisy,
            clean,
            prompt,
            camera,
            timestep_tokens,
            augmentation_tokens,
            num_frame_per_block=block_frames,
            sequence_length=latent_frames * int(model["frame_sequence_length"]),
        )
        target = self.diffusion.scheduler.training_target(clean, noise, timestep)
        weight = self.diffusion.scheduler.training_weight(timestep).view(
            batch_size,
            latent_frames,
            1,
            1,
            1,
        )
        mask = first_frame_loss_mask(batch_size, latent_frames, dropped)
        loss = weighted_masked_mse(prediction, target, mask, weight)
        if not bool(torch.isfinite(loss).item()):
            raise BackendContractError(f"non-finite Wan Stage1 loss: {loss.item()}")
        loss_scale = int(grad_accum) * int(self.topology.sp_size)
        (loss / loss_scale).backward()
        plan_payload = {
            "sample_ids": list(batch["sample_ids"]),
            "start_frames": list(batch["start_frames"]),
            "source_frame_indices": [list(value) for value in batch["source_frame_indices"]],
        }
        fingerprint = hashlib.blake2s(repr(plan_payload).encode()).hexdigest()
        noise_seed = int(rng_identity[:16], 16)
        identity = BatchIdentity(
            sample_ids=tuple(str(value) for value in batch["sample_ids"]),
            start_frames=tuple(int(value) for value in batch["start_frames"]),
            noise_seeds=tuple(noise_seed + index for index in range(batch_size)),
            checkpoint_id=self.checkpoint_id,
            plan_fingerprint=fingerprint,
        )
        return MicrobatchResult(identity=identity, losses={"flow_matching": loss.item()})


def build_stage1_runtime(config: Mapping[str, Any]) -> Wan5BStage1Runtime:
    """Build Stage1 and initialize its live/EMA weights from Stage0.5 EMA."""

    import torch

    topology = initialize_torchrun(int(config["distributed"]["sequence_parallel_size"]))
    declared_world = int(config["distributed"]["world_size"])
    if int(topology.raw_world_size) != declared_world:
        raise BackendContractError(
            "Wan torchrun world size differs from distributed.world_size: "
            f"{topology.raw_world_size} != {declared_world}"
        )
    checkpoint = config["checkpoint"]
    full_resume = str(checkpoint.get("mode")) == "full_resume"
    probe_runtime(
        config,
        family="wan22_ti2v_5b",
        require_cuda=True,
        require_transformer_weights=not full_resume,
        validate_index_contents=False,
    ).require_ready()

    initialization_seed = model_init_seed("wan22_ti2v_5b", int(config["data"]["seed"]))
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    if full_resume:
        diffusion = build_diffusion_architecture(config)
        text_encoder, vae = build_online_codec_components(config)
        initialization_receipt = {
            "schema": "solarwm.wan22-full-resume-initialization.v1",
            "initialization_id": "pending:full-resume",
        }
    else:
        diffusion, text_encoder, vae, _ = build_online_components(config)
    device = torch.device("cuda", topology.local_rank)
    text_encoder.to(device)
    vae.to(device)
    diffusion.module.train().requires_grad_(True)
    diffusion.module = wrap_transformer_fsdp(diffusion.module, config, topology)

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
    restored_weights = None
    if not full_resume:
        restored_weights = load_weights_only_checkpoint(
            config=config,
            path=str(checkpoint["path"]),
            diffusion=diffusion,
        )
        initialization_receipt = {
            "schema": "solarwm.wan22-weights-only-initialization.v1",
            "initialization_id": restored_weights.identity,
            "source_step": restored_weights.source_step,
            "source_path": str(restored_weights.path),
            "standalone": restored_weights.standalone,
            "weights": "ema",
        }
    ema_config = config["train"]["ema"]
    ema = ShardedEMA(
        diffusion.module,
        decay=float(ema_config["decay"]),
        device=device,
        dtype=torch.float32,
    )

    stream_seed = int(config["data"]["seed"]) * 100003 + int(topology.dp_rank) * 1024
    torch.manual_seed(stream_seed)
    torch.cuda.manual_seed_all(stream_seed)
    codec = Wan5BOnlineCodec(
        vae,
        text_encoder,
        pixel_frames=int(config["data"]["pixel_frames"]),
        height=int(config["data"]["height"]),
        width=int(config["data"]["width"]),
        frame_sequence_length=int(config["model"]["frame_sequence_length"]),
    )
    loader = build_raw_dataloader(config, topology)
    runtime = Wan5BStage1Runtime(
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
    if full_resume:
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


def run_stage1_training(config: Mapping[str, Any]) -> int:
    """Run the configured finite Stage1 job and return a process exit code."""

    owner = os.environ.get(_TORCHRUN_OWNER_ENV, "backend").strip().lower()
    if owner not in {"backend", "caller"}:
        raise BackendContractError(
            f"{_TORCHRUN_OWNER_ENV} must be backend or caller, got {owner!r}"
        )
    runtime = None
    try:
        runtime = build_stage1_runtime(config)
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
                f"Wan Stage1 training stopped at step {completed_step}, expected {max_steps}"
            )
        return 0
    finally:
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()
        if owner == "backend":
            cleanup_torchrun()


__all__ = [
    "Wan5BStage1Runtime",
    "build_stage1_runtime",
    "run_stage1_training",
    "sample_per_block_timestep_indices",
]
