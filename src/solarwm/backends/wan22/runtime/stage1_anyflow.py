"""Executable Wan2.2 TI2V-5B Stage1 teacher-forcing AnyFlow v1.5.

The objective uses globally assigned sample types, shifted scalar ``(t, r)``
pairs, a bounded central
difference along the predicted velocity, Gaussian timestep weighting, and
adaptive non-diffusion rescaling.  Every model call goes through the root
``WanDiffusion`` adapter so FSDP pre/post-forward hooks remain active.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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

from ..anyflow import bounded_difference_timesteps, sample_time_pairs
from ..objectives import apply_timestep_shift
from .assets import WanAssetLayout
from .checkpoint import load_anyflow_weights_only_checkpoint, load_full_checkpoint
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
    _bind_full_resume_initialization,
    expand_timesteps_to_tokens,
)
from .stage1 import Wan5BStage1Runtime

_TORCHRUN_OWNER_ENV = "SOLARWM_TORCHRUN_LIFECYCLE_OWNER"
AnyFlowForward = Callable[[Any, Any, Any, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class AnyFlowV15Result:
    """Auditable outputs from one AnyFlow objective evaluation."""

    loss: Any
    prediction: Any
    target: Any
    sample_type: Any
    timestep: Any
    endpoint_timestep: Any
    noise: Any
    noisy: Any
    weight: Any
    loss_mask: Any
    forward_count: int


def _prefix_broadcast(value: Any, reference: Any) -> Any:
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    try:
        return value.expand(reference.shape)
    except RuntimeError as exc:
        raise BackendContractError(
            f"AnyFlow value {tuple(value.shape)} cannot broadcast to {tuple(reference.shape)}"
        ) from exc


def _masked_per_sample_mse(prediction: Any, target: Any, mask: Any) -> Any:
    import torch

    if prediction.shape != target.shape or prediction.ndim < 1:
        raise BackendContractError("AnyFlow prediction/target layout mismatch")
    valid = _prefix_broadcast(
        torch.as_tensor(mask, device=prediction.device, dtype=torch.bool),
        prediction,
    )
    diff = torch.where(
        valid,
        prediction.float() - target.float(),
        torch.zeros_like(prediction, dtype=torch.float32),
    )
    dims = tuple(range(1, prediction.ndim))
    return diff.square().sum(dim=dims) / valid.float().sum(dim=dims).clamp_min(1.0)


def _masked_mean(value: Any, mask: Any) -> Any:
    import torch

    valid = _prefix_broadcast(
        torch.as_tensor(mask, device=value.device, dtype=torch.bool),
        value,
    )
    dims = tuple(range(1, value.ndim))
    return torch.where(valid, value.float(), torch.zeros_like(value, dtype=torch.float32)).sum(
        dim=dims
    ) / valid.float().sum(dim=dims).clamp_min(1.0)


def adaptive_rescale_non_diffusion_losses(
    per_sample_loss: Any,
    is_diffusion: Any,
    *,
    gather_fn: Callable[[Any], Any] | None = None,
) -> Any:
    """Apply the reference logical-DP adaptive scale with empty-class safety."""

    import torch
    import torch.distributed as dist

    if per_sample_loss.ndim != 1 or per_sample_loss.shape != is_diffusion.shape:
        raise BackendContractError("AnyFlow adaptive-rescale inputs must be equal 1-D tensors")
    detached = per_sample_loss.detach()
    mask = is_diffusion.detach().to(dtype=torch.bool)
    if gather_fn is not None:
        losses = gather_fn(detached)
        masks = gather_fn(mask)
        global_loss = (
            losses.to(device=detached.device).reshape(-1)
            if torch.is_tensor(losses)
            else torch.cat([item.to(device=detached.device).reshape(-1) for item in losses])
        )
        global_mask = (
            masks.to(device=mask.device).reshape(-1)
            if torch.is_tensor(masks)
            else torch.cat([item.to(device=mask.device).reshape(-1) for item in masks])
        ).bool()
    elif dist.is_available() and dist.is_initialized():
        loss_parts = [torch.empty_like(detached) for _ in range(dist.get_world_size())]
        mask_parts = [torch.empty_like(mask) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_parts, detached)
        dist.all_gather(mask_parts, mask)
        global_loss = torch.cat(loss_parts)
        global_mask = torch.cat(mask_parts).bool()
    else:
        global_loss = detached
        global_mask = mask
    reference = global_loss[global_mask]
    reference = reference[torch.isfinite(reference)]
    if reference.numel() == 0 or bool(mask.all().item()):
        return per_sample_loss
    mean_diffusion = reference.mean().to(per_sample_loss)
    scale = mean_diffusion / (detached + 1.0e-5)
    scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
    return torch.where(mask, per_sample_loss, per_sample_loss * scale)


def anyflow_v15_objective(
    *,
    clean: Any,
    condition: Mapping[str, Any],
    model_u: AnyFlowForward,
    scheduler: Any,
    loss_mask: Any,
    logical_dp_rank: int,
    logical_dp_world_size: int,
    shift: float,
    num_train_timesteps: int,
    epsilon: float,
    diffusion_ratio: float,
    consistency_ratio: float,
    guidance: float,
    negative_prompt_embeds: Any | None,
    generator: Any | None = None,
    noise: Any | None = None,
    gather_fn: Callable[[Any], Any] | None = None,
) -> AnyFlowV15Result:
    """Evaluate the AnyFlow-v1.5 objective without stepping state."""

    import torch

    if clean.ndim != 5 or clean.shape[0] < 1 or clean.shape[1] < 1:
        raise BackendContractError("AnyFlow clean latents must have shape [B,T,C,H,W]")
    if not math.isfinite(float(guidance)) or float(guidance) <= 0:
        raise BackendContractError("AnyFlow guidance must be finite and positive")
    batch_size, latent_frames = clean.shape[:2]
    pairs = sample_time_pairs(
        batch_size,
        logical_dp_rank=int(logical_dp_rank),
        logical_dp_world_size=int(logical_dp_world_size),
        diffusion_ratio=float(diffusion_ratio),
        consistency_ratio=float(consistency_ratio),
        generator=generator,
        device=clean.device,
    )
    t_sample = apply_timestep_shift(pairs.t, float(shift)) * int(num_train_timesteps)
    r_sample = apply_timestep_shift(pairs.r, float(shift)) * int(num_train_timesteps)
    t_raw = t_sample[:, None].expand(batch_size, latent_frames).clone()
    r_raw = r_sample[:, None].expand(batch_size, latent_frames).clone()
    mask = torch.as_tensor(loss_mask, device=clean.device, dtype=torch.bool)
    if tuple(mask.shape) != (batch_size, latent_frames):
        raise BackendContractError(
            f"AnyFlow loss mask must have shape {(batch_size, latent_frames)}"
        )
    # Stage1 AnyFlow preserves and excludes the TI2V first-latent anchor.
    anchored = ~mask[:, 0]
    t_raw[anchored, 0] = 0.0
    r_raw[anchored, 0] = 0.0

    if noise is None:
        sampled_noise = (
            torch.randn_like(clean)
            if generator is None
            else torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        )
    else:
        sampled_noise = torch.as_tensor(noise, device=clean.device, dtype=clean.dtype)
    if sampled_noise.shape != clean.shape:
        raise BackendContractError("AnyFlow noise shape differs from clean latents")
    sigma = t_raw.float() / float(num_train_timesteps)
    z_noisy = (
        (1.0 - sigma[..., None, None, None]) * clean.float()
        + sigma[..., None, None, None] * sampled_noise.float()
    ).to(clean.dtype)
    z_noisy[anchored, 0] = clean[anchored, 0]
    # Form the BF16 velocity before promoting it for objective arithmetic.
    velocity = (sampled_noise - clean).float()

    u_cond = model_u(z_noisy, t_raw, r_raw, condition)
    forward_count = 1
    if float(guidance) != 1.0:
        if negative_prompt_embeds is None:
            raise BackendContractError(
                "guided AnyFlow requires the negative prompt embedding asset"
            )
        negative = torch.as_tensor(
            negative_prompt_embeds,
            device=clean.device,
            dtype=condition["prompt_embeds"].dtype,
        )
        if negative.ndim == 2:
            negative = negative.unsqueeze(0)
        if negative.ndim != 3 or negative.shape[0] != 1:
            raise BackendContractError(
                "AnyFlow negative_prompt_embeds must have shape [L,D] or [1,L,D]"
            )
        negative_condition = {"prompt_embeds": negative.expand(batch_size, -1, -1)}
        with torch.no_grad():
            u_uncond = model_u(z_noisy, t_raw, t_raw, negative_condition)
        target = float(guidance) * velocity + (1.0 - float(guidance)) * u_uncond.float()
        forward_count += 1
    else:
        target = velocity

    with torch.no_grad():
        v_pred = model_u(z_noisy, t_raw, t_raw, condition)
    forward_count += 1
    t_plus, t_minus = bounded_difference_timesteps(
        t_raw,
        r_raw,
        epsilon=float(epsilon),
        num_train_timesteps=int(num_train_timesteps),
    )
    t_plus[anchored, 0] = 0.0
    t_minus[anchored, 0] = 0.0
    plus_step = ((t_plus - t_raw) / float(num_train_timesteps))[..., None, None, None]
    minus_step = ((t_raw - t_minus) / float(num_train_timesteps))[..., None, None, None]
    z_plus = (z_noisy.float() + v_pred.float() * plus_step).to(clean.dtype)
    z_minus = (z_noisy.float() - v_pred.float() * minus_step).to(clean.dtype)
    z_plus[anchored, 0] = clean[anchored, 0]
    z_minus[anchored, 0] = clean[anchored, 0]
    with torch.no_grad():
        u_plus = model_u(z_plus, t_plus, r_raw, condition)
        u_minus = model_u(z_minus, t_minus, r_raw, condition)
    forward_count += 2
    denominator = (t_plus - t_minus).float().clamp_min(1.0e-6)
    derivative = (u_plus.detach().float() - u_minus.detach().float()) / denominator[
        ..., None, None, None
    ]
    prediction = u_cond.float() + (t_raw - r_raw).float()[..., None, None, None] * derivative
    residual = (prediction - target.detach()) / float(guidance)
    per_sample = _masked_per_sample_mse(
        residual,
        torch.zeros_like(residual),
        mask,
    )
    per_sample = adaptive_rescale_non_diffusion_losses(
        per_sample,
        pairs.is_diffusion,
        gather_fn=gather_fn,
    )
    # The shared scheduler preserves the flattened lookup surface;
    # AnyFlow's reduction is frame-shaped, so restore [B,T] before masking.
    weight = scheduler.training_weight(t_raw).reshape_as(t_raw)
    per_sample = per_sample * _masked_mean(weight, mask)
    loss = per_sample.mean()
    if not bool(torch.isfinite(loss).item()):
        raise BackendContractError(f"non-finite Wan AnyFlow-v1.5 loss: {loss.item()}")
    return AnyFlowV15Result(
        loss=loss,
        prediction=prediction,
        target=target.detach(),
        sample_type=pairs.sample_type,
        timestep=t_raw,
        endpoint_timestep=r_raw,
        noise=sampled_noise,
        noisy=z_noisy,
        weight=weight,
        loss_mask=mask,
        forward_count=forward_count,
    )


def load_negative_prompt_embedding(config: Mapping[str, Any], *, device: Any) -> Any:
    """Load and validate the AnyFlow guidance embedding tensor."""

    import torch

    layout = WanAssetLayout.from_config(config)
    path = layout.anyflow_negative_embedding
    if path is None or not path.is_file():
        raise BackendContractError("AnyFlow negative embedding asset is missing")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or "negative_prompt_embeds" not in loaded:
        raise BackendContractError("AnyFlow embedding must contain negative_prompt_embeds")
    value = loaded["negative_prompt_embeds"]
    if not isinstance(value, torch.Tensor) or value.ndim not in {2, 3}:
        raise BackendContractError("AnyFlow negative prompt embedding has invalid shape")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.shape[0] != 1:
        raise BackendContractError("AnyFlow negative prompt embedding batch must be one")
    return value.to(device=device, dtype=torch.bfloat16)


class Wan5BStage1AnyFlowRuntime(Wan5BStage1Runtime):
    """Online-codec/FSDP runtime for the supported AnyFlow profiles."""

    def __init__(self, *args: Any, negative_prompt_embeds: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.negative_prompt_embeds = negative_prompt_embeds

    def train_microbatch(self, micro_index: int, grad_accum: int) -> MicrobatchResult:
        import torch

        batch = next(self.batches)
        camera = {
            key: value.to(self.device, non_blocking=True) for key, value in batch["camera"].items()
        }
        if self.codec is None:
            raise BackendContractError("Wan Stage1 AnyFlow requires the online T5/VAE codec")
        pixels = batch["pixels"].to(self.device, non_blocking=True)
        encoded = self.codec.encode_batch(
            sample_ids=batch["sample_ids"],
            pixels=pixels,
            captions=batch["prompts"],
            camera=camera,
        )
        clean = encoded["latents"]
        condition = {"prompt_embeds": encoded["prompt_embeds"]}
        camera = encoded["camera"]
        batch_size, latent_frames = clean.shape[:2]
        train = self.config["train"]
        model = self.config["model"]
        frame_tokens = int(model["frame_sequence_length"])
        block_frames = int(model["num_frame_per_block"])
        augmentation = torch.zeros(
            (batch_size, latent_frames),
            device=self.device,
            dtype=torch.float32,
        )
        augmentation_tokens = expand_timesteps_to_tokens(augmentation, frame_tokens)

        def model_u(sample: Any, t_value: Any, r_value: Any, cond: Mapping[str, Any]) -> Any:
            return self.diffusion.forward_train_tf(
                sample,
                clean,
                cond,
                camera,
                expand_timesteps_to_tokens(t_value, frame_tokens),
                augmentation_tokens,
                num_frame_per_block=block_frames,
                sequence_length=latent_frames * frame_tokens,
                r_timestep_tokens=expand_timesteps_to_tokens(r_value, frame_tokens),
                augmentation_r_timestep_tokens=augmentation_tokens,
            )

        mask = torch.ones(
            (batch_size, latent_frames),
            device=self.device,
            dtype=torch.bool,
        )
        mask[:, 0] = False
        result = anyflow_v15_objective(
            clean=clean,
            condition=condition,
            model_u=model_u,
            scheduler=self.diffusion.scheduler,
            loss_mask=mask,
            logical_dp_rank=int(self.topology.dp_rank) * int(grad_accum) + int(micro_index),
            logical_dp_world_size=int(self.topology.dp_world_size) * int(grad_accum),
            shift=float(model["timestep_shift"]),
            num_train_timesteps=int(train["num_train_timesteps"]),
            epsilon=float(train["anyflow_epsilon"]),
            diffusion_ratio=float(train["anyflow_diffusion_ratio"]),
            consistency_ratio=float(train["anyflow_consistency_ratio"]),
            guidance=float(train["anyflow_fuse_guidance_scale"]),
            negative_prompt_embeds=self.negative_prompt_embeds,
        )
        (result.loss / (int(grad_accum) * int(self.topology.sp_size))).backward()
        plan = {
            "sample_ids": list(batch["sample_ids"]),
            "start_frames": list(batch["start_frames"]),
            "source_frame_indices": [list(value) for value in batch["source_frame_indices"]],
        }
        fingerprint = hashlib.blake2s(repr(plan).encode()).hexdigest()
        seed = (
            int(self.config["data"]["seed"]) * 1_000_003
            + self.global_step * int(grad_accum)
            + int(micro_index)
        )
        identity = BatchIdentity(
            sample_ids=tuple(str(value) for value in batch["sample_ids"]),
            start_frames=tuple(int(value) for value in batch["start_frames"]),
            noise_seeds=tuple(seed + index for index in range(batch_size)),
            checkpoint_id=self.checkpoint_id,
            plan_fingerprint=fingerprint,
        )
        counts = torch.bincount(result.sample_type.long(), minlength=3)
        return MicrobatchResult(
            identity=identity,
            losses={
                "anyflow_v1_5": float(result.loss.item()),
                "anyflow_diffusion_samples": float(counts[0].item()),
                "anyflow_consistency_samples": float(counts[1].item()),
                "anyflow_forward_map_samples": float(counts[2].item()),
            },
        )


def build_stage1_anyflow_runtime(config: Mapping[str, Any]) -> Wan5BStage1AnyFlowRuntime:
    """Build Stage1 AnyFlow and upgrade the Stage0.5-FM objective weights."""

    import torch

    topology = initialize_torchrun(int(config["distributed"]["sequence_parallel_size"]))
    if int(topology.raw_world_size) != int(config["distributed"]["world_size"]):
        raise BackendContractError("Wan AnyFlow torchrun world size differs from config")
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
    if not full_resume:
        restored_weights, delta_keys = load_anyflow_weights_only_checkpoint(
            config=config,
            path=str(checkpoint["path"]),
            diffusion=diffusion,
        )
        initialization_receipt = {
            "schema": "solarwm.wan22-anyflow-initialization.v1",
            "initialization_id": restored_weights.identity,
            "source_step": restored_weights.source_step,
            "source_path": str(restored_weights.path),
            "standalone": restored_weights.standalone,
            "weights": "ema",
            "initialized_delta_keys": list(delta_keys),
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
    negative = load_negative_prompt_embedding(config, device=device)
    runtime = Wan5BStage1AnyFlowRuntime(
        config,
        diffusion=diffusion,
        codec=codec,
        batches=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        ema=ema,
        topology=topology,
        negative_prompt_embeds=negative,
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


def run_stage1_anyflow_training(config: Mapping[str, Any]) -> int:
    """Run Stage1 AnyFlow with the shared finite-step training engine."""

    owner = os.environ.get(_TORCHRUN_OWNER_ENV, "backend").strip().lower()
    if owner not in {"backend", "caller"}:
        raise BackendContractError(
            f"{_TORCHRUN_OWNER_ENV} must be backend or caller, got {owner!r}"
        )
    runtime = None
    try:
        runtime = build_stage1_anyflow_runtime(config)
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
        completed = TrainingEngine(runtime, policy, event_sink=sink).run()
        if completed != max_steps:
            raise BackendContractError(
                f"Wan AnyFlow stopped at step {completed}, expected {max_steps}"
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
    "AnyFlowV15Result",
    "Wan5BStage1AnyFlowRuntime",
    "adaptive_rescale_non_diffusion_losses",
    "anyflow_v15_objective",
    "build_stage1_anyflow_runtime",
    "load_negative_prompt_embedding",
    "run_stage1_anyflow_training",
]
