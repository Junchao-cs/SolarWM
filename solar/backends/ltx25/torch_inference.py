"""Official one-stage LTX sampling shared by inference and validation."""

from __future__ import annotations

import io
import itertools
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save

from solar.data.index import select_index_rows
from solar.errors import BackendContractError
from solar.inference import GeneratedSample, InferenceCase, encode_compare_mp4
from solar.runtime.distributed import collective_call

from .checkpoint import InferenceAdapterCheckpoint, StrictModelLoadReceipt
from .geometry import STABLE_GEOMETRY
from .inference import InferencePlan
from .official_codec import OfficialDiffVAEDecoder
from .torch_data import TorchBatch

if TYPE_CHECKING:
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    from .torch_model import LoRARuntime, StrictLoadedModel


def _pick_tensor(
    tensors: Mapping[str, torch.Tensor], names: Sequence[str], *, source: Path
) -> torch.Tensor:
    found = [tensors[name] for name in names if name in tensors]
    if len(found) != 1 or not isinstance(found[0], torch.Tensor):
        raise BackendContractError(f"{source} must contain exactly one of {tuple(names)!r}")
    return found[0]


def load_negative_caption(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = Path(path)
    if source.suffix != ".safetensors":
        raise BackendContractError("LTX negative caption cache must be safetensors")
    tensors = load_file(str(source), device="cpu")
    caption = _pick_tensor(
        tensors,
        ("video_prompt_embeds", "caption_embedding"),
        source=source,
    )
    mask = _pick_tensor(
        tensors,
        ("prompt_attention_mask", "caption_mask"),
        source=source,
    )
    if caption.ndim == 2:
        caption = caption.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if tuple(caption.shape) != (1, 1024, 4096) or tuple(mask.shape) != (1, 1024):
        raise BackendContractError("LTX negative caption cache geometry differs")
    if not caption.is_floating_point() or not bool(torch.isfinite(caption).all()):
        raise BackendContractError("LTX negative caption cache is not finite floating point")
    mask = mask.to(torch.int64)
    if (
        not bool(((mask == 0) | (mask == 1)).all())
        or not bool(mask.any())
        or bool((mask[:, 1:] < mask[:, :-1]).any())
    ):
        raise BackendContractError("LTX negative caption mask is not binary left padding")
    return (
        caption.to(device=device, dtype=torch.bfloat16).contiguous(),
        mask.to(device=device).contiguous(),
    )


def _adapter_path(checkpoint: InferenceAdapterCheckpoint, weights: str) -> Path:
    selected = str(weights).lower()
    if selected not in {"live", "ema"}:
        raise BackendContractError("LTX adapter weights must be live or ema")
    if checkpoint.weights != selected:
        raise BackendContractError("LTX adapter checkpoint role differs from requested weights")
    path = checkpoint.tensor_path
    if not path.is_file() or path.is_symlink():
        raise BackendContractError("LTX adapter checkpoint tensor is no longer a real file")
    return path


def load_adapter_checkpoint(
    lora: LoRARuntime,
    checkpoint: InferenceAdapterCheckpoint,
    *,
    weights: str,
) -> None:
    values = collective_call(
        lambda: load_file(str(_adapter_path(checkpoint, weights)), device="cpu"),
        dist=dist,
        label="LTX adapter checkpoint read",
        error_type=BackendContractError,
    )
    collective_call(
        lambda: lora.load_state_dict(values, broadcast=False),
        dist=dist,
        label="LTX adapter checkpoint apply",
        error_type=BackendContractError,
    )


def inference_cases(
    source: Any,
    plan: InferencePlan,
    *,
    camera_translation_transform: str,
    sample_count: int | None = None,
    selection_seed: int = 0,
) -> tuple[InferenceCase, ...]:
    transform = str(camera_translation_transform).strip().lower()
    if transform not in {"linear", "logd4"}:
        raise BackendContractError(
            "LTX inference camera_translation_transform must be linear or logd4"
        )
    rows = tuple(source.rows)
    if sample_count is not None:
        rows = select_index_rows(rows, sample_count=sample_count, seed=selection_seed)
    result = []
    for slot, row in enumerate(rows):
        factory = getattr(source, "case_for_row", None)
        if callable(factory):
            result.append(
                factory(
                    row,
                    slot=slot,
                    plan=plan,
                    camera_translation_transform=transform,
                )
            )
            continue
        batch = source.get(row.sample_id)
        prompt = str(row.values.get("caption") or "")
        result.append(
            InferenceCase(
                slot=slot,
                sample_id=row.sample_id,
                prompt=prompt,
                start_frame=batch.start_frame,
                noise_seed=plan.spec.seed + slot,
                camera_fingerprint=source.case_fingerprint(row.sample_id, batch),
                metadata={
                    "key": row.key,
                    "plan_fingerprint": batch.plan_fingerprint,
                    "source_pixel_frames": STABLE_GEOMETRY.pixel_frames,
                    "output_pixel_frames": STABLE_GEOMETRY.pixel_frames,
                    "train_latent_frames": STABLE_GEOMETRY.latent_frames,
                    "rollout_latent_frames": STABLE_GEOMETRY.latent_frames,
                    "generation_mode": "bidirectional",
                    "sample_solver": "stg-euler",
                    "num_inference_steps": plan.spec.num_inference_steps,
                    "camera_translation_transform": transform,
                    "artifact_valid": True,
                },
            )
        )
    return tuple(result)


def _patchified_noise(*, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    tokens = torch.randn(
        (1, STABLE_GEOMETRY.video_tokens, STABLE_GEOMETRY.latent_channels),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    return (
        tokens.view(
            1,
            STABLE_GEOMETRY.latent_frames,
            STABLE_GEOMETRY.latent_height,
            STABLE_GEOMETRY.latent_width,
            STABLE_GEOMETRY.latent_channels,
        )
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )


def _x0(sample: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (sample.float() - velocity.float() * sigma.float()).to(sample.dtype)


def _guided_x0(
    conditioned: torch.Tensor,
    unconditioned: torch.Tensor,
    perturbed: torch.Tensor,
    plan: InferencePlan,
) -> torch.Tensor:
    guidance = plan.guidance
    cond = conditioned.float()
    prediction = (
        cond
        + (float(guidance.cfg_scale) - 1.0) * (cond - unconditioned.float())
        + float(guidance.stg_scale) * (cond - perturbed.float())
    )
    if not math.isclose(float(guidance.rescale_scale), 0.0):
        std = prediction.std()
        if not bool(torch.isfinite(std).item()) or float(std.item()) == 0.0:
            raise BackendContractError("LTX guidance prediction has invalid variance")
        factor = (
            float(guidance.rescale_scale) * cond.std() / std + 1.0 - float(guidance.rescale_scale)
        )
        prediction = prediction * factor
    return prediction.to(conditioned.dtype)


def _perturbation(
    *, perturbed: bool, blocks: tuple[int, ...], device: torch.device
) -> BatchedPerturbationConfig:
    from ltx_core.guidance.perturbations import (
        BatchedPerturbationConfig,
        Perturbation,
        PerturbationConfig,
        PerturbationType,
    )

    config = (
        PerturbationConfig(
            [
                Perturbation(
                    type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                    blocks=list(blocks),
                )
            ]
        )
        if perturbed
        else PerturbationConfig.empty()
    )
    return BatchedPerturbationConfig(
        [config],
        num_blocks=48,
        device=device,
        dtype=torch.bfloat16,
    )


@dataclass
class LTX25Sampler:
    model: torch.nn.Module
    source: Any
    negative_caption: torch.Tensor
    negative_mask: torch.Tensor
    plan: InferencePlan
    device: torch.device

    def _velocity(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor,
        viewmats: torch.Tensor,
        camera_k: torch.Tensor,
        *,
        caption: torch.Tensor,
        mask: torch.Tensor,
        perturbed: bool,
    ) -> torch.Tensor:
        first_mask = torch.zeros(
            (1, STABLE_GEOMETRY.latent_frames),
            device=self.device,
            dtype=torch.bool,
        )
        first_mask[:, 0] = True
        result = self.model(
            video_latent=latent,
            sigma=sigma.reshape(1),
            caption_embedding=caption,
            first_frame_mask=first_mask,
            camera={
                "viewmats": viewmats,
                "K": camera_k,
            },
            caption_mask=mask,
            perturbations=_perturbation(
                perturbed=perturbed,
                blocks=self.plan.guidance.stg_blocks,
                device=self.device,
            ),
        )
        if tuple(result.shape) != (1, *STABLE_GEOMETRY.latent_shape):
            raise BackendContractError("LTX inference velocity geometry differs")
        if not bool(torch.isfinite(result).all()):
            raise BackendContractError("LTX inference velocity contains NaN or Inf")
        return result

    def sample(self, case: InferenceCase) -> torch.Tensor:
        def preflight() -> tuple[torch.Tensor, ...]:
            batch: TorchBatch = self.source.get(case.sample_id)
            if (
                batch.start_frame != case.start_frame
                or self.source.case_fingerprint(case.sample_id, batch) != case.camera_fingerprint
            ):
                raise BackendContractError("LTX inference case identity differs from its index")
            first = batch.first_frame_latent.to(
                self.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            caption = batch.video_prompt_embeds.to(
                self.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            mask = batch.prompt_attention_mask.to(
                self.device,
                dtype=torch.int64,
                non_blocking=True,
            )
            viewmats = batch.relative_w2c.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            camera_k = batch.camera_k.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            generator = torch.Generator(device=self.device).manual_seed(case.noise_seed)
            latent = _patchified_noise(generator=generator, device=self.device)
            latent[:, :, :1].copy_(first)
            sigmas = torch.tensor(
                self.plan.sigmas.copy(),
                device=self.device,
                dtype=torch.float32,
            )
            return first, caption, mask, viewmats, camera_k, latent, sigmas

        first, caption, mask, viewmats, camera_k, latent, sigmas = collective_call(
            preflight,
            dist=dist,
            label=f"LTX inference reader preflight for {case.sample_id}",
            error_type=BackendContractError,
        )
        was_training = self.model.training
        self.model.eval()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        try:
            with torch.inference_mode(), autocast:
                for sigma, sigma_next in itertools.pairwise(sigmas):
                    conditioned = self._velocity(
                        latent,
                        sigma,
                        viewmats,
                        camera_k,
                        caption=caption,
                        mask=mask,
                        perturbed=False,
                    )
                    unconditioned = self._velocity(
                        latent,
                        sigma,
                        viewmats,
                        camera_k,
                        caption=self.negative_caption,
                        mask=self.negative_mask,
                        perturbed=False,
                    )
                    perturbed = self._velocity(
                        latent,
                        sigma,
                        viewmats,
                        camera_k,
                        caption=caption,
                        mask=mask,
                        perturbed=True,
                    )
                    timestep = sigma.expand(1, 1, STABLE_GEOMETRY.latent_frames, 1, 1).clone()
                    timestep[:, :, :1] = 0.0
                    denoised = _guided_x0(
                        _x0(latent, conditioned, timestep),
                        _x0(latent, unconditioned, timestep),
                        _x0(latent, perturbed, timestep),
                        self.plan,
                    )
                    denoised[:, :, :1].copy_(first)
                    velocity = ((latent.float() - denoised.float()) / sigma.float()).to(
                        latent.dtype
                    )
                    latent = (
                        latent.float() + velocity.float() * (sigma_next.float() - sigma.float())
                    ).to(latent.dtype)
                    if not bool(torch.isfinite(latent).all()):
                        raise BackendContractError("LTX Euler trajectory became non-finite")
        finally:
            self.model.train(was_training)
        if not torch.equal(latent[:, :, :1], first):
            raise BackendContractError("LTX inference changed the clean first latent")
        return latent.contiguous()


def _mp4(frames: torch.Tensor, *, fps: int) -> bytes:
    import av

    value = (
        frames[0].permute(1, 2, 3, 0).mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    )
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = STABLE_GEOMETRY.width
        stream.height = STABLE_GEOMETRY.height
        stream.pix_fmt = "yuv420p"
        for array in value:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


class LTX25InferenceAdapter:
    family = "ltx25_video"

    def __init__(
        self,
        sampler: LTX25Sampler,
        decoder: OfficialDiffVAEDecoder,
        *,
        model_receipt: StrictModelLoadReceipt,
    ) -> None:
        self.sampler = sampler
        self.decoder = decoder
        self.model_receipt = model_receipt

    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        latent = self.sampler.sample(case)
        frames = self.decoder.decode(latent, seed=case.noise_seed)
        batch = self.sampler.source.get(case.sample_id)
        reference_latent = batch.video_latent
        if reference_latent.ndim == 4:
            reference_latent = reference_latent.unsqueeze(0)
        reference_latent = reference_latent.to(
            device=self.sampler.device,
            dtype=torch.bfloat16,
            non_blocking=True,
        )
        reference = self.decoder.decode(reference_latent, seed=case.noise_seed)
        metrics = _finite_generation_metrics(
            latent=latent,
            decoded=frames,
            reference_decoded=reference,
        )
        latent_bytes = save(
            {"generated_latents": latent.detach().cpu().to(torch.bfloat16).contiguous()},
            metadata={"weights_id": weights_id},
        )
        return GeneratedSample(
            artifacts={
                "compare.mp4": encode_compare_mp4(
                    reference,
                    frames,
                    fps=self.sampler.plan.spec.fps,
                    layout="bcthw",
                    value_range="zero_one",
                ),
                "latents.safetensors": latent_bytes,
                "video.mp4": _mp4(frames, fps=self.sampler.plan.spec.fps),
            },
            shape=tuple(int(item) for item in frames.shape),
            dtype="float32",
            metrics=metrics,
            provenance={
                "model_load_receipt": {
                    "provider_identity": self.model_receipt.provider_identity,
                    "ltx_core_version": self.model_receipt.ltx_core_version,
                    "strict_state_dict": self.model_receipt.strict_state_dict,
                    "adapter_target_count": self.model_receipt.adapter_target_count,
                    "adapter_trainable_parameters": (
                        self.model_receipt.adapter_trainable_parameters
                    ),
                },
                "decoder": self.decoder.implementation_class,
                "decoder_mode": "chunked_eager",
                "solver": "stg-euler",
                "num_inference_steps": self.sampler.plan.spec.num_inference_steps,
            },
        )


def _finite_generation_metrics(
    *,
    latent: torch.Tensor,
    decoded: torch.Tensor,
    reference_decoded: torch.Tensor,
) -> dict[str, float]:
    """Fail before publication unless latent and both decoded videos are finite."""

    values = {
        "latent": latent.detach(),
        "decoded": decoded.detach(),
        "reference_decoded": reference_decoded.detach(),
    }
    for name, value in values.items():
        if not bool(torch.isfinite(value).all().item()):
            raise BackendContractError(f"LTX inference {name} contains NaN or Inf")
    return {
        "finite_fraction": 1.0,
        "latent_finite_fraction": 1.0,
        "decoded_finite_fraction": 1.0,
        "reference_decoded_finite_fraction": 1.0,
        "latent_min": float(values["latent"].min().item()),
        "latent_max": float(values["latent"].max().item()),
        "latent_mean": float(values["latent"].mean(dtype=torch.float32).item()),
        "decoded_min": float(values["decoded"].min().item()),
        "decoded_max": float(values["decoded"].max().item()),
        "decoded_mean": float(values["decoded"].mean(dtype=torch.float32).item()),
    }


def build_inference_model(
    loaded: StrictLoadedModel,
    checkpoint: InferenceAdapterCheckpoint,
    *,
    weights: str,
) -> tuple[torch.nn.Module, LoRARuntime]:
    from .torch_model import LTX25SequenceParallelModel, inject_lora

    loaded.backbone.requires_grad_(False)
    core, lora = inject_lora(loaded.core)
    load_adapter_checkpoint(lora, checkpoint, weights=weights)
    loaded.backbone.transformer = LTX25SequenceParallelModel(core)
    loaded.backbone.eval().requires_grad_(False)
    return loaded.backbone, lora


__all__ = [
    "LTX25InferenceAdapter",
    "LTX25Sampler",
    "build_inference_model",
    "inference_cases",
    "load_adapter_checkpoint",
    "load_negative_caption",
]
