"""MiniMax-H3 generation packaging and shared-engine inference adapter."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solarwm.inference import GeneratedSample, InferenceCase, encode_compare_mp4

from .artifacts import H3ArtifactBatch
from .stage0p5 import H3Stage0p5Core


def camera_fingerprint(batch: H3ArtifactBatch) -> str:
    digest = hashlib.blake2s()
    digest.update(batch.camera_viewmats.detach().cpu().numpy().tobytes())
    digest.update(batch.camera_K.detach().cpu().numpy().tobytes())
    if batch.rollout_camera_viewmats is not None:
        digest.update(batch.rollout_camera_viewmats.detach().cpu().numpy().tobytes())
        digest.update(batch.rollout_camera_K.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def decode_h3_latents(video_vae: Any, latents: Any, *, device: Any) -> Any:
    """Run official latent de-normalization and the 47->158 VisualVAE decode."""

    import torch

    mean = torch.tensor(video_vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(video_vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    decoder_input = latents.to(device=device, dtype=torch.float32) * std + mean
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ),
    ):
        decoded = video_vae.decode(decoder_input, return_dict=False)[0]
    pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, -1, 1, 1, 1)
    decoded = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1)
    if tuple(decoded.shape) != (1, 3, 158, 768, 1344):
        raise RuntimeError(
            f"H3 VisualVAE decoded {tuple(decoded.shape)}, expected [1,3,158,768,1344]"
        )
    return decoded


def _mp4_bytes(decoded: Any, *, fps: int = 24) -> bytes:
    import imageio.v2 as imageio
    import numpy as np
    import torch

    frames = (
        decoded[0].permute(1, 2, 3, 0).mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    )
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        imageio.mimwrite(
            str(path),
            [np.asarray(frame) for frame in frames],
            fps=int(fps),
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


class H3InferenceAdapter:
    family = "minimax_h3"

    def __init__(
        self,
        *,
        core: H3Stage0p5Core,
        video_vae: Any,
        batches: Mapping[str, H3ArtifactBatch],
        device: Any,
        num_inference_steps: int,
    ) -> None:
        self.core = core
        self.video_vae = video_vae
        self.batches = dict(batches)
        self.device = device
        self.num_inference_steps = int(num_inference_steps)

    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        try:
            batch = self.batches[case.sample_id]
        except KeyError as exc:
            raise ValueError(f"no H3 artifact batch for case {case.sample_id!r}") from exc
        if camera_fingerprint(batch) != case.camera_fingerprint:
            raise ValueError("H3 inference case camera identity differs")
        latents = self.core.generate(
            batch,
            noise_seed=case.noise_seed,
            num_inference_steps=self.num_inference_steps,
        )
        return package_generated(
            latents,
            video_vae=self.video_vae,
            device=self.device,
            weights_id=weights_id,
            num_inference_steps=self.num_inference_steps,
            reference_latents=batch.target_latents,
        )


def package_generated(
    latents: Any,
    *,
    video_vae: Any,
    device: Any,
    weights_id: str,
    num_inference_steps: int,
    reference_latents: Any | None = None,
    sample_solver: str = "shifted-euler-data-ward",
) -> GeneratedSample:
    """Decode/package already-generated latents on the single artifact writer."""

    import torch
    from safetensors.torch import save

    decoded = decode_h3_latents(video_vae, latents, device=device)
    ground_truth = None
    if reference_latents is not None:
        reference = reference_latents
        if reference.ndim == 4:
            reference = reference.unsqueeze(0)
        ground_truth = decode_h3_latents(video_vae, reference, device=device)
    metrics = _finite_generation_metrics(
        latents=latents,
        decoded=decoded,
        reference_decoded=ground_truth,
    )
    artifacts = {
        "generated.safetensors": save(
            {"latents": latents.detach().to("cpu", dtype=torch.bfloat16)}
        ),
        "generated.mp4": _mp4_bytes(decoded),
    }
    if ground_truth is not None:
        artifacts["compare.mp4"] = encode_compare_mp4(
            ground_truth,
            decoded,
            fps=24,
            layout="bcthw",
            value_range="zero_one",
        )
    return GeneratedSample(
        artifacts=artifacts,
        shape=tuple(int(value) for value in decoded.shape),
        dtype="float32",
        metrics=metrics,
        provenance={
            "weights_id": weights_id,
            "preencode_version": "h3.158f.v1",
            "sampler": sample_solver,
            "solver": sample_solver,
            "num_inference_steps": int(num_inference_steps),
            "num_sigma_points": int(num_inference_steps),
            "video_shift": 12.0,
            "audio_shift": 3.0,
        },
    )


def _finite_generation_metrics(
    *,
    latents: Any,
    decoded: Any,
    reference_decoded: Any | None,
) -> dict[str, float]:
    """Fail before publication unless generated latent/decoded tensors are finite."""

    import torch

    values = {
        "latent": latents.detach(),
        "decoded": decoded.detach(),
    }
    if reference_decoded is not None:
        values["reference_decoded"] = reference_decoded.detach()
    for name, value in values.items():
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"H3 inference {name} contains NaN or Inf")
    metrics = {
        "finite_fraction": 1.0,
        "latent_finite_fraction": 1.0,
        "decoded_finite_fraction": 1.0,
        "latent_min": float(values["latent"].min().item()),
        "latent_max": float(values["latent"].max().item()),
        "latent_mean": float(values["latent"].mean(dtype=torch.float32).item()),
        "decoded_min": float(values["decoded"].min().item()),
        "decoded_max": float(values["decoded"].max().item()),
        "decoded_mean": float(values["decoded"].mean(dtype=torch.float32).item()),
    }
    if reference_decoded is not None:
        metrics["reference_decoded_finite_fraction"] = 1.0
    return metrics


__all__ = [
    "H3InferenceAdapter",
    "camera_fingerprint",
    "decode_h3_latents",
    "package_generated",
]
