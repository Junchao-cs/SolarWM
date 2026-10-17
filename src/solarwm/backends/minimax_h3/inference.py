"""Shared-engine inference adapter for MiniMax-H3 Stage0.5."""

from __future__ import annotations

import hashlib
import io
import subprocess
import tempfile
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solarwm.inference import GeneratedSample, InferenceCase, encode_compare_mp4
from solarwm.runtime.media import resolve_ffmpeg_executable

from .artifacts import H3ArtifactBatch
from .stage0p5 import H3Stage0p5Core


def camera_fingerprint(batch: H3ArtifactBatch) -> str:
    digest = hashlib.blake2s()
    digest.update(batch.camera_viewmats.detach().cpu().numpy().tobytes())
    digest.update(batch.camera_K.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def decoded_pixel_frames(latent_frames: int) -> int:
    """The VisualVAE writes ``17n+5`` pixel frames for the ``5n+2`` latent frames it was given."""
    if latent_frames < 2 or (latent_frames - 2) % 5:
        raise RuntimeError(f"H3 latents hold {latent_frames} frames, which is not 5n+2")
    return 17 * ((latent_frames - 2) // 5) + 5


def decode_h3_latents(video_vae: Any, latents: Any, *, device: Any) -> Any:
    """Run official latent de-normalization and the 5n+2 -> 17n+5 VisualVAE decode."""

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
    expected = (
        1,
        3,
        decoded_pixel_frames(int(latents.shape[-3])),
        int(latents.shape[-2]) * 16,
        int(latents.shape[-1]) * 16,
    )
    if tuple(decoded.shape) != expected:
        raise RuntimeError(
            f"H3 VisualVAE decoded {tuple(decoded.shape)}, expected {list(expected)}"
        )
    return decoded


def decode_h3_audio_latents(audio_vae: Any, latents: Any, *, device: Any) -> Any:
    """Denormalize `[2,32,263]` latents and decode a 32 kHz stereo waveform."""

    import torch

    if tuple(latents.shape) != (2, 32, 263):
        raise RuntimeError(f"H3 AudioVAE received {tuple(latents.shape)}, expected [2,32,263]")
    mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
    decoder_input = latents.to(device=device, dtype=torch.float32) * std + mean
    with torch.no_grad():
        decoded = audio_vae.decode(decoder_input, return_dict=False)[0]
    waveform = decoded.float().permute(1, 0, 2).contiguous()
    expected_samples = 263 * int(audio_vae.hop_length)
    if tuple(waveform.shape) != (1, 2, expected_samples):
        raise RuntimeError(
            f"H3 AudioVAE decoded {tuple(waveform.shape)}, expected [1,2,{expected_samples}]"
        )
    if not bool(torch.isfinite(waveform).all().item()):
        raise FloatingPointError("H3 AudioVAE decoded waveform contains NaN or Inf")
    return waveform


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


def _wav_bytes(audio: Any, *, sampling_rate: int) -> bytes:
    """Encode `[1,2,samples]` float audio as PCM16 WAV."""

    import torch

    if audio.ndim != 3 or tuple(audio.shape[:2]) != (1, 2):
        raise RuntimeError(f"H3 decoded audio is not [1,2,samples]: {tuple(audio.shape)}")
    interleaved = (
        audio[0]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(-1, 1)
        .transpose(0, 1)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .contiguous()
        .numpy()
    )
    payload = io.BytesIO()
    with wave.open(payload, "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(int(sampling_rate))
        output.writeframes(interleaved.tobytes())
    return payload.getvalue()


def _mux_mp4_audio(video: bytes, audio: bytes, *, duration_seconds: float = 158 / 24) -> bytes:
    """Copy H.264 video and add the generated soundtrack as AAC."""

    executable = resolve_ffmpeg_executable()
    if not 0 < duration_seconds < float("inf"):
        raise ValueError("H3 audio mux duration must be finite and positive")
    with tempfile.TemporaryDirectory(prefix="solarwm-h3-audio-") as directory:
        root = Path(directory)
        video_path = root / "video.mp4"
        audio_path = root / "audio.wav"
        output_path = root / "generated-with-audio.mp4"
        video_path.write_bytes(video)
        audio_path.write_bytes(audio)
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            f"apad=whole_dur={duration_seconds},atrim=duration={duration_seconds}",
            "-t",
            str(duration_seconds),
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, check=False, timeout=120)
        if result.returncode or not output_path.is_file():
            detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg failed to mux H3 audio: {detail}")
        return output_path.read_bytes()


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
    audio_latents: Any | None = None,
    audio_vae: Any | None = None,
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
    if (audio_latents is None) != (audio_vae is None):
        raise RuntimeError("H3 audio packaging requires both audio_latents and audio_vae")
    audio = None
    sampling_rate = None
    if audio_latents is not None:
        audio = decode_h3_audio_latents(audio_vae, audio_latents, device=device)
        sampling_rate = int(audio_vae.config.sampling_rate)
    metrics = _finite_generation_metrics(
        latents=latents,
        decoded=decoded,
        reference_decoded=ground_truth,
        audio_latents=audio_latents,
        audio=audio,
    )
    video_bytes = _mp4_bytes(decoded)
    latent_tensors = {
        "latents": latents.detach().to("cpu", dtype=torch.bfloat16),
    }
    if audio_latents is not None:
        latent_tensors["audio_latents"] = audio_latents.detach().to("cpu", dtype=torch.bfloat16)
    artifacts = {
        "generated.safetensors": save(latent_tensors),
        "generated.mp4": video_bytes,
    }
    if audio is not None and sampling_rate is not None:
        wav_bytes = _wav_bytes(audio, sampling_rate=sampling_rate)
        artifacts["generated.wav"] = wav_bytes
        artifacts["generated-with-audio.mp4"] = _mux_mp4_audio(video_bytes, wav_bytes)
    if ground_truth is not None:
        compare_bytes = encode_compare_mp4(
            ground_truth,
            decoded,
            fps=24,
            layout="bcthw",
            value_range="zero_one",
        )
        if audio is not None and sampling_rate is not None:
            compare_bytes = _mux_mp4_audio(compare_bytes, wav_bytes)
        artifacts["compare.mp4"] = compare_bytes
    return GeneratedSample(
        artifacts=artifacts,
        shape=tuple(int(value) for value in decoded.shape),
        dtype="float32",
        metrics=metrics,
        provenance={
            "weights_id": weights_id,
            "preencode_version": "h3.158f.v1",
            "sampler": "shifted-euler-data-ward",
            "solver": "shifted-euler-data-ward",
            "num_inference_steps": int(num_inference_steps),
            "num_sigma_points": int(num_inference_steps),
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "audio_generated": audio is not None,
            **(
                {
                    "audio_sampling_rate": sampling_rate,
                    "audio_latent_shape": [2, 32, 263],
                    "audio_codec": "pcm_s16le",
                    "muxed_audio_codec": "aac",
                }
                if audio is not None
                else {}
            ),
        },
    )


def _finite_generation_metrics(
    *,
    latents: Any,
    decoded: Any,
    reference_decoded: Any | None,
    audio_latents: Any | None = None,
    audio: Any | None = None,
) -> dict[str, float]:
    """Fail before publication unless generated latent/decoded tensors are finite."""

    import torch

    values = {
        "latent": latents.detach(),
        "decoded": decoded.detach(),
    }
    if reference_decoded is not None:
        values["reference_decoded"] = reference_decoded.detach()
    if audio_latents is not None:
        values["audio_latent"] = audio_latents.detach()
    if audio is not None:
        values["audio"] = audio.detach()
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
    if audio_latents is not None and audio is not None:
        audio_float = audio.detach().float()
        metrics.update(
            {
                "audio_latent_finite_fraction": 1.0,
                "audio_finite_fraction": 1.0,
                "audio_peak": float(audio_float.abs().max().item()),
                "audio_mean": float(audio_float.mean().item()),
                "audio_rms": float(audio_float.square().mean().sqrt().item()),
            }
        )
    return metrics


__all__ = [
    "H3InferenceAdapter",
    "camera_fingerprint",
    "decode_h3_audio_latents",
    "decode_h3_latents",
    "package_generated",
]
