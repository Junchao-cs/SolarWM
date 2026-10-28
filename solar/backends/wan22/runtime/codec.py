"""One Wan5B online codec shared by raw training and offline preencoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from solar.errors import DataContractError
from solar.preencode.contracts import (
    EncoderContract,
    TensorSpec,
    validate_encoded_tensors,
)


class Wan5BOnlineCodec:
    """Encode pixels/text while preserving already-canonical camera tensors."""

    def __init__(
        self,
        vae: Any,
        text_encoder: Any,
        *,
        pixel_frames: int,
        height: int = 480,
        width: int = 864,
        frame_sequence_length: int = 405,
    ) -> None:
        latent_frames = 1 + (int(pixel_frames) - 1) // 4
        self.vae = vae
        self.text_encoder = text_encoder
        self.contract = EncoderContract(
            schema="solarwm.encoder.v1",
            family="wan22_ti2v_5b",
            format_version="solarwm.wan22-ti2v-5b.online.v1",
            pixel_frames=int(pixel_frames),
            latent_frames=latent_frames,
            height=int(height),
            width=int(width),
            camera_convention="first-frame-relative-w2c-fp32",
            tensors=(
                TensorSpec(
                    "latents",
                    (latent_frames, 48, int(height) // 16, int(width) // 16),
                    "bfloat16",
                ),
                TensorSpec("prompt_embeds", (512, 4096), "bfloat16"),
                TensorSpec(
                    "camera_viewmats",
                    (latent_frames * int(frame_sequence_length), 4, 4),
                    "float32",
                ),
                TensorSpec(
                    "camera_K",
                    (latent_frames * int(frame_sequence_length), 3, 3),
                    "float32",
                ),
            ),
            extras={
                "vae_temporal_alignment": "0,1,5,9,...",
                "caption_source": "frozen",
                "camera_source": "preserve",
            },
        )

    def encode(
        self,
        *,
        sample_id: str,
        pixels: Any,
        caption: str,
        camera: Any,
        seed: int,
    ) -> Mapping[str, Any]:
        del seed
        if not sample_id:
            raise DataContractError("Wan online codec requires a sample_id")
        if not isinstance(camera, Mapping) or set(camera) != {"viewmats", "K"}:
            raise DataContractError("Wan online codec camera must contain viewmats and K")
        try:
            import torch
        except ImportError as exc:
            raise DataContractError("Wan online codec requires torch") from exc
        if tuple(pixels.shape) != (
            self.contract.pixel_frames,
            3,
            self.contract.height,
            self.contract.width,
        ):
            raise DataContractError(
                f"Wan pixels have shape {tuple(pixels.shape)}, expected "
                f"[{self.contract.pixel_frames},3,{self.contract.height},{self.contract.width}]"
            )
        with torch.no_grad():
            latents = self.vae.encode(pixels.unsqueeze(0).permute(0, 2, 1, 3, 4).float())[0].to(
                torch.bfloat16
            )
            prompt = self.text_encoder([caption])["prompt_embeds"][0].to(torch.bfloat16)
        encoded = {
            "latents": latents,
            "prompt_embeds": prompt,
            "camera_viewmats": camera["viewmats"].to(torch.float32),
            "camera_K": camera["K"].to(torch.float32),
        }
        validate_encoded_tensors(encoded, self.contract)
        return encoded

    def encode_batch(
        self,
        *,
        sample_ids: Sequence[str],
        pixels: Any,
        captions: Sequence[str],
        camera: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Batched execution with the same per-sample tensor contract."""

        try:
            import torch
        except ImportError as exc:
            raise DataContractError("Wan online codec requires torch") from exc
        batch = len(sample_ids)
        if len(captions) != batch or int(pixels.shape[0]) != batch:
            raise DataContractError("Wan online batch identity and tensor sizes differ")
        with torch.no_grad():
            latents = self.vae.encode(pixels.permute(0, 2, 1, 3, 4).contiguous().float()).to(
                torch.bfloat16
            )
            prompts = self.text_encoder(captions)["prompt_embeds"].to(torch.bfloat16)
        return {
            "latents": latents,
            "prompt_embeds": prompts,
            "camera": {
                "viewmats": camera["viewmats"].to(torch.float32),
                "K": camera["K"].to(torch.float32),
            },
        }

    def encode_stage2_batch(
        self,
        *,
        sample_ids: Sequence[str],
        first_pixels: Any,
        captions: Sequence[str],
        camera: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Encode the real first-frame anchor while preserving rollout geometry."""

        try:
            import torch
        except ImportError as exc:
            raise DataContractError("Wan online codec requires torch") from exc
        batch = len(sample_ids)
        if len(captions) != batch or int(first_pixels.shape[0]) != batch:
            raise DataContractError("Wan online batch identity and tensor sizes differ")
        expected = (batch, 1, 3, self.contract.height, self.contract.width)
        if tuple(first_pixels.shape) != expected:
            raise DataContractError(
                f"Wan Stage2 first-frame pixels have shape {tuple(first_pixels.shape)}, "
                f"expected {expected}"
            )
        with torch.no_grad():
            first_latents = self.vae.encode(
                first_pixels.permute(0, 2, 1, 3, 4).contiguous().float()
            ).to(torch.bfloat16)
            if int(first_latents.shape[1]) != 1:
                raise DataContractError(
                    "Wan Stage2 first-frame VAE encode must produce one latent frame"
                )
            # Wan5BVAE.encode returns [B,T,C,H,W] as a view of a contiguous
            # [B,C,T,H,W] allocation. Preserve that layout so randn_like uses
            # the same index-to-random-value mapping as full-clip encoding.
            latents = first_latents.new_zeros(
                (
                    batch,
                    first_latents.shape[2],
                    self.contract.latent_frames,
                    *first_latents.shape[3:],
                )
            ).permute(0, 2, 1, 3, 4)
            latents[:, :1].copy_(first_latents)
            prompts = self.text_encoder(captions)["prompt_embeds"].to(torch.bfloat16)
        return {
            "latents": latents,
            "prompt_embeds": prompts,
            "camera": {
                "viewmats": camera["viewmats"].to(torch.float32),
                "K": camera["K"].to(torch.float32),
            },
        }


def pack_first_frame_mask(
    *,
    batch_size: int,
    pixel_frames: int,
    latent_height: int,
    latent_width: int,
    device: Any,
    dtype: Any,
) -> Any:
    """Pack the official pixel-frame condition into four latent channels."""

    import torch

    if pixel_frames < 1 or (pixel_frames - 1) % 4:
        raise DataContractError("Wan I2V pixel frames must have form 4n+1")
    mask = torch.ones(
        int(batch_size),
        1,
        int(pixel_frames),
        int(latent_height),
        int(latent_width),
        device=device,
        dtype=dtype,
    )
    mask[:, :, 1:] = 0
    mask = torch.cat(
        [mask[:, :, :1].repeat_interleave(4, dim=2), mask[:, :, 1:]],
        dim=2,
    )
    latent_frames = 1 + (int(pixel_frames) - 1) // 4
    return (
        mask.view(
            int(batch_size),
            1,
            latent_frames,
            4,
            int(latent_height),
            int(latent_width),
        )
        .squeeze(1)
        .contiguous()
    )


def combine_i2v_condition(condition_latents: Any, *, pixel_frames: int) -> Any:
    """Concatenate the official four-channel mask and 16-channel VAE input."""

    import torch

    if condition_latents.ndim != 5:
        raise DataContractError("A14B condition latents must be [B,T,C,H,W]")
    batch, latent_frames, channels, height, width = condition_latents.shape
    expected_frames = 1 + (int(pixel_frames) - 1) // 4
    if (latent_frames, channels) != (expected_frames, 16):
        raise DataContractError(
            "A14B condition latent geometry differs from the 16-channel contract"
        )
    mask = pack_first_frame_mask(
        batch_size=batch,
        pixel_frames=int(pixel_frames),
        latent_height=height,
        latent_width=width,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    return torch.cat([mask, condition_latents], dim=2)


def build_official_i2v_y(pixels: Any, vae: Any) -> Any:
    """Encode ``[first image, zeros...]`` and prepend the packed mask."""

    import torch

    if pixels.ndim != 5 or int(pixels.shape[2]) != 3:
        raise DataContractError("A14B pixels must be [B,T,3,H,W]")
    condition_pixels = torch.zeros_like(pixels)
    condition_pixels[:, 0] = pixels[:, 0]
    condition_latents = vae.encode(condition_pixels.permute(0, 2, 1, 3, 4).contiguous().float())
    return combine_i2v_condition(
        condition_latents.to(torch.bfloat16),
        pixel_frames=int(pixels.shape[1]),
    )


class WanA14BOnlineCodec:
    """Official Wan2.2 I2V-A14B VAE, text, and 20-channel condition codec."""

    def __init__(
        self,
        vae: Any,
        text_encoder: Any,
        *,
        pixel_frames: int,
        height: int = 480,
        width: int = 832,
        frame_sequence_length: int = 1560,
    ) -> None:
        latent_frames = 1 + (int(pixel_frames) - 1) // 4
        self.vae = vae
        self.text_encoder = text_encoder
        self.contract = EncoderContract(
            schema="solarwm.encoder.v1",
            family="wan22_i2v_a14b",
            format_version="solarwm.wan22-i2v-a14b.online.v1",
            pixel_frames=int(pixel_frames),
            latent_frames=latent_frames,
            height=int(height),
            width=int(width),
            camera_convention="first-frame-relative-w2c-fp32",
            tensors=(
                TensorSpec(
                    "latents",
                    (latent_frames, 16, int(height) // 8, int(width) // 8),
                    "bfloat16",
                ),
                TensorSpec(
                    "i2v_y",
                    (latent_frames, 20, int(height) // 8, int(width) // 8),
                    "bfloat16",
                ),
                TensorSpec("prompt_embeds", (512, 4096), "bfloat16"),
                TensorSpec(
                    "camera_viewmats",
                    (latent_frames * int(frame_sequence_length), 4, 4),
                    "float32",
                ),
                TensorSpec(
                    "camera_K",
                    (latent_frames * int(frame_sequence_length), 3, 3),
                    "float32",
                ),
            ),
            extras={
                "vae": "Wan2.1_VAE.pth",
                "condition": "official-first-frame-mask-plus-latent",
                "caption_source": "frozen",
                "camera_source": "preserve",
            },
        )

    def encode_batch(
        self,
        *,
        sample_ids: Sequence[str],
        pixels: Any,
        captions: Sequence[str],
        camera: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        import torch

        batch = len(sample_ids)
        if len(captions) != batch or int(pixels.shape[0]) != batch:
            raise DataContractError("A14B online batch identity and tensor sizes differ")
        with torch.no_grad():
            latents = self.vae.encode(pixels.permute(0, 2, 1, 3, 4).contiguous().float()).to(
                torch.bfloat16
            )
            i2v_y = build_official_i2v_y(pixels, self.vae)
            prompts = self.text_encoder(captions)["prompt_embeds"].to(torch.bfloat16)
        return {
            "latents": latents,
            "i2v_y": i2v_y,
            "prompt_embeds": prompts,
            "camera": {
                "viewmats": camera["viewmats"].to(torch.float32),
                "K": camera["K"].to(torch.float32),
            },
        }


__all__ = [
    "Wan5BOnlineCodec",
    "WanA14BOnlineCodec",
    "build_official_i2v_y",
    "combine_i2v_condition",
    "pack_first_frame_mask",
]
