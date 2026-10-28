"""Official DiffVAE and Gemma4 owners for LTX inference/preencoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from solar.errors import BackendContractError

from .artifact import TensorArtifact
from .checkpoint import StrictCodecLoadReceipt
from .geometry import STABLE_GEOMETRY


def _qualified_class(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _bf16_artifact(value: torch.Tensor, shape: tuple[int, ...]) -> TensorArtifact:
    tensor = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(tensor.shape) != shape:
        raise BackendContractError(f"official LTX tensor shape {tuple(tensor.shape)} != {shape}")
    if not bool(torch.isfinite(tensor).all()):
        raise BackendContractError("official LTX tensor contains NaN or Inf")
    payload = tensor.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    return TensorArtifact("BF16", shape, payload)


def _i64_artifact(value: torch.Tensor, shape: tuple[int, ...]) -> TensorArtifact:
    tensor = value.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if tuple(tensor.shape) != shape:
        raise BackendContractError(f"official LTX tensor shape {tuple(tensor.shape)} != {shape}")
    return TensorArtifact("I64", shape, tensor.numpy().astype("<i8", copy=False).tobytes())


class OfficialDiffVAEDecoder:
    """Chunked-eager owner for the diffusion video VAE."""

    def __init__(self, checkpoint_path: str | Path, *, device: torch.device) -> None:
        from ltx_core.model.video_vae import is_diffusion_video_vae
        from ltx_core.model.video_vae.transformer import DiffVAEMode
        from ltx_pipelines.utils.blocks import VideoDecoder

        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        if not is_diffusion_video_vae(self.checkpoint_path):
            raise BackendContractError("LTX requires the official diffusion Video VAE")
        self.decoder = VideoDecoder(
            self.checkpoint_path,
            dtype=torch.bfloat16,
            device=device,
            diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
        )

    @property
    def implementation_class(self) -> str:
        return _qualified_class(self.decoder)

    def decode(self, latent: torch.Tensor, *, seed: int) -> torch.Tensor:
        from ltx_core.model.video_vae import AUTO_TILING
        from ltx_core.model.video_vae.transformer import DiffVAEMode
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.helpers import (
            ensure_tiling_config,
            tiling_scale_factors_for_vae,
        )

        if tuple(latent.shape) != (1, 128, 20, 16, 24):
            raise BackendContractError("DiffVAE input must be [1,128,20,16,24]")
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        scale_factors = tiling_scale_factors_for_vae(self.checkpoint_path)
        tiling = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1,
                frames=STABLE_GEOMETRY.pixel_frames,
                height=STABLE_GEOMETRY.height,
                width=STABLE_GEOMETRY.width,
                fps=24.0,
            ),
            diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
            device=self.device,
        )
        with torch.inference_mode():
            chunks = tuple(
                self.decoder(
                    latent,
                    tiling,
                    generator=generator,
                    dtype=torch.bfloat16,
                )
            )
        if not chunks or any(
            not isinstance(chunk, torch.Tensor) or chunk.ndim != 4 or chunk.shape[-1] != 3
            for chunk in chunks
        ):
            raise BackendContractError("official DiffVAE returned invalid FHWC chunks")
        frames = torch.cat(chunks, dim=0)
        if tuple(frames.shape) != (153, 512, 768, 3):
            raise BackendContractError("official DiffVAE output geometry differs")
        if not bool(torch.isfinite(frames).all()) or not bool(
            ((frames >= 0) & (frames <= 1)).all()
        ):
            raise BackendContractError("official DiffVAE output is not finite [0,1]")
        return frames.permute(3, 0, 1, 2).unsqueeze(0).float().contiguous()

    def close(self) -> None:
        del self.decoder
        torch.cuda.empty_cache()


class OfficialOnlineCodec:
    """Official direct VAE encoder and Gemma4 preconnector feature extractor."""

    def __init__(
        self,
        *,
        transformer_path: str | Path,
        video_vae_path: str | Path,
        gemma4_path: str | Path,
        device: torch.device,
        identity: str,
    ) -> None:
        from ltx_trainer import model_loader

        self.device = device
        self.identity = identity
        self.video_vae = (
            model_loader.load_video_vae_encoder(
                video_vae_path,
                device=device,
                dtype=torch.bfloat16,
            )
            .eval()
            .requires_grad_(False)
        )
        self.text_encoder = (
            model_loader.load_text_encoder(
                gemma4_path,
                device=device,
                dtype=torch.bfloat16,
                load_in_8bit=False,
            )
            .eval()
            .requires_grad_(False)
        )
        paths = model_loader.embedding_weight_paths(transformer_path, gemma4_path)
        processor = (
            model_loader.load_embeddings_processor(
                paths,
                gemma_model_path=gemma4_path,
                device=device,
                dtype=torch.bfloat16,
            )
            .eval()
            .requires_grad_(False)
        )
        self.feature_extractor = processor.feature_extractor
        del processor

    @property
    def video_vae_class(self) -> str:
        return _qualified_class(self.video_vae)

    @property
    def feature_extractor_class(self) -> str:
        return _qualified_class(self.feature_extractor)

    def encode_video(
        self,
        frames: object,
    ) -> tuple[TensorArtifact, TensorArtifact]:
        value = torch.as_tensor(frames)
        if tuple(value.shape) == (153, 512, 768, 3):
            value = value.permute(0, 3, 1, 2)
        if tuple(value.shape) != (153, 3, 512, 768):
            raise BackendContractError("official LTX codec frames must be [153,3,512,768]")
        value = value.to(dtype=torch.float32)
        if value.max() > 1:
            value = value / 255.0
        if not bool(torch.isfinite(value).all()) or not bool(((value >= 0) & (value <= 1)).all()):
            raise BackendContractError("official LTX codec frames must be finite [0,1]")
        video = (
            value.permute(1, 0, 2, 3)
            .unsqueeze(0)
            .contiguous()
            .mul(2.0)
            .sub_(1.0)
            .to(device=self.device, dtype=torch.bfloat16)
        )
        with torch.inference_mode():
            latent = self.video_vae(video)
        if tuple(latent.shape) != (1, 128, 20, 16, 24):
            raise BackendContractError("official LTX VAE returned the wrong latent geometry")
        encoded = _bf16_artifact(latent[0], (128, 20, 16, 24))
        first = _bf16_artifact(latent[0, :, :1], (128, 1, 16, 24))
        return encoded, first

    def encode_prompt(self, caption: str) -> tuple[TensorArtifact, TensorArtifact]:
        if not isinstance(caption, str) or not caption.strip():
            raise BackendContractError("official LTX codec requires a nonempty caption")
        with torch.inference_mode():
            encoded = self.text_encoder.encode([caption])
            if not isinstance(encoded, (list, tuple)) or len(encoded) != 1:
                raise BackendContractError("official Gemma must return one encoded prompt")
            hidden, mask = encoded[0]
            video_prompt, _audio_prompt = self.feature_extractor(hidden, mask, "left")
        prompt = video_prompt[0]
        prompt_mask = mask[0]
        if bool((prompt_mask[1:] < prompt_mask[:-1]).any()):
            raise BackendContractError("official Gemma caption mask is not left padded")
        return (
            _bf16_artifact(prompt, (1024, 4096)),
            _i64_artifact(prompt_mask, (1024,)),
        )

    def close(self) -> None:
        del self.video_vae
        del self.text_encoder
        del self.feature_extractor
        torch.cuda.empty_cache()


def codec_receipt(
    *,
    provider_identity: str,
    video_vae_class: str,
    gemma_feature_extractor_class: str = "",
    video_vae_operation: str = "diffvae_decode",
    video_vae_encoder_class: str = "",
) -> StrictCodecLoadReceipt:
    return StrictCodecLoadReceipt(
        provider_identity=provider_identity,
        video_vae_class=video_vae_class,
        diffvae_mode=("direct" if video_vae_operation == "direct_encode" else "chunked_eager"),
        gemma_feature_extractor_class=gemma_feature_extractor_class,
        caption_cache_stage=(
            "gemma4_feature_extractor_preconnector" if gemma_feature_extractor_class else ""
        ),
        video_vae_operation=video_vae_operation,
        video_vae_encoder_class=video_vae_encoder_class,
    )


__all__ = [
    "OfficialDiffVAEDecoder",
    "OfficialOnlineCodec",
    "codec_receipt",
]
