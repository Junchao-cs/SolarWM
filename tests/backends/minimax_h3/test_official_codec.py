from __future__ import annotations

from types import SimpleNamespace

import torch

from solar.backends.minimax_h3.official_codec import OfficialH3Codec


class _Posterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value

    def sample(self, *, generator: torch.Generator) -> torch.Tensor:
        del generator
        return self.value


class _VAE(torch.nn.Module):
    def __init__(self, *, channels: int, audio: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.config = SimpleNamespace(
            latents_mean=[0.0] * channels,
            latents_std=[1.0] * channels,
        )
        self.audio = audio
        self.input_dtype: torch.dtype | None = None
        self.attention_backend: str | None = None
        self.moves: list[str] = []

    def to(self, device):
        self.moves.append(str(device))
        return super().to(device)

    def set_attention_backend(self, name: str) -> None:
        self.attention_backend = name

    def encode(self, value: torch.Tensor, *, return_dict: bool):
        assert return_dict is False
        self.input_dtype = value.dtype
        if self.audio:
            result = torch.zeros((2, 32, 263), dtype=value.dtype, device=value.device)
        else:
            result = torch.zeros((1, 16, 1, 1, 1), dtype=value.dtype, device=value.device)
        return (_Posterior(result),)


def _codec(*, video_vae: _VAE, audio_vae: _VAE) -> OfficialH3Codec:
    text_encoder = _VAE(channels=1, audio=False)
    return OfficialH3Codec(
        text_encoder=text_encoder,
        tokenizer=object(),
        processor=object(),
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=torch.device("cpu"),
        encoder_identity="test-codec",
    )


def test_official_codec_uses_native_attention_for_fp32_vae_encoders() -> None:
    video_vae = _VAE(channels=16, audio=False)
    audio_vae = _VAE(channels=32, audio=True)
    codec = _codec(video_vae=video_vae, audio_vae=audio_vae)

    codec._video_latents(torch.zeros((1, 3, 1, 1, 1), dtype=torch.uint8), seed=42)
    codec._silence_latents()

    assert video_vae.input_dtype == torch.float32
    assert audio_vae.input_dtype == torch.float32
    assert video_vae.attention_backend == "native"
    assert audio_vae.attention_backend == "native"


def test_online_codec_offloads_conditioners_between_encode_and_training() -> None:
    video_vae = _VAE(channels=16, audio=False)
    audio_vae = _VAE(channels=32, audio=True)
    codec = _codec(video_vae=video_vae, audio_vae=audio_vae)

    codec.offload_encoders()
    codec.activate_encoders()

    assert codec.text_encoder.moves == ["cpu", "cpu"]
    assert video_vae.moves == ["cpu", "cpu"]
    assert audio_vae.moves == ["cpu"]
