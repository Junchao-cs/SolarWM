from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import pytest

import solarwm.backends.minimax_h3.inference as h3_inference
from solarwm.backends.minimax_h3.inference import decode_h3_audio_latents


def test_audio_decode_and_wav_contract() -> None:
    torch = pytest.importorskip("torch")

    class AudioVAE:
        config = SimpleNamespace(
            latents_mean=[0.0] * 32,
            latents_std=[1.0] * 32,
            sampling_rate=32_000,
        )
        hop_length = 800

        def decode(self, latents, *, return_dict):
            assert tuple(latents.shape) == (2, 32, 263)
            assert return_dict is False
            return (torch.full((2, 1, 263 * self.hop_length), 0.25),)

    latents = torch.zeros((2, 32, 263))
    audio = decode_h3_audio_latents(AudioVAE(), latents, device=torch.device("cpu"))
    assert tuple(audio.shape) == (1, 2, 210_400)
    payload = h3_inference._wav_bytes(audio, sampling_rate=32_000)
    with wave.open(io.BytesIO(payload), "rb") as source:
        assert source.getnchannels() == 2
        assert source.getframerate() == 32_000
        assert source.getnframes() == 210_400


def test_audio_packaging_publishes_wav_and_muxed_video(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        h3_inference,
        "decode_h3_latents",
        lambda vae, latents, *, device: torch.zeros((1, 3, 2, 4, 4)),
    )
    monkeypatch.setattr(
        h3_inference,
        "decode_h3_audio_latents",
        lambda vae, latents, *, device: torch.zeros((1, 2, 8)),
    )
    monkeypatch.setattr(h3_inference, "_mp4_bytes", lambda decoded: b"video")
    monkeypatch.setattr(h3_inference, "_wav_bytes", lambda audio, *, sampling_rate: b"wav")
    monkeypatch.setattr(h3_inference, "_mux_mp4_audio", lambda video, audio: b"muxed")
    audio_vae = SimpleNamespace(config=SimpleNamespace(sampling_rate=32_000))

    packaged = h3_inference.package_generated(
        torch.zeros((1, 3, 2, 4, 4)),
        video_vae=object(),
        device=torch.device("cpu"),
        weights_id="weights",
        num_inference_steps=2,
        audio_latents=torch.zeros((2, 32, 1)),
        audio_vae=audio_vae,
    )

    assert packaged.artifacts["generated.mp4"] == b"video"
    assert packaged.artifacts["generated.wav"] == b"wav"
    assert packaged.artifacts["generated-with-audio.mp4"] == b"muxed"
