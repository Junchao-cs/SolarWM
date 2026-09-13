from __future__ import annotations

import pytest

from solarwm.backends.wan22.runtime.components import Wan5BVAE


def test_wan5b_streaming_decode_keeps_one_cache_across_temporal_tiles() -> None:
    torch = pytest.importorskip("torch")

    class _Module:
        def __init__(self) -> None:
            self.clear_calls = 0
            self.chunk_sizes: list[int] = []
            self.first = True

        def clear_cache(self) -> None:
            self.clear_calls += 1
            self.first = True

        def cached_decode(self, value: object, _scale: object) -> object:
            latent_frames = int(value.shape[2])
            self.chunk_sizes.append(latent_frames)
            pixel_frames = 1 + 4 * (latent_frames - 1) if self.first else 4 * latent_frames
            self.first = False
            return torch.zeros((1, 3, pixel_frames, 1, 1), dtype=torch.bfloat16)

    module = _Module()
    vae = Wan5BVAE.__new__(Wan5BVAE)
    vae.module = module
    vae._mean = torch.zeros(48, dtype=torch.float32)
    vae._std = torch.ones(48, dtype=torch.float32)
    latents = torch.zeros((1, 240, 48, 1, 1), dtype=torch.bfloat16)

    decoded = vae.decode_streaming(latents, chunk_latent_frames=60)

    assert module.chunk_sizes == [60, 60, 60, 60]
    assert module.clear_calls == 2
    assert decoded.device.type == "cpu"
    assert decoded.shape == (1, 957, 3, 1, 1)


def test_wan5b_concatenating_stream_decode_preserves_batch_support() -> None:
    torch = pytest.importorskip("torch")

    class _Module:
        def __init__(self) -> None:
            self.first = True

        def clear_cache(self) -> None:
            self.first = True

        def cached_decode(self, value: object, _scale: object) -> object:
            latent_frames = int(value.shape[2])
            pixel_frames = 1 + 4 * (latent_frames - 1) if self.first else 4 * latent_frames
            self.first = False
            return torch.zeros((1, 3, pixel_frames, 1, 1), dtype=torch.bfloat16)

    vae = Wan5BVAE.__new__(Wan5BVAE)
    vae.module = _Module()
    vae._mean = torch.zeros(48, dtype=torch.float32)
    vae._std = torch.ones(48, dtype=torch.float32)
    latents = torch.zeros((2, 6, 48, 1, 1), dtype=torch.bfloat16)

    decoded = vae.decode_streaming(latents, chunk_latent_frames=3)

    assert decoded.shape == (2, 21, 3, 1, 1)
