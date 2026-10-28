from __future__ import annotations

import pytest

from solar.backends.wan22.runtime.codec import Wan5BOnlineCodec


def test_stage2_codec_encodes_only_first_frame_and_preserves_rollout_shape() -> None:
    torch = pytest.importorskip("torch")

    class FakeVAE:
        def __init__(self) -> None:
            self.input_shape: tuple[int, ...] | None = None

        def encode(self, pixels: object) -> object:
            self.input_shape = tuple(pixels.shape)
            return torch.full((2, 1, 48, 2, 3), 7.0, dtype=torch.float32)

    class FakeTextEncoder:
        def __call__(self, captions: object) -> dict[str, object]:
            return {"prompt_embeds": torch.ones((len(captions), 4, 5), dtype=torch.float32)}

    vae = FakeVAE()
    codec = Wan5BOnlineCodec(
        vae,
        FakeTextEncoder(),
        pixel_frames=81,
        height=32,
        width=48,
    )
    camera = {
        "viewmats": torch.eye(4).repeat(2, 1, 1, 1),
        "K": torch.eye(3).repeat(2, 1, 1, 1),
    }

    encoded = codec.encode_stage2_batch(
        sample_ids=["a", "b"],
        first_pixels=torch.zeros((2, 1, 3, 32, 48)),
        captions=["one", "two"],
        camera=camera,
    )

    assert vae.input_shape == (2, 3, 1, 32, 48)
    assert encoded["latents"].shape == (2, 21, 48, 2, 3)
    assert encoded["latents"].dtype == torch.bfloat16
    assert encoded["latents"].stride() == (6048, 6, 126, 3, 1)
    assert torch.equal(
        encoded["latents"][:, :1],
        torch.full((2, 1, 48, 2, 3), 7.0, dtype=torch.bfloat16),
    )
    assert torch.count_nonzero(encoded["latents"][:, 1:]) == 0

    legacy_layout = torch.zeros((2, 48, 21, 2, 3), dtype=torch.bfloat16).permute(0, 2, 1, 3, 4)
    torch.manual_seed(91)
    legacy_noise = torch.randn_like(legacy_layout)
    torch.manual_seed(91)
    optimized_noise = torch.randn_like(encoded["latents"])
    assert torch.equal(optimized_noise, legacy_noise)
