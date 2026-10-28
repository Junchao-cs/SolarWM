from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from solar.backends.wan22 import create_backend
from solar.config import load_config
from solar.errors import BackendContractError

ROOT = Path(__file__).resolve().parents[3]
A14B_81F_CONFIG = ROOT / "configs/solar_wm/wan22_i2v_a14b/train_stage0p5_fm_81f.yaml"
A14B_153F_CONFIG = ROOT / "configs/solar_wm/wan22_i2v_a14b/train_stage0p5_fm_153f.yaml"


def test_official_a14b_online_codec_builds_mask_and_condition_latents() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.codec import WanA14BOnlineCodec

    class FakeVAE:
        def encode(self, value: object) -> object:
            batch, _, pixels, height, width = value.shape
            latent_frames = 1 + (pixels - 1) // 4
            return torch.zeros(
                (batch, latent_frames, 16, height // 8, width // 8),
                dtype=torch.float32,
            )

    class FakeText:
        def __call__(self, captions: object) -> dict[str, object]:
            return {"prompt_embeds": torch.zeros((len(captions), 512, 4096), dtype=torch.float32)}

    codec = WanA14BOnlineCodec(
        FakeVAE(),
        FakeText(),
        pixel_frames=5,
        height=8,
        width=8,
        frame_sequence_length=1,
    )
    encoded = codec.encode_batch(
        sample_ids=("sample",),
        pixels=torch.ones((1, 5, 3, 8, 8), dtype=torch.bfloat16),
        captions=("caption",),
        camera={
            "viewmats": torch.eye(4).reshape(1, 1, 4, 4).expand(1, 2, 4, 4),
            "K": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, 3, 3),
        },
    )
    assert tuple(encoded["latents"].shape) == (1, 2, 16, 1, 1)
    assert tuple(encoded["i2v_y"].shape) == (1, 2, 20, 1, 1)
    assert torch.equal(
        encoded["i2v_y"][0, 0, :4, 0, 0],
        torch.ones(4, dtype=torch.bfloat16),
    )
    assert torch.equal(
        encoded["i2v_y"][0, 1, :4, 0, 0],
        torch.zeros(4, dtype=torch.bfloat16),
    )


def test_a14b_vae_uses_post_cast_reciprocal_scale() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.components import WanA14BVAE

    vae = object.__new__(WanA14BVAE)
    vae._mean = torch.tensor(WanA14BVAE.mean, dtype=torch.float32)
    vae._std = torch.tensor(WanA14BVAE.std, dtype=torch.float32)
    reference = torch.empty((), dtype=torch.bfloat16)
    scale = vae._scale(reference)
    expected = 1.0 / torch.tensor(WanA14BVAE.std, dtype=torch.float32).to(torch.bfloat16)
    assert torch.equal(scale[1], expected)


def test_a14b_vae_decode_reports_nonfinite_input_ranges() -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.components import WanA14BVAE
    from solar.errors import BackendContractError

    class FakeModule:
        def decode(self, value: object) -> object:
            sample = value[:, :3].clone()
            sample[..., 0, 0, 0] = torch.nan
            return SimpleNamespace(sample=sample)

    vae = object.__new__(WanA14BVAE)
    vae._mean = torch.tensor(WanA14BVAE.mean, dtype=torch.float32)
    vae._std = torch.tensor(WanA14BVAE.std, dtype=torch.float32)
    vae.module = FakeModule()
    latents = torch.zeros((1, 2, 16, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(BackendContractError, match="normalized_latent_absmax=0"):
        vae.decode(latents)


def test_a14b_vae_architecture_is_official_wan21_not_patchified_wan22() -> None:
    torch = pytest.importorskip("torch")
    from diffusers import AutoencoderKLWan

    with torch.device("meta"):
        module = AutoencoderKLWan()
    assert module.config.base_dim == 96
    assert module.config.decoder_base_dim is None
    assert module.config.temperal_downsample == [False, True, True]
    assert module.config.patch_size is None
    assert tuple(module.encoder.conv_in.weight.shape[1:]) == (3, 3, 3, 3)
    assert tuple(module.quant_conv.weight.shape[:2]) == (32, 32)


@pytest.mark.parametrize("failure", ("shape", "key"))
def test_a14b_vae_rejects_non_exact_state_dict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    torch = pytest.importorskip("torch")
    import diffusers

    from solar.backends.wan22.runtime.components import WanA14BVAE

    class TinyVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 3))

    state = {"weight": torch.zeros(2, 4)}
    if failure == "key":
        state = {
            "weight": torch.zeros(2, 3),
            "unexpected": torch.zeros(1),
        }
    monkeypatch.setattr(diffusers, "AutoencoderKLWan", TinyVAE)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: state)
    with pytest.raises(BackendContractError, match=r"cannot load Wan2.1 A14B VAE"):
        WanA14BVAE(tmp_path / "vae.pt")


def test_a14b_dual_weights_loader_keeps_live_and_ema_distinct(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.checkpoint import (
        load_live_and_ema_weights_checkpoint,
    )
    from solar.training.ema import ShardedEMA

    source = tmp_path / "model.pt"
    live_model = torch.nn.Linear(2, 2)
    ema_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        live_model.weight.fill_(1.0)
        live_model.bias.fill_(2.0)
        ema_model.weight.fill_(3.0)
        ema_model.bias.fill_(4.0)
    torch.save(
        {
            "generator": {
                f"model.{key}": value.detach().clone()
                for key, value in live_model.state_dict().items()
            },
            "generator_ema": {
                f"model.{key}": value.detach().clone()
                for key, value in ema_model.state_dict().items()
            },
            "global_step": 22000,
            "ema_num_updates": 22000,
            "config": {
                "model": {
                    "family": "wan22_i2v_a14b",
                    "camera_translation_transform": "linear",
                },
                "train": {"stage": "stage0p5", "objective": "flow_matching"},
            },
        },
        source,
    )
    config = load_config(A14B_153F_CONFIG).mutable_copy()
    config["checkpoint"]["path"] = str(source)
    target = torch.nn.Linear(2, 2)
    ema = ShardedEMA(target, decay=0.9999, device="cpu", dtype=torch.float32)
    restored = load_live_and_ema_weights_checkpoint(
        config=config,
        path=source,
        diffusion=SimpleNamespace(module=target),
        ema=ema,
    )
    assert restored.source_step == 22000
    assert torch.equal(target.weight, live_model.weight)
    assert torch.equal(target.bias, live_model.bias)
    assert torch.equal(ema.shadow["weight"], ema_model.weight)
    assert torch.equal(ema.shadow["bias"], ema_model.bias)
    assert ema.num_updates == 0


def test_stage0p5_ema_initialization_sets_live_and_ema(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from solar.backends.wan22.runtime.checkpoint import (
        load_live_and_ema_weights_checkpoint,
    )
    from solar.training.ema import ShardedEMA

    source = tmp_path / "model.pt"
    live_model = torch.nn.Linear(2, 2)
    ema_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        live_model.weight.fill_(1.0)
        live_model.bias.fill_(2.0)
        ema_model.weight.fill_(3.0)
        ema_model.bias.fill_(4.0)
    torch.save(
        {
            "generator": {
                f"model.{key}": value.detach().clone()
                for key, value in live_model.state_dict().items()
            },
            "generator_ema": {
                f"model.{key}": value.detach().clone()
                for key, value in ema_model.state_dict().items()
            },
            "global_step": 10000,
            "ema_num_updates": 10000,
            "config": {
                "model": {
                    "family": "wan22_i2v_a14b",
                    "camera_translation_transform": "linear",
                },
                "train": {"stage": "stage0p5", "objective": "flow_matching"},
            },
        },
        source,
    )
    config = load_config(A14B_153F_CONFIG).mutable_copy()
    config["checkpoint"]["path"] = str(source)
    config["checkpoint"]["source_step"] = 10000
    config["checkpoint"]["weights"] = ["ema", "ema"]
    target = torch.nn.Linear(2, 2)
    ema = ShardedEMA(target, decay=0.9999, device="cpu", dtype=torch.float32)
    load_live_and_ema_weights_checkpoint(
        config=config,
        path=source,
        diffusion=SimpleNamespace(module=target),
        ema=ema,
    )
    assert torch.equal(target.weight, ema_model.weight)
    assert torch.equal(target.bias, ema_model.bias)
    assert torch.equal(ema.shadow["weight"], ema_model.weight)
    assert torch.equal(ema.shadow["bias"], ema_model.bias)
    assert ema.num_updates == 0


@pytest.mark.parametrize(
    ("path", "encoding"),
    (
        (A14B_81F_CONFIG, "online"),
        (A14B_153F_CONFIG, "preencoded"),
        (A14B_153F_CONFIG, "online"),
    ),
)
def test_a14b_stage0p5_public_dispatch_is_wired(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    encoding: str,
) -> None:
    from solar.backends.wan22.runtime import readiness, stage0p5

    config = load_config(path).mutable_copy()
    config["data"]["encoding"] = encoding
    create_backend(family="wan22_i2v_a14b").validate_config(config)

    class Ready:
        def require_ready(self) -> None:
            return None

    monkeypatch.setattr(readiness, "probe_runtime", lambda *args, **kwargs: Ready())
    monkeypatch.setattr(stage0p5, "run_stage0p5_training", lambda _: 23)
    assert create_backend(family="wan22_i2v_a14b").train(config) == 23
