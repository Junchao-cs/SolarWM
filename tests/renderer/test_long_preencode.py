from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from solarwm.renderer import preencode

encoders = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.encoders")
AutoencoderKLMiniMaxH3 = pytest.importorskip(
    "diffusers.models.autoencoders.autoencoder_kl_minimax_h3"
).AutoencoderKLMiniMaxH3
DiagonalGaussianDistribution = pytest.importorskip(
    "diffusers.models.autoencoders.vae"
).DiagonalGaussianDistribution
MiniMaxH3ModularPipeline = pytest.importorskip(
    "diffusers.modular_pipelines.minimax_h3.modular_pipeline"
).MiniMaxH3ModularPipeline


class TinyVAE(torch.nn.Module):
    """Exercise H3's real temporal chunking with a cheap deterministic spatial encoder."""

    def __init__(self, token_drop=3):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1, bias=False)
        self.config = SimpleNamespace(
            clip_length=17, token_drop=token_drop, latents_mean=(0.2, -0.1), latents_std=(1.2, 0.7)
        )
        self.clips = []
        self.whole_calls = 0

    def _encode_clip(self, pixels):
        return AutoencoderKLMiniMaxH3._encode_clip(self, pixels)

    def encode(self, pixels, return_dict=False):
        self.whole_calls += 1
        moments = AutoencoderKLMiniMaxH3._encode(self, pixels.to(self.encoder.weight.dtype))
        return (DiagonalGaussianDistribution(moments),)


@pytest.fixture
def runtime(monkeypatch):
    def encode_clip(self, pixels):
        self.clips.append((pixels.shape[2], pixels.is_contiguous()))
        mean = pixels[:, :2, ::4]
        logvar = pixels[:, 1:3, ::4] * 0.01 - 3
        return torch.cat((mean, logvar), dim=1)

    monkeypatch.setattr(AutoencoderKLMiniMaxH3, "_encode_clip", encode_clip)
    monkeypatch.setattr(MiniMaxH3ModularPipeline, "max_duration", property(lambda self: 15.0))
    monkeypatch.setattr(
        encoders, "encode_vae_condition", inspect.unwrap(encoders.encode_vae_condition)
    )
    monkeypatch.delenv("OMNI_MAX_SECONDS", raising=False)
    monkeypatch.delenv("OMNI_CPU_OFFLOAD", raising=False)


def pixels(frames):
    # Match the non-contiguous C,T,H,W view produced by H3's ref2va encoder.
    return torch.randint(0, 256, (frames, 2, 2, 3), dtype=torch.uint8).permute(3, 0, 1, 2)[None]


@pytest.mark.parametrize("frames,token_drop", [(1513, 3), (1530, 0), (1535, 3)])
def test_long_condition_matches_released_posterior_without_whole_video_encode(
    runtime, frames, token_drop
):
    source = pixels(frames)
    vae = TinyVAE(token_drop)
    expected = encoders.encode_vae_condition(vae, source, (0.4, 0.5, 0.6), (0.2, 0.3, 0.4), 123)
    vae.whole_calls = 0
    vae.clips.clear()
    vae._solar_renderer_long_video = True
    preencode._install_long_video_support()

    actual = encoders.encode_vae_condition(vae, source, (0.4, 0.5, 0.6), (0.2, 0.3, 0.4), 123)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    assert vae.whole_calls == 0
    assert vae.clips == [(17, True)] * ((frames + 16) // 17)


@pytest.mark.parametrize("frames", [1, 158, 1500])
def test_short_conditions_keep_released_encoder(runtime, frames):
    source = pixels(frames)
    vae = TinyVAE()
    expected = encoders.encode_vae_condition(vae, source, (0, 0, 0), (1, 1, 1))
    vae.whole_calls = 0
    vae._solar_renderer_long_video = True
    preencode._install_long_video_support()
    actual = encoders.encode_vae_condition(vae, source, (0, 0, 0), (1, 1, 1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert vae.whole_calls == 1


def test_unmarked_vae_keeps_original_condition_and_clip_paths(runtime):
    preencode._install_long_video_support()
    vae = TinyVAE()
    encoders.encode_vae_condition(vae, pixels(1530), (0, 0, 0), (1, 1, 1))
    assert vae.whole_calls == 1
    source = torch.zeros(1, 3, 34, 2, 2)[:, :, :17]
    vae._encode_clip(source)
    assert vae.clips[-1] == (17, False)
    vae._solar_renderer_long_video = True
    vae._encode_clip(source)
    assert vae.clips[-1] == (17, True)


def test_long_condition_only_normalizes_one_clip_at_a_time(runtime):
    from torch.utils._python_dispatch import TorchDispatchMode

    converted_frames = []

    class WatchNormalization(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if (
                func is torch.ops.aten._to_copy.default
                and args[0].dtype == torch.uint8
                and kwargs.get("dtype") == torch.float32
            ):
                converted_frames.append(args[0].shape[2])
            return func(*args, **kwargs)

    vae = TinyVAE()
    vae._solar_renderer_long_video = True
    preencode._install_long_video_support()
    with WatchNormalization():
        encoders.encode_vae_condition(vae, pixels(1535), (0, 0, 0), (1, 1, 1))
    assert converted_frames == [17] * 91


def test_long_video_dispatchers_are_installed_once(runtime):
    def installed():
        return (
            MiniMaxH3ModularPipeline.max_duration,
            AutoencoderKLMiniMaxH3._encode_clip,
            encoders.encode_vae_condition,
        )

    preencode._install_long_video_support()
    first = installed()
    for _ in range(3):
        preencode._install_long_video_support()
        assert installed() == first
    vae = TinyVAE()
    vae._solar_renderer_long_video = True
    encoders.encode_vae_condition(vae, pixels(1513), (0, 0, 0), (1, 1, 1))
    assert vae.whole_calls == 0


@pytest.fixture
def pipeline_factory(runtime, monkeypatch):
    import diffusers
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    class Pipeline:
        pretrained_component_names = ("vae",)

        @property
        def max_duration(self):
            return MiniMaxH3ModularPipeline.max_duration.fget(self)

        def load_components(self, **kwargs):
            self.vae = TinyVAE()

        def update_components(self, **kwargs):
            self.__dict__.update(kwargs)

    blocks = SimpleNamespace(
        get_workflow=lambda _workflow: SimpleNamespace(
            sub_blocks={name: object() for name in ("before_encode", "text_encoder", "vae_encoder")}
        )
    )
    monkeypatch.setattr(diffusers, "ComponentsManager", lambda: object())
    monkeypatch.setattr(
        diffusers.ModularPipeline, "from_pretrained", lambda *_args: SimpleNamespace(blocks=blocks)
    )
    monkeypatch.setattr(
        SequentialPipelineBlocks,
        "from_blocks_dict",
        lambda *_args: SimpleNamespace(init_pipeline=lambda *_args, **_kwargs: Pipeline()),
    )


def test_pipeline_duration_and_vae_settings_are_local_to_each_build(pipeline_factory, monkeypatch):
    first = preencode.build_pipelines("cpu", max_seconds=61)
    second = preencode.build_pipelines("cpu", max_seconds=181)
    short = preencode.build_pipelines("cpu")
    monkeypatch.setenv("OMNI_MAX_SECONDS", "91")
    environment = preencode.build_pipelines("cpu")
    explicit = preencode.build_pipelines("cpu", max_seconds=31)
    for pipelines, limit in (
        (first, 61),
        (second, 181),
        (short, 15),
        (environment, 91),
        (explicit, 31),
    ):
        assert [pipe.max_duration for pipe in pipelines] == [limit] * 3
        assert pipelines[0].vae is pipelines[1].vae is pipelines[2].vae
        assert getattr(pipelines[0].vae, "_solar_renderer_long_video", False) == (limit != 15)
    assert first[0].vae is not second[0].vae


@pytest.mark.parametrize("limit", [0, -1, float("inf"), float("nan")])
def test_invalid_duration_fails_before_loading_models(runtime, limit):
    with pytest.raises(ValueError, match="finite and positive"):
        preencode.build_pipelines("cpu", max_seconds=limit)
