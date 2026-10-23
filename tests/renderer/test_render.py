from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from solarwm.backends.minimax_h3.lora import merge_lora_delta
from solarwm.renderer import render


@pytest.mark.parametrize(
    "seconds,held,expected",
    [
        (0, 158, (158, 158)),
        (0, 3000, (158, 158)),
        (1, 3000, (158, 24)),
        (10, 3000, (243, 240)),
        (60, 2000, (1450, 1440)),
        (60, 1440, (1433, 1433)),
        (60, 158, (158, 158)),
        (1e308, 1440, (1433, 1433)),
    ],
)
def test_requested_duration_is_aligned_without_exceeding_structure(seconds, held, expected):
    assert render.clip_frames(seconds, held) == expected


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf"), 0.001])
def test_invalid_duration_is_rejected(seconds):
    with pytest.raises(ValueError, match="seconds"):
        render.clip_frames(seconds, 2000)


def test_too_short_structure_is_rejected_before_encoding():
    with pytest.raises(ValueError, match="at least 158"):
        render.clip_frames(60, 157)


def test_two_loras_round_into_bf16_only_once():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16))
    model[0].weight.data.fill_(1)
    state = {"0.lora_A.weight": torch.ones(1, 1), "0.lora_B.weight": torch.full((1, 1), 0.003)}
    stacked = render.stack_lora_pairs((state, 1, 1), (state, 1, 1))
    merge_lora_delta(model, stacked, alpha=None, rank=None)
    expected = torch.tensor([[1.006]], dtype=torch.bfloat16)
    torch.testing.assert_close(model[0].weight, expected, rtol=0, atol=0)
    # Each individual delta rounds away if merged into BF16 weights separately.
    sequential = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16))
    sequential[0].weight.data.fill_(1)
    for _ in range(2):
        merge_lora_delta(sequential, state, alpha=1, rank=1)
    assert not torch.equal(model[0].weight, sequential[0].weight)


def test_stacked_loras_preserve_each_adapter_rank_and_alpha():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16))
    before = model[0].weight.detach().float().clone()
    first = {
        "0.lora_A.weight": torch.tensor([[1.0, 0.5]]),
        "0.lora_B.weight": torch.tensor([[0.5], [0.25], [0.125]]),
    }
    second = {
        "0.lora_A.weight": torch.tensor([[0.25, 0.5], [1.0, -0.5]]),
        "0.lora_B.weight": torch.tensor([[0.5, 0.25], [0.25, -0.5], [0.125, 1.0]]),
    }
    stacked = render.stack_lora_pairs((first, 1, 2), (second, 2, 1))
    merge_lora_delta(model, stacked, alpha=None, rank=None)
    expected = before + 2 * first["0.lora_B.weight"] @ first["0.lora_A.weight"]
    expected += 0.5 * second["0.lora_B.weight"] @ second["0.lora_A.weight"]
    torch.testing.assert_close(model[0].weight, expected.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("frames", [158, 243])
def test_renderer_loads_merged_weights_and_duration_matched_silence(monkeypatch, frames):
    from solarwm.backends.minimax_h3 import artifacts, optional
    from solarwm.renderer import core

    real_core = core.H3Stage0p5Core
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16))
    model[0].weight.data.fill_(1)
    state = {"0.lora_A.weight": torch.ones(1, 1), "0.lora_B.weight": torch.full((1, 1), 0.003)}
    monkeypatch.setattr(render, "RANK", 1)
    monkeypatch.setattr(render, "lora_pairs", lambda _: (state, 1, 1))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        optional, "load_transformer", lambda *a, **k: SimpleNamespace(transformer=model)
    )
    monkeypatch.setattr(optional, "load_conditioners", lambda *a, **k: "conditioners")
    lengths = {158: 263, 170: 283}

    def silence(_path, *, pixel_frames):
        length = lengths[pixel_frames]
        return torch.arange(length).expand(2, 32, length).to(torch.bfloat16), {}

    monkeypatch.setattr(artifacts, "load_silence_latents", silence)

    class Core:
        audio_latents = staticmethod(real_core.audio_latents)

        def __init__(self, wrapped, audio, device, **kwargs):
            self.model, self.audio = wrapped, audio

    monkeypatch.setattr(core, "H3Stage0p5Core", Core)
    loaded, conditioners, device = render.load_renderer(
        "/unused/h3", "control", "fewstep", "cuda:0", frames=frames
    )
    assert loaded.model is model and conditioners == "conditioners" and device.type == "cuda"
    assert not model.training and not model[0].weight.requires_grad
    torch.testing.assert_close(model[0].weight, torch.tensor([[1.006]]).bfloat16(), rtol=0, atol=0)
    assert loaded.audio.shape == (2, 32, real_core.audio_latents(5 * ((frames - 5) // 17) + 2))
    source = silence("", pixel_frames=170 if frames > 158 else 158)[0]
    torch.testing.assert_close(loaded.audio[..., :20], source[..., :20], rtol=0, atol=0)
    torch.testing.assert_close(loaded.audio[..., -20:], source[..., -20:], rtol=0, atol=0)


def cli_inputs(tmp_path):
    h3 = tmp_path / "h3"
    (h3 / "transformer_ref").mkdir(parents=True)
    arguments = ["--h3", str(h3), "--out", str(tmp_path / "out.mp4")]
    for option in ("control", "fewstep", "structure", "prompt"):
        path = tmp_path / option
        path.write_text("placeholder")
        arguments.extend(["--" + option, str(path)])
    return arguments


@pytest.mark.parametrize("seconds", ["-1", "nan", "inf", "0.001"])
def test_cli_rejects_invalid_durations(tmp_path, seconds):
    with pytest.raises(SystemExit) as error:
        render.parse([*cli_inputs(tmp_path), "--seconds", seconds])
    assert error.value.code == 2


def test_main_propagates_duration_and_releases_encoder_before_model(tmp_path, monkeypatch):
    from solarwm.renderer import preencode

    events = []
    monkeypatch.setattr(preencode, "frame_count", lambda _: 2000)
    monkeypatch.setattr(preencode, "frame_rate", lambda _: 24.0)

    class Encoder:
        def __del__(self):
            events.append("released")

    def load_encoder(_h3, _device, *, seconds):
        assert seconds == 60
        return Encoder()

    def encode(encoder, sample, member, *, frames):
        assert isinstance(encoder, Encoder)
        assert frames == 1450 and sample["observations"][0]["frames"] == "all"
        events.append("encoded")

    def load_renderer(*args, causal, frames):
        assert events == ["encoded", "released"]
        assert causal and frames == 1450
        return "renderer"

    def write(renderer, member, out, seed, forwards, *, keep):
        assert renderer == "renderer" and keep == 1440 and forwards == 4
        return 0.0

    monkeypatch.setattr(render, "load_encoder", load_encoder)
    monkeypatch.setattr(render, "encode_sample", encode)
    monkeypatch.setattr(render, "free_cuda", lambda: None)
    monkeypatch.setattr(render, "load_renderer", load_renderer)
    monkeypatch.setattr(render, "render_member", write)
    render.main([*cli_inputs(tmp_path), "--seconds", "60", "--causal"])


@pytest.mark.parametrize("keep,expected", [(0, 243), (240, 240)])
def test_video_writer_crops_to_requested_duration(tmp_path, monkeypatch, keep, expected):
    from solarwm.backends.minimax_h3 import inference
    from solarwm.renderer import data

    monkeypatch.setattr(data.H3OmniReader, "read", lambda *a: "batch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(
        inference, "decode_h3_latents", lambda *a, **k: torch.zeros(1, 3, 243, 1, 1)
    )

    def mp4(decoded):
        assert decoded.shape[2] == expected
        return b"encoded video"

    monkeypatch.setattr(inference, "_mp4_bytes", mp4)
    model = SimpleNamespace(generate=lambda *a, **k: "latents")
    output = tmp_path / "video.mp4"
    render.render_member(
        (model, SimpleNamespace(video_vae="vae"), torch.device("cpu")),
        "member",
        output,
        keep=keep,
    )
    assert output.read_bytes() == b"encoded video"
