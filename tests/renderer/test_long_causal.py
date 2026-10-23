"""Exercise long-stream row packing through the real sampler with a tiny CPU model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from solarwm.backends.minimax_h3.artifacts import H3VisualObservation
from solarwm.backends.minimax_h3.flow import make_shifted_schedule
from solarwm.renderer.causal import H3OmniCausalCore


class RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        size = kwargs["token_tags"].numel()
        assert kwargs["attention_mask"].sequence_length == size
        assert kwargs["audio_hidden_states"].shape[1] == kwargs["audio_indices"].numel()
        assert kwargs["hidden_states"].shape[1] == kwargs["video_indices"].numel()
        indices = torch.cat(
            [kwargs[name] for name in ("video_indices", "audio_indices", "text_indices")]
        )
        torch.testing.assert_close(indices.sort().values, torch.arange(size))
        return torch.zeros_like(kwargs["hidden_states"]), kwargs["audio_hidden_states"]


class SmallCausalCore(H3OmniCausalCore):
    @staticmethod
    def latent_grid(batch):
        return (2, 2)

    def _audio(self, frames, generator):
        # Distinct rows reveal both channel offsets after the sampler selects its audio.
        return torch.arange(2 * self.audio_latents(frames) * 32).reshape(1, -1, 32).float()


def make_batch(frames, *, structure=True):
    observations = [H3VisualObservation(torch.ones(24, 1, 2, 2), (0,))]
    if structure:
        observations.insert(
            0, H3VisualObservation(torch.ones(24, frames, 2, 2), tuple(range(frames)))
        )
    return SimpleNamespace(
        target_latents=torch.ones(24, frames, 2, 2),
        prompt_embeds=torch.ones(2, 5120),
        text_token_tags=torch.ones(2, dtype=torch.long),
        text_prompt_embeds=torch.ones(2, 5120),
        text_prompt_tags=torch.ones(2, dtype=torch.long),
        visual_observations=tuple(observations),
    )


def make_core(chunk=5):
    model = RecordingModel()
    return SmallCausalCore(
        model, torch.zeros(2, 32, 263), torch.device("cpu"), {"chunk": chunk, "window": 2}
    )


def stream(core, batch, *, cache_structure=False):
    return core.stream(
        batch,
        generator=torch.Generator().manual_seed(4),
        schedule=make_shifted_schedule(2),
        exit_index=0,
        cache_structure=cache_structure,
    )


@pytest.mark.parametrize("frames,chunk", [(7, 5), (47, 5), (52, 5), (62, 3)])
def test_stream_preserves_trained_condition_positions_and_audio_channels(frames, chunk):
    core = make_core(chunk)
    clip, _inputs, times, _jitters = stream(core, make_batch(frames))
    assert clip.shape == (1, 24, frames, 2, 2)
    assert torch.isfinite(clip).all()
    layout = next(iter(core._layouts.values()))
    source = layout.source
    # Per-chunk shifts must never accumulate in the reusable full-document layout.
    np.testing.assert_array_equal(layout.position_ids.numpy(), source.position_ids)
    positions = layout.position_ids
    per_channel = core.audio_latents(frames)
    heard = core.audio_latents(min(frames, 47))
    audio_rows = torch.cat((torch.arange(heard), torch.arange(per_channel, per_channel + heard)))
    expected_audio = core._audio(frames, None)[:, audio_rows]
    audio_positions = positions[layout.audio_indices[audio_rows]]
    original_text = positions[layout.text_indices]
    original_first = positions[source.observation_video_indices[-1]]
    assert len(core.model.calls) == 2 * len(times)

    for c, (denoise, commit) in enumerate(
        zip(core.model.calls[::2], core.model.calls[1::2], strict=True)
    ):
        start, stop = c * chunk, min((c + 1) * chunk, frames)
        target = positions[source.target_video_indices[start:stop]]
        shift = 0.0
        if frames > 47 and stop > 47:
            # The matching full training chunk has the same phase of the five-latent
            # video clock. Its prompt/first/audio offsets must recur in the long stream.
            trained_start = max(f for f in range(0, 48 - chunk, chunk) if (start - f) % 5 == 0)
            trained_time = positions[source.target_video_indices[trained_start], 0]
            shift = float(target[0, 0] - trained_time)
            assert shift > 0
        expected_text = original_text.clone()
        expected_text[:, 0] += shift
        expected_first = original_first.clone()
        expected_first[0] += shift
        expected_audio_positions = audio_positions.clone()
        expected_audio_positions[:, 0] += shift

        for call in (denoise, commit):
            packed = call["position_ids"]
            torch.testing.assert_close(packed[call["text_indices"]], expected_text)
            torch.testing.assert_close(packed[call["audio_indices"]], expected_audio_positions)
            torch.testing.assert_close(call["audio_hidden_states"], expected_audio)
            video = packed[call["video_indices"]]
            torch.testing.assert_close(video[-(stop - start) :], target)
            torch.testing.assert_close(video[-(stop - start) - 1], expected_first)
            # Only the last two structure chunks are recomputed, at original positions.
            structure_start = max(0, (c - 1) * chunk)
            structure_positions = positions[source.observation_video_indices[structure_start:stop]]
            torch.testing.assert_close(video[: -(stop - start) - 1], structure_positions)
            target_plan = call["attention_mask"].segments[-1]
            assert target_plan[3] == (() if c == 0 else (("clean", c - 1),))
        assert commit["attention_mask"].segments[-1][4] == ("clean", c)


def test_long_stream_requires_structure_recomputation_before_model_execution():
    core = make_core()
    with pytest.raises(ValueError, match="cache_structure=False"):
        stream(core, make_batch(52), cache_structure=True)
    assert core.model.calls == []


@pytest.mark.parametrize("frames,structure", [(47, True), (52, False)])
def test_structure_cache_is_allowed_when_condition_positions_do_not_stale_it(frames, structure):
    core = make_core()
    clip, *_ = stream(core, make_batch(frames, structure=structure), cache_structure=True)
    assert clip.shape == (1, 24, frames, 2, 2)
    assert core.model.calls
