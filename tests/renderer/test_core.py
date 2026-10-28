from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from solar.backends.minimax_h3.artifacts import H3ArtifactBatch, H3VisualObservation
from solar.backends.minimax_h3.flow import make_shifted_schedule
from solar.backends.minimax_h3.segmented_attention import segment_attention
from solar.renderer.causal import H3OmniCausalCore
from solar.renderer.core import H3Stage0p5Core
from solar.renderer.observations import observation_profile_from_train_config


class PackedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        video = kwargs["hidden_states"]
        size = kwargs["token_tags"].numel()
        packed = torch.zeros(1, size, 4)
        packed[:, kwargs["video_indices"]] = video[..., :4]
        packed[:, kwargs["text_indices"]] = kwargs["encoder_hidden_states"][..., :4].float()
        qkv = packed.unsqueeze(2)
        plan = kwargs["attention_mask"]
        if plan is None:
            attended = torch.nn.functional.scaled_dot_product_attention(
                qkv.transpose(1, 2), qkv.transpose(1, 2), qkv.transpose(1, 2)
            ).transpose(1, 2)
        else:
            attended = segment_attention(qkv, qkv, qkv, plan, 0)
        output = attended.squeeze(2)[:, kwargs["video_indices"]].repeat(1, 1, 24)
        return output * self.scale, kwargs["audio_hidden_states"]


class SmallCore(H3Stage0p5Core):
    @staticmethod
    def latent_grid(batch):
        return (2, 2)


class SmallCausalCore(H3OmniCausalCore):
    @staticmethod
    def latent_grid(batch):
        return (2, 2)


def batch(frames=7):
    prompt = torch.ones(2, 5120)
    tags = torch.ones(2, dtype=torch.long)
    return H3ArtifactBatch(
        sample_id="synthetic",
        start_frame=0,
        plan_fingerprint="synthetic",
        target_latents=torch.ones(24, frames, 2, 2),
        anchor_latents=None,
        prompt_embeds=prompt,
        text_token_tags=tags,
        source_frame_indices=None,
        camera_viewmats=None,
        camera_K=None,
        source_fps=24.0,
        text_prompt_embeds=prompt,
        text_prompt_tags=tags,
        visual_observations=(
            H3VisualObservation(torch.ones(24, frames, 2, 2), tuple(range(frames))),
            H3VisualObservation(torch.ones(24, 1, 2, 2), (0,)),
        ),
    )


@pytest.mark.parametrize("causal", [False, True])
def test_renderer_forward_backward_and_sampling_are_finite_and_repeatable(causal):
    model = PackedModel()
    options = {"observations": observation_profile_from_train_config({})}
    silence = torch.zeros(2, 32, 263)
    core = (
        SmallCausalCore(model, silence, torch.device("cpu"), {"chunk": 5, "window": 2}, **options)
        if causal
        else SmallCore(model, silence, torch.device("cpu"), **options)
    )
    loss = core.forward_loss(batch(), noise_seed=42)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.scale.grad)
    before = model.calls
    generated = core.generate(batch(), noise_seed=10, num_inference_steps=5, renoise=True)
    repeated = core.generate(batch(), noise_seed=10, num_inference_steps=5, renoise=True)
    assert generated.shape == (1, 24, 7, 2, 2)
    assert torch.isfinite(generated).all()
    torch.testing.assert_close(generated, repeated, rtol=0, atol=0)
    assert model.calls > before


def test_causal_stream_cached_and_recomputed_structure_agree():
    core = SmallCausalCore(
        PackedModel(), torch.zeros(2, 32, 263), torch.device("cpu"), {"chunk": 5, "window": 2}
    )
    outputs = []
    for cached in (True, False):
        with torch.no_grad():
            generated, *_ = core.stream(
                batch(),
                generator=torch.Generator().manual_seed(4),
                schedule=make_shifted_schedule(5),
                exit_index=3,
                cache_structure=cached,
            )
        outputs.append(generated)
    torch.testing.assert_close(*outputs, rtol=1e-6, atol=1e-6)


def test_renderer_supports_no_visual_observation():
    core = SmallCore(PackedModel(), torch.zeros(2, 32, 263), torch.device("cpu"))
    result = core.generate(
        replace(batch(), visual_observations=None), noise_seed=0, num_inference_steps=2
    )
    assert result.shape == (1, 24, 7, 2, 2)
