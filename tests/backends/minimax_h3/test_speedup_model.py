"""Wiring of the H3 speedup switches into the transformer, on a tiny random model (one GPU).

The model runs cuDNN attention with FlashAttention-2's deterministic backward in deterministic
mode, so that a forward/backward is reproducible bit for bit and every comparison is bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h3 = pytest.importorskip(
    "diffusers.models.transformers.transformer_minimax_h3", exc_type=ImportError
)
pytest.importorskip("triton")
pytest.importorskip("flash_attn")

from solar.backends.minimax_h3 import (  # noqa: E402
    fast_attention,
    fused_ops,
    fused_prope,
    stage_scope,
)
from solar.backends.minimax_h3.layout import build_stage0p5_layout  # noqa: E402
from solar.backends.minimax_h3.torch_layout import build_row_timesteps  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

TINY = {
    "num_attention_heads": 4,
    "attention_head_dim": 128,
    "hidden_size": 512,
    "num_layers": 3,
    "num_refiner_layers": 1,
    "ffn_dim": 1024,
    "text_dim": 64,
    "time_embed_hidden_dim": 256,
    "time_embed_dim": 128,
}
FP32_MODULES = ("proj_in", "proj_out", "time_embedder")


@pytest.fixture
def h3_model():
    # Imported here, not at collection: the module compiles FlexAttention at import.
    from solar.backends.minimax_h3 import model

    return model


@pytest.fixture
def tiny(monkeypatch, h3_model):
    """``run()`` -> (video output, video gradient, block parameter gradients), camera forward."""
    monkeypatch.setattr(fast_attention, "_BACKEND", "cudnn")
    monkeypatch.setattr(stage_scope, "_CUDNN_STAGES", None)
    monkeypatch.setattr(h3_model, "_PROPE_PLAN", False)
    monkeypatch.setattr(fused_prope, "_ENABLED", False)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)

    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = h3_model.SolarMiniMaxH3Transformer3DModel.from_config(TINY)
    for name, parameter in model.named_parameters():
        dtype = torch.float32 if any(key in name for key in FP32_MODULES) else torch.bfloat16
        parameter.data = parameter.data.to(dtype)
        # Gradients of the transformer blocks only: the token refiner's attention backward is
        # outside the switches and need not be deterministic.
        parameter.requires_grad_(name.startswith("transformer_blocks."))
    model = model.to("cuda")

    layout = build_stage0p5_layout(np.ones(37, dtype=np.int64), 47, 8, 12, 11).to("cuda")
    gen = torch.Generator(device="cuda").manual_seed(1)
    video = torch.randn(1, layout.video_indices.numel(), 96, device="cuda", generator=gen)
    audio = torch.randn(1, layout.audio_indices.numel(), 32, device="cuda", generator=gen)
    prompt = torch.randn(1, 37, 64, device="cuda", generator=gen).bfloat16()
    views = torch.eye(4, device="cuda").repeat(1, 47, 1, 1)
    views[..., :3, 3] = torch.randn(1, 47, 3, device="cuda", generator=gen)
    intrinsics = torch.eye(3, device="cuda").repeat(1, 47, 1, 1)
    times, indices = build_row_timesteps(
        layout,
        torch.tensor(0.6, device="cuda"),
        torch.tensor(0.7, device="cuda"),
        condition_video_timestep=0.999,
        device="cuda",
    )

    def run():
        model.zero_grad(set_to_none=True)
        leaf = video.clone().requires_grad_(True)
        out, _ = model(
            hidden_states=leaf,
            audio_hidden_states=audio,
            encoder_hidden_states=prompt,
            timestep=times,
            timestep_indices=indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            attention_mask=None,
            fused_prope=True,
            prope_token_indices=layout.camera_video_indices,
            prope_frame_ids=layout.camera_frame_ids,
            cam_viewmats=views,
            cam_K=intrinsics,
            return_dict=False,
        )
        out.float().square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        return [out.detach(), leaf.grad, *grads]

    try:
        yield run
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _assert_bitwise(actual, expected):
    assert len(actual) == len(expected)
    for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(a, e), f"tensor {index} (output, video gradient, parameter gradients)"


def test_tiny_model_baseline_is_reproducible(tiny):
    before = fast_attention.COUNTERS["deterministic_backward"]
    _assert_bitwise(tiny(), tiny())
    assert fast_attention.COUNTERS["deterministic_backward"] > before


@pytest.mark.parametrize("fused", [False, True])
def test_prope_plan_is_bitwise_to_per_layer_prope(tiny, h3_model, monkeypatch, fused):
    monkeypatch.setattr(fused_prope, "_ENABLED", fused)
    expected = tiny()
    monkeypatch.setattr(h3_model, "_PROPE_PLAN", True)
    before = dict(h3_model.PROPE_PLAN_COUNTERS)
    actual = tiny()
    # Built by the first of the three blocks, reused by the other two.
    delta = {key: h3_model.PROPE_PLAN_COUNTERS[key] - before[key] for key in before}
    assert delta == {"built": 1, "reused": 2}
    _assert_bitwise(actual, expected)


def test_fused_block_forward_wiring_is_bitwise_with_eager_kernels(tiny, h3_model, monkeypatch):
    """``SOLARWM_H3_FUSED=1`` replaces the block forward and the q/k norm + rotary; with the
    kernels swapped for the eager formulas they implement, the model must not change at all
    (the kernels themselves are compared with those formulas in test_speedup_kernels)."""
    expected = tiny()
    calls = {"rmsnorm_modulate": 0, "gated_residual": 0, "qk_norm_rope": 0}

    def rmsnorm_modulate(norm, x, scale, shift, index):
        calls["rmsnorm_modulate"] += 1
        return norm(x) * (1.0 + scale.index_select(0, index)) + shift.index_select(0, index)

    def gated_residual(residual, gate, index, update):
        calls["gated_residual"] += 1
        return residual + gate.index_select(0, index) * update

    def qk_norm_rope(norm, x, cos, sin):
        calls["qk_norm_rope"] += 1
        return h3._apply_rotary_emb(norm(x), cos, sin)

    for name, value in (
        ("rmsnorm_modulate", rmsnorm_modulate),
        ("gated_residual", gated_residual),
        ("qk_norm_rope", qk_norm_rope),
    ):
        monkeypatch.setattr(fused_ops, name, value)
    monkeypatch.setattr(h3_model, "_FUSED", True)
    monkeypatch.setattr(
        h3_model.MiniMaxH3TransformerBlock, "forward", h3_model._fused_block_forward
    )
    actual = tiny()
    assert all(calls.values()), calls
    _assert_bitwise(actual, expected)


def test_fused_swiglu_patch_is_bitwise(tiny, monkeypatch):
    from diffusers.models.activations import SwiGLU

    expected = tiny()
    monkeypatch.setattr(SwiGLU, "forward", fused_ops._fused_swiglu_forward)
    before = fused_ops.COUNTERS.get("swiglu", 0)
    actual = tiny()
    assert fused_ops.COUNTERS.get("swiglu", 0) > before
    _assert_bitwise(actual, expected)
