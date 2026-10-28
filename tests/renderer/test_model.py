from __future__ import annotations

import builtins

import pytest

from solar.backends.minimax_h3.layout import H3VisualObservationSpec
from solar.renderer.causal import (
    causal_rows,
    row_times,
    training_plan,
    unique_times,
    with_clean_copy,
)
from solar.renderer.core import H3Stage0p5Core

torch = pytest.importorskip("torch")
upstream = pytest.importorskip(
    "diffusers.models.transformers.transformer_minimax_h3",
    reason="requires the SolarRenderer dependency set with Diffusers >= 0.40 and MiniMax-H3",
    exc_type=ImportError,
)


TINY_CONFIG = {
    "num_attention_heads": 1,
    "attention_head_dim": 128,
    "hidden_size": 128,
    "num_layers": 1,
    "num_refiner_layers": 1,
    "ffn_dim": 256,
    "in_channels": 24,
    "audio_in_channels": 32,
    "text_dim": 32,
    "freq_dim": 32,
    "time_embed_hidden_dim": 128,
    "time_embed_dim": 64,
    "rope_freq_dim": 16,
}


def tiny_model():
    from solar.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel

    with torch.random.fork_rng():
        torch.manual_seed(716)
        return SolarMiniMaxH3Transformer3DModel.from_config(TINY_CONFIG)


def packed_inputs(model, *, teacher=False):
    core = H3Stage0p5Core(model, None, torch.device("cpu"))
    layout = core.layout(
        torch.tensor([1, 1]),
        anchors=("first",),
        observations=(H3VisualObservationSpec(2, 2, 2, frames=(0, 1)),),
        grid=(2, 2),
        frames=2,
    )
    rows = causal_rows(layout.source, structure_index=0, chunk=1)
    plan, clean = training_plan(rows, window=1, teacher=teacher)
    if teacher:
        arrays = with_clean_copy(layout)
    else:
        arrays = {
            name: getattr(layout, name)
            for name in (
                "position_ids",
                "token_tags",
                "video_indices",
                "audio_indices",
                "text_indices",
            )
        }
    times, inverse = unique_times(
        row_times(plan.sequence_length, rows, target_times=[0.3, 0.6], audio=0.2, clean=clean),
        torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(717)
    arguments = {
        "hidden_states": torch.randn(1, len(arrays["video_indices"]), 96, generator=generator),
        "audio_hidden_states": torch.randn(
            1, len(arrays["audio_indices"]), 32, generator=generator
        ),
        "encoder_hidden_states": torch.randn(
            1, len(arrays["text_indices"]), 32, generator=generator
        ),
        "timestep": times,
        "timestep_indices": inverse,
        "return_dict": False,
        **arrays,
    }
    return arguments, plan, rows


def differentiable_inputs(arguments):
    result = dict(arguments)
    for name in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        result[name] = arguments[name].detach().clone().requires_grad_()
    return result


def output_and_gradients(model, arguments, **options):
    result = model(**arguments, **options)
    loss = result[0].square().mean() + result[1].square().mean()
    differentiated = (
        arguments["hidden_states"],
        arguments["audio_hidden_states"],
        arguments["encoder_hidden_states"],
        model.transformer_blocks[0].attn.to_q.weight,
        model.transformer_blocks[0].attn.to_v.weight,
    )
    gradients = torch.autograd.grad(loss, differentiated, retain_graph=True)
    assert all(torch.isfinite(value).all() for value in (*result, *gradients))
    assert all(torch.count_nonzero(value) > 0 for value in gradients)
    return result, gradients


@pytest.mark.parametrize("explicit_control", [False, True])
def test_tiny_adapter_preserves_native_bidirectional_forward_and_backward(explicit_control):
    from solar.backends.minimax_h3.model import SolarMiniMaxH3AttnProcessor

    model = tiny_model()
    native = upstream.MiniMaxH3Transformer3DModel.from_config(TINY_CONFIG)
    native.load_state_dict(model.state_dict(), strict=True)
    assert set(model.state_dict()) == set(native.state_dict())
    attention = model.transformer_blocks[0].attn
    assert isinstance(attention.processor, SolarMiniMaxH3AttnProcessor)
    assert attention.h3_layer_index == 0
    arguments, _, _ = packed_inputs(model)
    cosine, sine = model.rope(arguments["position_ids"])
    assert cosine.shape[-1] == sine.shape[-1] == 96
    options = {"fused_prope": False} if explicit_control else {}
    actual, actual_gradients = output_and_gradients(
        model, differentiable_inputs(arguments), **options
    )
    expected, expected_gradients = output_and_gradients(native, differentiable_inputs(arguments))
    assert actual[0].shape == (1, len(arguments["video_indices"]), 96)
    assert actual[1].shape == (1, len(arguments["audio_indices"]), 32)
    for actual_value, expected_value in zip(
        (*actual, *actual_gradients), (*expected, *expected_gradients), strict=True
    ):
        torch.testing.assert_close(actual_value, expected_value, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_tiny_model_segment_plan_matches_dense_attention_and_excludes_future(
    teacher, checkpointing, monkeypatch
):
    model = tiny_model()
    reference = tiny_model()
    if checkpointing:
        model.enable_gradient_checkpointing()
        reference.enable_gradient_checkpointing()
    arguments, plan, rows = packed_inputs(model, teacher=teacher)
    dense = torch.zeros(plan.sequence_length, plan.sequence_length, dtype=torch.bool)
    for start, stop, spans, _, _ in plan.segments:
        for begin, end in spans:
            dense[start:stop, begin:end] = True
    differentiable = differentiable_inputs(arguments)
    original_import = builtins.__import__

    def without_renderer(name, *args, **kwargs):
        if name == "solar.renderer" or name.startswith("solar.renderer."):
            raise AssertionError("H3 attention must not import the renderer extension")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_renderer)
    actual, actual_gradients = output_and_gradients(model, differentiable, attention_mask=plan)
    expected, expected_gradients = output_and_gradients(
        reference,
        differentiable_inputs(arguments),
        attention_mask=dense,
        fused_prope=False,
    )
    for actual_value, expected_value in zip(
        (*actual, *actual_gradients), (*expected, *expected_gradients), strict=True
    ):
        torch.testing.assert_close(actual_value, expected_value, rtol=3e-5, atol=3e-6)
    video_rows = arguments["video_indices"].tolist()
    first_output = video_rows.index(rows.target[0][0])
    future_rows = [rows.structure[1][0], rows.target[1][0]]
    if teacher:
        future_rows.append(rows.sequence_length + 1)
    future_inputs = [video_rows.index(row) for row in future_rows]
    gradient = torch.autograd.grad(
        actual[0][:, first_output].square().sum(),
        differentiable["hidden_states"],
    )[0]
    assert torch.count_nonzero(gradient[:, future_inputs]) == 0
    changed = dict(arguments)
    changed["hidden_states"] = arguments["hidden_states"].clone()
    changed["hidden_states"][:, future_inputs] += 10
    with torch.no_grad():
        changed_output = model(**changed, attention_mask=plan)[0]
    torch.testing.assert_close(
        changed_output[:, first_output], actual[0][:, first_output], rtol=0, atol=0
    )
