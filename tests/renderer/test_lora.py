from __future__ import annotations

import pytest
import torch

from solar.backends.minimax_h3.lora import merge_lora_delta, seed_lora_from_released
from solar.errors import BackendContractError


def test_merge_matches_explicit_lora_forward():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
    before = model[0].weight.detach().clone()
    down, up = torch.randn(2, 4), torch.randn(3, 2)
    state = {"0.lora_A.weight": down, "0.lora_B.weight": up}
    assert merge_lora_delta(model, state, alpha=4, rank=2, scale=0.5) == 1
    values = torch.randn(7, 4)
    torch.testing.assert_close(model(values), values @ before.T + values @ down.T @ up.T)


def test_invalid_lora_does_not_partially_modify_model():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
    before = model[0].weight.detach().clone()
    state = {
        "0.lora_A.weight": torch.ones(2, 4),
        "0.lora_B.weight": torch.ones(3, 2),
        "missing.lora_A.weight": torch.ones(2, 4),
        "missing.lora_B.weight": torch.ones(3, 2),
    }
    with pytest.raises(BackendContractError, match="absent"):
        merge_lora_delta(model, state, alpha=2, rank=2)
    torch.testing.assert_close(before, model[0].weight, rtol=0, atol=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_lora_is_rejected(value):
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
    with pytest.raises(BackendContractError, match="non-finite"):
        merge_lora_delta(
            model,
            {"0.lora_A.weight": torch.full((2, 4), value), "0.lora_B.weight": torch.ones(3, 2)},
            alpha=2,
            rank=2,
        )


def test_seed_wider_adapter_preserves_released_delta():
    live = {"layer.lora_A.weight": torch.randn(4, 3), "layer.lora_B.weight": torch.randn(2, 4)}
    released = {"layer.lora_A.weight": torch.randn(2, 3), "layer.lora_B.weight": torch.randn(2, 2)}
    assert seed_lora_from_released(live, released, live_scale=0.5, released_scale=2.0) == 1
    torch.testing.assert_close(
        0.5 * live["layer.lora_B.weight"] @ live["layer.lora_A.weight"],
        2.0 * released["layer.lora_B.weight"] @ released["layer.lora_A.weight"],
    )
