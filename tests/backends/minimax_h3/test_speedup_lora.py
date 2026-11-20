"""PEFT LoRA forward speedups (``SOLARWM_H3_LORA_FAST``, ``SOLARWM_H3_LORA_ADDMM``) against PEFT."""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from peft.tuners.lora import layer as lora_layer  # noqa: E402

from solar.backends.minimax_h3 import lora as h3_lora  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def install_fast_forward(monkeypatch):
    """``install_fast_lora_forward`` with PEFT's own forward restored afterwards."""
    if getattr(lora_layer.Linear, "_solarwm_fast_forward", False):
        pytest.skip("PEFT's forward was already replaced at import (SOLARWM_H3_LORA_FAST=1)")
    monkeypatch.setattr(lora_layer.Linear, "forward", lora_layer.Linear.forward)
    monkeypatch.setattr(lora_layer.Linear, "_solarwm_fast_forward", False, raising=False)
    return h3_lora.install_fast_lora_forward


def _lora_model(widths, *, rank, alpha):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        layers = [torch.nn.Linear(a, b, bias=False) for a, b in itertools.pairwise(widths)]
        net = torch.nn.Sequential(*layers).to(DEVICE, torch.bfloat16)
        config = peft.LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules=[str(index) for index in range(len(layers))],
            bias="none",
            init_lora_weights=True,
        )
        model = peft.get_peft_model(net, config, autocast_adapter_dtype=False)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = (torch.randn_like(parameter.data.float()) * 0.05).bfloat16()
    return model


def _run(model, x, grad=None):
    model.zero_grad(set_to_none=True)
    leaf = x.clone().requires_grad_(True)
    out = model(leaf)
    if grad is None:
        out.float().square().mean().backward()
    else:
        out.backward(grad)
    trainable = [p.grad for p in model.parameters() if p.requires_grad]
    return [out.detach(), leaf.grad, *trainable]


def _input(*shape, seed=1):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(*shape, device=DEVICE, generator=generator).bfloat16()


def test_fast_lora_forward_is_bitwise_to_peft(install_fast_forward, monkeypatch):
    monkeypatch.setattr(h3_lora, "_ADDMM", False)
    model = _lora_model((512, 768, 512), rank=64, alpha=64)
    x = _input(4, 300, 512)
    expected = _run(model, x)
    install_fast_forward()
    before = dict(h3_lora._FAST_FORWARD_COUNTERS)
    actual = _run(model, x)
    assert h3_lora._FAST_FORWARD_COUNTERS["fast"] == before["fast"] + 2
    assert h3_lora._FAST_FORWARD_COUNTERS["peft"] == before["peft"]
    for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(a, e), f"tensor {index} (out, dx, LoRA grads)"


def test_fast_lora_forward_keeps_peft_when_scaling_is_not_one(install_fast_forward):
    model = _lora_model((512, 768, 512), rank=8, alpha=16)
    x = _input(2, 100, 512)
    expected = _run(model, x)
    install_fast_forward()
    before = dict(h3_lora._FAST_FORWARD_COUNTERS)
    actual = _run(model, x)
    assert h3_lora._FAST_FORWARD_COUNTERS["peft"] == before["peft"] + 2
    assert h3_lora._FAST_FORWARD_COUNTERS["fast"] == before["fast"]
    for a, e in zip(actual, expected, strict=True):
        assert torch.equal(a, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_lora_addmm_epilogue_is_closer_to_fp64_with_identical_gradients(
    install_fast_forward, monkeypatch
):
    model = _lora_model((1024, 1536), rank=64, alpha=64)
    x = _input(1, 2048, 1024)
    grad = _input(1, 2048, 1536, seed=2)
    install_fast_forward()
    monkeypatch.setattr(h3_lora, "_ADDMM", False)
    two_adds = _run(model, x, grad)
    monkeypatch.setattr(h3_lora, "_ADDMM", True)
    before = h3_lora._FAST_FORWARD_COUNTERS.get("addmm", 0)
    epilogue = _run(model, x, grad)
    assert h3_lora._FAST_FORWARD_COUNTERS.get("addmm", 0) == before + 1

    layer = model.base_model.model[0]
    base = layer.base_layer.weight.double()
    down = layer.lora_A["default"].weight.double()
    up = layer.lora_B["default"].weight.double()
    xd = x.double()
    gold = xd @ base.t() + (xd @ down.t()) @ up.t()

    def rel(value):
        return ((value.double() - gold).norm() / gold.norm()).item()

    # One BF16 rounding of base + B(A x) instead of rounding B(A x) and then the sum.
    assert rel(epilogue[0]) <= rel(two_adds[0]), (rel(epilogue[0]), rel(two_adds[0]))
    for name, a, b in zip(("dx", "dA", "dB"), epilogue[1:], two_adds[1:], strict=True):
        assert torch.equal(a, b), name
