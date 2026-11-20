"""H3 Triton speedup kernels against the eager torch chains they replace (single GPU).

Each kernel is called directly; its switch (``SOLARWM_H3_FUSED``, ``SOLARWM_H3_SWIGLU_FUSED``,
``SOLARWM_H3_PROPE_FUSED``, ``SOLARWM_H3_PROPE_V2``, ``SOLARWM_H3_A2A_COPY``) only selects it at
the call sites. Kernels that keep every BF16 rounding of the eager chain are compared bitwise.
Tier-N1 kernels (a reordered FP32 reduction) may differ from eager on rare elements; they must
stay as close to an FP64 reference as the eager chain.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from solar.backends.minimax_h3 import copy_kernels, fused_ops, fused_prope, torch_prope  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the H3 Triton kernels need a CUDA device"
)

H3_MODULE = "diffusers.models.transformers.transformer_minimax_h3"


def _generator(seed: int):
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(*shape, generator, scale=1.0, shift=0.0):
    values = torch.randn(*shape, device="cuda", generator=generator) * scale + shift
    return values.bfloat16()


def _rel(actual, reference) -> float:
    reference = reference.double()
    return ((actual.double() - reference).norm() / reference.norm()).item()


def _with_grad(fn, *inputs, grad):
    leaves = [value.detach().clone().requires_grad_(True) for value in inputs]
    out = fn(*leaves)
    out.backward(grad)
    return out.detach(), *(leaf.grad for leaf in leaves)


def _assert_n1(fused, eager, gold, label, *, max_mismatch=1e-3, slack=0.02):
    """Rare last-bit differences only, and no further from FP64 than the eager chain."""
    mismatch = (fused != eager).double().mean().item()
    fused_error, eager_error = _rel(fused, gold), _rel(eager, gold)
    assert mismatch <= max_mismatch, f"{label}: {mismatch:.3e} of the elements differ from eager"
    assert fused_error <= eager_error * (1 + slack), (
        f"{label}: FP64 distance fused {fused_error:.5e} vs eager {eager_error:.5e}"
    )


def _assert_within_one_ulp(actual, expected, label):
    """Results of a single BF16 rounding of an FP32 sum whose last bit may differ."""
    actual, expected = actual.float(), expected.float()
    _, exponent = torch.frexp(torch.maximum(actual.abs(), expected.abs()))
    ulp = torch.ldexp(torch.ones_like(actual), exponent - 8)
    excess = ((actual - expected).abs() - ulp).max().item()
    assert excess <= 0, f"{label}: a difference exceeds one BF16 ulp by {excess:.3e}"


def _rms_norm(width: int, eps: float, generator):
    norm = torch.nn.RMSNorm(width, eps=eps).to("cuda", torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(width, device="cuda", generator=generator) * 0.2 + 1)
    return norm.requires_grad_(False)


def _fp64_rms_norm(x, norm):
    eps = norm.eps
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm.weight.double()


@pytest.mark.parametrize(("rows", "width"), [(515, 5376), (300, 128)])
def test_rmsnorm_modulate_matches_eager_within_n1(rows, width):
    gen = _generator(0)
    norm = _rms_norm(width, 1e-5, gen)
    scale = _randn(18, width, generator=gen, scale=0.3)
    shift = _randn(18, width, generator=gen, scale=0.3)
    index = torch.randint(0, 18, (rows,), device="cuda", generator=gen)
    x = _randn(1, rows, width, generator=gen, scale=3.0, shift=0.5)
    grad = _randn(1, rows, width, generator=gen)

    def eager(value):
        modulation = 1.0 + scale.index_select(0, index)
        return norm(value) * modulation + shift.index_select(0, index)

    before = fused_ops.COUNTERS["rmsnorm_modulate"]
    eager_out, eager_grad = _with_grad(eager, x, grad=grad)
    fused_out, fused_grad = _with_grad(
        lambda value: fused_ops.rmsnorm_modulate(norm, value, scale, shift, index), x, grad=grad
    )
    assert fused_ops.COUNTERS["rmsnorm_modulate"] == before + 1

    xd = x.double().requires_grad_(True)
    gold = _fp64_rms_norm(xd, norm) * (1.0 + scale.double().index_select(0, index))
    gold = gold + shift.double().index_select(0, index)
    gold.backward(grad.double())
    _assert_n1(fused_out, eager_out, gold.detach(), "forward")
    _assert_n1(fused_grad, eager_grad, xd.grad, "input gradient", max_mismatch=1e-2)


@pytest.mark.parametrize("width", [5376, 1000])
def test_gated_residual_is_bitwise_to_eager(width):
    gen = _generator(1)
    rows = 515
    gate = _randn(18, width, generator=gen, scale=0.3)
    index = torch.randint(0, 18, (rows,), device="cuda", generator=gen)
    residual = _randn(1, rows, width, generator=gen)
    update = _randn(1, rows, width, generator=gen)
    grad = _randn(1, rows, width, generator=gen)

    before = fused_ops.COUNTERS["gated_residual"]
    expected = _with_grad(
        lambda r, u: r + gate.index_select(0, index) * u, residual, update, grad=grad
    )
    actual = _with_grad(
        lambda r, u: fused_ops.gated_residual(r, gate, index, u), residual, update, grad=grad
    )
    assert fused_ops.COUNTERS["gated_residual"] == before + 1
    for name, a, e in zip(("out", "d residual", "d update"), actual, expected, strict=True):
        assert torch.equal(a, e), name


def test_qk_norm_rope_matches_eager_within_n1():
    h3 = pytest.importorskip(H3_MODULE, exc_type=ImportError)
    gen = _generator(2)
    rows, heads = 257, 4
    rope = h3.MiniMaxH3RotaryPosEmbed(16).to("cuda")
    cos, sin = rope(torch.rand(rows, 3, device="cuda", generator=gen) * 40)
    norm = _rms_norm(128, 1e-5, gen)
    x = _randn(1, rows, heads, 128, generator=gen, scale=2.0, shift=0.3)
    grad = _randn(1, rows, heads, 128, generator=gen)

    before = fused_ops.COUNTERS["qk_norm_rope"]
    eager_out, eager_grad = _with_grad(
        lambda value: h3._apply_rotary_emb(norm(value), cos, sin), x, grad=grad
    )
    fused_out, fused_grad = _with_grad(
        lambda value: fused_ops.qk_norm_rope(norm, value, cos, sin), x, grad=grad
    )
    assert fused_ops.COUNTERS["qk_norm_rope"] == before + 1

    xd = x.double().requires_grad_(True)
    gold = h3._apply_rotary_emb(_fp64_rms_norm(xd, norm), cos.double(), sin.double())
    gold.backward(grad.double())
    _assert_n1(fused_out, eager_out, gold.detach(), "forward")
    _assert_n1(fused_grad, eager_grad, xd.grad, "input gradient", max_mismatch=1e-2)


def test_qk_norm_matches_rmsnorm_within_n1():
    gen = _generator(3)
    norm = _rms_norm(128, 1e-6, gen)
    x = _randn(1, 300, 4, 128, generator=gen, scale=3.0)
    grad = _randn(1, 300, 4, 128, generator=gen)

    before = fused_ops.COUNTERS.get("qk_norm", 0)
    eager_out, eager_grad = _with_grad(norm, x, grad=grad)
    fused_out, fused_grad = _with_grad(lambda value: fused_ops.qk_norm(norm, value), x, grad=grad)
    assert fused_ops.COUNTERS.get("qk_norm", 0) == before + 1

    xd = x.double().requires_grad_(True)
    gold = _fp64_rms_norm(xd, norm)
    gold.backward(grad.double())
    _assert_n1(fused_out, eager_out, gold.detach(), "forward")
    _assert_n1(fused_grad, eager_grad, xd.grad, "input gradient", max_mismatch=1e-2)


@pytest.mark.parametrize("width", [2600, 1024])
def test_swiglu_is_bitwise_to_eager(width):
    gen = _generator(4)
    projected = _randn(1, 257, 2 * width, generator=gen, scale=2.0)
    grad = _randn(1, 257, width, generator=gen)

    def eager(value):
        hidden, gate = value.chunk(2, dim=-1)
        return hidden * torch.nn.functional.silu(gate)

    before = fused_ops.COUNTERS.get("swiglu", 0)
    expected = _with_grad(eager, projected, grad=grad)
    actual = _with_grad(fused_ops.swiglu, projected, grad=grad)
    assert fused_ops.COUNTERS.get("swiglu", 0) == before + 1
    for name, a, e in zip(("out", "input gradient"), actual, expected, strict=True):
        assert torch.equal(a, e), name


def _prope_matrices(sequence: int, generator):
    views = torch.eye(4, device="cuda").repeat(1, sequence, 1, 1)
    rotations = torch.randn(1, sequence, 3, 3, device="cuda", generator=generator)
    views[..., :3, :3] = torch.linalg.qr(rotations)[0]
    views[..., :3, 3] = torch.randn(1, sequence, 3, device="cuda", generator=generator) * 3
    intrinsics = torch.eye(3, device="cuda").repeat(1, sequence, 1, 1)
    return torch_prope.prope_matrices(views.bfloat16(), intrinsics.bfloat16())


def _einsum_suffix(features, matrix):
    return torch.cat(
        (features[..., :96], torch_prope._project(features[..., 96:128], matrix=matrix)), dim=-1
    )


def _fp64_projection(features, matrix, *, transpose=False):
    """FP64 ``M x`` (or ``M^T x``) of every 4-vector of the channels [96, 128)."""
    batch, heads, sequence, _ = features.shape
    tiled = features[..., 96:].double().reshape(batch, heads, sequence, 8, 4)
    equation = "bsji,bhspj->bhspi" if transpose else "bsij,bhspj->bhspi"
    projected = torch.einsum(equation, matrix.double(), tiled).reshape(batch, heads, sequence, 32)
    return torch.cat((features[..., :96].double(), projected), dim=-1)


def _assert_projection_n1(fused, eager, gold, label):
    """Channels [0, 96) are copied (bitwise); [96, 128) are one BF16 rounding of a 4-term sum."""
    assert torch.equal(fused[..., :96], eager[..., :96]), f"{label}: copied channels differ"
    _assert_within_one_ulp(fused[..., 96:], eager[..., 96:], label)
    _assert_n1(fused, eager, gold, label)


@pytest.mark.parametrize(("sequence", "heads"), [(517, 4), (1797, 2)])
def test_prope_project_matches_einsum_path_within_n1(sequence, heads, monkeypatch):
    monkeypatch.setattr(fused_prope, "_V2", False)
    gen = _generator(5)
    matrices = _prope_matrices(sequence, gen)
    # BSHD memory viewed as BHSD, as the model passes q/k/v.
    x = _randn(1, sequence, heads, 128, generator=gen).transpose(1, 2)
    grad = _randn(1, sequence, heads, 128, generator=gen).transpose(1, 2)
    for name, matrix in zip(("query", "key/value", "output"), matrices, strict=True):
        before = fused_prope.COUNTERS["fused"]
        eager_out, eager_grad = _with_grad(
            lambda value, m=matrix: _einsum_suffix(value, m), x, grad=grad
        )
        fused_out, fused_grad = _with_grad(
            lambda value, m=matrix: fused_prope.prope_project(value, m), x, grad=grad
        )
        assert fused_prope.COUNTERS["fused"] == before + 1
        _assert_projection_n1(fused_out, eager_out, _fp64_projection(x, matrix), f"{name} fwd")
        gold_grad = _fp64_projection(grad, matrix, transpose=True)
        _assert_projection_n1(fused_grad, eager_grad, gold_grad, f"{name} bwd")
    repeat = fused_prope.prope_project(x, matrices[0])
    assert torch.equal(repeat, fused_prope.prope_project(x, matrices[0]))


def _eager_rope_prope(value, cos, sin, matrix, apply_rotary):
    rotated = apply_rotary(value, cos, sin).transpose(1, 2)
    return _einsum_suffix(rotated, matrix).transpose(1, 2)


def test_rope_prope_matches_eager_rotary_and_projection():
    h3 = pytest.importorskip(H3_MODULE, exc_type=ImportError)
    gen = _generator(6)
    sequence, heads = 517, 4
    rope = h3.MiniMaxH3RotaryPosEmbed(16).to("cuda")
    # The planned SGF path passes rotary tables already cast to the activation dtype.
    cos, sin = (
        table.bfloat16() for table in rope(torch.rand(sequence, 3, device="cuda", generator=gen))
    )
    _, matrix, _ = _prope_matrices(sequence, gen)
    x = _randn(1, sequence, heads, 128, generator=gen)
    grad = _randn(1, sequence, heads, 128, generator=gen)

    before = fused_prope.COUNTERS.get("rope_prope", 0)
    eager_out, eager_grad = _with_grad(
        lambda value: _eager_rope_prope(value, cos, sin, matrix, h3._apply_rotary_emb),
        x,
        grad=grad,
    )
    fused_out, fused_grad = _with_grad(
        lambda value: fused_prope.rope_prope(value, cos, sin, matrix), x, grad=grad
    )
    assert fused_prope.COUNTERS.get("rope_prope", 0) == before + 1

    xd = x.double().requires_grad_(True)
    gold = _eager_rope_prope(xd, cos.double(), sin.double(), matrix.double(), h3._apply_rotary_emb)
    gold.backward(grad.double())
    # MM-RoPE on [0, 96) keeps every rounding of _apply_rotary_emb; PRoPE on [96, 128) is N1.
    for name, fused, eager, reference in (
        ("forward", fused_out, eager_out, gold.detach()),
        ("input gradient", fused_grad, eager_grad, xd.grad),
    ):
        _assert_projection_n1(fused, eager, reference, name)


@pytest.mark.parametrize(("sequence", "heads"), [(517, 4), (1797, 2)])
def test_prope_v2_kernels_are_bitwise_to_v1(sequence, heads, monkeypatch):
    gen = _generator(7)
    query_matrix, kv_matrix, _ = _prope_matrices(sequence, gen)
    x = _randn(1, sequence, heads, 128, generator=gen)
    grad = _randn(1, sequence, heads, 128, generator=gen)
    cos = _randn(sequence, 96, generator=gen)
    sin = _randn(sequence, 96, generator=gen)
    results = {}
    for v2 in (False, True):
        monkeypatch.setattr(fused_prope, "_V2", v2)
        projected = _with_grad(
            lambda value: fused_prope.prope_project(value, query_matrix),
            x.transpose(1, 2),
            grad=grad.transpose(1, 2),
        )
        rotated = _with_grad(
            lambda value: fused_prope.rope_prope(value, cos, sin, kv_matrix), x, grad=grad
        )
        with torch.no_grad():
            into = torch.empty_like(x)
            fused_prope.rope_prope(x, cos, sin, kv_matrix, out=into)
        results[v2] = (*projected, *rotated, into)
    names = ("prope fwd", "prope bwd", "rope_prope fwd", "rope_prope bwd", "rope_prope out=")
    for name, v1, v2 in zip(names, results[False], results[True], strict=True):
        assert torch.equal(v1, v2), name


@pytest.mark.parametrize(("outer", "inner", "width"), [(300, 2, 1792), (77, 4, 3000)])
def test_block_transpose_is_a_pure_copy(outer, inner, width):
    source = _randn(outer, inner, width, generator=_generator(8))
    actual = copy_kernels.block_transpose(source, outer, inner, width)
    assert actual.shape == (inner, outer, width)
    assert torch.equal(actual, source.transpose(0, 1).contiguous())
