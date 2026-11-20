"""H3 attention speedups against the attention paths they replace.

* ``SOLARWM_H3_ATTENTION=cudnn`` (``fast_attention``): cuDNN fused SDPA instead of FA2.
* ``SOLARWM_H3_W6=segments`` and ``SOLARWM_H3_STAGE1_PREFIX_REUSE=1`` (``w6_attention``).
* ``SOLARWM_H3_SGF_PLAN``, ``SGF_PIECES``, ``SGF_PREFIX_REUSE``, ``SGF_KV_WINDOW`` and the SGF use
  of ``PROPE_FUSED`` (``sgf_attention``), at the attention level in one process.

Switches are toggled through their module attributes. Paths that only regroup or reuse the same
operations are compared bitwise; kernels with a different accumulation order must stay as close
to an FP64 reference as the path they replace.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from solar.backends.minimax_h3 import fast_attention, stage_scope, torch_prope  # noqa: E402
from solar.backends.minimax_h3 import sgf_attention as sa  # noqa: E402
from solar.backends.minimax_h3 import w6_attention as w6  # noqa: E402
from solar.backends.minimax_h3.layout import (  # noqa: E402
    build_stage0p5_layout,
    build_stage1_layout,
)
from solar.backends.minimax_h3.mask import build_stage1_window_mask  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

H3_MODULE = "diffusers.models.transformers.transformer_minimax_h3"
HEADS = 4


@pytest.fixture
def deterministic():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _generator(seed: int):
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(*shape, generator, scale=1.0):
    return (torch.randn(*shape, device="cuda", generator=generator) * scale).bfloat16()


def _rel(actual, reference) -> float:
    reference = reference.double()
    return ((actual.double() - reference).norm() / reference.norm()).item()


def _with_grads(fn, q, k, v, grad):
    leaves = [value.detach().clone().requires_grad_(True) for value in (q, k, v)]
    out = fn(*leaves)
    out.backward(grad)
    return out.detach(), *(leaf.grad for leaf in leaves)


def _sdpa(q, k, v, mask=None):
    """``[B, S, H, D]`` scaled dot-product attention (any dtype, PyTorch's backend choice).

    FP64 references run on the CPU: with torch 2.6 on an sm90 GPU the CUDA FP64 softmax returned
    wrong rows for some lengths (631, 751, 871), while CPU FP64 and CUDA FP32 agreed.
    """
    if q.dtype == torch.float64 and q.is_cuda:
        mask = None if mask is None else mask.cpu()
        return _sdpa(q.cpu(), k.cpu(), v.cpu(), mask).to(q.device)
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
    )
    return out.transpose(1, 2)


def _delta(counters, before):
    return {key: counters[key] - before.get(key, 0) for key in counters}


def _fp64(*tensors):
    return tuple(value.double() for value in tensors)


def _assert_accuracy(candidate, reference, gold, *, factor, names=("out", "dq", "dk", "dv")):
    for name, c, r, g in zip(names, candidate, reference, gold, strict=True):
        error, bound = _rel(c, g), _rel(r, g)
        assert error <= factor * bound, (
            f"{name}: FP64 distance {error:.4e} vs reference {bound:.4e}"
        )


# ------------------------------------------------------------------------- cuDNN attention


def _attention_inputs(sequence, keys, heads, seed):
    gen = _generator(seed)
    return (
        _randn(1, sequence, heads, 128, generator=gen, scale=2.0),
        _randn(1, keys, heads, 128, generator=gen, scale=2.0),
        _randn(1, keys, heads, 128, generator=gen),
        _randn(1, sequence, heads, 128, generator=gen),
    )


@cuda
@pytest.mark.parametrize(("sequence", "keys", "heads"), [(300, 700, 2), (1000, 1000, 4)])
def test_cudnn_attention_matches_cudnn_sdpa_and_flash_accuracy(sequence, keys, heads):
    flash_attn = pytest.importorskip("flash_attn")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q, k, v, grad = _attention_inputs(sequence, keys, heads, seed=10)
    before = fast_attention.COUNTERS["cudnn_forward"]
    ours = _with_grads(fast_attention.cudnn_attention, q, k, v, grad)
    assert fast_attention.COUNTERS["cudnn_forward"] == before + 1
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        sdpa = _with_grads(_sdpa, q, k, v, grad)
    flash = _with_grads(flash_attn.flash_attn_func, q, k, v, grad)
    gold = _with_grads(_sdpa, *_fp64(q, k, v, grad))

    assert torch.equal(ours[0], sdpa[0]), "forward differs from F.sdpa's cuDNN kernel"
    with torch.no_grad():
        assert torch.equal(fast_attention.cudnn_attention(q, k, v), ours[0])
    # Same algorithm family as FlashAttention-2: BF16 inputs, FP32 accumulation and LSE.
    _assert_accuracy(ours, flash, gold, factor=1.1)


@cuda
def test_cudnn_attention_deterministic_backward_is_reproducible(deterministic):
    flash_attn = pytest.importorskip("flash_attn")
    q, k, v, grad = _attention_inputs(500, 900, 2, seed=11)
    before = fast_attention.COUNTERS["deterministic_backward"]
    first = _with_grads(fast_attention.cudnn_attention, q, k, v, grad)
    second = _with_grads(fast_attention.cudnn_attention, q, k, v, grad)
    assert fast_attention.COUNTERS["deterministic_backward"] == before + 2
    for name, a, b in zip(("out", "dq", "dk", "dv"), first, second, strict=True):
        assert torch.equal(a, b), name
    flash = _with_grads(
        lambda *qkv: flash_attn.flash_attn_func(*qkv, deterministic=True), q, k, v, grad
    )
    gold = _with_grads(_sdpa, *_fp64(q, k, v, grad))
    _assert_accuracy(first, flash, gold, factor=1.1)


@cuda
def test_cudnn_attention_keep_attention_checkpoint_is_bitwise_to_plain(deterministic):
    pytest.importorskip("flash_attn")
    from solar.backends.minimax_h3.fsdp import block_checkpoint_wrapper

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(256, 3 * 256, bias=False)
            self.out = torch.nn.Linear(256, 256, bias=False)

        def forward(self, x):
            q, k, v = self.qkv(x).unflatten(-1, (3, 2, 128)).unbind(2)
            return x + self.out(fast_attention.cudnn_attention(q, k, v).flatten(2))

    x = _randn(1, 600, 256, generator=_generator(12))

    def run(save_attention):
        with torch.random.fork_rng():
            torch.manual_seed(1)
            blocks = torch.nn.ModuleList(Block() for _ in range(3)).to("cuda", torch.bfloat16)
        wrapped = [block_checkpoint_wrapper(b, save_attention=save_attention) for b in blocks]
        leaf = x.clone().requires_grad_(True)
        hidden = leaf
        for block in wrapped:
            hidden = block(hidden)
        hidden.float().square().mean().backward()
        return [hidden.detach(), leaf.grad, *(p.grad for p in blocks.parameters())]

    plain, kept = run(False), run(True)
    for index, (a, b) in enumerate(zip(plain, kept, strict=True)):
        assert torch.equal(a, b), f"tensor {index}"


def test_cudnn_switch_follows_the_stage_scope(monkeypatch):
    monkeypatch.setattr(stage_scope, "_CURRENT", None)
    monkeypatch.setattr(stage_scope, "_CUDNN_STAGES", None)
    monkeypatch.setattr(fast_attention, "_BACKEND", "flash")
    with fast_attention.cudnn_scope():
        assert not fast_attention.cudnn_enabled()
    monkeypatch.setattr(fast_attention, "_BACKEND", "cudnn")
    stage_scope.set_stage("stage2")
    assert fast_attention.cudnn_enabled()
    monkeypatch.setattr(stage_scope, "_CUDNN_STAGES", frozenset({"stage0p5", "stage1"}))
    assert not fast_attention.cudnn_enabled()
    with fast_attention.cudnn_scope():
        assert fast_attention.cudnn_enabled()
    assert not fast_attention.cudnn_enabled()
    stage_scope.set_stage("stage1")
    assert fast_attention.cudnn_enabled()


# --------------------------------------------------------------------------------- W6 (Stage1)


def _w6_layout():
    # 14 text + 16 anchor + 18 audio + 2 x 720 target rows = 1488; nine 5-frame chunks, so the
    # six-chunk window excludes old clean chunks.
    return build_stage1_layout(np.ones(14, dtype=np.int64), 45, 8, 8, 9)


def test_w6_plan_covers_exactly_the_stage1_window_mask():
    source = _w6_layout()
    plan = w6.build_w6_plan(source.to("cpu"))
    covered = np.zeros((plan.sequence_length, plan.sequence_length), dtype=bool)
    for start, stop, spans in plan.segments:
        for begin, end in spans:
            assert not covered[start:stop, begin:end].any(), "key spans overlap"
            covered[start:stop, begin:end] = True
    np.testing.assert_array_equal(covered, build_stage1_window_mask(source))


def _w6_case():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    source = _w6_layout()
    layout = source.to("cuda")
    plan = w6.build_w6_plan(layout)
    mask = torch.from_numpy(build_stage1_window_mask(source)).to("cuda")
    q, k, v, grad = _attention_inputs(plan.sequence_length, plan.sequence_length, HEADS, seed=20)

    def masked(*qkv):
        return _sdpa(*qkv, mask=mask)

    # The same masked attention as one fused BF16 kernel (FP32 accumulation, one rounding).
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        reference = _with_grads(masked, q, k, v, grad)
    gold = _with_grads(masked, *_fp64(q, k, v, grad))
    return layout, plan, (q, k, v, grad), reference, gold


@cuda
def test_w6_segment_attention_matches_masked_attention_accuracy():
    _, plan, inputs, reference, gold = _w6_case()
    before = dict(w6.COUNTERS)
    ours = _with_grads(lambda *qkv: w6.w6_attention(*qkv, plan), *inputs)
    assert _delta(w6.COUNTERS, before) == {"forward": 1, "backward": 1}
    # Per-run dK/dV are rounded to BF16 before their FP32 sum over runs (WORK_LOG r02: 1.13x
    # and 1.21x the compiled Flex reference's dK/dV FP64 distance).
    _assert_accuracy(ours, reference, gold, factor=1.5)


@cuda
def test_w6_deterministic_backward_is_reproducible(deterministic):
    pytest.importorskip("flash_attn")
    _, plan, inputs, reference, gold = _w6_case()
    first = _with_grads(lambda *qkv: w6.w6_attention(*qkv, plan), *inputs)
    second = _with_grads(lambda *qkv: w6.w6_attention(*qkv, plan), *inputs)
    for name, a, b in zip(("out", "dq", "dk", "dv"), first, second, strict=True):
        assert torch.equal(a, b), name
    _assert_accuracy(first, reference, gold, factor=1.5)


@cuda
def test_w6_prefix_reuse_is_bitwise_to_the_full_document():
    layout = _w6_layout().to("cuda")
    plan = w6.build_w6_plan(layout)
    suffix = w6.H3W6Suffix(plan, layout)
    prefix = suffix.prefix_rows
    gen = _generator(21)
    first = [_randn(1, plan.sequence_length, HEADS, 128, generator=gen) for _ in range(3)]
    # A later AnyFlow evaluation: other noisy rows, the same condition/audio/clean rows.
    noisy = [_randn(1, suffix.suffix_rows, HEADS, 128, generator=gen) for _ in range(3)]
    second = [
        torch.cat((value[:, :prefix], rows), dim=1)
        for value, rows in zip(first, noisy, strict=True)
    ]
    cache = w6.H3W6PrefixCache()
    before = dict(w6.REUSE_COUNTERS)
    with torch.no_grad():
        full_first = w6.w6_attention(*first, plan)
        full_second = w6.w6_attention(*second, plan)
        captured = w6.w6_run_attention(*first, w6.H3W6Run(plan, suffix, cache, "capture"), 0)
        reused = w6.w6_run_attention(*noisy, w6.H3W6Run(plan, suffix, cache, "suffix"), 0)
    assert w6.REUSE_COUNTERS["captured_layers"] == before["captured_layers"] + 1
    assert w6.REUSE_COUNTERS["suffix_layers"] == before["suffix_layers"] + 1
    assert torch.equal(captured, full_first)
    assert torch.equal(reused, full_second[:, prefix:])


# ------------------------------------------------------------------------------ SGF (Stage2)

SGF_TAGS = np.ones(61, dtype=np.int64)
SGF_GRID = (8, 12)
SGF_AUDIO = 33


@pytest.fixture
def sgf(monkeypatch):
    """Rotary module, cameras and attention kernels of the SGF tests (upstream switches)."""
    h3 = pytest.importorskip(H3_MODULE, exc_type=ImportError)
    pytest.importorskip("triton")  # the planned SGF path imports the fused PRoPE module
    flash_attn = pytest.importorskip("flash_attn")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from solar.backends.minimax_h3 import fused_prope

    for name, value in (("_PLAN_ENABLED", False), ("_PIECES", False), ("_KV_WINDOW", False)):
        monkeypatch.setattr(sa, name, value)
    monkeypatch.setattr(fused_prope, "_ENABLED", False)
    gen = _generator(30)
    views = torch.eye(4, device="cuda").repeat(1, 51, 1, 1)
    rotations = torch.randn(1, 51, 3, 3, device="cuda", generator=gen)
    views[..., :3, :3] = torch.linalg.qr(rotations)[0]
    views[..., :3, 3] = torch.randn(1, 51, 3, device="cuda", generator=gen)
    intrinsics = torch.eye(3, device="cuda").repeat(1, 51, 1, 1) * 0.9
    intrinsics[..., 2, 2] = 1.0

    def attend(q, k, v):
        return flash_attn.flash_attn_func(
            q.contiguous(), k.contiguous(), v.contiguous(), deterministic=True
        )

    rope = h3.MiniMaxH3RotaryPosEmbed(16).to("cuda")

    def control(layout, mode, *, plan, **fields):
        monkeypatch.setattr(sa, "_PLAN_ENABLED", plan)
        value = sa.H3SGFAttention(layout, views, intrinsics, mode, **fields)
        return sa.prepare_sgf_rotaries(value, rope)

    def attention(q, k, v, control, layer):
        return sa.sgf_attention(
            q, k, v, control=control, layer=layer, apply_rotary=h3._apply_rotary_emb, attend=attend
        )

    def gold(q, k, v, control, layer):
        """Upstream SGF attention in FP64 on the BF16 paths' own camera matrices and rotary
        tables (both are computed in the activation dtype), so that only the rounding of the
        activations separates the BF16 paths from it."""
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sa, "prope_qkv_separate", _bf16_camera_prope)
            return sa.sgf_attention(
                *_fp64(q, k, v),
                control=control,
                layer=layer,
                apply_rotary=lambda x, cos, sin: h3._apply_rotary_emb(
                    x, cos.bfloat16(), sin.bfloat16()
                ),
                attend=_sdpa,
            )

    return SimpleNamespace(control=control, attention=attention, gold=gold, prope=fused_prope)


def _bf16_camera_prope(q, k, v, *, query_viewmats, query_K, key_viewmats, key_K):
    """``prope_qkv_separate`` with the camera arithmetic in BF16 and the projection in the
    dtype of q/k/v."""

    def transforms(views, intrinsics):
        return torch_prope._prepare_suffix_transforms(
            viewmats=torch_prope.logd4_relative_viewmats(views.to(torch.bfloat16)),
            Ks=intrinsics.to(torch.bfloat16),
        )

    def suffix(features, transform):
        return torch.cat((features[..., :96], transform(features[..., 96:128])), dim=-1)

    apply_q, _, apply_output = transforms(query_viewmats, query_K)
    _, apply_kv, _ = transforms(key_viewmats, key_K)
    return (
        suffix(q, apply_q),
        suffix(k, apply_kv),
        suffix(v, apply_kv),
        partial(suffix, transform=apply_output),
    )


def _sgf_layout(mode):
    if mode == "replay":
        return build_stage1_layout(SGF_TAGS, 50, *SGF_GRID, SGF_AUDIO).to("cuda")
    return build_stage0p5_layout(SGF_TAGS, 5, *SGF_GRID, SGF_AUDIO).to("cuda")


def _replay(sgf, control, q, k, v, grad, layer=0, *, gold=False):
    attention = sgf.gold if gold else sgf.attention
    return _with_grads(lambda *qkv: attention(*qkv, control, layer), q, k, v, grad)


def test_sgf_forward_plan_is_bitwise_to_upstream_attention(sgf):
    gen = _generator(31)
    before = dict(sa.PLAN_COUNTERS)
    for mode, chunks in (("rollout", (3, 7)), ("replay", (0,))):
        layout = _sgf_layout(mode)
        rows, sequence = layout.rows_per_video_frame, layout.sequence_length
        for chunk in chunks:
            cache = sa.H3RawKVCache()
            for index in range(max(0, chunk - 5), chunk):
                for layer in range(2):
                    history = [_randn(1, 5 * rows, HEADS, 128, generator=gen) for _ in range(2)]
                    cache.layers.setdefault(layer, []).append((index, *history))
            fields = {"chunk_index": chunk, "cache": cache, "commit_cache": False}
            upstream = sgf.control(layout, mode, plan=False, **fields)
            planned = sgf.control(layout, mode, plan=True, **fields)
            assert upstream.plan is None and planned.plan is not None
            for layer in range(2):
                q, k, v, grad = (_randn(1, sequence, HEADS, 128, generator=gen) for _ in range(4))
                runs = []
                for control in (upstream, planned, planned):  # the last reuses the plan
                    if mode == "replay":
                        runs.append(_replay(sgf, control, q, k, v, grad, layer))
                    else:
                        with torch.no_grad():
                            runs.append((sgf.attention(q, k, v, control, layer),))
                for other in runs[1:]:
                    for a, b in zip(runs[0], other, strict=True):
                        assert torch.equal(a, b), f"{mode} chunk {chunk} layer {layer}"
    # Three planned controls, each built once and reused in the other three calls.
    assert _delta(sa.PLAN_COUNTERS, before) == {"built": 3, "reused": 9}


@pytest.mark.parametrize(("pieces", "fused"), [(True, False), (False, True), (True, True)])
def test_sgf_replay_pieces_and_fused_prope_keep_fp64_accuracy(sgf, monkeypatch, pieces, fused):
    layout = _sgf_layout("replay")
    gen = _generator(32)
    q, k, v, grad = (_randn(1, layout.sequence_length, HEADS, 128, generator=gen) for _ in range(4))
    planned = sgf.control(layout, "replay", plan=True)
    baseline = _replay(sgf, planned, q, k, v, grad)
    monkeypatch.setattr(sa, "_PIECES", pieces)
    monkeypatch.setattr(sgf.prope, "_ENABLED", fused)
    pieced = sa.PIECES_COUNTERS["replay_layers"]
    rope_prope = sgf.prope.COUNTERS.get("rope_prope", 0)
    candidate = _replay(sgf, planned, q, k, v, grad)
    assert sa.PIECES_COUNTERS["replay_layers"] == pieced + int(pieces)
    assert (sgf.prope.COUNTERS.get("rope_prope", 0) > rope_prope) == fused
    if not fused:
        # Pieces take the same rows from one split: the same forward operations.
        assert torch.equal(candidate[0], baseline[0])
    upstream = sgf.control(layout, "replay", plan=False)
    gold = _replay(sgf, upstream, *_fp64(q, k, v, grad), gold=True)
    _assert_accuracy(candidate, baseline, gold, factor=1.02)


@pytest.mark.parametrize("fused", [False, True])
def test_sgf_rollout_prefix_reuse_and_kv_window(sgf, monkeypatch, fused):
    layout = _sgf_layout("rollout")
    chunk_rows = 5 * layout.rows_per_video_frame
    prefix = layout.sequence_length - chunk_rows
    layers, chunks, steps = 2, 7, 2
    gen = _generator(33)
    prefix_rows = {
        layer: [_randn(1, prefix, HEADS, 128, generator=gen) for _ in range(3)]
        for layer in range(layers)
    }
    chunk_inputs = {
        (chunk, step, layer): [_randn(1, chunk_rows, HEADS, 128, generator=gen) for _ in range(3)]
        for chunk in range(chunks)
        for step in range(steps + 1)
        for layer in range(layers)
    }

    def rollout(*, plan, reuse, window, gold=False):
        """Denoising evaluations and a commit per chunk; the prefix rows never change."""
        monkeypatch.setattr(sa, "_KV_WINDOW", window)
        cache = sa.H3RawKVCache()
        prefix_kv = sa.H3SGFPrefixKV() if reuse else None
        outputs = []
        with torch.no_grad():
            for chunk in range(chunks):
                for step in range(steps + 1):
                    control = sgf.control(
                        layout,
                        "rollout",
                        plan=plan,
                        chunk_index=chunk,
                        cache=cache,
                        commit_cache=step == steps,
                        prefix_kv=prefix_kv,
                    )
                    suffix = prefix_kv is not None and prefix_kv.mode == "suffix"
                    for layer in range(layers):
                        rows = chunk_inputs[(chunk, step, layer)]
                        if not suffix:
                            rows = [
                                torch.cat((p, c), dim=1)
                                for p, c in zip(prefix_rows[layer], rows, strict=True)
                            ]
                        out = (sgf.gold if gold else sgf.attention)(*rows, control, layer)
                        outputs.append(out[:, -chunk_rows:])
                    if prefix_kv is not None:
                        prefix_kv.mode = "suffix"
        return torch.cat(outputs, dim=1)

    upstream = rollout(plan=False, reuse=False, window=False)
    monkeypatch.setattr(sgf.prope, "_ENABLED", fused)
    captured = sa.PREFIX_REUSE_COUNTERS["captured_layers"]
    reused = rollout(plan=True, reuse=True, window=False)
    assert sa.PREFIX_REUSE_COUNTERS["captured_layers"] == captured + layers
    window = dict(sa.KV_WINDOW_COUNTERS)
    windowed = rollout(plan=True, reuse=True, window=True)
    # Built at the first suffix evaluation of every chunk, reused in its later ones.
    assert _delta(sa.KV_WINDOW_COUNTERS, window) == {
        "built": chunks * layers,
        "reused": (chunks * steps - 1) * layers,
    }
    assert torch.equal(windowed, reused)
    if fused:
        monkeypatch.setattr(sgf.prope, "_ENABLED", False)
        gold = rollout(plan=False, reuse=False, window=False, gold=True)
        _assert_accuracy((reused,), (upstream,), (gold,), factor=1.02, names=("out",))
    else:
        assert torch.equal(reused, upstream)
