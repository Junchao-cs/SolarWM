"""Exercise Reactor's integration with the native causal layout, sampler and source."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from solar.backends.minimax_h3.flow import make_shifted_schedule
from solar.backends.minimax_h3.layout import H3VisualObservationSpec, build_stage0p5_layout
from solar.backends.minimax_h3.segmented_attention import H3SegmentPlan, segment_attention
from solar.renderer.causal import H3OmniCausalMixin, causal_rows

torch = pytest.importorskip("torch")

ADAPTER = Path(__file__).resolve().parents[2] / "deploy/solar_renderer/reactor"


@pytest.fixture
def reactor_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ADAPTER))
    return SimpleNamespace(
        sampling=importlib.import_module("solarrenderer_sampling"),
        optimizations=importlib.import_module("solarrenderer_optimizations"),
        assets=importlib.import_module("solarrenderer_assets"),
        staging=importlib.import_module("stage_source"),
    )


class SmallCausalCore(H3OmniCausalMixin):
    """Keep the real row/cache operations while replacing large model projections."""

    device = torch.device("cpu")
    causal: ClassVar = {"chunk": 5, "window": 6}

    def _document(self, reference, generator):
        frames = 47
        specs = [H3VisualObservationSpec(frames, 2, 2, tuple(range(frames)))]
        if reference:
            specs.append(H3VisualObservationSpec(1, 2, 2, (0,)))
        source = build_stage0p5_layout(
            [1, 1], frames, 2, 2, 2, keyframe_anchors=(), observations=tuple(specs)
        )
        layout = SimpleNamespace(
            source=source,
            token_tags=torch.from_numpy(source.token_tags),
            position_ids=torch.from_numpy(source.position_ids),
        )
        prompt = torch.randn(1, 2, 4, generator=generator)
        observations = torch.randn(1, source.num_condition_video_rows, 96, generator=generator)
        rows = causal_rows(source, 0, 5)
        return prompt, layout, observations, rows, (frames, (2, 2))

    def _audio(self, frames, generator):
        return torch.randn(1, 4, 4, generator=generator)

    @staticmethod
    def audio_latents(frames):
        return 2  # Two rows per channel in this small document.

    def _broadcast_logical_sample(self, value):
        pass

    def _run(self, *, video_rows, audio_rows, prompt, arrays, times, plan, model=None):
        packed = torch.zeros(1, len(times), 4)
        packed[:, arrays["video_indices"]] = video_rows[..., :4]
        packed[:, arrays["text_indices"]] = prompt
        packed[:, arrays["audio_indices"]] = audio_rows
        packed *= torch.from_numpy(times)[None, :, None] + 0.1
        qkv = packed.unsqueeze(2)
        result = segment_attention(qkv, qkv, qkv, plan, 0)
        self.last_cache = plan.cache
        return result.squeeze(2)[:, arrays["video_indices"]].repeat(1, 1, 24) * 0.1


@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("cache_structure", [False, True])
def test_incremental_sampler_matches_native_ten_blocks(
    reactor_modules, seed, reference, cache_structure
):
    native_core, incremental_core = SmallCausalCore(), SmallCausalCore()
    native_rng = torch.Generator().manual_seed(seed)
    incremental_rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        expected, *_ = native_core.stream(
            reference,
            generator=native_rng,
            schedule=make_shifted_schedule(5, shift=12.0),
            exit_index=3,
            cache_structure=cache_structure,
        )
        run = reactor_modules.sampling.SolarCausalRun(
            incremental_core, reference, incremental_rng, cache_structure=cache_structure
        )
        chunks = [run.step() for _ in range(10)]
    assert run.complete
    torch.testing.assert_close(torch.cat(chunks, dim=2), expected, rtol=0, atol=0)
    assert torch.equal(native_rng.get_state(), incremental_rng.get_state())
    assert native_core.last_cache.layers.keys() == run.cache.layers.keys()
    for layer, entries in native_core.last_cache.layers.items():
        assert entries.keys() == run.cache.layers[layer].keys()
        for name, (key, value) in entries.items():
            actual_key, actual_value = run.cache.get(layer, name)
            torch.testing.assert_close(actual_key, key, rtol=0, atol=0)
            torch.testing.assert_close(actual_value, value, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="complete"):
        run.step()


@pytest.mark.parametrize("reference", [False, True])
def test_static_prefix_preserves_native_causal_conditioning(reactor_modules, reference):
    with torch.no_grad():
        expected, *_ = SmallCausalCore().stream(
            reference,
            generator=torch.Generator().manual_seed(0),
            schedule=make_shifted_schedule(5, shift=12.0),
            exit_index=3,
            cache_structure=True,
        )
        run = reactor_modules.optimizations.StaticPrefixRun(
            SmallCausalCore(),
            reference,
            torch.Generator().manual_seed(0),
            cache_structure=True,
        )
        actual = torch.cat([run.step() for _ in range(10)], dim=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert ("prefix", 0) in run.cache.layers[0]


@pytest.mark.parametrize("wrapped", [False, True], ids=["merged-native", "legacy-peft"])
def test_text_refiner_cache_accepts_both_renderer_models(reactor_modules, wrapped):
    original = torch.nn.Linear(2, 2, bias=False)
    model = SimpleNamespace(token_refiner=original)
    core = SimpleNamespace(
        model=SimpleNamespace(get_base_model=lambda: model) if wrapped else model
    )
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    with torch.no_grad():
        expected = original(value)
        changed_expected = original(value + 1)
    calls = []
    original.register_forward_hook(lambda *args: calls.append(1))

    reactor_modules.optimizations.cache_text_refiner(core)
    with torch.no_grad():
        first = model.token_refiner(value)
        repeated = model.token_refiner(value.clone())
        torch.testing.assert_close(first, expected)
        assert repeated is first
        assert len(calls) == 1
        torch.testing.assert_close(model.token_refiner(value + 1), changed_expected)
        assert len(calls) == 2
    with torch.enable_grad():
        uncached = model.token_refiner(value)
        assert uncached.requires_grad
        assert len(calls) == 3


@pytest.mark.parametrize("wrapped", [False, True], ids=["merged-native", "legacy-peft"])
def test_adaln_precomputation_accepts_both_renderer_models(reactor_modules, monkeypatch, wrapped):
    from solar.renderer.causal import CLEAN_LEVEL, STATIC_TIME

    # Only the parameter-dtype helper is needed; the real H3 model would import GPU dependencies.
    monkeypatch.setitem(
        sys.modules,
        "solar.backends.minimax_h3.model",
        SimpleNamespace(get_parameter_dtype=lambda model: next(model.parameters()).dtype),
    )

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

        def forward(self, embedding):
            return embedding * self.weight[0], embedding * self.weight[1]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.adaln_proj = Projection()
            self.transformer_blocks = torch.nn.ModuleList([block])

        def _time_condition(self, times, _target, *, parameter_dtype):
            return times.to(parameter_dtype(self))[:, None]

    model = Model()
    projection = model.transformer_blocks[0].adaln_proj
    core = SimpleNamespace(
        model=SimpleNamespace(get_base_model=lambda: model) if wrapped else model,
        device=torch.device("cpu"),
    )

    def run(*, times):
        # Model weights are gone after precomputation, but the native embeddings retain FP32.
        embedding = torch.tensor(times, dtype=torch.float32).unique(sorted=True)[:, None]
        return model.transformer_blocks[0].adaln_proj(embedding)

    core._run = run
    schedule = make_shifted_schedule(5, shift=12.0)
    time_sets = [[STATIC_TIME], [CLEAN_LEVEL]]
    for level in [CLEAN_LEVEL, *(1 - float(s) for s in schedule.sigmas[:4])]:
        time_sets.extend(([STATIC_TIME, CLEAN_LEVEL, level], [CLEAN_LEVEL, level]))
    time_sets = [torch.tensor(times, dtype=torch.float32).numpy() for times in time_sets]
    with torch.no_grad():
        expected = [tuple(t.clone() for t in run(times=times)) for times in time_sets]
    result = reactor_modules.optimizations.precompute_adaln(core)
    assert result["adaln_weight_bytes_removed"] == projection.weight.numel() * 4
    assert isinstance(
        model.transformer_blocks[0].adaln_proj, reactor_modules.optimizations.ModulationTable
    )
    with torch.no_grad():
        for times, target in zip(time_sets, expected, strict=True):
            actual = core._run(times=times)
            for value, reference in zip(actual, target, strict=True):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
        with pytest.raises(ValueError, match="Unprepared timestep set"):
            core._run(times=[0.12345])


def test_fa4_targets_the_integrated_model_attention_entrypoint(reactor_modules, monkeypatch):
    backend = importlib.import_module("solar.backends.minimax_h3")
    model = SimpleNamespace(segment_attention=segment_attention)
    calls = []

    def attention(query, key, value, **kwargs):
        calls.append(kwargs)
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)

    monkeypatch.setattr(backend, "model", model, raising=False)
    monkeypatch.setitem(
        sys.modules, "flash_attn.cute.interface", SimpleNamespace(flash_attn_func=attention)
    )
    reactor_modules.optimizations.install_fa4()
    plan = H3SegmentPlan(((0, 3, ((0, 3),), (), None),), 3)
    qkv = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4) * 0.1
    expected = segment_attention(qkv, qkv, qkv, plan, 0)
    actual = model.segment_attention(qkv, qkv, qkv, plan, 0)
    assert calls == [{"causal": False, "num_splits": 1}]
    torch.testing.assert_close(actual, expected)


def test_staged_source_preserves_assets_and_detects_modified_files(reactor_modules, tmp_path):
    source = tmp_path / "repository"
    renderer = source / "solar/renderer"
    (renderer / "assets").mkdir(parents=True)
    (renderer / "render.py").write_text("# integrated renderer\n")
    (renderer / "assets/silence.safetensors").write_bytes(b"renderer asset")
    for name in ("LICENSE", "NOTICE"):
        (source / name).write_text(name)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    destination = tmp_path / ".solarwm-source"
    provenance = reactor_modules.staging.stage_source(source, destination)
    assert not provenance["solarwm_source_has_local_changes"]
    assert "solar/renderer/assets/silence.safetensors" in provenance["files"]
    assert json.loads((destination / "provenance.json").read_text()) == provenance
    config = tmp_path / "solarrenderer.yaml"
    config.write_text("source: .solarwm-source\n")
    settings = reactor_modules.assets.read_settings(config, tmp_path / "weights")
    assert settings.source == destination
    assert settings.scratch.is_dir()
    (destination / "solar/renderer/render.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="Staged source changed"):
        reactor_modules.assets.read_settings(config, tmp_path / "weights")


def test_source_archive_can_stage_without_git_and_cannot_overwrite_source(
    reactor_modules, tmp_path
):
    source = tmp_path / "source-archive"
    renderer = source / "solar/renderer"
    renderer.mkdir(parents=True)
    (renderer / "render.py").write_text("# integrated renderer\n")
    for name in ("LICENSE", "NOTICE"):
        (source / name).write_text(name)
    for destination in (source, source / "solar", renderer):
        with pytest.raises(ValueError, match="overwrite"):
            reactor_modules.staging.stage_source(source, destination)
        assert (renderer / "render.py").is_file()
    provenance = reactor_modules.staging.stage_source(source, tmp_path / ".solarwm-source")
    assert provenance["solarwm_revision"] is None
    assert provenance["source_archive"]
    assert provenance["files"]
