from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval" / "solar_renderer"


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(EVAL))
    for name in ("bench", "render_benchmark"):
        spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def test_benchmark_uses_integrated_renderer_and_keeps_split_outputs(
    benchmark, monkeypatch, tmp_path
):
    from solar.renderer import render

    assert benchmark.render is render
    cases = [
        benchmark.bench.Case(split, "same/id", "prompt", "structure.mp4", "first.png", None)
        for split in ("hard", "heldout")
    ]
    monkeypatch.setattr(benchmark.bench, "load", lambda *_: cases)
    encoded = {}
    renders = []
    calls = []

    def encode(encoder, sample, member):
        assert encoder == "encoder"
        assert sample["target"] == "structure.mp4"
        assert sample["observations"] == [
            {"media": "structure.mp4", "frames": "all"},
            {"media": "first.png", "frames": [0]},
        ]
        assert member not in encoded
        encoded[member] = sample

    def load_renderer(*args, **kwargs):
        calls.append((args, kwargs))
        return "renderer"

    def render_member(renderer, member, output, seed, forwards):
        assert renderer == "renderer"
        assert member in encoded
        renders.append((Path(output).relative_to(tmp_path), seed, forwards))
        return 0.0

    monkeypatch.setattr(render, "load_encoder", lambda *_: "encoder")
    monkeypatch.setattr(render, "encode_sample", encode)
    monkeypatch.setattr(render, "free_cuda", lambda: None)
    monkeypatch.setattr(render, "load_renderer", load_renderer)
    monkeypatch.setattr(render, "render_member", render_member)
    benchmark.main(
        [
            "--bench",
            "dataset",
            "--out",
            str(tmp_path),
            "--h3",
            "model",
            "--control",
            "control",
            "--fewstep",
            "fewstep",
            "--causal",
            "--seed",
            "7",
        ]
    )
    assert len(encoded) == 2
    assert renders == [(Path("hard/same/id.mp4"), 7, 4), (Path("heldout/same/id.mp4"), 7, 4)]
    assert calls == [(("model", "control", "fewstep", "cuda:0"), {"causal": True})]


def test_completed_benchmark_does_not_load_models(benchmark, monkeypatch, tmp_path):
    case = benchmark.bench.Case("hard", "done", "prompt", "structure.mp4", "first.png", None)
    output = case.render(tmp_path)
    output.parent.mkdir()
    output.write_bytes(b"already rendered")
    monkeypatch.setattr(benchmark.bench, "load", lambda *_: [case])
    monkeypatch.setattr(
        benchmark.render, "load_encoder", lambda *_: pytest.fail("loaded completed batch")
    )
    benchmark.main(
        [
            "--bench",
            "dataset",
            "--out",
            str(tmp_path),
            "--h3",
            "model",
            "--control",
            "control",
            "--fewstep",
            "fewstep",
        ]
    )


@pytest.mark.parametrize("args", [["--shards", "0"], ["--shard", "-1"], ["--shard", "1"]])
def test_invalid_benchmark_shard_fails_before_reading_dataset(benchmark, args):
    with pytest.raises(SystemExit, match="2"):
        benchmark.main(
            [
                "--bench",
                "missing",
                "--out",
                "out",
                "--h3",
                "model",
                "--control",
                "control",
                "--fewstep",
                "fewstep",
                *args,
            ]
        )


@pytest.mark.parametrize(
    "script", ["render_benchmark", "metrics", "gt_metrics", "summary", "judge", "elo"]
)
def test_eval_cli_help_from_outside_repository(script, tmp_path):
    if script in {"metrics", "gt_metrics"}:
        pytest.importorskip("torch")
    result = subprocess.run(
        [sys.executable, str(EVAL / f"{script}.py"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
