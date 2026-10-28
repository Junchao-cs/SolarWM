from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

_REPOSITORY = Path(__file__).resolve().parents[2]


def test_every_torch_load_in_public_runtime_is_weights_only() -> None:
    failures: list[str] = []
    sources = [
        source
        for root in ("solar", "deploy/solar_wm")
        for source in (_REPOSITORY / root).rglob("*.py")
    ]
    for source in sorted(sources):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr == "load"
                and isinstance(function.value, ast.Name)
                and function.value.id == "torch"
            ):
                continue
            weights_only = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "weights_only"),
                None,
            )
            if not (isinstance(weights_only, ast.Constant) and weights_only.value is True):
                relative = source.relative_to(_REPOSITORY)
                failures.append(f"{relative}:{node.lineno}")
    assert not failures, "unsafe torch.load call(s): " + ", ".join(failures)


def test_weights_only_loader_does_not_execute_reduce_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    sentinel = tmp_path / "unsafe-reduce-executed"

    class Payload:
        def __reduce__(self):
            return os.system, (f"touch {sentinel}",)

    payload = tmp_path / "payload.pt"
    torch.save({"value": Payload()}, payload)

    with pytest.raises(Exception, match="Weights only load failed"):
        torch.load(payload, map_location="cpu", weights_only=True)
    assert not sentinel.exists()
