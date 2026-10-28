"""Check the relocated node pack without a ComfyUI install, models, browser or GPU."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "deploy/solar_renderer/comfyui"


@pytest.fixture
def node_pack(monkeypatch, tmp_path):
    """Supply only the ComfyUI host interfaces needed to load the real plugin."""
    pytest.importorskip("torch")
    module_names = [
        "comfy",
        "comfy.lora",
        "comfy.patcher_extension",
        "comfy.utils",
        "comfy.model_management",
        "comfy.nested_tensor",
        "comfy.quant_ops",
        "comfy.ldm",
        "comfy.ldm.minimax",
        "comfy.ldm.minimax.model",
        "comfy_extras",
        "comfy_extras.nodes_minimax_h3",
        "folder_paths",
        "latent_preview",
        "server",
        "aiohttp",
        "aiohttp.web",
    ]
    modules = {}
    for name in module_names:
        module = modules[name] = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)

    def host_only(*args, **kwargs):
        raise AssertionError("A loading/layout test unexpectedly invoked the ComfyUI model runtime")

    for name in (
        "time_shift_sigma",
        "PackedLayout",
        "pack_audio",
        "patchify_video",
        "rope_rotation_table",
        "unpatchify_video",
    ):
        setattr(modules["comfy.ldm.minimax.model"], name, host_only)
    modules["comfy_extras.nodes_minimax_h3"]._empty_av_latent = host_only
    modules["folder_paths"].get_filename_list = lambda folder: []
    input_dir = tmp_path / "input"
    modules["folder_paths"].get_input_directory = lambda: str(input_dir)

    routes = {}

    def register(method):
        def route(path):
            def decorate(handler):
                routes[method, path] = handler
                return handler

            return decorate

        return route

    modules["server"].PromptServer = SimpleNamespace(
        instance=SimpleNamespace(routes=SimpleNamespace(get=register("GET"), post=register("POST")))
    )
    modules["aiohttp.web"].json_response = lambda data, **kwargs: data

    # Reproduce ComfyUI's file loader and the documented custom_nodes symlink.
    custom_nodes = tmp_path / "custom_nodes"
    custom_nodes.mkdir()
    link = custom_nodes / "solar_renderer"
    link.symlink_to(PACK, target_is_directory=True)
    name = "_test_comfyui_solar_renderer"
    spec = importlib.util.spec_from_file_location(name, link / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    old_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
        assert sys.path == old_path
        yield SimpleNamespace(module=module, input_dir=input_dir, routes=routes, host=modules)
    finally:
        for loaded in list(sys.modules):
            if loaded.startswith(name + "."):
                sys.modules.pop(loaded)


def test_comfyui_symlink_loads_all_nodes_and_repository_resources(node_pack):
    module = node_pack.module
    assert set(module.NODE_CLASS_MAPPINGS) == {
        "SolarRendererLoraLoader",
        "SolarRendererConditioning",
        "SolarRendererModel",
        "SolarRendererSigmas",
        "SolarRendererFastAttention",
        "SolarRendererPrompt",
        "SolarRendererWebCapture",
        "SolarRendererCausalSampler",
    }
    assert set(module.NODE_DISPLAY_NAME_MAPPINGS) == set(module.NODE_CLASS_MAPPINGS)
    assert module._paths.ROOT == ROOT
    assert Path(module.app.EXAMPLES) == ROOT / "examples/solar_renderer"
    assert (Path(module.app.WEB) / "index.html").is_file()
    assert ("GET", "/solarrenderer") in node_pack.routes
    assert ("POST", "/solarrenderer/api/graph") in node_pack.routes


def test_comfyui_uses_the_integrated_silence_artifact(node_pack):
    path = Path(node_pack.module.nodes.SILENCE)
    assert path == ROOT / "solar/renderer/assets/h3_silence_153_158_170.safetensors"
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    assert header["silence_170f"]["shape"] == [2, 32, 283]
    assert header["silence_170f"]["dtype"] == "BF16"
    assert str(path) == node_pack.module.causal.SILENCE


def test_comfyui_long_silence_preserves_both_encoded_edges(node_pack):
    import torch

    safetensors = pytest.importorskip("safetensors.torch")
    node_pack.host["comfy.utils"].load_torch_file = safetensors.load_file
    causal = node_pack.module.causal
    original = causal._silence(283)
    extended = causal._silence(400)
    assert original.shape == (1, 32, 2, 283)
    assert extended.shape == (1, 32, 2, 400)
    torch.testing.assert_close(extended[..., :20], original[..., :20])
    torch.testing.assert_close(extended[..., -20:], original[..., -20:])
    torch.testing.assert_close(extended[..., 20:-20], original[..., 141:142].expand(1, 32, 2, 360))


@pytest.mark.parametrize("first_frame", [False, True])
def test_comfyui_prompt_loads_the_integrated_caption_helper(node_pack, first_frame):
    nodes = node_pack.module.nodes
    (prompt,) = nodes.SolarRendererPrompt().build(
        "semantic",
        first_frame,
        "A car travels along a highway.",
        "[Shot 1] " + "A blue car travels along the road in neon light. " * 35,
        "N/A",
    )
    assert "one flat colour per class" in prompt
    assert ("<Video 2> is the first frame" in prompt) is first_frame
    assert prompt.endswith("non_diegetic_music:\nN/A")
    assert Path(nodes._caption_helpers().__file__) == (
        ROOT / "skills/solarrenderer-caption/scripts/conditions.py"
    )


def test_comfyui_page_discovers_and_copies_namespaced_examples(node_pack):
    app = node_pack.module.app
    example = next(e for e in app._examples() if e["id"] == "sem-highway_synthwave")
    assert example["first"] is True
    chosen = asyncio.run(app.use_example(SimpleNamespace(match_info={"name": example["id"]})))
    root = ROOT / "examples/solar_renderer" / example["id"]
    for key, filename in (("structure", "structure.mp4"), ("first", "first.png")):
        assert (node_pack.input_dir / chosen[key]).read_bytes() == (root / filename).read_bytes()
    assert chosen["prompt"] == (root / "prompt.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("causal", [False, True])
def test_comfyui_graph_and_workflows_keep_resolvable_node_names(node_pack, causal):
    app = node_pack.module.app
    graph = app.graph("low_memory", "input.mp4", "first.png", "prompt", 7, 1344, 768, 158, causal)
    expected = "SolarRendererCausalSampler" if causal else "SamplerCustomAdvanced"
    assert graph["16"]["class_type"] == expected
    assert graph["11"]["inputs"]["first_frame"] == ["10", 0]
    assert graph["2"]["inputs"]["lora_name"] == (
        app.CAUSAL_FEWSTEP_LORA if causal else app.FEWSTEP_LORA
    )
    for node in graph.values():
        if node["class_type"].startswith("SolarRenderer"):
            assert node["class_type"] in node_pack.module.NODE_CLASS_MAPPINGS
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph
    for workflow in (PACK / "workflows").glob("*.json"):
        for node in json.loads(workflow.read_text())["nodes"]:
            if node["type"].startswith("SolarRenderer"):
                assert node["type"] in node_pack.module.NODE_CLASS_MAPPINGS
