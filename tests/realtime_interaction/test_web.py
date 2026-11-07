"""Exercise deployment configuration and the web API without loading GPU models."""

import importlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from deploy.solar_wm.shared.configuration import load_scenes


@pytest.fixture(params=["standard", "memory"])
def backend_module(request):
    return importlib.import_module("deploy.solar_wm.backends." + request.param)


@pytest.fixture(params=[False, True], ids=["original-only", "with-streamrender"])
def deployment(request, tmp_path, backend_module):
    weights = tmp_path / "models"
    weights.mkdir()
    for directory in ("SolarWM-h3-33B-base", "SolarWM-h3-33B-sgf-stage2-158f-fix"):
        (weights / directory).mkdir()
    (weights / "h3_silence_153_158_170.safetensors").touch()
    memory = weights / "SolarWM-h3-33B-memory-stage1-158f"
    memory.mkdir()
    for name in ("adapter.pt", "checkpoint-manifest.json", "COMPLETE.json"):
        (memory / name).touch()
    if request.param:
        (weights / "causal_vit24_step6000.safetensors").touch()
    configure = importlib.import_module(backend_module.__name__ + ".configure").configure
    load_config = importlib.import_module(
        backend_module.__name__ + ".runtime.configuration"
    ).load_config
    config = load_config(configure(weights, tmp_path / "deployment"))
    return config, request.param


def test_configured_vae_options_and_default_are_served(deployment, tmp_path, backend_module):
    State = importlib.import_module(backend_module.__name__ + ".runtime.state").State
    serve = importlib.import_module(backend_module.__name__ + ".server").serve
    config, has_streamrender = deployment
    scenes = load_scenes(config["scenes"])
    assert list(scenes) == ["snow", "fantasy_world"]
    expected_ids = ["current", "streamrender"] if has_streamrender else ["current"]
    expected_default = "streamrender" if has_streamrender else "current"
    state = State(scenes, "web-test", dict(config, port=0, media_port=0))
    assert [row["id"] for row in state.snapshot()["vae_options"]] == expected_ids
    assert state.snapshot()["vae_backend"] == expected_default

    servers = serve(state, tmp_path / "output")
    try:
        base = f"http://127.0.0.1:{servers[0].server_address[1]}"
        with urlopen(base + "/status", timeout=5) as response:
            status = json.load(response)
        assert [row["id"] for row in status["vae_options"]] == expected_ids
        assert status["vae_backend"] == expected_default
        with urlopen(base + "/gallery/manifest.json", timeout=5) as response:
            gallery = json.load(response)
        assert [item["id"] for item in gallery["items"]] == ["snow", "fantasy_world"]
        for item in gallery["items"]:
            with urlopen(base + item["thumbnail"], timeout=5) as response:
                assert response.read() == Path(scenes[item["id"]]["first"]).read_bytes()
        with urlopen(base + "/", timeout=5) as response:
            assert response.status == 200
            assert response.headers.get_content_type() == "text/html"

        state.update(ready=True)
        state.request("/prepare", {"scene_id": "snow"})
        assert state.next_command(0)["op"] == "prepare"
        state.prepare_progress("ready", ready=True)
        body = {
            "scene_id": "snow",
            "preparation_id": state.snapshot()["scene_preparation"]["id"],
        }
        if not has_streamrender:
            invalid = Request(
                base + "/start",
                data=json.dumps(dict(body, vae_backend="streamrender")).encode(),
                headers={"Content-Type": "application/json", "Origin": base},
            )
            with pytest.raises(HTTPError) as error:
                urlopen(invalid, timeout=5)
            assert error.value.code == 409
            assert json.load(error.value)["error"] == "VAE backend is unavailable"
            error.value.close()
        start = Request(
            base + "/start",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Origin": base},
        )
        with urlopen(start, timeout=5) as response:
            status = json.load(response)
        assert status["start_pending"]
        assert status["vae_backend"] == expected_default
        assert state.next_command(0)["backend"] == expected_default
        # Active or queued generation cannot switch decoders. Between sessions,
        # each advertised decoder can be selected without preparing DiT again.
        with pytest.raises(RuntimeError, match="busy"):
            state.request("/start", dict(body, vae_backend="current"))
        state.update(start_pending=False, active=True)
        with pytest.raises(RuntimeError, match="busy"):
            state.request("/start", dict(body, vae_backend="current"))
        state.request("/stop", {})
        assert state.next_command(0)["op"] == "stop"
        state.update(active=False)
        for selected in [*expected_ids, expected_default]:
            state.request("/start", dict(body, vae_backend=selected))
            assert state.next_command(0)["backend"] == selected
            state.update(start_pending=False, active=True)
            state.request("/stop", {})
            assert state.next_command(0)["op"] == "stop"
            state.update(active=False)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
