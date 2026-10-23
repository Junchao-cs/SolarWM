"""Exercise the real application hooks and model bookkeeping without a GPU."""

import asyncio
import io
import subprocess
import sys
from types import SimpleNamespace

import av
import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from solarrenderer import SolarRenderer
from solarrenderer_media import validate_reference, validate_structure
from solarrenderer_model import SolarAnchor, SolarInput, SolarModel, SolarResult
from solarrenderer_types import SolarState


def app():
    model = SolarRenderer()
    model.state = SolarState()
    messages = []

    async def send(message):
        messages.append(message)

    model.send = send
    model.output = SimpleNamespace(flush=lambda: None)
    return model, messages


def test_refusal_prevents_model_call():
    model, _ = app()
    model.engine = SimpleNamespace(generate=lambda _: pytest.fail("refused step reached model"))
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    with pytest.raises(CommandError, match="structure"):
        asyncio.run(model.start())


def test_anchor_acknowledgment_and_fixed_prompt():
    model, messages = app()
    model.state._structure = b"validated video"
    with pytest.raises(CommandError):
        asyncio.run(model.start())
    asyncio.run(model.set_prompt("first prompt"))
    asyncio.run(model.start())
    first = asyncio.run(model.process_input())
    assert first.anchor.prompt == "first prompt"
    assert asyncio.run(model.process_input()).anchor is first.anchor
    asyncio.run(model.set_prompt("next video"))
    assert asyncio.run(model.process_input()).anchor.prompt == "first prompt"
    result = SolarResult(world_id=first.world_id, chunk_index=1, frames=None, complete=False)
    asyncio.run(model.process_output(StepOutcome(result=result, elapsed=0.5)))
    assert asyncio.run(model.process_input()).anchor is None
    assert messages[-1].chunks == 1 and messages[-1].elapsed == 0.5
    asyncio.run(model.start())
    next_input = asyncio.run(model.process_input())
    assert next_input.world_id == first.world_id + 1
    assert next_input.anchor.prompt == "next video"


def test_generate_reads_only_input():
    model, _ = app()
    supplied = SolarInput(world_id=11)
    result = SolarResult(world_id=11, chunk_index=1, frames=None, complete=False)

    def generate(value):
        assert value is supplied
        return result

    model.engine = SimpleNamespace(generate=generate)
    model.state = None
    assert model.generate(supplied) is result


def test_failure_does_not_acknowledge_or_complete():
    model, messages = app()
    error = RuntimeError("compute failed")
    with pytest.raises(RuntimeError, match="compute failed"):
        asyncio.run(model.process_output(StepOutcome(error=error, elapsed=0.1)))
    assert model.state._applied_id is None and model.state._chunks == 0
    assert not messages


def test_pause_completion_and_session_reset():
    model, _ = app()
    model.state._structure = b"video"
    model.state.prompt = "scene"
    asyncio.run(model.start())
    asyncio.run(model.set_paused(True))
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    asyncio.run(model.set_paused(False))
    result = SolarResult(
        world_id=1,
        chunk_index=10,
        frames=np.zeros((5, 2, 2, 3), np.uint8),
        complete=True,
    )
    output = asyncio.run(model.process_output(StepOutcome(result=result, elapsed=0.2)))
    assert output.main_video.shape == (5, 2, 2, 3)
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    calls = []
    model.engine = SimpleNamespace(reset=lambda: calls.append("reset"))
    model.on_session_ended()
    assert calls == ["reset"]
    assert not hasattr(SolarRenderer, "on_disconnected")
    assert SolarState().seed == 0 and SolarState()._anchor is None


@pytest.mark.parametrize(
    "command,mime", [("set_structure", "video/mp4"), ("set_reference", "image/png")]
)
def test_bad_upload_refused(command, mime):
    model, messages = app()
    with pytest.raises(CommandError):
        asyncio.run(getattr(model, command)(UploadedFile(name="bad", mime_type=mime, data=b"bad")))
    assert model.state._structure is None and model.state._reference is None
    assert not messages


def test_model_bookkeeping_and_failure():
    backend = SimpleNamespace(begin=lambda _: None, reset=lambda: None, step=lambda: (None, False))
    model = SolarModel()
    model._backend = backend
    with pytest.raises(ValueError):
        model.generate(SolarInput(world_id=1))
    anchor = SolarAnchor(structure=b"video", reference=None, prompt="scene", seed=7)
    assert model.generate(SolarInput(world_id=1, anchor=anchor)).chunk_index == 1

    def fail():
        raise RuntimeError("failed")

    backend.step = fail
    with pytest.raises(RuntimeError):
        model.generate(SolarInput(world_id=1))
    backend.step = lambda: (None, True)
    assert model.generate(SolarInput(world_id=1)).chunk_index == 2
    with pytest.raises(RuntimeError, match="complete"):
        model.generate(SolarInput(world_id=1))
    model.reset()
    assert model.generate(SolarInput(world_id=2, anchor=anchor)).chunk_index == 1


def test_model_imports_without_runtime():
    script = """
import sys
class BlockRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("reactor_runtime"):
            raise RuntimeError("model imported Runtime")
sys.meta_path.insert(0, BlockRuntime())
import solarrenderer_model
import solarrenderer_backend
import solarrenderer_distributed
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_four_gpu_runner_configuration(monkeypatch, tmp_path):
    import solarrenderer
    from solarrenderer_distributed import FourModel

    calls = {}
    settings = object()
    monkeypatch.setattr(solarrenderer, "get_weights_path", lambda: str(tmp_path))
    monkeypatch.setattr(solarrenderer, "read_settings", lambda *_: settings)

    class Runner:
        def __init__(self, model, **kwargs):
            calls.update(model=model, **kwargs)

        def start(self):
            calls["started"] = True

    monkeypatch.setattr(solarrenderer, "DistributedRunner", Runner)
    SolarRenderer().load(tmp_path / "solarrenderer.yaml")
    assert calls["model"] is FourModel
    assert calls["world_size"] == 4
    assert calls["init_process_group"] is True
    assert calls["load_kwargs"] == {"settings": settings}
    assert calls["started"]


def test_four_model_uses_runner_rank_context(monkeypatch):
    import solarrenderer_distributed as distributed

    calls = []
    monkeypatch.setattr(
        distributed,
        "FourBackend",
        lambda *args: calls.append(args) or SimpleNamespace(reset=lambda: None),
    )
    model = distributed.FourModel()
    model.device, model.rank, model.world_size = "cuda:2", 2, 4
    settings = object()
    model.load(settings)
    assert calls == [(settings, "cuda:2", 2, 4)]
    model.reset()


def test_validated_reference_is_only_staged():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    data = buffer.getvalue()
    model, _ = app()
    asyncio.run(model.set_reference(UploadedFile(name="red.png", mime_type="image/png", data=data)))
    assert model.state._reference == data
    assert not model.state._active
    asyncio.run(model.clear_reference())
    assert model.state._reference is None


@pytest.mark.parametrize("frames,valid", [(5, False), (158, True), (170, True)])
def test_video_minimum_and_bounded_decode(frames, valid):
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if valid:
        validate_structure(buffer.getvalue())
    else:
        with pytest.raises(ValueError, match="158"):
            validate_structure(buffer.getvalue())


def test_image_pixel_limit():
    buffer = io.BytesIO()
    Image.new("1", (4097, 4096)).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="megapixel"):
        validate_reference(buffer.getvalue())
