"""Reuse this checkout's integrated renderer inside an isolated model process."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

from solarrenderer_model import SolarAnchor, SolarSettings


class SolarBackend:
    def __init__(self, settings: SolarSettings):
        self.settings = settings
        for path in (
            settings.source / "src/solarwm/renderer/render.py",
            settings.h3 / "modular_model_index.json",
            settings.control,
            settings.fewstep,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing SolarRenderer asset: {path}")
        sys.path.insert(0, str(settings.source / "src"))
        self.native = importlib.import_module("solarwm.renderer.render")
        expected = (settings.source / "src/solarwm/renderer/render.py").resolve()
        if Path(self.native.__file__).resolve() != expected:
            raise RuntimeError(
                f"SolarWM was already imported from {self.native.__file__}; "
                f"start a fresh worker using {settings.source}."
            )
        self.renderer = self.run = self.decoder = None
        self.device = "cuda:0"

    def reset(self):
        self.run = self.decoder = None

    def begin(self, anchor: SolarAnchor):
        import torch
        from solarrenderer_decode import SolarDecoder
        from solarrenderer_sampling import SolarCausalRun

        from solarwm.renderer.data import H3OmniStream

        self.reset()
        # The native entrypoint never holds Qwen and the transformer on the card together.
        self.renderer = None
        self.native.free_cuda()
        with tempfile.TemporaryDirectory(prefix="solar-", dir=self.settings.scratch) as directory:
            directory = Path(directory)
            structure = directory / "structure.mp4"
            structure.write_bytes(anchor.structure)
            first = directory / "first.png"
            if anchor.reference is not None:
                first.write_bytes(anchor.reference)
            member = directory / "member.safetensors"
            encoder = self.native.load_encoder(str(self.settings.h3), self.device)
            sample = self.native.sample_of(
                "reactor",
                anchor.prompt,
                str(structure),
                str(first) if anchor.reference is not None else None,
            )
            self.native.encode_sample(encoder, sample, str(member))
            del encoder
            self.native.free_cuda()
            self.renderer = self.native.load_renderer(
                str(self.settings.h3),
                str(self.settings.control),
                str(self.settings.fewstep),
                self.device,
                causal=True,
            )
            stream = object.__new__(H3OmniStream)
            stream.root, stream.current_plan_fingerprint, stream.aligned_control = (
                "/",
                "render",
                False,
            )
            batch = stream.read({"id": "reactor", "member": str(member)})
        core, conditioners, device = self.renderer
        torch.manual_seed(anchor.seed)
        with torch.no_grad():
            self.run = SolarCausalRun(
                core, batch, torch.Generator(device=device).manual_seed(anchor.seed)
            )
        self.decoder = SolarDecoder(conditioners.video_vae, self.run.frames, device)

    def step(self):
        import torch

        block = self.run.step()
        decoded = self.decoder.push(block)
        frames = None
        if decoded is not None:
            mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
            std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
            pixels = (decoded.float() * std + mean).clamp(0, 1)
            frames = (
                pixels[0].permute(1, 2, 3, 0).mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
            )
        return frames, self.run.complete
