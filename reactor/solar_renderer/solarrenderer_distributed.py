"""Four-GPU sequence parallelism with native conditioning and incremental decoding."""

import tempfile
from pathlib import Path

from solarrenderer_backend import SolarBackend
from solarrenderer_model import SolarModel


class FourBackend(SolarBackend):
    def __init__(self, settings, device, rank, world_size):
        super().__init__(settings)
        from solarwm.backends.minimax_h3.distributed import init_sequence_parallel

        self.device, self.rank = device, rank
        init_sequence_parallel(world_size)

    def begin(self, anchor):
        import torch
        import torch.distributed as dist
        from solarrenderer_decode import SolarDecoder
        from solarrenderer_optimizations import (
            StaticPrefixRun,
            cache_text_refiner,
            install_fa4,
            precompute_adaln,
        )

        from solarwm.renderer.data import H3OmniStream

        self.reset()
        self.renderer = None
        self.native.free_cuda()
        directory = None
        result = [None]
        if self.rank == 0:
            try:
                directory = tempfile.TemporaryDirectory(
                    prefix="solar-sp4-", dir=self.settings.scratch
                )
                root = Path(directory.name)
                structure, reference, member = (
                    root / "structure.mp4",
                    root / "first.png",
                    root / "member.safetensors",
                )
                structure.write_bytes(anchor.structure)
                if anchor.reference is not None:
                    reference.write_bytes(anchor.reference)
                encoder = self.native.load_encoder(str(self.settings.h3), self.device)
                sample = self.native.sample_of(
                    "reactor",
                    anchor.prompt,
                    str(structure),
                    str(reference) if anchor.reference is not None else None,
                )
                self.native.encode_sample(encoder, sample, str(member))
                del encoder
                self.native.free_cuda()
                result[0] = {"member": str(member)}
            except Exception as exc:
                result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(result, src=0)
        if "error" in result[0]:
            if directory is not None:
                directory.cleanup()
            raise RuntimeError(result[0]["error"])
        stream = object.__new__(H3OmniStream)
        stream.root, stream.current_plan_fingerprint, stream.aligned_control = (
            "/",
            "render",
            False,
        )
        batch = None
        read_error = None
        try:
            batch = stream.read({"id": "reactor", "member": result[0]["member"]})
        except Exception as exc:
            read_error = f"Rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, read_error)
        if directory is not None:
            directory.cleanup()
        if any(error is not None for error in errors):
            raise RuntimeError("; ".join(error for error in errors if error is not None))
        self.renderer = self.native.load_renderer(
            str(self.settings.h3),
            str(self.settings.control),
            str(self.settings.fewstep),
            self.device,
            causal=True,
        )
        core, conditioners, device = self.renderer
        precompute_adaln(core)
        cache_text_refiner(core)
        install_fa4()
        from solarrenderer_vae import install

        install(conditioners.video_vae, batched=True, compile_mode=None)
        torch.manual_seed(anchor.seed)
        with torch.no_grad():
            self.run = StaticPrefixRun(
                core,
                batch,
                torch.Generator(device=device).manual_seed(anchor.seed),
                cache_structure=True,
            )
        self.decoder = SolarDecoder(conditioners.video_vae, self.run.frames, device)


class FourModel(SolarModel):
    def load(self, settings):
        self._backend = FourBackend(settings, self.device, self.rank, self.world_size)
