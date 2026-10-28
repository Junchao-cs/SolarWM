"""Render structure videos with SolarRenderer's released causal LoRA pair."""

from __future__ import annotations

import asyncio
from pathlib import Path

import av
from PIL import Image
from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    UploadedFile,
    connected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner
from solarrenderer_assets import read_settings
from solarrenderer_distributed import FourModel
from solarrenderer_media import validate_reference, validate_structure
from solarrenderer_model import SolarAnchor, SolarInput, SolarResult
from solarrenderer_types import SolarOutput, SolarState
from solarrenderer_types import StateUpdate as SolarStatus


class SolarRenderer(ReactorApp):
    """Render a finite video from structure, optional appearance reference, and prompt."""

    state: SolarState

    def load(self, config_path: Path | None) -> None:
        if config_path is None:
            raise ValueError("SolarRenderer requires solarrenderer.yaml.")
        settings = read_settings(config_path, Path(get_weights_path()))
        self.engine = DistributedRunner(
            FourModel,
            world_size=4,
            load_kwargs={"settings": settings},
            init_process_group=True,
            call_timeout=1800,
        )
        self.engine.start()

    def _status(self, elapsed: float | None = None) -> SolarStatus:
        s = self.state
        return SolarStatus(
            prompt=s.prompt,
            seed=s.seed,
            paused=s.paused,
            has_structure=s._structure is not None,
            has_reference=s._reference is not None,
            world_id=s._world_id,
            chunks=s._chunks,
            frames=s._frames,
            active=s._active,
            complete=s._complete,
            elapsed=elapsed,
        )

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await self.send(self._status())

    @session_ended
    def on_session_ended(self) -> None:
        self.engine.reset()
        self.output.flush()

    @event(
        name="set_structure",
        description=(
            "Select a structure video for the next Start. Fails with invalid_video for "
            "corrupt, oversized or too-short input."
        ),
    )
    async def set_structure(
        self,
        video: UploadedFile = InputField(  # noqa: B008 - Runtime schema declaration
            moderate=True,
            description="Structure video covering at least 158 frames at 24 FPS.",
        ),
    ) -> SolarStatus:
        try:
            await asyncio.to_thread(validate_structure, video.data)
        except (ValueError, av.error.FFmpegError) as exc:
            raise CommandError("invalid_video", str(exc)) from exc
        self.state._structure = video.data
        status = self._status()
        await self.send(status)
        return status

    @event(
        name="set_reference",
        description=(
            "Select an optional aligned appearance image for the next Start. Fails with "
            "invalid_image for unsupported, oversized or corrupt images."
        ),
    )
    async def set_reference(
        self,
        image: UploadedFile = InputField(  # noqa: B008 - Runtime schema declaration
            moderate=True,
            description="PNG, JPEG or WebP aligned with the structure video's first frame.",
        ),
    ) -> SolarStatus:
        try:
            await asyncio.to_thread(validate_reference, image.data)
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            raise CommandError("invalid_image", str(exc)) from exc
        self.state._reference = image.data
        status = self._status()
        await self.send(status)
        return status

    @event(
        name="clear_reference",
        description="Omit the optional appearance reference from the next video.",
    )
    async def clear_reference(self) -> SolarStatus:
        self.state._reference = None
        status = self._status()
        await self.send(status)
        return status

    @event(
        name="set_prompt",
        description=(
            "Stage the six-part rendering prompt for the next Start. The current video "
            "keeps its original prompt."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            moderate=True,
            max_length=16000,
            description="Rendering prompt for the next video.",
        ),
    ) -> SolarStatus:
        self.state.prompt = prompt
        status = self._status()
        await self.send(status)
        return status

    @event(name="set_seed", description="Stage the random seed for the next Start.")
    async def set_seed(
        self,
        seed: int = InputField(ge=0, le=2**32 - 1, description="Random seed for the next video."),
    ) -> SolarStatus:
        self.state.seed = seed
        status = self._status()
        await self.send(status)
        return status

    @event(
        name="set_paused",
        description="Pause or resume generation. Already buffered video may still play.",
    )
    async def set_paused(
        self, paused: bool = InputField(description="Whether to pause generation.")
    ) -> SolarStatus:
        self.state.paused = paused
        status = self._status()
        await self.send(status)
        return status

    @event(
        name="start",
        description=(
            "Begin a new finite video from selected inputs. Fails with missing_video or "
            "missing_prompt until both are supplied. Replaces the current video."
        ),
    )
    async def start(self) -> SolarStatus:
        s = self.state
        if s._structure is None:
            raise CommandError("missing_video", "Select a structure video first.")
        if not s.prompt.strip():
            raise CommandError("missing_prompt", "Supply a rendering prompt first.")
        s._anchor = SolarAnchor(
            structure=s._structure, reference=s._reference, prompt=s.prompt, seed=s.seed
        )
        s._world_id += 1
        s._chunks = s._frames = 0
        s._active, s._complete, s.paused = True, False, False
        self.output.flush()
        status = self._status()
        await self.send(status)
        return status

    async def process_input(self) -> SolarInput:
        s = self.state
        if not s._active:
            raise ApplicationError("Select inputs and press Start to render a video.")
        if s.paused:
            raise ApplicationError("paused")
        return SolarInput(
            world_id=s._world_id,
            anchor=s._anchor if s._applied_id != s._world_id else None,
        )

    def generate(self, input: SolarInput) -> SolarResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> SolarOutput | None:
        if outcome.error is not None:
            # Unexpected model/driver failures must end the session, never count as a successful
            # block.
            raise outcome.error
        result: SolarResult = outcome.result
        s = self.state
        s._applied_id, s._chunks = result.world_id, result.chunk_index
        s._frames += 0 if result.frames is None else len(result.frames)
        s._complete, s._active = result.complete, not result.complete
        await self.send(self._status(outcome.elapsed))
        return None if result.frames is None else SolarOutput(main_video=result.frames)
