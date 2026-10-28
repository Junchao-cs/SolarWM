"""Runtime-independent contracts and ownership of one SolarRenderer rollout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SolarSettings:
    source: Path
    h3: Path
    control: Path
    fewstep: Path
    scratch: Path


@dataclass(frozen=True)
class SolarAnchor:
    """Validated original media bytes; native preprocessing happens in the model."""

    structure: bytes
    reference: bytes | None
    prompt: str
    seed: int


@dataclass(frozen=True)
class SolarInput:
    """Request a block from world_id; carry its anchor only until acknowledged."""

    world_id: int
    anchor: SolarAnchor | None = None


@dataclass(frozen=True)
class SolarResult:
    """Completed block facts, with RGB uint8 [T,H,W,3] media when available."""

    world_id: int
    chunk_index: int
    frames: np.ndarray | None
    complete: bool


class SolarModel:
    """Hold a finite video and advance only after a successful backend step.

    New world IDs require an anchor. Completed worlds require a new Start.
    Reset forgets the current world without changing the next caller's inputs.
    """

    def __init__(self):
        self._backend = None
        self._world_id = None
        self._chunk = 0
        self._complete = False

    def load(self, settings: SolarSettings) -> None:
        from solarrenderer_backend import SolarBackend

        self._backend = SolarBackend(settings)

    def generate(self, input: SolarInput) -> SolarResult:
        if self._backend is None:
            raise RuntimeError("Load the model before generating.")
        if input.world_id != self._world_id:
            if input.anchor is None:
                raise ValueError("A new world requires its structure, prompt and seed.")
            self._backend.begin(input.anchor)
            self._world_id, self._chunk, self._complete = input.world_id, 0, False
        if self._complete:
            raise RuntimeError("The video is complete; start a new world.")
        frames, complete = self._backend.step()
        self._chunk += 1
        self._complete = complete
        return SolarResult(
            world_id=self._world_id,
            chunk_index=self._chunk,
            frames=frames,
            complete=complete,
        )

    def reset(self) -> None:
        if self._backend is not None:
            self._backend.reset()
        self._world_id, self._chunk, self._complete = None, 0, False
