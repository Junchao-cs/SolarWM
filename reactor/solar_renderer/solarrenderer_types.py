"""Client state, generated video and explicit progress broadcasts."""

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)
from solarrenderer_model import SolarAnchor


class SolarState(InputState):
    prompt: str = InputField(
        default="",
        moderate=True,
        max_length=16000,
        description=(
            "Six-part rendering prompt, captured when Start is pressed. Changes apply to "
            "the next video."
        ),
    )
    seed: int = InputField(
        default=0, ge=0, le=2**32 - 1, description="Random seed captured at Start."
    )
    paused: bool = InputField(
        default=False,
        description="Pause at the next block boundary; already buffered frames may still play.",
    )
    _structure: bytes | None = None
    _reference: bytes | None = None
    _anchor: SolarAnchor | None = None
    _world_id: int = 0
    _applied_id: int | None = None
    _chunks: int = 0
    _frames: int = 0
    _active: bool = False
    _complete: bool = False


class SolarOutput(Output):
    main_video: Video


class StateUpdate(ModelMessage):
    """Full shared state after a command, connection or generated block."""

    prompt: str = MessageField(description="Prompt staged for the next Start.")
    seed: int = MessageField(description="Seed staged for the next Start.")
    paused: bool = MessageField(description="Whether generation is paused.")
    has_structure: bool = MessageField(
        description="Whether a valid structure video has been selected."
    )
    has_reference: bool = MessageField(
        description="Whether an appearance reference has been selected."
    )
    world_id: int = MessageField(description="ID of the current video, zero before Start.")
    chunks: int = MessageField(description="Completed generation blocks in the current video.")
    frames: int = MessageField(description="RGB frames emitted in the current video.")
    active: bool = MessageField(description="Whether a video is being generated.")
    complete: bool = MessageField(description="Whether the finite video has completed.")
    elapsed: float | None = MessageField(
        description="Runtime-measured seconds for the last block; null on state transitions."
    )
