"""A small, auditable optimizer-step state machine shared by backends."""

from __future__ import annotations

import gc
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from solarwm.errors import BackendContractError


@dataclass(frozen=True)
class BatchIdentity:
    sample_ids: tuple[str, ...]
    start_frames: tuple[int, ...]
    noise_seeds: tuple[int, ...]
    checkpoint_id: str
    plan_fingerprint: str

    def __post_init__(self) -> None:
        size = len(self.sample_ids)
        if size < 1:
            raise BackendContractError("microbatch identity has no samples")
        if len(self.start_frames) != size or len(self.noise_seeds) != size:
            raise BackendContractError("microbatch identity fields have different lengths")
        if not self.checkpoint_id or not self.plan_fingerprint:
            raise BackendContractError("microbatch identity lacks checkpoint/plan identity")


@dataclass(frozen=True)
class MicrobatchResult:
    identity: BatchIdentity
    losses: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.losses:
            raise BackendContractError("microbatch produced no loss fields")
        for name, value in self.losses.items():
            if not name or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise BackendContractError(f"non-finite or invalid loss {name!r}: {value!r}")


@dataclass(frozen=True)
class GradientStatus:
    finite: bool
    norm: float

    def __post_init__(self) -> None:
        if self.finite and not math.isfinite(self.norm):
            raise BackendContractError("finite gradients reported a non-finite norm")


@dataclass(frozen=True)
class StepPolicy:
    max_steps: int
    grad_accum: int = 1
    save_every: int = 0
    validate_every: int = 0
    validation_steps: tuple[int, ...] = ()
    save_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.grad_accum < 1:
            raise BackendContractError("max_steps and grad_accum must be positive")
        if self.save_every < 0 or self.validate_every < 0:
            raise BackendContractError("save/validate intervals must be non-negative")
        if any(step < 1 for step in (*self.validation_steps, *self.save_steps)):
            raise BackendContractError("validation_steps must be positive")

    def should_save(self, step: int) -> bool:
        return step in self.save_steps or bool(self.save_every and step % self.save_every == 0)

    def should_validate(self, step: int) -> bool:
        return step in self.validation_steps or bool(
            self.validate_every and step % self.validate_every == 0
        )


class TrainingRuntime(Protocol):
    """Heavy backend hook surface; every method has one lifecycle meaning."""

    @property
    def global_step(self) -> int: ...

    def zero_grad(self) -> None: ...

    def train_microbatch(self, micro_index: int, grad_accum: int) -> MicrobatchResult: ...

    def assert_sp_peer_identity(self, identity: BatchIdentity) -> None: ...

    def prepare_optimizer_step(self) -> GradientStatus: ...

    def optimizer_step(self) -> None: ...

    def scheduler_step(self) -> None: ...

    def ema_update(self, step: int) -> None: ...

    def set_global_step(self, step: int) -> None: ...

    def save_checkpoint(self, step: int) -> str: ...

    def validate(self, step: int) -> Mapping[str, Any]: ...


EventSink = Callable[[Mapping[str, Any]], None]


@contextmanager
def suspend_automatic_cycle_collection():
    restore_gc = gc.isenabled()
    if restore_gc:
        gc.disable()
    try:
        yield
    finally:
        if restore_gc:
            gc.enable()


class JsonlEventSink:
    """Append canonical finite JSON records to one rank-owned log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: Mapping[str, Any]) -> None:
        try:
            payload = (
                json.dumps(
                    event,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode()
        except (TypeError, ValueError) as exc:
            raise BackendContractError(f"invalid training event: {exc}") from exc
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)


class TrainingEngine:
    """Run finite updates; unsafe steps never reach optimizer, EMA, save, or validation."""

    def __init__(
        self,
        runtime: TrainingRuntime,
        policy: StepPolicy,
        *,
        event_sink: EventSink | None = None,
    ) -> None:
        self.runtime = runtime
        self.policy = policy
        self.event_sink = event_sink or (lambda _: None)

    def run(self) -> int:
        with suspend_automatic_cycle_collection():
            while self.runtime.global_step < self.policy.max_steps:
                self._step()
            return self.runtime.global_step

    def _step(self) -> None:
        next_step = self.runtime.global_step + 1
        self.runtime.zero_grad()
        microbatches: list[MicrobatchResult] = []
        try:
            for micro_index in range(self.policy.grad_accum):
                result = self.runtime.train_microbatch(micro_index, self.policy.grad_accum)
                self.runtime.assert_sp_peer_identity(result.identity)
                microbatches.append(result)
        except Exception:
            self.runtime.zero_grad()
            raise

        gradient = self.runtime.prepare_optimizer_step()
        if not gradient.finite or not math.isfinite(gradient.norm):
            self.runtime.zero_grad()
            raise BackendContractError(
                f"non-finite gradients before optimizer step {next_step}: {gradient.norm}"
            )

        self.runtime.optimizer_step()
        self.runtime.scheduler_step()
        self.runtime.set_global_step(next_step)
        self.runtime.ema_update(next_step)
        self.runtime.zero_grad()

        losses = _mean_losses(microbatches)
        event: dict[str, Any] = {
            "schema": "solarwm.training-step.v1",
            "event": "optimizer_step",
            "step": next_step,
            "grad_accum": self.policy.grad_accum,
            "gradient_norm": gradient.norm,
            "losses": losses,
            "microbatches": [
                {"identity": asdict(result.identity), "losses": dict(result.losses)}
                for result in microbatches
            ],
        }
        self.event_sink(event)

        crossed_boundary = False
        if self.policy.should_save(next_step):
            checkpoint_id = self.runtime.save_checkpoint(next_step)
            if not checkpoint_id:
                raise BackendContractError("checkpoint hook returned an empty identity")
            self.event_sink(
                {
                    "schema": "solarwm.training-event.v1",
                    "event": "checkpoint",
                    "step": next_step,
                    "checkpoint_id": checkpoint_id,
                }
            )
            crossed_boundary = True
        if self.policy.should_validate(next_step):
            report = self.runtime.validate(next_step)
            self.event_sink(
                {
                    "schema": "solarwm.training-event.v1",
                    "event": "validation",
                    "step": next_step,
                    "report": dict(report),
                }
            )
            crossed_boundary = True
        if crossed_boundary:
            gc.collect()


def _mean_losses(results: Sequence[MicrobatchResult]) -> dict[str, float]:
    names = set(results[0].losses)
    if any(set(result.losses) != names for result in results):
        raise BackendContractError("loss fields changed inside one optimizer step")
    return {
        name: sum(float(result.losses[name]) for result in results) / len(results)
        for name in sorted(names)
    }
