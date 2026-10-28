from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest

from solar.errors import BackendContractError
from solar.training import (
    BatchIdentity,
    GradientStatus,
    JsonlEventSink,
    MicrobatchResult,
    StepPolicy,
    TrainingEngine,
)


class FakeRuntime:
    def __init__(self, *, gradient: GradientStatus | None = None) -> None:
        self.global_step = 0
        self.gradient = gradient or GradientStatus(True, 1.5)
        self.calls: list[str] = []

    def zero_grad(self) -> None:
        self.calls.append("zero")

    def train_microbatch(self, micro_index: int, grad_accum: int) -> MicrobatchResult:
        self.calls.append(f"micro{micro_index}/{grad_accum}")
        return MicrobatchResult(
            BatchIdentity(
                sample_ids=(f"sample-{micro_index}",),
                start_frames=(micro_index,),
                noise_seeds=(100 + micro_index,),
                checkpoint_id="init-digest",
                plan_fingerprint=f"plan-{micro_index}",
            ),
            {"loss": 1.0 + micro_index},
        )

    def assert_sp_peer_identity(self, identity: BatchIdentity) -> None:
        self.calls.append(f"sp:{identity.sample_ids[0]}")

    def prepare_optimizer_step(self) -> GradientStatus:
        self.calls.append("grad")
        return self.gradient

    def optimizer_step(self) -> None:
        self.calls.append("optimizer")

    def scheduler_step(self) -> None:
        self.calls.append("scheduler")

    def ema_update(self, step: int) -> None:
        self.calls.append(f"ema:{step}")

    def set_global_step(self, step: int) -> None:
        self.global_step = step

    def save_checkpoint(self, step: int) -> str:
        self.calls.append(f"save:{step}")
        return f"checkpoint-{step}"

    def validate(self, step: int) -> dict:
        self.calls.append(f"validate:{step}")
        return {"passed": True}


def test_engine_orders_accumulation_update_ema_save_and_validation(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    sink = JsonlEventSink(tmp_path / "events.jsonl")
    engine = TrainingEngine(
        runtime,
        StepPolicy(
            max_steps=2,
            grad_accum=2,
            save_every=2,
            validation_steps=(1,),
        ),
        event_sink=sink,
    )
    assert engine.run() == 2
    assert runtime.calls.index("optimizer") < runtime.calls.index("ema:1")
    assert "validate:1" in runtime.calls
    assert "save:2" in runtime.calls
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    first_step = records[0]
    assert first_step["losses"]["loss"] == 1.5
    assert first_step["microbatches"][0]["identity"]["sample_ids"] == ["sample-0"]


def test_nonfinite_gradient_stops_before_mutating_state() -> None:
    runtime = FakeRuntime(gradient=GradientStatus(False, float("nan")))
    with pytest.raises(BackendContractError, match="non-finite gradients"):
        TrainingEngine(runtime, StepPolicy(max_steps=1)).run()
    assert runtime.global_step == 0
    assert "optimizer" not in runtime.calls
    assert not any(call.startswith(("ema", "save", "validate")) for call in runtime.calls)


def test_nonfinite_loss_stops_before_gradient_check() -> None:
    with pytest.raises(BackendContractError, match="non-finite"):
        MicrobatchResult(
            BatchIdentity(("s",), (0,), (1,), "checkpoint", "plan"),
            {"loss": float("inf")},
        )


def test_engine_suspends_automatic_cycle_collection_during_training() -> None:
    class GCRuntime(FakeRuntime):
        def train_microbatch(self, micro_index: int, grad_accum: int) -> MicrobatchResult:
            assert not gc.isenabled()
            return super().train_microbatch(micro_index, grad_accum)

        def validate(self, step: int) -> dict:
            assert not gc.isenabled()
            return super().validate(step)

    gc.enable()
    runtime = GCRuntime()
    assert (
        TrainingEngine(
            runtime,
            StepPolicy(max_steps=1, validation_steps=(1,)),
        ).run()
        == 1
    )
    assert gc.isenabled()
