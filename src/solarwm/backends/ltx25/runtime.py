"""Heavy-runtime provider protocol and shared-engine adapters for LTX-2.5.

The core package never imports Torch or LTX-Core at module import time.
The runtime supplies one provider implementing this protocol. Receipts
make successful imports insufficient: the provider must prove exact base,
codec, adapter, and shared-checkpoint identities before a step is allowed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from solarwm.checkpoint import (
    CheckpointContract,
    VerifiedCheckpoint,
    assert_resume_compatible,
    verify_checkpoint,
)
from solarwm.errors import BackendContractError
from solarwm.inference import InferenceAdapter, InferenceCase
from solarwm.runtime.safe_state import (
    decode_numpy_rng_state,
    decode_python_rng_state,
    encode_numpy_rng_state,
    encode_python_rng_state,
)
from solarwm.training import GradientStatus, MicrobatchResult, TrainingRuntime

from .checkpoint import (
    LORA_CHECKPOINT_CONTRACT,
    BaseCheckpointInspection,
    InferenceAdapterCheckpoint,
    StrictCodecLoadReceipt,
    StrictModelLoadReceipt,
    runtime_fingerprint,
)
from .codec import LTX25OnlineCodec, RawSample
from .inference import InferencePlan

RUNTIME_PROVIDER_API = "solarwm.ltx25.runtime-provider.v1"
RUNTIME_ENTRYPOINT_GROUP = "solarwm.ltx25_runtime"
DEFAULT_RUNTIME_PROVIDER = "solarwm.backends.ltx25.official:create_provider"
REQUIRED_CHECKPOINT_COMPONENTS = (
    "adapter",
    "ema",
    "optimizer",
    "scheduler",
    "runtime",
)


def _host_rng_state() -> dict[str, Any]:
    import numpy as np

    return {
        "python_rng_state": encode_python_rng_state(random.getstate()),
        "numpy_rng_state": encode_numpy_rng_state(np.random.get_state()),
    }


def _restore_host_rng_state(state: Mapping[str, Any]) -> None:
    import numpy as np

    random.setstate(decode_python_rng_state(state["python_rng_state"]))
    np.random.set_state(decode_numpy_rng_state(state["numpy_rng_state"]))


@dataclass(frozen=True)
class ProviderCheck:
    """One provider-owned readiness fact."""

    name: str
    status: str
    detail: str
    required: bool = True
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or self.status not in {"pass", "fail", "warning"}:
            raise BackendContractError("provider readiness check is malformed")
        if not self.detail.strip():
            raise BackendContractError("provider readiness check lacks detail")


@dataclass(frozen=True)
class TrainingRuntimeBundle:
    runtime: TrainingRuntime
    model_receipt: StrictModelLoadReceipt
    codec_receipt: StrictCodecLoadReceipt | None = None
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class InferenceRuntimeBundle:
    adapter: InferenceAdapter
    cases: Sequence[InferenceCase]
    model_receipt: StrictModelLoadReceipt
    codec_receipt: StrictCodecLoadReceipt
    adapter_checkpoint_manifest_digest: str
    non_writer_run: Callable[[Sequence[InferenceCase], str], None] | None = None
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class PreencodeRuntimeBundle:
    codec: LTX25OnlineCodec
    samples: Iterable[RawSample]
    codec_receipt: StrictCodecLoadReceipt
    close: Callable[[], None] | None = None


class LTX25RuntimeProvider(Protocol):
    api_version: str
    identity: str

    def readiness(self, config: Mapping[str, Any], action: str) -> Sequence[ProviderCheck]: ...

    def create_training_runtime(
        self,
        config: Mapping[str, Any],
        inspection: BaseCheckpointInspection,
        validation_plan: InferencePlan,
    ) -> TrainingRuntimeBundle: ...

    def create_inference_runtime(
        self,
        config: Mapping[str, Any],
        inspection: BaseCheckpointInspection,
        plan: InferencePlan,
        adapter_checkpoint: InferenceAdapterCheckpoint,
    ) -> InferenceRuntimeBundle: ...

    def create_preencode_runtime(self, config: Mapping[str, Any]) -> PreencodeRuntimeBundle: ...


@dataclass(frozen=True)
class ProviderResolution:
    provider: LTX25RuntimeProvider | None
    entrypoint: str
    error: str = ""


def _runtime_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("runtime", {})
    return value if isinstance(value, Mapping) else {}


def _entrypoint_candidates(config: Mapping[str, Any]) -> tuple[str, ...]:
    runtime = _runtime_mapping(config)
    explicit = str(runtime.get("provider_entrypoint") or "").strip()
    if explicit:
        return (explicit,)
    environment = os.environ.get("SOLARWM_LTX25_RUNTIME_PROVIDER", "").strip()
    if environment:
        return (environment,)
    discovered: list[str] = []
    points = importlib.metadata.entry_points()
    selected = (
        points.select(group=RUNTIME_ENTRYPOINT_GROUP)
        if hasattr(points, "select")
        else points.get(RUNTIME_ENTRYPOINT_GROUP, ())
    )
    for point in sorted(selected, key=lambda item: item.name):
        discovered.append(f"entrypoint://{point.name}")
    return tuple(discovered) or (DEFAULT_RUNTIME_PROVIDER,)


def _validate_provider(provider: Any, entrypoint: str) -> LTX25RuntimeProvider:
    if getattr(provider, "api_version", None) != RUNTIME_PROVIDER_API:
        raise BackendContractError(f"LTX provider {entrypoint!r} has unsupported api_version")
    if not str(getattr(provider, "identity", "")).strip():
        raise BackendContractError(f"LTX provider {entrypoint!r} has no identity")
    for name in (
        "readiness",
        "create_training_runtime",
        "create_inference_runtime",
        "create_preencode_runtime",
    ):
        if not callable(getattr(provider, name, None)):
            raise BackendContractError(f"LTX provider {entrypoint!r} lacks callable {name}")
    return provider


def _load_entrypoint(value: str) -> LTX25RuntimeProvider:
    if value.startswith("entrypoint://"):
        requested = value.removeprefix("entrypoint://")
        points = importlib.metadata.entry_points()
        selected = (
            points.select(group=RUNTIME_ENTRYPOINT_GROUP, name=requested)
            if hasattr(points, "select")
            else [
                point
                for point in points.get(RUNTIME_ENTRYPOINT_GROUP, ())
                if point.name == requested
            ]
        )
        if len(selected) != 1:
            raise BackendContractError(f"LTX runtime entry point {requested!r} is not unique")
        factory = selected[0].load()
    else:
        module_name, separator, attribute = value.partition(":")
        if not separator or not module_name or not attribute:
            raise BackendContractError("runtime.provider_entrypoint must use 'module.path:factory'")
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    if not callable(factory):
        raise BackendContractError(f"LTX runtime provider factory {value!r} is not callable")
    return _validate_provider(factory(), value)


def resolve_runtime_provider(
    config: Mapping[str, Any],
    *,
    injected: LTX25RuntimeProvider | None = None,
) -> ProviderResolution:
    """Discover exactly one provider without hiding import/protocol failures."""

    if injected is not None:
        try:
            provider = _validate_provider(injected, "<injected>")
        except Exception as exc:  # provider boundary normalization
            return ProviderResolution(None, "<injected>", f"{type(exc).__name__}: {exc}")
        return ProviderResolution(provider, "<injected>")
    candidates = _entrypoint_candidates(config)
    if len(candidates) != 1:
        return ProviderResolution(
            None,
            ",".join(candidates),
            "multiple LTX runtime providers were discovered; configure one explicitly",
        )
    entrypoint = candidates[0]
    try:
        provider = _load_entrypoint(entrypoint)
    except Exception as exc:  # provider boundary normalization
        return ProviderResolution(
            None,
            entrypoint,
            f"{type(exc).__name__}: {exc}",
        )
    return ProviderResolution(provider, entrypoint)


def checkpoint_contract(config: Mapping[str, Any]) -> CheckpointContract:
    model = config["model"]
    data = config["data"]
    distributed = config["distributed"]
    if not all(isinstance(value, Mapping) for value in (model, data, distributed)):
        raise BackendContractError("LTX checkpoint contract requires model/data/distributed")
    transform = str(model["camera_translation_transform"])
    generation = str(data["generation"])
    fingerprint = runtime_fingerprint(
        camera_translation_transform=transform,
        data_generation=generation,
    )
    extras = {"runtime": fingerprint}
    return CheckpointContract(
        family="ltx25_video",
        stage="stage0p5",
        causal_mode="bidirectional",
        objective="native_rectified_flow",
        objective_variant="shifted_logit_normal",
        camera_translation_transform=transform,
        parameterization="lora-r384-alpha384",
        sp_size=int(distributed["sequence_parallel_size"]),
        data_generation=generation,
        extras=extras,
    )


def inference_checkpoint_contract(config: Mapping[str, Any]) -> CheckpointContract:
    """Return the portable, optimizer-free LTX inference-weight contract."""

    model = config["model"]
    distributed = config["distributed"]
    if not all(isinstance(value, Mapping) for value in (model, distributed)):
        raise BackendContractError("LTX inference checkpoint contract requires model/distributed")
    transform = str(model["camera_translation_transform"])
    extras = {
        "lora": {
            "rank": LORA_CHECKPOINT_CONTRACT.rank,
            "alpha": LORA_CHECKPOINT_CONTRACT.alpha,
            "dropout": LORA_CHECKPOINT_CONTRACT.dropout,
            "target_count": LORA_CHECKPOINT_CONTRACT.target_count,
            "trainable_parameters": LORA_CHECKPOINT_CONTRACT.trainable_parameters,
            "base_scale_tables_trainable": (LORA_CHECKPOINT_CONTRACT.base_scale_tables_trainable),
        }
    }
    return CheckpointContract(
        family="ltx25_video",
        stage="stage0p5",
        causal_mode="bidirectional",
        objective="native_rectified_flow",
        objective_variant="shifted_logit_normal",
        camera_translation_transform=transform,
        parameterization="lora-r384-alpha384",
        sp_size=1,
        data_generation="inference-portable:v1",
        extras=extras,
    )


def validate_ltx_inference_checkpoint(
    checkpoint: VerifiedCheckpoint,
    config: Mapping[str, Any],
    *,
    weights: str,
) -> None:
    """Validate one optimizer-free LTX release transaction for inference only."""

    selected = str(weights).strip().lower()
    if selected not in {"live", "ema"}:
        raise BackendContractError("LTX inference checkpoint weights must be live or ema")
    expected = inference_checkpoint_contract(config)
    assert_resume_compatible(expected, checkpoint.contract)
    component = "adapter" if selected == "live" else "ema"
    paths = {item.path for item in checkpoint.files}
    required = {f"{component}/model.safetensors"}
    if paths != required:
        raise BackendContractError(f"LTX inference checkpoint inventory differs: {sorted(paths)}")
    metadata = checkpoint.metadata
    expected_metadata = {
        "schema": "solarwm.ltx25.inference-checkpoint.v1",
        "selected_weights": selected,
        "lora": expected.extras["lora"],
        "optimizer_state_present": False,
        "scheduler_state_present": False,
    }
    drift = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if drift:
        raise BackendContractError(f"LTX inference checkpoint metadata differs: {drift}")


def validate_ltx_checkpoint(
    checkpoint: VerifiedCheckpoint,
    config: Mapping[str, Any],
    *,
    model_load_receipt: Mapping[str, Any] | None = None,
) -> None:
    """Apply the LTX semantic and component inventory on the shared store."""

    expected = checkpoint_contract(config)
    assert_resume_compatible(expected, checkpoint.contract)
    roots = {Path(item.path).parts[0] for item in checkpoint.files}
    missing = sorted(set(REQUIRED_CHECKPOINT_COMPONENTS) - roots)
    if missing:
        raise BackendContractError(f"LTX shared checkpoint lacks required components: {missing}")
    metadata = checkpoint.metadata
    if model_load_receipt is not None and metadata.get("model_load_receipt") != dict(
        model_load_receipt
    ):
        raise BackendContractError("LTX checkpoint does not bind its strict model load")


class VerifiedTrainingRuntime:
    """Delegate model math while verifying every checkpoint transaction."""

    def __init__(
        self,
        delegate: TrainingRuntime,
        *,
        config: Mapping[str, Any],
        model_receipt: StrictModelLoadReceipt,
        output_dir: str | Path,
    ) -> None:
        self.delegate = delegate
        self.config = config
        self.model_receipt = model_receipt
        self.output_dir = Path(output_dir).resolve()

    @property
    def global_step(self) -> int:
        return self.delegate.global_step

    def zero_grad(self) -> None:
        self.delegate.zero_grad()

    def train_microbatch(self, micro_index: int, grad_accum: int) -> MicrobatchResult:
        return self.delegate.train_microbatch(micro_index, grad_accum)

    def assert_sp_peer_identity(self, identity: Any) -> None:
        self.delegate.assert_sp_peer_identity(identity)

    def prepare_optimizer_step(self) -> GradientStatus:
        return self.delegate.prepare_optimizer_step()

    def optimizer_step(self) -> None:
        self.delegate.optimizer_step()

    def scheduler_step(self) -> None:
        self.delegate.scheduler_step()

    def ema_update(self, step: int) -> None:
        self.delegate.ema_update(step)

    def set_global_step(self, step: int) -> None:
        self.delegate.set_global_step(step)

    def save_checkpoint(self, step: int) -> str:
        raw = self.delegate.save_checkpoint(step)
        if not raw:
            raise BackendContractError("LTX runtime returned an empty checkpoint path")
        path = Path(raw)
        if not path.is_absolute():
            path = self.output_dir / "checkpoints" / path
        try:
            import torch.distributed as dist
        except ImportError:
            dist = None  # type: ignore[assignment]
        distributed = dist is not None and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        result = [""]
        error = [""]
        if rank == 0:
            try:
                verified = verify_checkpoint(path)
                if verified.step != step:
                    raise BackendContractError("LTX checkpoint step differs from training engine")
                validate_ltx_checkpoint(
                    verified,
                    self.config,
                    model_load_receipt=self.model_receipt.as_dict(),
                )
                result[0] = verified.manifest_digest
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        if distributed:
            dist.broadcast_object_list(error, src=0)
            dist.broadcast_object_list(result, src=0)
        if error[0]:
            raise BackendContractError(f"LTX committed checkpoint verification failed: {error[0]}")
        if not result[0]:
            raise BackendContractError("LTX checkpoint verification returned no identity")
        return result[0]

    def validate(self, step: int) -> Mapping[str, Any]:
        return self.delegate.validate(step)


def require_training_runtime(value: Any) -> TrainingRuntime:
    required = (
        "zero_grad",
        "train_microbatch",
        "assert_sp_peer_identity",
        "prepare_optimizer_step",
        "optimizer_step",
        "scheduler_step",
        "ema_update",
        "set_global_step",
        "save_checkpoint",
        "validate",
    )
    if not hasattr(value, "global_step") or any(
        not callable(getattr(value, name, None)) for name in required
    ):
        raise BackendContractError("provider returned an incomplete TrainingRuntime")
    return value


__all__ = [
    "DEFAULT_RUNTIME_PROVIDER",
    "REQUIRED_CHECKPOINT_COMPONENTS",
    "RUNTIME_ENTRYPOINT_GROUP",
    "RUNTIME_PROVIDER_API",
    "InferenceRuntimeBundle",
    "LTX25RuntimeProvider",
    "PreencodeRuntimeBundle",
    "ProviderCheck",
    "ProviderResolution",
    "TrainingRuntimeBundle",
    "VerifiedTrainingRuntime",
    "checkpoint_contract",
    "inference_checkpoint_contract",
    "require_training_runtime",
    "resolve_runtime_provider",
    "validate_ltx_checkpoint",
    "validate_ltx_inference_checkpoint",
]
