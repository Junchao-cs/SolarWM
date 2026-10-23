"""Public Wan2.2 backend boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from solarwm.errors import BackendContractError

from .codec import validate_preencode_config
from .contracts import WAN_FAMILIES, validate_wan_config
from .generation import resolve_generation_plan


@dataclass(frozen=True)
class Wan22Backend:
    """Validated Wan2.2 family plugin."""

    family: str

    def validate_config(self, config: Mapping[str, Any]) -> None:
        action = str(config.get("action", "")).strip().lower()
        if action == "preencode":
            validate_preencode_config(config, expected_family=self.family)
            return
        validate_wan_config(config, expected_family=self.family)
        if action == "infer":
            # Inference and inline validation resolve the same immutable plan.
            resolve_generation_plan(config)

    def _unsupported_training_route(self) -> NoReturn:
        raise BackendContractError(
            f"Wan2.2 training configuration for {self.family} has no executable "
            "supported dispatch; see the matching guide under docs/solarwm/backends/"
        )

    def train(self, config: Mapping[str, Any]) -> int:
        self.validate_config(config)
        train = config.get("train", {})
        data = config.get("data", {})
        if (
            self.family == "wan22_ti2v_5b"
            and isinstance(train, Mapping)
            and isinstance(data, Mapping)
            and str(train.get("stage")) == "stage0p5"
            and str(train.get("objective")) == "flow_matching"
            and (
                (
                    str(data.get("encoding")) == "online"
                    and int(data.get("pixel_frames", 0)) in {81, 153}
                )
                or (
                    str(data.get("encoding")) == "preencoded"
                    and int(data.get("pixel_frames", 0)) in {81, 153}
                )
            )
        ):
            from .runtime.readiness import require_training_runtime

            require_training_runtime(config, family=self.family)
            from .runtime.stage0p5 import run_stage0p5_training

            return run_stage0p5_training(config)
        if (
            self.family == "wan22_i2v_a14b"
            and isinstance(train, Mapping)
            and isinstance(data, Mapping)
            and str(train.get("stage")) == "stage0p5"
            and str(train.get("objective")) == "flow_matching"
            and (
                (
                    str(data.get("encoding")) == "online"
                    and int(data.get("pixel_frames", 0)) in {81, 153}
                )
                or (
                    str(data.get("encoding")) == "preencoded"
                    and int(data.get("pixel_frames", 0)) == 153
                )
            )
        ):
            from .runtime.readiness import require_training_runtime

            require_training_runtime(config, family=self.family)
            from .runtime.stage0p5 import run_stage0p5_training

            return run_stage0p5_training(config)
        if (
            self.family == "wan22_ti2v_5b"
            and isinstance(train, Mapping)
            and isinstance(data, Mapping)
            and str(train.get("stage")) == "stage1"
            and str(train.get("causal_mode")) == "teacher_forcing"
            and str(train.get("objective")) == "flow_matching"
            and str(data.get("encoding")) == "online"
            and int(data.get("pixel_frames", 0)) == 81
        ):
            from .runtime.readiness import require_training_runtime

            require_training_runtime(config, family=self.family)
            from .runtime.stage1 import run_stage1_training

            return run_stage1_training(config)
        if (
            self.family == "wan22_ti2v_5b"
            and isinstance(train, Mapping)
            and isinstance(data, Mapping)
            and str(train.get("stage")) == "stage1"
            and str(train.get("causal_mode")) == "teacher_forcing"
            and str(train.get("objective")) == "anyflow_forward_map"
            and str(train.get("objective_variant")) == "v1_5"
            and str(data.get("encoding")) == "online"
            and int(data.get("pixel_frames", 0)) == 81
        ):
            from .runtime.readiness import require_training_runtime

            require_training_runtime(config, family=self.family)
            from .runtime.stage1_anyflow import run_stage1_anyflow_training

            return run_stage1_anyflow_training(config)
        if (
            self.family == "wan22_ti2v_5b"
            and isinstance(train, Mapping)
            and isinstance(data, Mapping)
            and str(train.get("stage")) == "stage2"
            and str(train.get("causal_mode")) == "self_gradient_forcing"
            and str(train.get("objective")) == "flow_matching"
            and str(data.get("encoding")) == "online"
            and int(data.get("pixel_frames", 0)) == 81
        ):
            from .runtime.readiness import require_training_runtime

            require_training_runtime(
                config,
                family=self.family,
                require_transformer_weights=False,
            )
            from .runtime.stage2 import run_stage2_training

            return run_stage2_training(config)
        self._unsupported_training_route()

    def readiness(self, config: Mapping[str, Any], *, require_cuda: bool = False) -> Any:
        """Inspect dependencies, assets, indexes, and optionally CUDA."""

        self.validate_config(config)
        from .runtime.readiness import probe_runtime

        train = config.get("train", {})
        stage2 = isinstance(train, Mapping) and str(train.get("stage")) == "stage2"
        return probe_runtime(
            config,
            family=self.family,
            require_cuda=require_cuda,
            require_transformer_weights=not stage2,
        )

    def infer(self, config: Mapping[str, Any]) -> int:
        self.validate_config(config)
        from .runtime.readiness import probe_runtime

        train = config.get("train", {})
        stage2 = isinstance(train, Mapping) and str(train.get("stage")) == "stage2"
        probe_runtime(
            config,
            family=self.family,
            require_cuda=True,
            require_transformer_weights=not stage2,
        ).require_ready()
        if self.family == "wan22_ti2v_5b" and stage2:
            from .runtime.stage2 import run_stage2_inference

            run_stage2_inference(config)
            return 0
        from .runtime.inference import run_wan_inference

        run_wan_inference(config)
        return 0

    def preencode(self, config: Mapping[str, Any]) -> int:
        self.validate_config(config)
        # Preencoding intentionally allocates only the VAE and UMT5 codec, not
        # transformer/FlashAttention training assets.  Its provider performs
        # the narrower CUDA and file checks through the action-aware probe.
        from .runtime.readiness import probe_runtime

        probe_runtime(config, family=self.family, require_cuda=True).require_ready()
        from .runtime.preencode import run_wan_preencode

        run_wan_preencode(config)
        return 0


def create_backend(*, family: str) -> Wan22Backend:
    """Create one strictly named Wan backend without importing torch."""

    normalized = str(family).strip().lower()
    if normalized not in WAN_FAMILIES:
        raise BackendContractError(
            f"solarwm.backends.wan22 does not implement family {family!r}; "
            f"expected one of {sorted(WAN_FAMILIES)}"
        )
    return Wan22Backend(family=normalized)
