"""Canonical supported model, stage, and objective routes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from solarwm.errors import ConfigurationError

_ALIASES = {
    "wan22_ti2v_5b": "wan22_ti2v_5b",
    "wan2.2-ti2v-5b": "wan22_ti2v_5b",
    "wan22_i2v_a14b": "wan22_i2v_a14b",
    "wan2.2-i2v-a14b": "wan22_i2v_a14b",
    "ltx25_video": "ltx25_video",
    "ltx-2.5": "ltx25_video",
    "minimax_h3": "minimax_h3",
    "minimax-h3": "minimax_h3",
}


@dataclass(frozen=True, order=True)
class Route:
    family: str
    stage: str
    causal_mode: str
    objective: str
    variant: str = ""

    @property
    def key(self) -> str:
        suffix = f":{self.variant}" if self.variant else ""
        return f"{self.family}:{self.stage}:{self.causal_mode}:{self.objective}{suffix}"


_SUPPORTED = frozenset(
    {
        # Wan2.2 TI2V-5B routes.
        Route("wan22_ti2v_5b", "stage0p5", "bidirectional", "flow_matching"),
        Route("wan22_ti2v_5b", "stage1", "teacher_forcing", "flow_matching"),
        Route(
            "wan22_ti2v_5b",
            "stage1",
            "teacher_forcing",
            "anyflow_forward_map",
            "v1_5",
        ),
        Route("wan22_ti2v_5b", "stage2", "self_gradient_forcing", "flow_matching"),
        # Wan2.2 A14B exposes only its Stage0.5 route.
        Route("wan22_i2v_a14b", "stage0p5", "bidirectional", "flow_matching"),
        # Independent backbones; config label FM maps to their native flow.
        Route("ltx25_video", "stage0p5", "bidirectional", "native_rectified_flow"),
        Route("minimax_h3", "stage0p5", "bidirectional", "flow_matching"),
        Route("minimax_h3", "stage1", "teacher_forcing", "anyflow_forward_map", "v1_5"),
        Route("minimax_h3", "stage2", "self_gradient_forcing", "flow_matching"),
    }
)


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def resolve_route(config: Mapping[str, Any]) -> Route:
    model = _mapping(config, "model")
    train = _mapping(config, "train")
    family_raw = str(model.get("family", "")).strip().lower()
    family = _ALIASES.get(family_raw)
    if family is None:
        raise ConfigurationError(f"unsupported model.family {family_raw!r}")
    stage = str(train.get("stage", model.get("stage", ""))).strip().lower()
    causal = str(train.get("causal_mode", "")).strip().lower()
    objective = str(train.get("objective", "")).strip().lower()
    variant = str(train.get("objective_variant", "")).strip().lower()
    if objective == "anyflow_forward_map" and not variant:
        variant = "v1_5"
    if family == "ltx25_video" and objective == "flow_matching":
        objective = "native_rectified_flow"
    return Route(family, stage, causal, objective, variant)


def validate_route(config: Mapping[str, Any]) -> Route:
    action = str(config.get("action", "")).lower()
    if action == "preencode":
        model = _mapping(config, "model")
        family_raw = str(model.get("family", "")).strip().lower()
        if family_raw not in _ALIASES:
            raise ConfigurationError(f"unsupported model.family {family_raw!r}")
        # Preencoding does not have a causal training route.
        return Route(_ALIASES[family_raw], "preencode", "none", "none")

    route = resolve_route(config)
    if route not in _SUPPORTED:
        allowed = sorted(item.key for item in _SUPPORTED if item.family == route.family)
        raise ConfigurationError(
            f"unsupported route {route.key!r}; allowed for {route.family}: {allowed}"
        )

    data = _mapping(config, "data")
    frames = int(data.get("pixel_frames", 0))
    if route.family in {"wan22_ti2v_5b", "wan22_i2v_a14b"}:
        allowed_frames = {81, 153} if route.stage == "stage0p5" else {81}
        if frames not in allowed_frames:
            raise ConfigurationError(
                f"{route.family} {route.stage} requires pixel_frames in "
                f"{sorted(allowed_frames)}, got {frames}"
            )
    elif route.family == "ltx25_video" and frames != 153:
        raise ConfigurationError("LTX-2.5 Stage0.5 requires 153 frames")
    elif route.family == "minimax_h3" and frames != 158:
        raise ConfigurationError("MiniMax-H3 Stage0.5 requires 158 frames")

    transform = str(model_value(config, "camera_translation_transform", "linear")).lower()
    if transform not in {"linear", "logd4"}:
        raise ConfigurationError("camera_translation_transform must be linear or logd4")
    if route.family == "minimax_h3" and transform != "logd4":
        raise ConfigurationError("MiniMax-H3 requires logd4")
    return route


def model_value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return _mapping(config, "model").get(key, default)


def supported_routes() -> tuple[Route, ...]:
    return tuple(sorted(_SUPPORTED))
