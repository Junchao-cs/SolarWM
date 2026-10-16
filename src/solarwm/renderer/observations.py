"""Observation plans: which inputs a sample carries and how well each target row is known.

Every target video row has a *known degree* in H3's data-ward time: ``1`` is a row
whose content is given, ``0`` a row that is pure noise. A row at degree ``t``
holds ``t * x0 + (1 - t) * noise``, so a condition is a target row placed high on
the same axis the denoiser already uses. One plan therefore describes ordinary
generation, first/last-frame and prefix conditioning, inpainting, outpainting,
partially trusted observations and staggered schedules. The flow-matching target
``x0 - noise`` does not depend on the degree, so a plan changes the noising and the
row timesteps, never the loss expression.

The helpers use NumPy with an explicit generator so every pattern can be tested
without a GPU runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PATTERNS = (
    "given",
    "shared",
    "known_frames",
    "known_region",
    "soft_frames",
    "soft_region",
    "confidence_map",
    "ramp",
)
_DROPOUTS = ("camera", "anchor", "control", "observation")


@dataclass(frozen=True)
class H3ObservationProfile:
    """Sampling distribution over observation plans for one training run."""

    patterns: tuple[tuple[str, float], ...] = (("shared", 1.0),)
    hard_level: float = 0.999
    soft_min: float = 0.5
    levels: int = 1000
    dropout: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def dropout_probability(self, name: str) -> float:
        return dict(self.dropout).get(name, 0.0)

    def fingerprint(self) -> dict[str, Any]:
        """Checkpoint-visible description of the training distribution."""
        return {
            "patterns": {name: weight for name, weight in self.patterns},
            "hard_level": self.hard_level,
            "soft_min": self.soft_min,
            "levels": self.levels,
            "dropout": {name: value for name, value in self.dropout},
        }


@dataclass(frozen=True)
class H3ObservationPlan:
    """One sample's known-degree field and the optional inputs it keeps."""

    pattern: str
    base_time: float
    row_times: np.ndarray
    loss_rows: np.ndarray
    use_camera: bool = True
    use_anchor: bool = True
    use_control: bool = True
    observation_draws: np.ndarray = field(default_factory=lambda: np.ones(0))
    observation_dropout: float = 0.0

    def keeps_observation(self, index: int) -> bool:
        if index >= self.observation_draws.size:
            raise ValueError("a sample carries more visual observations than a plan covers")
        return bool(self.observation_draws[index] >= self.observation_dropout)

    @property
    def generated_fraction(self) -> float:
        return float(self.loss_rows.mean())


def observation_profile_from_train_config(train: Mapping[str, Any]) -> H3ObservationProfile | None:
    """Parse the optional ``train.observations`` block; ``None`` keeps the fixed recipe."""
    from solarwm.errors import ConfigurationError

    block = train.get("observations")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ConfigurationError("train.observations must be a mapping")
    unknown = sorted(set(block) - {"patterns", "hard_level", "soft_min", "levels", "dropout"})
    if unknown:
        raise ConfigurationError(f"train.observations has unsupported keys {unknown}")
    patterns = block.get("patterns", {"shared": 1.0})
    if not isinstance(patterns, Mapping) or not patterns:
        raise ConfigurationError("train.observations.patterns must be a non-empty mapping")
    weights: list[tuple[str, float]] = []
    for name in PATTERNS:
        if name not in patterns:
            continue
        weight = patterns[name]
        if isinstance(weight, bool) or not isinstance(weight, int | float) or weight < 0:
            raise ConfigurationError(
                f"train.observations.patterns.{name} must be a non-negative number"
            )
        if weight > 0:
            weights.append((name, float(weight)))
    extra = sorted(set(patterns) - set(PATTERNS))
    if extra:
        raise ConfigurationError(f"train.observations.patterns has unsupported patterns {extra}")
    if not weights:
        raise ConfigurationError("train.observations.patterns needs a positive weight")
    total = sum((weight for _, weight in weights))

    def unit(key: str, default: float) -> float:
        value = block.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigurationError(f"train.observations.{key} must be a number")
        return float(value)

    hard_level, soft_min = (unit("hard_level", 0.999), unit("soft_min", 0.5))
    levels = block.get("levels", 1000)
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2:
        raise ConfigurationError("train.observations.levels must be an integer of at least 2")
    if not 0.0 < soft_min < hard_level <= 1.0:
        raise ConfigurationError("train.observations needs 0 < soft_min < hard_level <= 1")
    if round(hard_level * levels) / levels != hard_level:
        raise ConfigurationError("train.observations.hard_level must lie on the level grid")
    dropout = block.get("dropout", {})
    if not isinstance(dropout, Mapping):
        raise ConfigurationError("train.observations.dropout must be a mapping")
    extra = sorted(set(dropout) - set(_DROPOUTS))
    if extra:
        raise ConfigurationError(f"train.observations.dropout has unsupported inputs {extra}")
    probabilities: list[tuple[str, float]] = []
    for name in _DROPOUTS:
        value = dropout.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, int | float) or (not 0 <= value <= 1):
            raise ConfigurationError(f"train.observations.dropout.{name} must lie in [0, 1]")
        probabilities.append((name, float(value)))
    return H3ObservationProfile(
        patterns=tuple(((name, weight / total) for name, weight in weights)),
        hard_level=hard_level,
        soft_min=soft_min,
        levels=int(levels),
        dropout=tuple(probabilities),
    )


@dataclass(frozen=True)
class H3InferenceObservations:
    """What a sampling call is given, beyond text: everything here is optional.

    ``row_degrees`` holds the known degree of every target row (``0`` rows are
    generated) and ``latents`` their ``[24,47,h,w]`` content. Rows with a positive
    degree stay fixed during sampling, so the rest is drawn conditionally on them.
    ``start_degrees`` marks rows that *start* from ``latents`` at that degree and are
    then denoised with the others: refining a rough result, or an observation with
    error ``s`` entered at ``1/(1+s)``.
    """

    latents: Any = None
    row_degrees: Any = None
    start_degrees: Any = None
    use_camera: bool = True
    use_anchor: bool = True
    use_visual_observations: bool = True
    control_gain: float = 1.0

    def __post_init__(self) -> None:
        fields = self.row_degrees is not None or self.start_degrees is not None
        if (self.latents is None) == fields:
            raise ValueError("in-place observations need latents and row or start degrees")
        if not 0.0 <= float(self.control_gain) <= 1.0:
            raise ValueError("control_gain must lie in [0, 1]")


def _shift(value: np.ndarray | float, shift: float) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    return shift * values / (1.0 + (shift - 1.0) * values)


def _unshift(value: float, shift: float) -> float:
    return float(value / (shift - (shift - 1.0) * value))


def _frame_subset(frames: int, rng: np.random.Generator) -> np.ndarray:
    """Frames given as observations: prefix, suffix, first and last, or scattered keyframes."""
    known = np.zeros(frames, dtype=bool)
    kind = int(rng.integers(4))
    limit = max(1, frames // 2)
    if kind == 0:
        known[: int(rng.integers(1, limit + 1))] = True
    elif kind == 1:
        known[frames - int(rng.integers(1, limit + 1)) :] = True
    elif kind == 2:
        known[[0, frames - 1]] = True
    else:
        count = int(rng.integers(1, min(4, frames - 1) + 1))
        known[rng.choice(frames, size=count, replace=False)] = True
    if known.all():
        known[int(rng.integers(frames))] = False
    return known


def _region(frames: int, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """``[frames, height, width]`` mask of the rows to regenerate.

    A box covering 10-60% of the frame, regenerated inside (edit, inpaint, removal)
    or outside (outpaint), over the whole clip or a sub-span of it.
    """
    area = rng.uniform(0.1, 0.6)
    aspect = np.exp(rng.uniform(-0.7, 0.7))
    box_h = int(np.clip(round(np.sqrt(area / aspect) * height), 1, height))
    box_w = int(np.clip(round(np.sqrt(area * aspect) * width), 1, width))
    top = int(rng.integers(0, height - box_h + 1))
    left = int(rng.integers(0, width - box_w + 1))
    box = np.zeros((height, width), dtype=bool)
    box[top : top + box_h, left : left + box_w] = True
    if rng.random() < 0.25 and (not box.all()):
        box = ~box
    regenerate = np.zeros((frames, height, width), dtype=bool)
    if rng.random() < 0.5:
        regenerate[:] = box
    else:
        span = int(rng.integers(1, frames + 1))
        start = int(rng.integers(0, frames - span + 1))
        regenerate[start : start + span] = box
    return regenerate


def _confidence(height: int, width: int, low: float, high: float, rng: np.random.Generator):
    """A smooth per-row confidence field with holes, as a reprojected memory would give."""
    coarse_h, coarse_w = (max(2, height // 6), max(2, width // 6))
    coarse = rng.uniform(low, high, size=(coarse_h, coarse_w))
    rows = np.linspace(0, coarse_h - 1, height)
    cols = np.linspace(0, coarse_w - 1, width)
    r0, c0 = (np.floor(rows).astype(int), np.floor(cols).astype(int))
    r1, c1 = (np.minimum(r0 + 1, coarse_h - 1), np.minimum(c0 + 1, coarse_w - 1))
    fr, fc = ((rows - r0)[:, None], (cols - c0)[None, :])
    smooth = (
        coarse[r0][:, c0] * (1 - fr) * (1 - fc)
        + coarse[r0][:, c1] * (1 - fr) * fc
        + coarse[r1][:, c0] * fr * (1 - fc)
        + coarse[r1][:, c1] * fr * fc
    )
    holes = rng.random((coarse_h, coarse_w)) < 0.2
    hole_rows = np.minimum(np.arange(height) * coarse_h // height, coarse_h - 1)
    hole_cols = np.minimum(np.arange(width) * coarse_w // width, coarse_w - 1)
    return (smooth, holes[hole_rows][:, hole_cols])


def sample_observation_plan(
    profile: H3ObservationProfile,
    *,
    frames: int,
    grid_height: int,
    grid_width: int,
    base_time: float,
    shift: float,
    rng: np.random.Generator,
    given: np.ndarray | None = None,
) -> H3ObservationPlan:
    """Draw one plan. ``base_time`` is the run's ordinary shifted video timestep.

    ``given`` is the data's own known-degree field for this sample, if it has one;
    the ``given`` pattern uses it and falls back to ``shared`` without one.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    if frames < 2 or grid_height < 1 or grid_width < 1:
        raise ValueError("observation plans need at least two frames and a non-empty grid")
    if not 0.0 <= base_time <= 1.0:
        raise ValueError(f"base_time must lie in [0, 1], got {base_time}")
    names = [name for name, _ in profile.patterns]
    weights = np.asarray([weight for _, weight in profile.patterns], dtype=np.float64)
    pattern = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    hard = profile.hard_level
    base = min(float(base_time), hard)
    times = np.full((frames, grid_height, grid_width), base, dtype=np.float64)
    if pattern == "given":
        if given is None:
            pattern = "shared"
        else:
            field_ = np.asarray(given, dtype=np.float64).reshape(times.shape)
            if field_.min() < 0.0 or field_.max() > 1.0:
                raise ValueError("a given known-degree field must lie in [0, 1]")
            times = np.where(field_ > 0.0, np.maximum(np.minimum(field_, hard), base), base)
    if pattern in ("known_frames", "soft_frames"):
        level = hard if pattern == "known_frames" else rng.uniform(profile.soft_min, hard)
        times[_frame_subset(frames, rng)] = max(level, base)
    elif pattern in ("known_region", "soft_region"):
        level = hard if pattern == "known_region" else rng.uniform(profile.soft_min, hard)
        times[~_region(frames, grid_height, grid_width, rng)] = max(level, base)
    elif pattern == "confidence_map":
        smooth, holes = _confidence(grid_height, grid_width, profile.soft_min, hard, rng)
        observed = _frame_subset(frames, rng)
        times[observed] = np.where(holes, base, np.maximum(smooth, base))[None]
    elif pattern == "ramp":
        sigma_u = _unshift(1.0 - base, shift)
        order = np.linspace(-0.5, 0.5, frames) * (1.0 if rng.random() < 0.5 else -1.0)
        staggered = np.clip(sigma_u + rng.uniform(0.2, 1.0) * order, 0.0, 1.0)
        times[:] = np.minimum(1.0 - _shift(staggered, shift), hard)[:, None, None]
    times = np.round(times * profile.levels) / profile.levels
    loss_rows = times < hard
    if not loss_rows.any():
        times[-1] = np.round(base * profile.levels) / profile.levels
        times[-1] = np.minimum(times[-1], hard - 1.0 / profile.levels)
        loss_rows = times < hard
    return H3ObservationPlan(
        pattern=pattern,
        base_time=float(np.round(base * profile.levels) / profile.levels),
        row_times=times.reshape(-1).astype(np.float32),
        loss_rows=loss_rows.reshape(-1),
        use_camera=bool(rng.random() >= profile.dropout_probability("camera")),
        use_anchor=bool(rng.random() >= profile.dropout_probability("anchor")),
        use_control=bool(rng.random() >= profile.dropout_probability("control")),
        observation_draws=rng.random(64),
        observation_dropout=profile.dropout_probability("observation"),
    )


def confidence_to_known_degree(error_std: Any) -> Any:
    """Known degree that matches an observation ``y = x0 + error_std * e``.

    ``t * y`` then has exactly the distribution of a row at degree ``t``, so the
    sampler draws from the posterior under that error model. An infinite error is
    a hole (degree zero).
    """
    values = np.asarray(error_std, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("observation error must be non-negative")
    return 1.0 / (1.0 + values)


__all__ = [
    "PATTERNS",
    "H3InferenceObservations",
    "H3ObservationPlan",
    "H3ObservationProfile",
    "confidence_to_known_degree",
    "observation_profile_from_train_config",
    "sample_observation_plan",
]
