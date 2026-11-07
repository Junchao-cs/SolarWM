"""Compile discrete keyboard states into smooth absolute camera trajectories.

The model is conditioned on camera poses, never on key symbols. A compact
offline WASD/IJKL schedule is therefore expanded on the decoded-video timeline,
integrated into one OpenCV-style camera-to-world matrix per pixel frame, and
only then sampled by the normal Wan camera/PRoPE path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import Iterable, Sequence

import numpy as np


ALLOWED_CAMERA_KEYS = frozenset("wasdijkl")
_CANONICAL_KEY_ORDER = "wasdijkl"
MIN_MOVE_SPEED_SCALE = 0.25
MAX_MOVE_SPEED_SCALE = 4.0


def normalize_move_speed_scale(value: float = 1.0) -> float:
    """Validate a user-selected translation multiplier without coercing text."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("move_speed_scale must be a number from 0.25 to 4.0")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("move_speed_scale must be finite and from 0.25 to 4.0") from exc
    if not math.isfinite(value) or not MIN_MOVE_SPEED_SCALE <= value <= MAX_MOVE_SPEED_SCALE:
        raise ValueError("move_speed_scale must be finite and from 0.25 to 4.0")
    return value


def parse_action_string(action_string: str) -> list[list[str]]:
    """Expand ``w-3,wl-2,none-1`` into one key list per latent interval."""

    if not isinstance(action_string, str):
        raise TypeError("action_string must be a string")
    normalized = "".join(action_string.replace("，", ",").split())
    if not normalized:
        raise ValueError("action_string is empty")

    raw_segments = normalized.split(",")
    if any(not segment for segment in raw_segments):
        raise ValueError("action_string has an empty segment (check commas)")

    expanded: list[list[str]] = []
    for segment in raw_segments:
        if "-" not in segment:
            raise ValueError(
                f"invalid segment {segment!r}: expected <keys>-<duration>"
            )
        keys_text, duration_text = segment.rsplit("-", 1)
        if not duration_text.isdigit() or int(duration_text) <= 0:
            raise ValueError(
                f"invalid duration in segment {segment!r}: expected a positive integer"
            )

        keys_text = keys_text.lower()
        if keys_text == "none":
            keys: list[str] = []
        else:
            if not keys_text:
                raise ValueError(
                    f"invalid segment {segment!r}: use 'none' for a neutral step"
                )
            invalid = sorted(set(keys_text) - ALLOWED_CAMERA_KEYS)
            if invalid:
                raise ValueError(
                    f"invalid camera key(s) {invalid!r} in {segment!r}; "
                    "allowed keys are w,a,s,d,i,j,k,l"
                )
            keys = [key for key in _CANONICAL_KEY_ORDER if key in keys_text]

        expanded.extend([list(keys) for _ in range(int(duration_text))])
    return expanded


def fit_action_steps(
    actions: Sequence[Sequence[str]],
    *,
    num_transitions: int,
) -> list[list[str]]:
    """Validate and neutral-pad actions to the requested latent intervals."""

    if num_transitions < 0:
        raise ValueError("num_transitions must be non-negative")
    normalized = [_normalize_keys(keys) for keys in actions]
    if len(normalized) > num_transitions:
        raise ValueError(
            f"action schedule has {len(normalized)} transitions, but the rollout "
            f"only has room for {num_transitions}"
        )
    return normalized + [[] for _ in range(num_transitions - len(normalized))]


def _normalize_keys(keys: Iterable[str]) -> list[str]:
    if isinstance(keys, str):
        keys = list(keys)
    normalized_set = {str(key).lower() for key in keys}
    invalid = sorted(normalized_set - ALLOWED_CAMERA_KEYS)
    if invalid:
        raise ValueError(
            f"invalid camera key(s) {invalid!r}; allowed keys are w,a,s,d,i,j,k,l"
        )
    return [key for key in _CANONICAL_KEY_ORDER if key in normalized_set]


def _rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    if axis == "x":
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float64,
        )
    if axis == "y":
        return np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )
    raise ValueError(f"unsupported rotation axis: {axis!r}")


def _move_towards_scalar(current: float, target: float, max_delta: float) -> float:
    if max_delta <= 0.0:
        return target
    difference = target - current
    if abs(difference) <= max_delta:
        return target
    return current + math.copysign(max_delta, difference)


def _move_towards_vector(
    current: np.ndarray,
    target: np.ndarray,
    max_delta: float,
) -> np.ndarray:
    if max_delta <= 0.0:
        return target.copy()
    difference = target - current
    distance = float(np.linalg.norm(difference))
    if distance <= max_delta or distance == 0.0:
        return target.copy()
    return current + difference * (max_delta / distance)


@dataclass(frozen=True)
class KeyboardCameraConfig:
    """Physical camera speeds used by the pixel-timeline controller."""

    frames_per_second: float = 16.0
    move_speed_per_second: float = 1.2
    pitch_speed_degrees_per_second: float = 24.0
    yaw_speed_degrees_per_second: float = 24.0
    acceleration_seconds: float = 0.25
    pitch_limit_degrees: float = 85.0
    ground_plane_translation: bool = False

    def __post_init__(self) -> None:
        values = (
            self.frames_per_second,
            self.move_speed_per_second,
            self.pitch_speed_degrees_per_second,
            self.yaw_speed_degrees_per_second,
            self.acceleration_seconds,
            self.pitch_limit_degrees,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("keyboard camera configuration must be finite")
        if self.frames_per_second <= 0:
            raise ValueError("frames_per_second must be positive")
        if self.move_speed_per_second < 0:
            raise ValueError("move_speed_per_second must be non-negative")
        if (
            self.pitch_speed_degrees_per_second < 0
            or self.yaw_speed_degrees_per_second < 0
        ):
            raise ValueError("rotation speeds must be non-negative")
        if self.acceleration_seconds < 0:
            raise ValueError("acceleration_seconds must be non-negative")
        if not 0 < self.pitch_limit_degrees < 90:
            raise ValueError("pitch_limit_degrees must be in (0, 90)")


class KeyboardCameraState:
    """Persistent velocity and pose state advanced once per pixel frame."""

    def __init__(
        self,
        config: KeyboardCameraConfig | None = None,
        *,
        initial_c2w: np.ndarray | None = None,
        initial_linear_velocity_local: np.ndarray | None = None,
        initial_pitch_speed_degrees_per_second: float = 0.0,
        initial_yaw_speed_degrees_per_second: float = 0.0,
    ) -> None:
        self.config = config or KeyboardCameraConfig()
        self._base_move_speed_per_second = self.config.move_speed_per_second
        self.move_speed_scale = 1.0
        if initial_c2w is None:
            self.c2w = np.eye(4, dtype=np.float64)
        else:
            initial_c2w = np.asarray(initial_c2w, dtype=np.float64)
            if initial_c2w.shape != (4, 4) or not np.isfinite(initial_c2w).all():
                raise ValueError("initial_c2w must be one finite 4x4 matrix")
            if not np.allclose(initial_c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
                raise ValueError("initial_c2w must be an SE(3) homogeneous matrix")
            self.c2w = initial_c2w.copy()

        if initial_linear_velocity_local is None:
            initial_linear_velocity_local = np.zeros(3, dtype=np.float64)
        initial_linear_velocity_local = np.asarray(
            initial_linear_velocity_local, dtype=np.float64
        )
        if (
            initial_linear_velocity_local.shape != (3,)
            or not np.isfinite(initial_linear_velocity_local).all()
        ):
            raise ValueError("initial_linear_velocity_local must be one finite 3-vector")
        angular_values = (
            initial_pitch_speed_degrees_per_second,
            initial_yaw_speed_degrees_per_second,
        )
        if not all(math.isfinite(float(value)) for value in angular_values):
            raise ValueError("initial angular speeds must be finite")

        self.linear_velocity_local = initial_linear_velocity_local.copy()
        self.pitch_speed_degrees_per_second = float(
            initial_pitch_speed_degrees_per_second
        )
        self.yaw_speed_degrees_per_second = float(
            initial_yaw_speed_degrees_per_second
        )
        rotation = self.c2w[:3, :3]
        self.pitch_rad = math.atan2(
            -float(rotation[1, 2]), float(rotation[1, 1])
        )

    def set_move_speed_scale(self, value: float = 1.0) -> float:
        """Change the next frame's target speed, preserving pose and momentum.

        Scale the original configured speed, never the previously scaled speed.
        Replacing this frozen config leaves the existing per-frame acceleration,
        braking, rotation speeds, and pitch limit unchanged.
        """

        scale = normalize_move_speed_scale(value)
        if scale != self.move_speed_scale:
            self.config = replace(
                self.config,
                move_speed_per_second=self._base_move_speed_per_second * scale,
            )
            self.move_speed_scale = scale
        return scale

    def step(self, keys: Iterable[str]) -> np.ndarray:
        """Advance exactly one output-video frame."""

        keys = _normalize_keys(keys)
        dt = 1.0 / float(self.config.frames_per_second)

        target_velocity_local = np.zeros(3, dtype=np.float64)
        target_velocity_local[0] = self.config.move_speed_per_second * (
            int("d" in keys) - int("a" in keys)
        )
        target_velocity_local[2] = self.config.move_speed_per_second * (
            int("w" in keys) - int("s" in keys)
        )
        target_pitch_speed = self.config.pitch_speed_degrees_per_second * (
            int("i" in keys) - int("k" in keys)
        )
        target_yaw_speed = self.config.yaw_speed_degrees_per_second * (
            int("l" in keys) - int("j" in keys)
        )

        if self.config.acceleration_seconds == 0.0:
            self.linear_velocity_local = target_velocity_local
            self.pitch_speed_degrees_per_second = target_pitch_speed
            self.yaw_speed_degrees_per_second = target_yaw_speed
        else:
            linear_delta = (
                self.config.move_speed_per_second
                * dt
                / self.config.acceleration_seconds
            )
            pitch_delta = (
                self.config.pitch_speed_degrees_per_second
                * dt
                / self.config.acceleration_seconds
            )
            yaw_delta = (
                self.config.yaw_speed_degrees_per_second
                * dt
                / self.config.acceleration_seconds
            )
            self.linear_velocity_local = _move_towards_vector(
                self.linear_velocity_local,
                target_velocity_local,
                linear_delta,
            )
            self.pitch_speed_degrees_per_second = _move_towards_scalar(
                self.pitch_speed_degrees_per_second,
                target_pitch_speed,
                pitch_delta,
            )
            self.yaw_speed_degrees_per_second = _move_towards_scalar(
                self.yaw_speed_degrees_per_second,
                target_yaw_speed,
                yaw_delta,
            )

        pitch_delta_rad = math.radians(
            self.pitch_speed_degrees_per_second * dt
        )
        yaw_delta_rad = math.radians(self.yaw_speed_degrees_per_second * dt)
        pitch_limit = math.radians(self.config.pitch_limit_degrees)
        proposed_pitch = self.pitch_rad + pitch_delta_rad
        if proposed_pitch > pitch_limit:
            pitch_delta_rad = pitch_limit - self.pitch_rad
            self.pitch_rad = pitch_limit
            self.pitch_speed_degrees_per_second = 0.0
        elif proposed_pitch < -pitch_limit:
            pitch_delta_rad = -pitch_limit - self.pitch_rad
            self.pitch_rad = -pitch_limit
            self.pitch_speed_degrees_per_second = 0.0
        else:
            self.pitch_rad = proposed_pitch

        rotation = self.c2w[:3, :3]
        translation = self.c2w[:3, 3].copy()
        rotation_new = (
            _rotation_matrix("y", yaw_delta_rad)
            @ rotation
            @ _rotation_matrix("x", pitch_delta_rad)
        )
        if self.config.ground_plane_translation:
            forward = rotation_new[:, 2].copy()
            right = rotation_new[:, 0].copy()
            forward[1] = 0.0
            right[1] = 0.0
            forward_norm = float(np.linalg.norm(forward))
            right_norm = float(np.linalg.norm(right))
            if forward_norm:
                forward /= forward_norm
            if right_norm:
                right /= right_norm
            world_velocity = (
                right * self.linear_velocity_local[0]
                + forward * self.linear_velocity_local[2]
            )
        else:
            world_velocity = rotation_new @ self.linear_velocity_local

        self.c2w = np.eye(4, dtype=np.float64)
        self.c2w[:3, :3] = rotation_new
        self.c2w[:3, 3] = translation + world_velocity * dt
        return self.c2w.astype(np.float32, copy=True)


def estimate_camera_velocity(
    c2w_poses: np.ndarray,
    *,
    frames_per_second: float,
    window_frames: int = 8,
) -> tuple[np.ndarray, float, float]:
    """Estimate local linear, pitch, and yaw velocity at a trajectory tail."""

    poses = np.asarray(c2w_poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"c2w_poses must have shape [N,4,4], got {poses.shape}")
    if poses.shape[0] < 2 or not np.isfinite(poses).all():
        return np.zeros(3, dtype=np.float64), 0.0, 0.0
    if not math.isfinite(float(frames_per_second)) or frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive and finite")

    count = min(max(int(window_frames), 2), poses.shape[0])
    tail = poses[-count:]
    duration = (count - 1) / float(frames_per_second)
    world_velocity = (tail[-1, :3, 3] - tail[0, :3, 3]) / duration
    local_velocity = tail[-1, :3, :3].T @ world_velocity

    rotations = tail[:, :3, :3]
    forward = rotations[:, :, 2]
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    pitch = np.unwrap(
        np.arctan2(-rotations[:, 1, 2], rotations[:, 1, 1])
    )
    times = np.arange(count, dtype=np.float64) / float(frames_per_second)
    centered = times - times.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        return local_velocity, 0.0, 0.0
    yaw_speed = math.degrees(float(np.dot(centered, yaw - yaw.mean()) / denominator))
    pitch_speed = math.degrees(
        float(np.dot(centered, pitch - pitch.mean()) / denominator)
    )
    return local_velocity, pitch_speed, yaw_speed


def actions_to_pixel_c2w_trajectory(
    actions: Sequence[Sequence[str]],
    *,
    config: KeyboardCameraConfig | None = None,
    frames_per_action: int = 4,
    include_initial_pose: bool = True,
    initial_c2w: np.ndarray | None = None,
    initial_linear_velocity_local: np.ndarray | None = None,
    initial_pitch_speed_degrees_per_second: float = 0.0,
    initial_yaw_speed_degrees_per_second: float = 0.0,
) -> np.ndarray:
    """Expand latent-interval actions and integrate one pose per pixel frame."""

    if int(frames_per_action) <= 0:
        raise ValueError("frames_per_action must be positive")
    state = KeyboardCameraState(
        config,
        initial_c2w=initial_c2w,
        initial_linear_velocity_local=initial_linear_velocity_local,
        initial_pitch_speed_degrees_per_second=(
            initial_pitch_speed_degrees_per_second
        ),
        initial_yaw_speed_degrees_per_second=initial_yaw_speed_degrees_per_second,
    )
    poses = [state.c2w.astype(np.float32, copy=True)] if include_initial_pose else []
    for keys in actions:
        normalized_keys = _normalize_keys(keys)
        for _ in range(int(frames_per_action)):
            poses.append(state.step(normalized_keys))
    if not poses:
        return np.empty((0, 4, 4), dtype=np.float32)
    return np.stack(poses).astype(np.float32, copy=False)


def action_string_to_pixel_c2w_trajectory(
    action_string: str,
    *,
    num_latent_transitions: int,
    config: KeyboardCameraConfig | None = None,
    frames_per_action: int = 4,
) -> np.ndarray:
    """Compile a compact schedule into a dense pixel-frame trajectory."""

    actions = fit_action_steps(
        parse_action_string(action_string),
        num_transitions=int(num_latent_transitions),
    )
    return actions_to_pixel_c2w_trajectory(
        actions,
        config=config,
        frames_per_action=frames_per_action,
    )


def actions_to_c2w_trajectory(
    actions: Sequence[Sequence[str]],
    *,
    config: KeyboardCameraConfig | None = None,
    include_initial_pose: bool = True,
    initial_c2w: np.ndarray | None = None,
) -> np.ndarray:
    """Compatibility helper: advance one pixel-frame step per supplied action."""

    return actions_to_pixel_c2w_trajectory(
        actions,
        config=config,
        frames_per_action=1,
        include_initial_pose=include_initial_pose,
        initial_c2w=initial_c2w,
    )


def action_string_to_c2w_trajectory(
    action_string: str,
    *,
    num_poses: int,
    config: KeyboardCameraConfig | None = None,
) -> np.ndarray:
    """Compatibility helper producing exactly ``num_poses`` pixel-step poses."""

    if num_poses <= 0:
        raise ValueError("num_poses must be positive")
    actions = fit_action_steps(
        parse_action_string(action_string),
        num_transitions=num_poses - 1,
    )
    return actions_to_c2w_trajectory(actions, config=config)


__all__ = [
    "ALLOWED_CAMERA_KEYS",
    "KeyboardCameraConfig",
    "KeyboardCameraState",
    "MIN_MOVE_SPEED_SCALE",
    "MAX_MOVE_SPEED_SCALE",
    "normalize_move_speed_scale",
    "action_string_to_c2w_trajectory",
    "action_string_to_pixel_c2w_trajectory",
    "actions_to_c2w_trajectory",
    "actions_to_pixel_c2w_trajectory",
    "estimate_camera_velocity",
    "fit_action_steps",
    "parse_action_string",
]
