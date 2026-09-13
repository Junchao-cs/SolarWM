"""Raw H3 decoding helpers for offline preencoding."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from solarwm.data.index import IndexRow
from solarwm.errors import DataContractError


def normalize_raw_source_windows(rows: Sequence[IndexRow]) -> tuple[IndexRow, ...]:
    """Use the first contiguous 158 frames when a generic raw row has no window."""

    normalized = []
    for row in rows:
        values = dict(row.values)
        has_start = "start_frame" in values
        has_indices = "source_frame_indices" in values
        if has_start and not has_indices:
            try:
                start = int(values["start_frame"])
            except (TypeError, ValueError) as exc:
                raise DataContractError(
                    f"H3 raw row {row.sample_id!r} has an invalid start frame"
                ) from exc
            values["source_frame_indices"] = list(range(start, start + 158))
        elif has_indices and not has_start:
            try:
                indices = tuple(int(value) for value in values["source_frame_indices"])
            except (TypeError, ValueError) as exc:
                raise DataContractError(
                    f"H3 raw row {row.sample_id!r} has invalid source indices"
                ) from exc
            if len(indices) != 158 or indices != tuple(range(indices[0], indices[0] + 158)):
                raise DataContractError(
                    f"H3 raw row {row.sample_id!r} source indices are not contiguous"
                )
            values["start_frame"] = indices[0]
        elif not has_start:
            values["start_frame"] = 0
            values["source_frame_indices"] = list(range(158))
        normalized.append(IndexRow.from_mapping(row.ordinal, values))
    return tuple(normalized)


def decode_resize_video(
    video_bytes: bytes,
    source_indices: Sequence[int],
    *,
    decord_num_threads: int = 2,
) -> tuple[np.ndarray, dict[str, int]]:
    """Decode the exact 158-frame window and apply H3's Lanczos cover crop."""

    try:
        import decord
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise DataContractError("H3 raw decoding requires decord and Pillow") from exc
    indices = tuple(int(value) for value in source_indices)
    if len(indices) != 158 or any(right != left + 1 for left, right in pairwise(indices)):
        raise DataContractError("H3 raw decoding requires 158 contiguous source indices")
    decord.bridge.set_bridge("native")
    try:
        reader = decord.VideoReader(
            io.BytesIO(video_bytes), num_threads=max(1, int(decord_num_threads))
        )
        frames = reader.get_batch(list(indices)).asnumpy()
    except Exception as exc:
        raise DataContractError(f"cannot decode the selected H3 video window: {exc}") from exc
    if tuple(frames.shape[:1]) != (158,) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise DataContractError(f"H3 decoded frames have shape {frames.shape}")
    source_h, source_w = int(frames.shape[1]), int(frames.shape[2])
    target_h, target_w = 768, 1344
    scale = max(target_w / source_w, target_h / source_h)
    resized_w = max(target_w, round(source_w * scale))
    resized_h = max(target_h, round(source_h * scale))
    left = max(0, (resized_w - target_w) // 2)
    top = max(0, (resized_h - target_h) // 2)
    output = np.empty((158, target_h, target_w, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
        if image.size != (resized_w, resized_h):
            image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        if image.size != (target_w, target_h):
            image = image.crop((left, top, left + target_w, top + target_h))
        output[index] = np.asarray(image, dtype=np.uint8)
    return output, {
        "source_h": source_h,
        "source_w": source_w,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "crop_top": top,
        "crop_left": left,
        "target_h": target_h,
        "target_w": target_w,
    }


def _invert_se3(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float32)
    rotation = np.swapaxes(matrices[..., :3, :3], -1, -2)
    output = np.zeros_like(matrices)
    output[..., :3, :3] = rotation
    output[..., :3, 3] = -np.einsum("...ij,...j->...i", rotation, matrices[..., :3, 3])
    output[..., 3, 3] = 1
    return output


def _load_intrinsics_candidate(
    archive: Any,
    intrinsics_bytes: bytes | None,
) -> np.ndarray | None:
    if intrinsics_bytes:
        loaded = np.load(io.BytesIO(intrinsics_bytes), allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                for key in ("camera_K", "K", "intrinsics"):
                    if key in loaded.files:
                        return np.asarray(loaded[key])
                raise DataContractError("H3 intrinsics NPZ requires camera_K, K, or intrinsics")
            finally:
                loaded.close()
        return np.asarray(loaded)
    files = set(archive.files)
    for key in ("camera_K", "K", "intrinsics"):
        if key in files:
            return np.asarray(archive[key])
    if {"fx", "fy"} <= files:
        fx, fy = np.asarray(archive["fx"]), np.asarray(archive["fy"])
        cx = np.asarray(archive["cx"]) if "cx" in files else 0.5
        cy = np.asarray(archive["cy"]) if "cy" in files else 0.5
        return np.stack(np.broadcast_arrays(fx, fy, cx, cy), axis=-1)
    return None


def _normalise_K(
    candidate: np.ndarray | None,
    source_indices: Sequence[int],
    transform: Mapping[str, int],
    *,
    enforce_focal_guard: bool = True,
) -> np.ndarray:
    count = len(source_indices)
    if candidate is None:
        output = np.zeros((count, 3, 3), dtype=np.float32)
        output[:, 0, 0] = 969.6969696969696 / (960.0 * 2.0)
        output[:, 1, 1] = 969.6969696969696 / (540.0 * 2.0)
        output[:, 0, 2] = output[:, 1, 2] = 0.5
        output[:, 2, 2] = 1
        return output
    value = np.asarray(candidate, dtype=np.float64)
    maximum = max(source_indices)
    if value.shape == (3, 3):
        matrices = np.broadcast_to(value, (count, 3, 3)).copy()
    elif value.ndim == 3 and value.shape[1:] == (3, 3):
        if value.shape[0] > maximum:
            matrices = value[np.asarray(source_indices)].copy()
        elif value.shape[0] == 1:
            matrices = np.broadcast_to(value, (count, 3, 3)).copy()
        else:
            raise DataContractError("H3 camera K cannot align to source indices")
    elif value.shape == (4,) or (value.ndim == 2 and value.shape[1] == 4):
        vectors = value.reshape(1, 4) if value.shape == (4,) else value
        if vectors.shape[0] > maximum:
            vectors = vectors[np.asarray(source_indices)]
        elif vectors.shape[0] == 1:
            vectors = np.broadcast_to(vectors, (count, 4))
        else:
            raise DataContractError("H3 camera intrinsics vectors cannot align")
        matrices = np.zeros((count, 3, 3), dtype=np.float64)
        matrices[:, 0, 0] = vectors[:, 0]
        matrices[:, 1, 1] = vectors[:, 1]
        matrices[:, 0, 2] = vectors[:, 2]
        matrices[:, 1, 2] = vectors[:, 3]
        matrices[:, 2, 2] = 1
    else:
        raise DataContractError(f"unsupported H3 camera K shape {value.shape}")
    normalized = bool(
        np.nanmax(np.abs(matrices[:, (0, 1), (0, 1)])) <= 4
        and np.nanmax(np.abs(matrices[:, (0, 1), (2, 2)])) <= 4
    )
    if normalized:
        matrices[:, 0, :] *= transform["source_w"]
        matrices[:, 1, :] *= transform["source_h"]
    matrices[:, 0, 0] *= transform["resized_w"] / transform["source_w"]
    matrices[:, 1, 1] *= transform["resized_h"] / transform["source_h"]
    matrices[:, 0, 2] = (
        matrices[:, 0, 2] * transform["resized_w"] / transform["source_w"] - transform["crop_left"]
    )
    matrices[:, 1, 2] = (
        matrices[:, 1, 2] * transform["resized_h"] / transform["source_h"] - transform["crop_top"]
    )
    matrices[:, 0, :] /= transform["target_w"]
    matrices[:, 1, :] /= transform["target_h"]
    matrices[:, 2, :] = (0, 0, 1)
    focal = matrices[:, (0, 1), (0, 1)]
    if not np.isfinite(matrices).all() or (
        enforce_focal_guard and (np.any(focal <= 0) or np.any(focal > 4))
    ):
        raise DataContractError("H3 camera K exceeds the normalized PRoPE guards")
    return matrices.astype(np.float32)


def decode_camera(
    camera_bytes: bytes,
    source_indices: Sequence[int],
    transform: Mapping[str, int],
    *,
    intrinsics_bytes: bytes | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select absolute C2W and normalized K rows for one exact H3 window."""

    with np.load(io.BytesIO(camera_bytes), allow_pickle=False) as archive:
        files = set(archive.files)
        if "vipe_status" in files:
            status = str(np.asarray(archive["vipe_status"]).reshape(-1)[0]).strip().lower()
            if status != "ok":
                raise DataContractError(f"VIPE camera status is {status!r}")
        if "vipe_c2w" in files:
            c2w_full = np.asarray(archive["vipe_c2w"], dtype=np.float32)
        elif "vipe_w2c" in files:
            c2w_full = _invert_se3(archive["vipe_w2c"])
        elif "camera_c2w" in files or "c2w" in files:
            key = "camera_c2w" if "camera_c2w" in files else "c2w"
            c2w_full = np.asarray(archive[key], dtype=np.float32)
        elif "camera_w2c" in files or "w2c" in files:
            key = "camera_w2c" if "camera_w2c" in files else "w2c"
            c2w_full = _invert_se3(archive[key])
        else:
            raise DataContractError("H3 camera NPZ has no explicit c2w/w2c array")
        K = _load_intrinsics_candidate(archive, intrinsics_bytes)
    if c2w_full.ndim != 3 or c2w_full.shape[1:] != (4, 4):
        raise DataContractError("H3 camera pose array must be [N,4,4]")
    if c2w_full.shape[0] <= max(source_indices):
        raise DataContractError("H3 camera track is shorter than the selected window")
    c2w = c2w_full[np.asarray(source_indices)].copy()
    if not np.isfinite(c2w).all():
        raise DataContractError("H3 camera c2w contains non-finite values")
    return c2w.astype(np.float32), _normalise_K(K, source_indices, transform)


__all__ = [
    "decode_camera",
    "decode_resize_video",
    "normalize_raw_source_windows",
]
