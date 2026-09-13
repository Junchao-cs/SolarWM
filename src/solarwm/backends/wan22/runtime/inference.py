"""Unified Wan inference/validation runner with an injectable generation adapter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from solarwm.config.loader import canonical_json
from solarwm.data.archive import RawSampleReader, TarShardReader
from solarwm.data.camera import CameraGuardError, CameraGuards, load_camera_npz
from solarwm.data.index import IndexRow, read_index, resolve_index_path, select_index_rows
from solarwm.data.sampling import SamplePlan, SamplingConfig, frame_offsets
from solarwm.data.transport import resolver_from_config
from solarwm.errors import BackendContractError, DataContractError
from solarwm.inference import (
    GeneratedFile,
    GeneratedSample,
    InferenceCase,
    InferenceEngine,
    encode_compare_mp4,
    publish_comparison_complete,
    publish_comparison_partition,
)
from solarwm.inference.validation_plan import (
    load_validation_plan,
    publish_validation_plan,
    validation_plan_key,
    validation_plan_payload,
)
from solarwm.runtime.create_only import publish_directory_no_replace
from solarwm.runtime.output_layout import (
    camera_inference_output_layout,
    cleanup_validation_staging,
    portable_output_component,
    public_validation_dir,
    validation_staging_root,
)

from ..generation import GenerationPass, GenerationPlan, resolve_generation_plan
from .data import build_camera_tokens, decode_video


@dataclass(frozen=True)
class WanGenerationSummary:
    output_dir: Path
    family: str
    cases: int
    passes: tuple[str, ...]
    weights_ids: Mapping[str, str]
    complete_digest: str
    validation_plan_source: str = ""
    validation_plan_path: str = ""
    publication_layout: str = ""
    publication_complete_path: str = ""
    publication_complete_digest: str = ""


@dataclass(frozen=True)
class _PreparedCase:
    pixels: Any
    camera: Mapping[str, Any]
    source_pixel_frames: int
    publication_c2w: np.ndarray | None = None
    publication_pixel_frames: int | None = None
    repeat_first_frame: bool = False


@dataclass(frozen=True)
class _MaterializedCandidate:
    raw: Any
    sample_plan: SamplePlan
    noise_seed: int
    start_adjusted: bool
    pass_rollouts: Mapping[str, int]
    source_pixel_frames: int
    camera_fingerprint: str
    prepared: _PreparedCase | None
    publication_source_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class _DeferredCameraCase:
    row: IndexRow
    sample_plan: SamplePlan
    noise_seed: int
    pass_rollouts: Mapping[str, int]
    source_pixel_frames: int
    prompt: str
    camera_fingerprint: str
    publication_source_frame_indices: tuple[int, ...]


@dataclass
class _DeferredCameraInputs:
    rows: tuple[IndexRow, ...]
    resolver: Any
    runtime_guards: CameraGuards
    manifest_guards: CameraGuards
    cases: dict[int, _DeferredCameraCase]
    shards: TarShardReader | None = None
    reader: RawSampleReader | None = None


class _FamilyAdapter:
    def __init__(self, provider: Any, family: str) -> None:
        self.provider = provider
        self.family = family

    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        return self.provider.generate(case, weights_id=weights_id)


def _atomic_write(path: Path, value: bytes) -> str:
    if path.exists():
        raise BackendContractError(f"generation control already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.blake2s(value).hexdigest()


def _checkpoint_inventory_id(path: Path) -> str:
    size = path.stat().st_size
    if size <= 0:
        raise BackendContractError(f"Wan inference checkpoint is empty: {path}")
    return f"inventory:file={path.name}:bytes={size}"


def _seed_for(video_id: str, step: int, slot: int, rank: int) -> int:
    key = f"{video_id}|{step}|{slot}|{rank}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") & 0x7FFFFFFF


def _flowmap_schedule(
    num_inference_steps: int,
    *,
    shift: float,
    num_train_timesteps: int,
    device: Any,
) -> tuple[Any, Any]:
    """Construct the validation sampler's FP32 schedule exactly."""

    import torch

    if num_inference_steps < 1 or shift <= 0 or num_train_timesteps < 1:
        raise BackendContractError("flow-map inference schedule values must be positive")
    base = torch.linspace(
        1.0,
        0.0,
        int(num_inference_steps) + 1,
        device=device,
        dtype=torch.float32,
    )
    shifted = float(shift) * base / (1.0 + (float(shift) - 1.0) * base)
    shifted[-1] = 0.0
    raw = shifted * float(num_train_timesteps)
    return raw[:-1].contiguous(), raw[1:].contiguous()


def _num_frames(row: IndexRow) -> int:
    manifest = row.values.get("manifest", {})
    video = manifest.get("video", {}) if isinstance(manifest, Mapping) else {}
    try:
        return int(row.values.get("num_frames") or video["num_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(
            f"validation row {row.sample_id!r} lacks source num_frames"
        ) from exc


def _source_fps(row: IndexRow) -> float:
    manifest = row.values.get("manifest", {})
    video = manifest.get("video", {}) if isinstance(manifest, Mapping) else {}
    try:
        value = float(row.values.get("fps") or video.get("fps") or 0.0)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"validation row {row.sample_id!r} has invalid fps") from exc
    if value <= 0 or not np.isfinite(value):
        raise DataContractError(f"validation row {row.sample_id!r} needs a positive finite fps")
    return value


def _source_variable_rollout_latents(
    *,
    num_frames: int,
    max_latent_frames: int,
    min_latent_frames: int,
    num_frame_per_block: int,
) -> int:
    """Map source length to a block-aligned Wan rollout length."""

    values = {
        "num_frames": int(num_frames),
        "max_latent_frames": int(max_latent_frames),
        "min_latent_frames": int(min_latent_frames),
        "num_frame_per_block": int(num_frame_per_block),
    }
    if any(value < 1 for value in values.values()):
        raise DataContractError("source-variable validation rollout values must be positive")
    block = values["num_frame_per_block"]
    if (
        values["max_latent_frames"] % block
        or values["min_latent_frames"] % block
        or values["min_latent_frames"] > values["max_latent_frames"]
    ):
        raise DataContractError(
            "source-variable validation rollout bounds must be ordered and "
            "num_frame_per_block aligned"
        )
    available = 1 + (values["num_frames"] - 1) // 4
    aligned = (available // block) * block
    selected = min(values["max_latent_frames"], aligned)
    if selected < values["min_latent_frames"]:
        raise DataContractError(
            f"source num_frames={values['num_frames']} cannot support minimum "
            f"rollout {values['min_latent_frames']} latents"
        )
    return selected


def _camera_length_rollout_latents(
    *,
    num_frames: int,
    source_fps: float,
    output_fps: float,
    num_frame_per_block: int,
) -> int:
    """Return the smallest chunk-aligned rollout covering the full camera."""

    if num_frames < 1 or num_frame_per_block < 1:
        raise DataContractError("camera-length rollout values must be positive")
    if (
        source_fps <= 0
        or output_fps <= 0
        or not np.isfinite(source_fps)
        or not np.isfinite(output_fps)
    ):
        raise DataContractError("camera-length rollout requires positive finite fps")

    publication_pixels = int(np.floor((num_frames - 1) * output_fps / source_fps)) + 1
    required_latents = 1 + int(np.ceil((publication_pixels - 1) / 4))
    block = int(num_frame_per_block)
    return max(block, ((required_latents + block - 1) // block) * block)


def _camera_publication_source_frame_indices(
    row: IndexRow,
    *,
    output_fps: float,
) -> tuple[int, ...]:
    """Map the complete source duration to exact output-frame camera rows."""

    num_frames = _num_frames(row)
    source_fps = _source_fps(row)
    if output_fps <= 0 or not np.isfinite(output_fps):
        raise DataContractError("camera publication requires positive finite output fps")
    pixel_frames = int(np.floor((num_frames - 1) * output_fps / source_fps)) + 1
    offsets = frame_offsets(
        SamplingConfig(
            seed=0,
            pixel_frames=pixel_frames,
            random_start=False,
            clip_seconds=(pixel_frames - 1) / output_fps,
            output_fps=output_fps,
            shuffle_buffer=1,
        ),
        source_fps,
    )
    if len(offsets) != pixel_frames or int(offsets[-1]) >= num_frames:
        raise DataContractError(
            f"camera publication source mapping is invalid for {row.sample_id!r}"
        )
    return tuple(int(value) for value in offsets)


def _camera_publication_identity(row: IndexRow) -> tuple[str, str]:
    physical_dataset = row.values.get("physical_generation") or _row_dataset(row)
    return (
        portable_output_component(
            physical_dataset,
            field=f"camera publication sample {row.sample_id!r} physical dataset",
        ),
        portable_output_component(
            row.values.get("clip_id"),
            field=f"camera publication sample {row.sample_id!r} clip_id",
        ),
    )


def _camera_publication_metadata(
    row: IndexRow,
    source_frame_indices: Sequence[int],
) -> dict[str, Any]:
    physical_dataset, publish_stem = _camera_publication_identity(row)
    indices = tuple(int(value) for value in source_frame_indices)
    if not indices:
        raise BackendContractError("camera dataset publication lacks source-frame mapping")
    return {
        "physical_dataset": physical_dataset,
        "publish_stem": publish_stem,
        "publication_pixel_frames": len(indices),
        "publication_source_frame_last": indices[-1],
        "camera_publication_convention": "authoritative_absolute_c2w",
    }


def _authoritative_publication_c2w(
    payload: bytes,
    *,
    array_key: str,
    source_frame_indices: Sequence[int],
) -> np.ndarray:
    """Select source absolute C2W rows and apply only the public FP64 cast."""

    matrices, storage = load_camera_npz(payload, array_key)
    if storage != "absolute_c2w":
        raise DataContractError(
            "camera dataset publication requires authoritative absolute C2W input"
        )
    source_indices = np.asarray(tuple(source_frame_indices), dtype=np.int64)
    if (
        source_indices.ndim != 1
        or not len(source_indices)
        or source_indices.min(initial=0) < 0
        or source_indices.max(initial=0) >= len(matrices)
    ):
        raise DataContractError(
            "camera publication frames are outside the authoritative trajectory"
        )
    return np.ascontiguousarray(matrices[source_indices], dtype=np.float64)


def _source_rollout_latent_frames(
    row: IndexRow,
    generation_pass: GenerationPass,
    *,
    num_frame_per_block: int,
    output_fps: float,
    camera_length: bool = False,
) -> int:
    if camera_length:
        try:
            return _camera_length_rollout_latents(
                num_frames=_num_frames(row),
                source_fps=_source_fps(row),
                output_fps=output_fps,
                num_frame_per_block=num_frame_per_block,
            )
        except DataContractError as exc:
            raise DataContractError(f"inference source {row.sample_id!r}: {exc}") from exc
    if not generation_pass.variable_rollout_by_source:
        return generation_pass.rollout_latent_frames
    try:
        return _source_variable_rollout_latents(
            num_frames=_num_frames(row),
            max_latent_frames=generation_pass.rollout_latent_frames,
            min_latent_frames=generation_pass.min_rollout_latent_frames,
            num_frame_per_block=num_frame_per_block,
        )
    except DataContractError as exc:
        raise DataContractError(f"validation source {row.sample_id!r}: {exc}") from exc


def _test_index_path(config: Mapping[str, Any], plan: GenerationPlan) -> Path:
    del plan
    return resolve_index_path(config["data"], "test_index")


def _plan_case(
    row: IndexRow,
    *,
    slot: int,
    pixel_frames: int,
    output_fps: float,
    base_noise_seed: int,
    variable_rollout_by_source: bool = False,
    start_at_first_frame: bool = False,
    hold_last_source_frame: bool = False,
) -> tuple[SamplePlan, int, bool]:
    source_fps = _source_fps(row)
    offsets = frame_offsets(
        SamplingConfig(
            seed=0,
            pixel_frames=pixel_frames,
            random_start=False,
            fixed_start_from_index=True,
            clip_seconds=(pixel_frames - 1) / output_fps,
            output_fps=output_fps,
            shuffle_buffer=1,
        ),
        source_fps,
    )
    source_frames = _num_frames(row)
    max_start = source_frames - int(offsets[-1]) - 1
    if max_start < 0:
        if not hold_last_source_frame or not start_at_first_frame:
            raise DataContractError(
                f"validation source {row.sample_id!r} is shorter than {pixel_frames} output frames"
            )
        max_start = 0
    raw_start = row.values.get("start_frame")
    if start_at_first_frame:
        start = 0
    elif raw_start is None:
        # Derive a repeatable source window from the selected recipe row and
        # its logical validation slot.
        seed = _seed_for(row.key, 0, 0, slot)
        start = int(np.random.RandomState(seed).randint(0, max_start + 1))
    else:
        try:
            start = int(raw_start)
        except (TypeError, ValueError) as exc:
            raise DataContractError(
                f"validation row {row.sample_id!r} has invalid fixed start"
            ) from exc
    start_adjusted = False
    if not 0 <= start <= max_start:
        if not variable_rollout_by_source:
            raise DataContractError(
                f"validation start {start} for {row.sample_id!r} is outside max {max_start}"
            )
        seed = _seed_for(
            f"variable-rollout-start:{row.key}",
            0,
            0,
            slot,
        )
        start = int(np.random.RandomState(seed).randint(0, max_start + 1))
        start_adjusted = True
    raw_noise_seed = row.values.get("validation_noise_seed")
    if raw_noise_seed is None:
        noise_seed = _seed_for("validation-noise", 0, slot, base_noise_seed)
    else:
        if type(raw_noise_seed) is not int or not 0 <= raw_noise_seed < 2**63:
            raise DataContractError(
                f"validation row {row.sample_id!r} has invalid validation_noise_seed"
            )
        noise_seed = raw_noise_seed
    indices = tuple(
        min(int(start + value), source_frames - 1) if hold_last_source_frame else int(start + value)
        for value in offsets
    )
    return (
        SamplePlan(
            sample_id=row.sample_id,
            key=row.key,
            shard=row.shard,
            row_ordinal=row.ordinal,
            repeat_ordinal=0,
            epoch=0,
            start_frame=start,
            source_frame_indices=indices,
            reader_rank=0,
            worker_id=0,
        ),
        noise_seed,
        start_adjusted,
    )


def _row_dataset(row: IndexRow) -> str:
    return str(row.values.get("dataset") or row.values.get("source_dataset") or "")


def _checkpoint_file(config: Mapping[str, Any]) -> Path:
    source = Path(str(config["checkpoint"]["path"])).resolve()
    if source.is_dir():
        from solarwm.checkpoint import verify_checkpoint

        try:
            verified = verify_checkpoint(source)
        except Exception as exc:
            raise BackendContractError(
                f"Wan inference checkpoint transaction is invalid: {exc}"
            ) from exc
        paths = {record.path for record in verified.files}
        if paths != {"model.pt"}:
            raise BackendContractError(
                "Wan inference checkpoint transaction must contain exactly model.pt"
            )
        contract = verified.contract
        expected = {
            "family": str(config["model"]["family"]),
            "stage": str(config["train"]["stage"]),
            "objective": str(config["train"]["objective"]),
            "camera_translation_transform": str(config["model"]["camera_translation_transform"]),
        }
        drift = {
            field: {"actual": getattr(contract, field), "expected": value}
            for field, value in expected.items()
            if str(getattr(contract, field)) != value
        }
        if drift:
            raise BackendContractError(
                f"Wan inference checkpoint transaction contract differs: {drift}"
            )
        source = verified.path / "model.pt"
    if not source.is_file():
        raise BackendContractError(f"Wan inference checkpoint is missing: {source}")
    return source


def _validate_checkpoint_config(
    saved: Any,
    current: Mapping[str, Any],
) -> None:
    if saved is None:
        raise BackendContractError(
            "Wan inference checkpoint lacks its training config; family/stage/camera "
            "compatibility cannot be proven"
        )
    if not isinstance(saved, Mapping):
        try:
            from omegaconf import OmegaConf

            saved = OmegaConf.to_container(saved, resolve=True)
        except Exception as exc:
            raise BackendContractError(
                "Wan inference checkpoint config is not a readable mapping"
            ) from exc
    if not isinstance(saved, Mapping):
        raise BackendContractError("Wan inference checkpoint config is not a mapping")
    saved_model = saved.get("model", {})
    saved_train = saved.get("train", {})
    if not isinstance(saved_model, Mapping):
        saved_model = {}
    if not isinstance(saved_train, Mapping):
        saved_train = {}
    expected = {
        "model.family": (
            saved_model.get("family"),
            str(current["model"]["family"]),
        ),
        "model.camera_translation_transform": (
            saved_model.get("camera_translation_transform", "linear"),
            str(current["model"]["camera_translation_transform"]),
        ),
        "train.stage": (
            saved_train.get("stage"),
            str(current["train"]["stage"]),
        ),
        "train.objective": (
            saved_train.get("objective"),
            str(current["train"]["objective"]),
        ),
    }
    drift: dict[str, Any] = {}
    for field, (actual, wanted) in expected.items():
        if str(actual) != wanted:
            drift[field] = {"actual": actual, "expected": wanted}
    if drift:
        raise BackendContractError(f"Wan inference checkpoint semantic contract differs: {drift}")


def _invert_se3(matrices: Any) -> Any:
    import torch

    rotation = matrices[..., :3, :3]
    rotation_t = rotation.transpose(-1, -2)
    result = torch.zeros_like(matrices)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_t, matrices[..., :3, 3])
    result[..., 3, 3] = 1
    return result


def _camera_window(
    camera: Mapping[str, Any],
    *,
    frame_sequence_length: int,
    start: int,
    end: int,
) -> Mapping[str, Any]:
    viewmats = camera["viewmats"][:, ::frame_sequence_length].float()
    if end > viewmats.shape[1] or not 0 <= start < end:
        raise BackendContractError("Wan inference camera window is out of bounds")
    c2w = _invert_se3(viewmats)
    relative_c2w = torch_matmul(viewmats[:, start : start + 1], c2w[:, start:end])
    rebased = _invert_se3(relative_c2w)
    tokens = (
        rebased.unsqueeze(2)
        .expand(-1, -1, frame_sequence_length, -1, -1)
        .reshape(rebased.shape[0], -1, 4, 4)
        .contiguous()
    )
    token_start = start * frame_sequence_length
    token_end = end * frame_sequence_length
    return {
        "viewmats": tokens,
        "K": camera["K"][:, token_start:token_end].contiguous(),
    }


def torch_matmul(left: Any, right: Any) -> Any:
    """Late-bound helper keeps importing this module allocation-free."""

    import torch

    return torch.matmul(left, right)


def _encode_mp4(frames: Any, *, fps: float) -> bytes:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise BackendContractError("Wan inference requires ffmpeg to encode MP4 artifacts")
    array = (
        ((frames[0].float().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .to(dtype=__import__("torch").uint8)
        .cpu()
        .numpy()
    )
    if array.ndim != 4 or array.shape[1] != 3:
        raise BackendContractError(f"Wan VAE decoded unexpected video shape {tuple(array.shape)}")
    array = np.transpose(array, (0, 2, 3, 1))
    height, width = int(array.shape[1]), int(array.shape[2])
    with tempfile.TemporaryDirectory(prefix="solarwm-wan-mp4-") as directory:
        target = Path(directory) / "sample.mp4"
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{float(fps):g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(target),
        ]
        result = subprocess.run(
            command,
            input=array.tobytes(),
            capture_output=True,
            check=False,
        )
        if result.returncode or not target.is_file():
            detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
            raise BackendContractError(f"ffmpeg failed to encode Wan output: {detail}")
        return target.read_bytes()


def _encode_compare_mp4(
    generated: Any,
    prepared: _PreparedCase,
    *,
    fps: float,
) -> bytes:
    """Adapt Wan tensors to the shared comparison encoder."""

    reference = prepared.pixels
    if prepared.repeat_first_frame:
        reference = reference[:1].expand(int(generated.shape[1]), -1, -1, -1)
    return encode_compare_mp4(
        reference.unsqueeze(0),
        generated,
        fps=fps,
        layout="btchw",
        value_range="minus_one_one",
    )


class _RawVideoFileWriter:
    """Stream RGB frames into a private H.264 file on the output filesystem."""

    def __init__(self, path: Path, *, fps: float, height: int, width: int) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise BackendContractError("Wan streaming inference requires ffmpeg")
        self.path = path
        self.height = int(height)
        self.width = int(width)
        self._closed = False
        self._process = subprocess.Popen(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{self.width}x{self.height}",
                "-r",
                f"{float(fps):g}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frames: np.ndarray) -> None:
        expected = (self.height, self.width, 3)
        if frames.ndim != 4 or tuple(frames.shape[1:]) != expected or frames.dtype != np.uint8:
            raise BackendContractError(
                "Wan streaming encoder received invalid RGB frames: "
                f"shape={frames.shape} dtype={frames.dtype} expected=[T,{expected}] uint8"
            )
        stream = self._process.stdin
        if self._closed or stream is None:
            raise BackendContractError("Wan streaming encoder is already closed")
        try:
            remaining = memoryview(np.ascontiguousarray(frames)).cast("B")
            while remaining:
                written = stream.write(remaining)
                if written is None or written <= 0:
                    raise BrokenPipeError("ffmpeg input made no progress")
                remaining = remaining[written:]
        except (BrokenPipeError, OSError) as exc:
            raise BackendContractError(
                f"ffmpeg stopped while encoding Wan streaming output: {self._error_tail()}"
            ) from exc

    def _error_tail(self) -> str:
        stream = self._process.stderr
        if stream is None:
            return "stderr unavailable"
        try:
            return stream.read().decode("utf-8", errors="replace")[-2000:]
        except OSError:
            return "stderr unavailable"

    def finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        detail = self._error_tail()
        returncode = self._process.wait()
        if returncode or not self.path.is_file() or self.path.stat().st_size <= 0:
            raise BackendContractError(f"ffmpeg failed to encode Wan streaming output: {detail}")
        with self.path.open("rb") as handle:
            os.fsync(handle.fileno())

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        self._process.wait()
        self._closed = True


@dataclass(frozen=True)
class _StreamedStage2Artifacts:
    video: GeneratedFile
    compare: GeneratedFile
    shape: tuple[int, ...]


def _temporary_mp4(root: Path, *, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    descriptor, value = tempfile.mkstemp(prefix=f".solarwm-{label}-", suffix=".mp4", dir=root)
    os.close(descriptor)
    return Path(value)


def _encode_stage2_streaming(
    vae: Any,
    latents: Any,
    prepared: _PreparedCase,
    *,
    target_pixel_frames: int,
    fps: float,
    temporary_root: Path,
    chunk_latent_frames: int,
) -> _StreamedStage2Artifacts:
    """Decode, compare, and encode a long Stage2 rollout with bounded host memory."""

    import torch

    if target_pixel_frames < 1:
        raise BackendContractError("Wan streaming publication length must be positive")
    expected_model_frames = 1 + 4 * (int(latents.shape[1]) - 1)
    pixels = prepared.pixels
    if pixels.ndim != 4 or int(pixels.shape[1]) != 3 or int(pixels.shape[0]) < 1:
        raise BackendContractError(
            f"Wan streaming comparison input has invalid shape {tuple(pixels.shape)}"
        )
    repeat_first = bool(prepared.repeat_first_frame)
    if not repeat_first and int(pixels.shape[0]) != target_pixel_frames:
        raise BackendContractError(
            "Wan streaming comparison input length differs from publication length"
        )

    video_path = _temporary_mp4(temporary_root, label="generate")
    compare_path = _temporary_mp4(temporary_root, label="compare")
    video_writer: _RawVideoFileWriter | None = None
    compare_writer: _RawVideoFileWriter | None = None
    decoded_model_frames = 0
    written_frames = 0
    last_generated: Any | None = None
    channels = 0
    height = 0
    width = 0

    def write_chunk(chunk: Any, *, reference_start: int) -> None:
        nonlocal video_writer, compare_writer, channels, height, width, last_generated
        if chunk.ndim != 5 or int(chunk.shape[0]) != 1 or int(chunk.shape[2]) != 3:
            raise BackendContractError(
                f"Wan streaming VAE returned invalid BTCHW tile {tuple(chunk.shape)}"
            )
        count = int(chunk.shape[1])
        channels, height, width = (int(value) for value in chunk.shape[2:])
        if video_writer is None:
            video_writer = _RawVideoFileWriter(
                video_path,
                fps=fps,
                height=height,
                width=width,
            )
            compare_writer = _RawVideoFileWriter(
                compare_path,
                fps=fps,
                height=height,
                width=width * 2,
            )
        if video_writer.height != height or video_writer.width != width:
            raise BackendContractError("Wan streaming VAE tile dimensions changed")

        generated = chunk[0].detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(generated).all().item()):
            raise BackendContractError("Wan streaming VAE decode produced non-finite pixels")
        generated = generated.clamp(-1, 1)
        generated_rgb = (
            ((generated + 1.0) * 127.5)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy()
        )
        video_writer.write(generated_rgb)

        if repeat_first:
            reference = pixels[:1].expand(count, -1, -1, -1)
        else:
            reference = pixels[reference_start : reference_start + count]
        if int(reference.shape[0]) != count or tuple(reference.shape[1:]) != (
            channels,
            height,
            width,
        ):
            raise BackendContractError("Wan streaming comparison tile dimensions differ")
        reference_rgb = (
            torch.nan_to_num(
                reference.detach().to(device="cpu", dtype=torch.float32).mul(0.5).add(0.5),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            .clamp_(0, 1)
            .mul_(255.0)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy()
        )
        generated_compare_rgb = (
            generated.mul(0.5)
            .add(0.5)
            .clamp_(0, 1)
            .mul_(255.0)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy()
        )
        compare_writer.write(np.concatenate((reference_rgb, generated_compare_rgb), axis=2))
        last_generated = chunk[:, -1:].detach().cpu()

    try:
        decode_chunks = getattr(vae, "decode_streaming_chunks", None)
        if not callable(decode_chunks):
            raise BackendContractError(
                "hour-scale Wan inference requires chunk-yielding VAE decode"
            )
        for chunk in decode_chunks(latents, chunk_latent_frames=chunk_latent_frames):
            count = int(chunk.shape[1])
            if decoded_model_frames + count > expected_model_frames:
                raise BackendContractError("Wan streaming VAE decoded too many frames")
            remaining = target_pixel_frames - written_frames
            if remaining > 0:
                published = chunk[:, : min(count, remaining)]
                write_chunk(published, reference_start=written_frames)
                written_frames += int(published.shape[1])
            decoded_model_frames += count
        if decoded_model_frames != expected_model_frames or last_generated is None:
            raise BackendContractError(
                "Wan streaming VAE frame count differs from latent-aligned output: "
                f"{decoded_model_frames} != {expected_model_frames}"
            )
        tail_frames = target_pixel_frames - written_frames
        if tail_frames:
            write_chunk(
                last_generated.expand(-1, tail_frames, -1, -1, -1),
                reference_start=written_frames,
            )
            written_frames += tail_frames
        if written_frames != target_pixel_frames:
            raise BackendContractError(
                "Wan streaming encoder frame count differs from publication length: "
                f"{written_frames} != {target_pixel_frames}"
            )
        if video_writer is None or compare_writer is None:
            raise BackendContractError("Wan streaming VAE produced no encodable frames")
        video_writer.finish()
        compare_writer.finish()
        return _StreamedStage2Artifacts(
            video=GeneratedFile(video_path),
            compare=GeneratedFile(compare_path),
            shape=(1, target_pixel_frames, channels, height, width),
        )
    except Exception:
        if video_writer is not None:
            video_writer.abort()
        if compare_writer is not None:
            compare_writer.abort()
        video_path.unlink(missing_ok=True)
        compare_path.unlink(missing_ok=True)
        raise


class CudaWanGenerationAdapter:
    """CUDA implementation of Wan samplers over logical DP/SP groups."""

    def __init__(self, config: Mapping[str, Any], plan: GenerationPlan) -> None:
        try:
            import torch
        except ImportError as exc:
            raise BackendContractError("Wan inference requires torch") from exc
        if not torch.cuda.is_available():
            raise BackendContractError(
                "Wan inference requires CUDA; inject an adapter for CPU contract tests"
            )
        from .components import build_online_components
        from .distributed import initialize_torchrun

        self.config = config
        self.plan = plan
        self.family = str(config["model"]["family"])
        sp_size = int(config["distributed"]["sequence_parallel_size"])
        self.topology = initialize_torchrun(sp_size)
        self.is_writer = self.topology.sp_rank == 0
        self.device = torch.device("cuda", self.topology.local_rank)
        self.checkpoint_path = _checkpoint_file(config)
        self.checkpoint_id = _checkpoint_inventory_id(self.checkpoint_path)
        self.diffusion, self.text_encoder, self.vae, self.base_report = build_online_components(
            config
        )
        inference_dtype = torch.bfloat16
        self.diffusion.module.eval().requires_grad_(False).to(
            device=self.device, dtype=inference_dtype
        )
        self.text_encoder.to(self.device, dtype=inference_dtype)
        self.vae.to(self.device, dtype=inference_dtype)
        self._loaded_role: str | None = None
        self._prepared: dict[int, _PreparedCase] = {}
        self._deferred_camera_inputs: _DeferredCameraInputs | None = None

    def sync(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        from .distributed import cleanup_torchrun

        try:
            self._close_deferred_camera_inputs()
        finally:
            cleanup_torchrun()

    def _close_deferred_camera_inputs(self) -> None:
        deferred = getattr(self, "_deferred_camera_inputs", None)
        if deferred is None:
            return
        self._deferred_camera_inputs = None
        try:
            if deferred.shards is not None:
                deferred.shards.close()
        finally:
            for slot in deferred.cases:
                self._prepared.pop(slot, None)
            deferred.cases.clear()
            deferred.reader = None
            deferred.shards = None

    def _materialize_deferred_camera_case(self, case: InferenceCase) -> None:
        deferred = getattr(self, "_deferred_camera_inputs", None)
        if deferred is None:
            raise BackendContractError("Wan adapter has no deferred camera inputs")
        descriptor = deferred.cases.get(case.slot)
        if descriptor is None:
            raise BackendContractError(
                f"Wan adapter has no deferred camera case {case.sample_id!r}"
            )
        if self._prepared:
            raise BackendContractError("Wan adapter retained another prepared camera case")
        metadata = case.metadata
        if (
            descriptor.row.sample_id != case.sample_id
            or descriptor.sample_plan.start_frame != case.start_frame
            or descriptor.noise_seed != case.noise_seed
            or descriptor.prompt != case.prompt
            or descriptor.camera_fingerprint != case.camera_fingerprint
            or descriptor.row.key != str(metadata.get("key", ""))
            or descriptor.row.shard != str(metadata.get("source_shard", ""))
            or descriptor.row.ordinal != int(metadata.get("source_row_ordinal", -1))
            or descriptor.source_pixel_frames != int(metadata.get("source_pixel_frames", -1))
            or descriptor.sample_plan.source_frame_indices[-1]
            != int(metadata.get("source_frame_last", -1))
            or len(descriptor.publication_source_frame_indices)
            != int(metadata.get("publication_pixel_frames", 0))
            or (
                descriptor.publication_source_frame_indices
                and descriptor.publication_source_frame_indices[-1]
                != int(metadata.get("publication_source_frame_last", -1))
            )
            or dict(descriptor.pass_rollouts)
            != dict(metadata.get("rollout_latent_frames_by_pass", {}))
        ):
            raise DataContractError(f"deferred camera case {case.sample_id!r} identity drifted")
        if deferred.reader is None:
            deferred.shards = TarShardReader(
                deferred.resolver,
                max_open=int(self.config["data"].get("tar_cache_size", 4)),
            )
            deferred.reader = RawSampleReader(deferred.rows, deferred.shards)
        raw = deferred.reader.materialize(descriptor.sample_plan)
        camera_fingerprint = hashlib.blake2s(raw.members["camera_member"]).hexdigest()
        if raw.caption != case.prompt or camera_fingerprint != case.camera_fingerprint:
            raise DataContractError(f"deferred camera case {case.sample_id!r} payload drifted")
        data = self.config["data"]
        configured_array_key = str(data["camera_array_key"])
        camera = build_camera_tokens(
            raw.members["camera_member"],
            descriptor.sample_plan.source_frame_indices,
            raw.manifest,
            source_fps=_source_fps(descriptor.row),
            output_fps=float(data.get("fps", 16.0)),
            frame_sequence_length=int(self.config["model"]["frame_sequence_length"]),
            guards=deferred.runtime_guards,
            manifest_guards=deferred.manifest_guards,
            configured_array_key=configured_array_key,
        )
        publication_indices = descriptor.publication_source_frame_indices
        decode_indices = publication_indices or descriptor.sample_plan.source_frame_indices
        video_manifest = raw.manifest.get("video", {})
        repeat_first_frame = bool(
            isinstance(video_manifest, Mapping)
            and str(video_manifest.get("frame_content", "")) == "static_first_frame"
        )
        if repeat_first_frame:
            decode_indices = decode_indices[:1]
        pixels = decode_video(
            raw.members["video_member"],
            decode_indices,
            height=int(data["height"]),
            width=int(data["width"]),
        )
        publication_c2w = None
        publication_pixel_frames = None
        if publication_indices:
            publication_c2w = _authoritative_publication_c2w(
                raw.members["camera_member"],
                array_key=configured_array_key,
                source_frame_indices=publication_indices,
            )
            publication_pixel_frames = len(publication_indices)
        self._prepared[case.slot] = _PreparedCase(
            pixels=pixels,
            camera=camera,
            source_pixel_frames=descriptor.source_pixel_frames,
            publication_c2w=publication_c2w,
            publication_pixel_frames=publication_pixel_frames,
            repeat_first_frame=repeat_first_frame,
        )

    def weight_id(self, role: str) -> str:
        if role not in {"live", "ema"}:
            raise BackendContractError(f"unknown Wan checkpoint weight role {role!r}")
        return f"{self.checkpoint_id}#{role}"

    def _checkpoint_state_field(self, role: str) -> str:
        if role == "live":
            return "generator"
        if role == "ema":
            return "generator_ema"
        raise BackendContractError(f"unknown Wan checkpoint weight role {role!r}")

    def _load_role(self, role: str) -> None:
        if self._loaded_role == role:
            return
        import torch

        from .checkpoint import normalize_model_state

        try:
            payload = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except Exception as exc:
            raise BackendContractError(
                f"cannot read Wan inference checkpoint {self.checkpoint_path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise BackendContractError("Wan inference checkpoint payload must be a mapping")
        _validate_checkpoint_config(
            payload.get("config"),
            self.config,
        )
        field = self._checkpoint_state_field(role)
        state = payload.get(field)
        if not isinstance(state, Mapping) or not state:
            raise BackendContractError(f"Wan inference checkpoint lacks {field}")
        normalized = normalize_model_state(state, field=field)
        result = self.diffusion.module.load_state_dict(normalized, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise BackendContractError(
                "Wan inference checkpoint load was not exact: "
                f"missing={result.missing_keys[:8]} "
                f"unexpected={result.unexpected_keys[:8]}"
            )
        self._loaded_role = role
        del payload, state, normalized
        torch.cuda.empty_cache()

    build_cases_returns_partition = True

    def build_cases(self, plan: GenerationPlan) -> tuple[InferenceCase, ...]:
        data = self.config["data"]
        validation = self.config["validation"]
        camera_length = (
            str(self.config.get("inference", {}).get("length", "fixed")).strip().lower() == "camera"
        )
        dataset_publication = camera_inference_output_layout(self.config) is not None
        if camera_length:
            self._close_deferred_camera_inputs()
        frozen_path_value = getattr(self, "validation_plan_path", None)
        frozen_path = Path(str(frozen_path_value)) if frozen_path_value is not None else None
        frozen_key = validation_plan_key("wan22", self.config)
        frozen_cases = (
            _load_validation_plan_collectively(
                frozen_path,
                backend="wan22",
                plan_key=frozen_key,
                expected_count=plan.sample_count,
            )
            if frozen_path is not None
            else None
        )
        self.validation_plan_source = "loaded" if frozen_cases is not None else "created"
        source_rows = read_index(_test_index_path(self.config, plan))
        if len(source_rows) < plan.sample_count:
            raise DataContractError(
                "recipe test index has fewer rows than validation.sample_count: "
                f"rows={len(source_rows)} sample_count={plan.sample_count}"
            )
        # Camera validity is known only after materialization. Shuffle the
        # complete recipe test index once, then assign slots only to successful
        # candidates. This keeps selection fixed without weakening guards or
        # duplicating a row when an earlier candidate is rejected.
        candidate_rows = select_index_rows(
            source_rows,
            sample_count=len(source_rows),
            seed=plan.selection_seed,
        )
        dp_world_size = int(self.topology.dp_world_size)
        dp_rank = int(self.topology.dp_rank)
        transport = data["transport"]
        runtime = self.config.get("runtime", {})
        resolver = resolver_from_config(
            str(transport["root"]),
            cache_dir=runtime.get("validation_cache_dir", transport.get("cache_dir")),
            max_gib=float(
                runtime.get(
                    "validation_cache_max_gib",
                    transport.get("cache_max_gib", 256),
                )
            ),
        )

        def validation_guard(name: str) -> float | None:
            raw = validation[name] if name in validation else data[name]
            return None if raw is None else float(raw)

        runtime_guards = CameraGuards(
            max_rel_translation=validation_guard("max_rel_translation"),
            max_camera_abs=validation_guard("max_camera_abs"),
        )
        manifest_guards = CameraGuards(
            max_rel_translation=float(data["max_rel_translation"]),
            max_camera_abs=float(data["max_camera_abs"]),
        )
        if camera_length:
            self._deferred_camera_inputs = _DeferredCameraInputs(
                rows=tuple(source_rows),
                resolver=resolver,
                runtime_guards=runtime_guards,
                manifest_guards=manifest_guards,
                cases={},
            )
        variable_rollout = camera_length or any(
            item.variable_rollout_by_source for item in plan.passes
        )
        dist, _, raw_world_size = _distributed_generation_context()
        sp_rank = int(getattr(self.topology, "sp_rank", 0))
        rejected_candidates: list[tuple[str, str]] = []
        selected_rows: list[tuple[int, IndexRow]] = []
        locally_materialized: dict[tuple[int, int], _MaterializedCandidate] = {}
        rows_by_id = {row.sample_id: row for row in source_rows}
        if len(rows_by_id) != len(source_rows):
            raise DataContractError("recipe test index contains duplicate sample IDs")
        with TarShardReader(resolver, max_open=int(data.get("tar_cache_size", 4))) as shards:
            # Candidate rows retain full-index ordinals, so random access must
            # address the complete source index rather than the seeded order.
            reader = RawSampleReader(source_rows, shards)
            camera_reader = RawSampleReader(
                source_rows,
                shards,
                member_fields=("camera_member",),
            )

            def materialize_candidate(row: IndexRow, slot: int) -> _MaterializedCandidate:
                pass_rollouts = {
                    item.name: _source_rollout_latent_frames(
                        row,
                        item,
                        num_frame_per_block=int(self.config["model"]["num_frame_per_block"]),
                        output_fps=float(data.get("fps", 16.0)),
                        camera_length=camera_length,
                    )
                    for item in plan.passes
                }
                source_latent_frames = max(pass_rollouts.values())
                source_pixel_frames = 1 + 4 * (source_latent_frames - 1)
                publication_source_frame_indices = (
                    _camera_publication_source_frame_indices(
                        row,
                        output_fps=float(data.get("fps", 16.0)),
                    )
                    if dataset_publication
                    else ()
                )
                sample_plan, noise_seed, start_adjusted = _plan_case(
                    row,
                    slot=slot,
                    pixel_frames=source_pixel_frames,
                    output_fps=float(data.get("fps", 16.0)),
                    base_noise_seed=plan.noise_seed,
                    variable_rollout_by_source=variable_rollout,
                    start_at_first_frame=camera_length,
                    hold_last_source_frame=camera_length,
                )
                raw = (camera_reader if camera_length else reader).materialize(sample_plan)
                pixels = None
                if not camera_length:
                    pixels = decode_video(
                        raw.members["video_member"],
                        sample_plan.source_frame_indices,
                        height=int(data["height"]),
                        width=int(data["width"]),
                    )
                camera = build_camera_tokens(
                    raw.members["camera_member"],
                    sample_plan.source_frame_indices,
                    raw.manifest,
                    source_fps=_source_fps(row),
                    output_fps=float(data.get("fps", 16.0)),
                    frame_sequence_length=int(self.config["model"]["frame_sequence_length"]),
                    guards=runtime_guards,
                    manifest_guards=manifest_guards,
                    configured_array_key=str(data["camera_array_key"]),
                )
                return _MaterializedCandidate(
                    raw=raw,
                    sample_plan=sample_plan,
                    noise_seed=noise_seed,
                    start_adjusted=start_adjusted,
                    pass_rollouts=pass_rollouts,
                    source_pixel_frames=source_pixel_frames,
                    camera_fingerprint=hashlib.blake2s(raw.members["camera_member"]).hexdigest(),
                    prepared=(
                        None
                        if camera_length
                        else _PreparedCase(
                            pixels=pixels,
                            camera=camera,
                            source_pixel_frames=source_pixel_frames,
                        )
                    ),
                    publication_source_frame_indices=publication_source_frame_indices,
                )

            def retain_candidate(
                row: IndexRow,
                slot: int,
                materialized: _MaterializedCandidate,
            ) -> None:
                if camera_length:
                    deferred = self._deferred_camera_inputs
                    if deferred is None or materialized.prepared is not None:
                        raise BackendContractError("Wan deferred camera state is invalid")
                    deferred.cases[slot] = _DeferredCameraCase(
                        row=row,
                        sample_plan=materialized.sample_plan,
                        noise_seed=materialized.noise_seed,
                        pass_rollouts=dict(materialized.pass_rollouts),
                        source_pixel_frames=materialized.source_pixel_frames,
                        prompt=str(materialized.raw.caption),
                        camera_fingerprint=materialized.camera_fingerprint,
                        publication_source_frame_indices=(
                            materialized.publication_source_frame_indices
                        ),
                    )
                    return
                if materialized.prepared is None:
                    raise BackendContractError("Wan fixed validation case lacks pixels")
                self._prepared[slot] = materialized.prepared

            if frozen_cases is not None:
                restored: list[InferenceCase] = []
                for case in frozen_cases:
                    if case.slot % dp_world_size != dp_rank:
                        continue
                    try:
                        row = rows_by_id[case.sample_id]
                    except KeyError as exc:
                        raise DataContractError(
                            f"frozen validation sample {case.sample_id!r} left the test index"
                        ) from exc
                    if (
                        str(case.metadata.get("key", "")) != row.key
                        or str(case.metadata.get("source_shard", "")) != row.shard
                        or int(case.metadata.get("source_row_ordinal", -1)) != row.ordinal
                    ):
                        raise DataContractError(
                            f"frozen validation sample {case.sample_id!r} index identity drifted"
                        )
                    try:
                        materialized = materialize_candidate(row, case.slot)
                    except CameraGuardError as exc:
                        raise DataContractError(
                            f"frozen validation sample {case.sample_id!r} changed camera validity"
                        ) from exc
                    if (
                        materialized.sample_plan.start_frame != case.start_frame
                        or materialized.noise_seed != case.noise_seed
                        or materialized.raw.caption != case.prompt
                        or materialized.camera_fingerprint != case.camera_fingerprint
                        or dict(materialized.pass_rollouts)
                        != dict(case.metadata.get("rollout_latent_frames_by_pass", {}))
                    ):
                        raise DataContractError(
                            f"frozen validation sample {case.sample_id!r} materialization drifted"
                        )
                    retain_candidate(row, case.slot, materialized)
                    if dataset_publication:
                        restored_metadata = dict(case.metadata)
                        restored_metadata.update(
                            _camera_publication_metadata(
                                row,
                                materialized.publication_source_frame_indices,
                            )
                        )
                        case = replace(case, metadata=restored_metadata)
                    restored.append(case)
                return tuple(restored)

            for candidate_ordinal, row in enumerate(candidate_rows):
                slot = len(selected_rows)
                evaluator_dp_rank = candidate_ordinal % dp_world_size
                local_status: dict[str, Any] | None = None
                materialized: _MaterializedCandidate | None = None
                if dist is None or (dp_rank == evaluator_dp_rank and sp_rank == 0):
                    try:
                        materialized = materialize_candidate(row, slot)
                        local_status = {
                            "candidate_ordinal": candidate_ordinal,
                            "sample_id": row.sample_id,
                            "accepted": True,
                            "error": None,
                        }
                    except CameraGuardError as exc:
                        local_status = {
                            "candidate_ordinal": candidate_ordinal,
                            "sample_id": row.sample_id,
                            "accepted": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    except Exception as exc:
                        local_status = {
                            "candidate_ordinal": candidate_ordinal,
                            "sample_id": row.sample_id,
                            "accepted": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                if dist is not None:
                    gathered: list[Any] = [None] * raw_world_size
                    dist.all_gather_object(gathered, local_status)
                    reports = [value for value in gathered if value is not None]
                    if len(reports) != 1:
                        raise BackendContractError(
                            "Wan validation candidate evaluation has invalid ownership: "
                            f"candidate={candidate_ordinal} reports={len(reports)}"
                        )
                    status = reports[0]
                else:
                    status = local_status
                if not isinstance(status, Mapping) or (
                    int(status.get("candidate_ordinal", -1)) != candidate_ordinal
                    or str(status.get("sample_id", "")) != row.sample_id
                ):
                    raise BackendContractError(
                        f"Wan validation candidate {candidate_ordinal} identity drifted"
                    )
                accepted = status.get("accepted")
                if accepted is None:
                    raise DataContractError(
                        f"validation candidate {row.sample_id!r} is invalid: {status.get('error')}"
                    )
                if accepted is False:
                    if len(rejected_candidates) < 8:
                        rejected_candidates.append(
                            (row.sample_id, str(status.get("error") or "camera guard rejected"))
                        )
                    continue
                selected_rows.append((candidate_ordinal, row))
                if materialized is not None and slot % dp_world_size == dp_rank:
                    locally_materialized[(candidate_ordinal, slot)] = materialized
                if len(selected_rows) == plan.sample_count:
                    break
            if len(selected_rows) != plan.sample_count:
                raise DataContractError(
                    "recipe test index has too few camera-safe validation candidates: "
                    f"valid={len(selected_rows)} sample_count={plan.sample_count} "
                    f"rows={len(source_rows)} first_rejections={rejected_candidates}"
                )

            cases: list[InferenceCase] = []
            for slot, (candidate_ordinal, row) in enumerate(selected_rows):
                if slot % dp_world_size != dp_rank:
                    continue
                materialized = locally_materialized.get((candidate_ordinal, slot))
                if materialized is None:
                    try:
                        materialized = materialize_candidate(row, slot)
                    except CameraGuardError as exc:
                        raise DataContractError(
                            f"validation candidate {row.sample_id!r} changed camera validity"
                        ) from exc
                sample_plan = materialized.sample_plan
                raw = materialized.raw
                publication_metadata: dict[str, Any] = {}
                if dataset_publication:
                    publication_metadata = _camera_publication_metadata(
                        row,
                        materialized.publication_source_frame_indices,
                    )
                cases.append(
                    InferenceCase(
                        slot=slot,
                        sample_id=row.sample_id,
                        prompt=raw.caption,
                        start_frame=sample_plan.start_frame,
                        noise_seed=materialized.noise_seed,
                        camera_fingerprint=materialized.camera_fingerprint,
                        metadata={
                            "key": row.key,
                            "source_shard": row.shard,
                            "source_row_ordinal": row.ordinal,
                            "dataset": _row_dataset(row),
                            "scene": row.values.get("scene"),
                            "source_num_frames": _num_frames(row),
                            "source_pixel_frames": materialized.source_pixel_frames,
                            "train_latent_frames": int(data["latent_frames"]),
                            "rollout_latent_frames_by_pass": materialized.pass_rollouts,
                            **({"rollout_length_source": "camera"} if camera_length else {}),
                            **publication_metadata,
                            "source_frame_last": sample_plan.source_frame_indices[-1],
                            "camera_translation_transform": str(
                                self.config["model"]["camera_translation_transform"]
                            ),
                            "selection_candidate_ordinal": candidate_ordinal,
                            "variable_rollout_start_adjusted": materialized.start_adjusted,
                            "validation_input_status": "selected",
                            "artifact_valid": True,
                        },
                    )
                )
                retain_candidate(row, slot, materialized)
        local_cases = tuple(cases)
        if frozen_path is not None:
            all_cases = _gather_partitioned_cases(local_cases)
            _publish_validation_plan_collectively(
                frozen_path,
                backend="wan22",
                plan_key=frozen_key,
                cases=all_cases,
                local_rank=int(getattr(self.topology, "local_rank", 0)),
            )
        return local_cases

    def _broadcast(self, tensor: Any) -> None:
        if self.topology.sp_size == 1:
            return
        import torch.distributed as dist

        from .sequence_parallel import get_sp_group

        source = self.topology.raw_rank - self.topology.sp_rank
        dist.broadcast(tensor, src=source, group=get_sp_group())

    def _conditions(
        self,
        case: InferenceCase,
        *,
        latent_frames: int,
    ) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], Any | None]:
        import torch

        from .codec import build_official_i2v_y

        prepared = self._prepared.get(case.slot)
        if prepared is None:
            raise BackendContractError(f"Wan adapter has no materialized case {case.sample_id!r}")
        metadata = case.metadata.get("generation_pass", {})
        output_latent_frames = (
            int(metadata.get("output_rollout_latent_frames", latent_frames))
            if isinstance(metadata, Mapping)
            else int(latent_frames)
        )
        output_pixel_frames = 1 + 4 * (output_latent_frames - 1)
        if output_pixel_frames > prepared.source_pixel_frames:
            raise BackendContractError("Wan generation pass exceeds materialized source window")
        condition_frames = 1 if self.family == "wan22_ti2v_5b" else output_pixel_frames
        pixels = (
            prepared.pixels[:condition_frames]
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=torch.bfloat16,
            )
        )
        with torch.no_grad():
            first_latent = self.vae.encode(pixels[:, :1].permute(0, 2, 1, 3, 4).contiguous()).to(
                torch.bfloat16
            )
            condition = self.text_encoder([case.prompt])
            model_y = (
                build_official_i2v_y(pixels, self.vae) if self.family == "wan22_i2v_a14b" else None
            )
        self._broadcast(first_latent)
        self._broadcast(condition["prompt_embeds"])
        if model_y is not None:
            self._broadcast(model_y)
        frame_sequence_length = int(self.config["model"]["frame_sequence_length"])
        tokens = latent_frames * frame_sequence_length
        output_tokens = output_latent_frames * frame_sequence_length
        viewmats = prepared.camera["viewmats"][:output_tokens]
        intrinsics = prepared.camera["K"][:output_tokens]
        if output_tokens < tokens:
            repeats = latent_frames - output_latent_frames
            viewmats = torch.cat(
                [
                    viewmats,
                    viewmats[-frame_sequence_length:].repeat(repeats, 1, 1),
                ],
                dim=0,
            )
            intrinsics = torch.cat(
                [
                    intrinsics,
                    intrinsics[-frame_sequence_length:].repeat(repeats, 1, 1),
                ],
                dim=0,
            )
        camera = {
            "viewmats": viewmats.unsqueeze(0).to(self.device),
            "K": intrinsics.unsqueeze(0).to(self.device),
        }
        return first_latent, condition, camera, model_y

    def _noise(self, shape: Sequence[int], generator: Any) -> Any:
        import torch

        value = torch.randn(
            tuple(int(item) for item in shape),
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._broadcast(value)
        return value

    def _bidirectional(
        self,
        generation_pass: GenerationPass,
        first_latent: Any,
        condition: Mapping[str, Any],
        camera: Mapping[str, Any],
        model_y: Any | None,
        generator: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        import torch

        from .scheduler import build_wan_flow_unipc_scheduler

        if generation_pass.solver != "unipc":
            raise BackendContractError("Wan bidirectional generation requires solver=unipc")
        latent_frames = generation_pass.rollout_latent_frames
        channels = int(self.config["model"]["latent_channels"])
        height = int(self.config["data"]["latent_shape"][-2])
        width = int(self.config["data"]["latent_shape"][-1])
        latents = self._noise((1, latent_frames, channels, height, width), generator)
        anchor_first = self.family == "wan22_ti2v_5b"
        if anchor_first:
            latents[:, 0] = first_latent[:, 0]
        scheduler = build_wan_flow_unipc_scheduler(
            num_train_timesteps=int(self.config["train"]["num_train_timesteps"]),
            shift=float(self.config["model"]["timestep_shift"]),
            num_inference_steps=generation_pass.num_inference_steps,
            device=self.device,
        )
        sequence = latent_frames * int(self.config["model"]["frame_sequence_length"])
        for timestep in scheduler.timesteps:
            per_frame = torch.full(
                (1, latent_frames),
                float(timestep.item()),
                device=self.device,
                dtype=torch.float32,
            )
            if anchor_first:
                per_frame[:, 0] = 0
            tokens = (
                per_frame.unsqueeze(-1)
                .expand(-1, -1, int(self.config["model"]["frame_sequence_length"]))
                .reshape(1, sequence)
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                flow = self.diffusion(
                    latents,
                    condition,
                    camera,
                    tokens,
                    i2v_y=model_y,
                    sequence_length=sequence,
                )
            if anchor_first:
                flow[:, 0] = 0
            latents = scheduler.step(flow.to(latents.dtype), timestep, latents, return_dict=False)[
                0
            ].to(torch.bfloat16)
            if anchor_first:
                latents[:, 0] = first_latent[:, 0]
        return latents, {
            "scheduler": "diffusers.UniPCMultistepScheduler",
            "prediction_type": "flow_prediction",
            "timesteps": [float(value.item()) for value in scheduler.timesteps],
            "flow_shift": float(self.config["model"]["timestep_shift"]),
        }

    def _autoregressive(
        self,
        generation_pass: GenerationPass,
        first_latent: Any,
        condition: Mapping[str, Any],
        camera: Mapping[str, Any],
        model_y: Any | None,
        generator: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        import torch

        from .scheduler import build_wan_flow_unipc_scheduler

        if self.family != "wan22_ti2v_5b":
            raise BackendContractError("A14B autoregressive generation is not supported")
        latent_frames = generation_pass.rollout_latent_frames
        chunk = int(self.config["model"]["num_frame_per_block"])
        if latent_frames % chunk:
            raise BackendContractError(
                "Wan autoregressive rollout must divide by num_frame_per_block"
            )
        channels = int(self.config["model"]["latent_channels"])
        height = int(self.config["data"]["latent_shape"][-2])
        width = int(self.config["data"]["latent_shape"][-1])
        output = torch.zeros(
            1,
            latent_frames,
            channels,
            height,
            width,
            device=self.device,
            dtype=torch.bfloat16,
        )
        output[:, 0] = first_latent[:, 0]
        maximum_prior = int(self.config["model"].get("max_prior_clean_chunks", 5))
        frame_sequence_length = int(self.config["model"]["frame_sequence_length"])
        context_noise = float(self.config["validation"].get("context_noise", 0.0))
        raw_t: list[float] = []
        raw_r: list[float] = []
        if generation_pass.solver == "flowmap":
            schedule_t, schedule_r = _flowmap_schedule(
                generation_pass.num_inference_steps,
                shift=float(self.config["model"]["timestep_shift"]),
                num_train_timesteps=int(self.config["train"]["num_train_timesteps"]),
                device=self.device,
            )
            raw_t = [float(value.item()) for value in schedule_t]
            raw_r = [float(value.item()) for value in schedule_r]
        elif generation_pass.solver != "unipc":
            raise BackendContractError(
                f"Wan autoregressive adapter does not implement {generation_pass.solver!r}"
            )

        for block in range(latent_frames // chunk):
            start = block * chunk
            end = start + chunk
            history_start = max(0, block - maximum_prior) * chunk
            clean_history = output[:, history_start:start] if start > history_start else None
            camera_window = _camera_window(
                camera,
                frame_sequence_length=frame_sequence_length,
                start=history_start,
                end=end,
            )
            latents = self._noise((1, chunk, channels, height, width), generator)
            if block == 0:
                latents[:, 0] = first_latent[:, 0]
            y_window = model_y[:, history_start:end] if model_y is not None else None

            if generation_pass.solver == "unipc":
                scheduler = build_wan_flow_unipc_scheduler(
                    num_train_timesteps=int(self.config["train"]["num_train_timesteps"]),
                    shift=float(self.config["model"]["timestep_shift"]),
                    num_inference_steps=generation_pass.num_inference_steps,
                    device=self.device,
                )
                pairs = [(value, None) for value in scheduler.timesteps]
            else:
                pairs = list(zip(schedule_t, schedule_r, strict=True))

            for timestep, target in pairs:
                per_frame = torch.full(
                    (1, chunk),
                    float(timestep.item()),
                    device=self.device,
                    dtype=torch.float32,
                )
                if block == 0:
                    per_frame[:, 0] = 0
                timestep_tokens = (
                    per_frame.unsqueeze(-1)
                    .expand(-1, -1, frame_sequence_length)
                    .reshape(1, chunk * frame_sequence_length)
                )
                target_tokens = None
                if target is not None:
                    target_per_frame = torch.full_like(per_frame, float(target.item()))
                    if block == 0:
                        target_per_frame[:, 0] = 0
                    target_tokens = (
                        target_per_frame.unsqueeze(-1)
                        .expand(-1, -1, frame_sequence_length)
                        .reshape(1, chunk * frame_sequence_length)
                    )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted = self.diffusion.forward_inference_window(
                        latents,
                        clean_history,
                        condition,
                        camera_window,
                        timestep_tokens,
                        num_frame_per_block=chunk,
                        i2v_y=y_window,
                        sequence_length=chunk * frame_sequence_length,
                        clean_history_timestep=context_noise,
                        r_timestep_tokens=target_tokens,
                    )
                if block == 0:
                    predicted[:, 0] = 0
                if target is None:
                    latents = scheduler.step(
                        predicted.to(latents.dtype),
                        timestep,
                        latents,
                        return_dict=False,
                    )[0].to(torch.bfloat16)
                else:
                    delta = (timestep.float() - target.float()) / float(
                        self.config["train"]["num_train_timesteps"]
                    )
                    latents = (latents.float() - delta * predicted.float()).to(torch.bfloat16)
                if block == 0:
                    latents[:, 0] = first_latent[:, 0]
            output[:, start:end] = latents
        schedule = {
            "scheduler": (
                "flowmap-adjacent-shifted"
                if generation_pass.solver == "flowmap"
                else "diffusers.UniPCMultistepScheduler-per-chunk"
            ),
            "max_prior_clean_chunks": maximum_prior,
            "chunk_latent_frames": chunk,
        }
        if raw_t:
            schedule["t"] = raw_t
            schedule["r"] = raw_r
        return output, schedule

    def _generation_pass(self, case: InferenceCase) -> GenerationPass:
        metadata = case.metadata.get("generation_pass")
        if not isinstance(metadata, Mapping):
            raise BackendContractError("Wan generation case lacks generation_pass metadata")
        return GenerationPass(
            name=str(metadata["name"]),
            weights=str(metadata["weights"]),
            mode=str(metadata["mode"]),
            solver=str(metadata["solver"]),
            num_inference_steps=int(metadata["num_inference_steps"]),
            rollout_latent_frames=int(metadata["rollout_latent_frames"]),
            min_rollout_latent_frames=int(
                metadata.get(
                    "min_rollout_latent_frames",
                    metadata["rollout_latent_frames"],
                )
            ),
            fixed_plan_pixel_frames=int(
                metadata.get(
                    "fixed_plan_pixel_frames",
                    1 + 4 * (int(metadata["rollout_latent_frames"]) - 1),
                )
            ),
            variable_rollout_by_source=bool(metadata.get("variable_rollout_by_source", False)),
        )

    def _provenance(
        self,
        generation_pass: GenerationPass,
        weights_id: str,
    ) -> Mapping[str, Any]:
        return {
            "generation_pass": asdict(generation_pass),
            "weights_id": weights_id,
            "checkpoint_id": self.checkpoint_id,
            "base_weight_inventory": self.base_report.initialization_receipt(),
            "camera_translation_transform": str(
                self.config["model"]["camera_translation_transform"]
            ),
        }

    def _generate_loaded(
        self,
        case: InferenceCase,
        generation_pass: GenerationPass,
        *,
        weights_id: str,
    ) -> GeneratedSample:
        import torch

        generator = torch.Generator(device=self.device)
        generator.manual_seed(case.noise_seed)
        first, condition, camera, model_y = self._conditions(
            case, latent_frames=generation_pass.rollout_latent_frames
        )
        with torch.no_grad():
            if generation_pass.mode == "bidirectional":
                latents, schedule = self._bidirectional(
                    generation_pass,
                    first,
                    condition,
                    camera,
                    model_y,
                    generator,
                )
            elif generation_pass.solver in {"unipc", "flowmap"}:
                latents, schedule = self._autoregressive(
                    generation_pass,
                    first,
                    condition,
                    camera,
                    model_y,
                    generator,
                )
            else:
                raise BackendContractError(
                    "self_forcing generation must inject the Stage2 adapter into run_wan_generation"
                )
            if not torch.isfinite(latents).all():
                raise BackendContractError("Wan generation produced non-finite latents")
            output_latent_frames = int(
                case.metadata.get("generation_pass", {}).get(
                    "output_rollout_latent_frames",
                    generation_pass.rollout_latent_frames,
                )
            )
            latents = latents[:, :output_latent_frames].contiguous()
            decoded = self.vae.decode(latents, use_cache=False)
            finite = torch.isfinite(decoded)
            all_finite = bool(finite.all().item())
            finite_fraction = (
                1.0 if all_finite else float(finite.to(dtype=torch.float64).mean().item())
            )
            if not all_finite:
                raise BackendContractError("Wan VAE decode produced non-finite pixels")
            fps = float(self.config["data"].get("fps", 16.0))
            video = _encode_mp4(decoded, fps=fps)
            prepared = self._prepared.get(case.slot)
            if prepared is None:
                raise BackendContractError(
                    f"Wan adapter has no comparison input for slot {case.slot}"
                )
            compare = _encode_compare_mp4(decoded, prepared, fps=fps)
        return GeneratedSample(
            artifacts={
                "compare.mp4": compare,
                "video.mp4": video,
                "schedule.json": canonical_json(dict(schedule)),
            },
            shape=tuple(int(value) for value in decoded.shape),
            dtype=str(decoded.dtype).removeprefix("torch."),
            metrics={"finite_fraction": finite_fraction},
            provenance=self._provenance(generation_pass, weights_id),
        )

    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        generation_pass = self._generation_pass(case)
        expected_id = self.weight_id(generation_pass.weights)
        if weights_id != expected_id:
            raise BackendContractError(
                f"Wan weights identity drift: {weights_id!r} != {expected_id!r}"
            )
        self._load_role(generation_pass.weights)
        return self._generate_loaded(
            case,
            generation_pass,
            weights_id=weights_id,
        )


class TrainingWanGenerationAdapter(CudaWanGenerationAdapter):
    """Expose live and sharded-EMA training weights to the common runner."""

    def __init__(self, runtime: Any) -> None:
        codec = getattr(runtime, "codec", None)
        if codec is None:
            raise BackendContractError(
                "inline Wan validation requires a raw online VAE/text codec; "
                "preencoded-only training must configure a separate raw "
                "validation runtime"
            )
        self.runtime = runtime
        self.config = runtime.config
        self.plan = resolve_generation_plan(self.config)
        self.family = str(runtime.family)
        self.topology = runtime.topology
        self.is_writer = int(getattr(self.topology, "sp_rank", 0)) == 0
        self.device = runtime.device
        self.diffusion = runtime.diffusion
        self.text_encoder = codec.text_encoder
        self.vae = codec.vae
        self._prepared: dict[int, _PreparedCase] = {}
        self.step = int(runtime.global_step)
        self.runtime_checkpoint_id = str(runtime.checkpoint_id)
        if not self.runtime_checkpoint_id:
            raise BackendContractError(
                "inline Wan validation lacks its runtime checkpoint identity"
            )

    def sync(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

    def weight_id(self, role: str) -> str:
        if role not in {"live", "ema"}:
            raise BackendContractError(f"unknown Wan runtime weight role {role!r}")
        return f"runtime:{self.runtime_checkpoint_id}:step-{self.step:06d}#{role}"

    def _provenance(
        self,
        generation_pass: GenerationPass,
        weights_id: str,
    ) -> Mapping[str, Any]:
        return {
            "generation_pass": asdict(generation_pass),
            "weights_id": weights_id,
            "runtime_checkpoint_id": self.runtime_checkpoint_id,
            "runtime_step": self.step,
            "initialization_receipt": dict(self.runtime.initialization_receipt),
            "camera_translation_transform": str(
                self.config["model"]["camera_translation_transform"]
            ),
        }

    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        if int(self.runtime.global_step) != self.step:
            raise BackendContractError("Wan training weights changed during inline validation")
        generation_pass = self._generation_pass(case)
        expected_id = self.weight_id(generation_pass.weights)
        if weights_id != expected_id:
            raise BackendContractError(
                f"Wan weights identity drift: {weights_id!r} != {expected_id!r}"
            )
        module = self.diffusion.module
        was_training = bool(module.training)
        module.eval()
        swap = (
            self.runtime.ema.swapped_into(module)
            if generation_pass.weights == "ema"
            else nullcontext()
        )
        try:
            with swap:
                return self._generate_loaded(
                    case,
                    generation_pass,
                    weights_id=weights_id,
                )
        finally:
            module.train(was_training)


def _pass_case(case: InferenceCase, generation_pass: GenerationPass) -> InferenceCase:
    metadata = dict(case.metadata)
    pass_metadata = asdict(generation_pass)
    rollouts = metadata.get("rollout_latent_frames_by_pass", {})
    if isinstance(rollouts, Mapping) and generation_pass.name in rollouts:
        resolved_rollout = int(rollouts[generation_pass.name])
        if metadata.get("rollout_length_source") == "camera":
            pass_metadata.update(
                rollout_latent_frames=resolved_rollout,
                min_rollout_latent_frames=resolved_rollout,
                fixed_plan_pixel_frames=1 + 4 * (resolved_rollout - 1),
            )
        else:
            pass_metadata["output_rollout_latent_frames"] = resolved_rollout
    metadata["generation_pass"] = pass_metadata
    return replace(case, metadata=metadata)


def _weight_ids(
    provider: Any,
    plan: GenerationPlan,
    explicit: Mapping[str, str] | None,
) -> dict[str, str]:
    roles = {generation_pass.weights for generation_pass in plan.passes}
    if explicit is not None:
        result = {role: str(explicit.get(role, "")) for role in roles}
    else:
        resolver = getattr(provider, "weight_id", None)
        if not callable(resolver):
            raise BackendContractError(
                "an injected Wan adapter must provide weight_id(role) or explicit weights_ids"
            )
        result = {role: str(resolver(role)) for role in roles}
    missing = [role for role, identity in result.items() if not identity]
    if missing:
        raise BackendContractError(f"Wan generation lacks stable weights IDs: {missing}")
    return result


def _collective_generation_failure(
    local_error: str | None,
    *,
    phase: str,
) -> None:
    try:
        import torch.distributed as dist
    except ImportError:
        dist = None
    if dist is not None and dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, (dist.get_rank(), local_error))
        failures = [item for item in gathered if item[1]]
    else:
        failures = [(0, local_error)] if local_error else []
    if failures:
        detail = "; ".join(f"rank={rank}: {error}" for rank, error in failures)
        raise BackendContractError(f"Wan generation {phase} failed collectively: {detail}")


def _distributed_generation_context() -> tuple[Any | None, int, int]:
    try:
        import torch.distributed as dist
    except ImportError:
        return None, 0, 1
    if not dist.is_available() or not dist.is_initialized():
        return None, 0, 1
    return dist, int(dist.get_rank()), int(dist.get_world_size())


def _broadcast_generation_object(value: Any, *, source: int = 0) -> Any:
    dist, rank, _ = _distributed_generation_context()
    if dist is None:
        return value
    payload = [value if rank == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def _load_validation_plan_collectively(
    path: Path,
    *,
    backend: str,
    plan_key: str,
    expected_count: int,
) -> tuple[InferenceCase, ...] | None:
    cases = load_validation_plan(
        path,
        backend=backend,
        plan_key=plan_key,
        expected_count=expected_count,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if cases is not None else None
    dist, _, world = _distributed_generation_context()
    if dist is None:
        return cases
    states: list[Any] = [None] * world
    dist.all_gather_object(states, digest)
    present = [value for value in states if value is not None]
    if not present:
        return None
    if len(present) != world:
        raise BackendContractError(
            "frozen Wan validation plan exists on only part of the distributed world"
        )
    if len(set(present)) != 1:
        raise BackendContractError("frozen Wan validation plan differs between nodes")
    if cases is None:
        raise BackendContractError("local frozen Wan validation plan disappeared")
    return cases


def _publish_validation_plan_collectively(
    path: Path,
    *,
    backend: str,
    plan_key: str,
    cases: Sequence[InferenceCase],
    local_rank: int,
) -> None:
    payload = validation_plan_payload(backend=backend, plan_key=plan_key, cases=cases)
    digest = hashlib.sha256(payload).hexdigest()
    dist, _, world = _distributed_generation_context()
    if dist is not None:
        digests: list[Any] = [None] * world
        dist.all_gather_object(digests, digest)
        if len(set(digests)) != 1:
            raise BackendContractError("Wan ranks disagree before freezing validation plan")
    local_error: str | None = None
    if local_rank == 0:
        try:
            publish_validation_plan(
                path,
                backend=backend,
                plan_key=plan_key,
                cases=cases,
            )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    if dist is None:
        if local_error is not None:
            raise BackendContractError(f"Wan validation plan publication failed: {local_error}")
        return
    failures: list[Any] = [None] * world
    dist.all_gather_object(failures, local_error)
    errors = [value for value in failures if value]
    if errors:
        raise BackendContractError(
            "Wan validation plan publication failed collectively: " + " | ".join(errors)
        )
    dist.barrier()


def _generation_topology(provider: Any) -> tuple[int, int, int, int, int]:
    """Return raw rank/world and logical DP/SP coordinates."""

    dist, raw_rank, raw_world = _distributed_generation_context()
    topology = getattr(provider, "topology", None)
    if topology is None:
        if dist is not None:
            raise BackendContractError(
                "distributed Wan generation adapters must expose logical topology"
            )
        return 0, 1, 0, 1, 0
    values = (
        int(topology.raw_rank),
        int(topology.raw_world_size),
        int(topology.dp_rank),
        int(topology.sp_rank),
    )
    if dist is not None and values[:2] != (raw_rank, raw_world):
        raise BackendContractError("Wan generation adapter topology differs from torch.distributed")
    if int(topology.dp_world_size) * int(topology.sp_size) != values[1]:
        raise BackendContractError("Wan generation topology is internally inconsistent")
    return (
        values[0],
        values[1],
        values[2],
        int(topology.dp_world_size),
        values[3],
    )


def _partition_generation_cases(
    cases: Sequence[InferenceCase],
    *,
    dp_rank: int,
    dp_world_size: int,
) -> tuple[InferenceCase, ...]:
    """Assign equal waves by logical DP while preserving SP peer symmetry."""

    if dp_world_size < 1 or not 0 <= dp_rank < dp_world_size:
        raise BackendContractError("Wan generation logical DP identity is invalid")
    slots = [int(case.slot) for case in cases]
    if len(slots) != len(set(slots)):
        raise BackendContractError("Wan generation cases contain duplicate slots")
    return tuple(case for ordinal, case in enumerate(cases) if ordinal % dp_world_size == dp_rank)


def _gather_partitioned_cases(
    local_cases: Sequence[InferenceCase],
) -> tuple[InferenceCase, ...]:
    """Gather lightweight case identities and collapse duplicate SP peers."""

    dist, _, world = _distributed_generation_context()
    if dist is None:
        return tuple(local_cases)
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, tuple(local_cases))
    by_slot: dict[int, InferenceCase] = {}
    for partition in gathered:
        if not isinstance(partition, tuple):
            raise BackendContractError("Wan generation gathered an invalid case partition")
        for case in partition:
            if not isinstance(case, InferenceCase):
                raise BackendContractError("Wan generation gathered a non-InferenceCase value")
            previous = by_slot.setdefault(int(case.slot), case)
            if previous != case:
                raise BackendContractError(f"Wan SP peers disagree on validation slot {case.slot}")
    return tuple(by_slot[slot] for slot in sorted(by_slot))


def _merge_generation_partitions(
    parts_root: Path,
    target: Path,
    *,
    cases: Sequence[InferenceCase],
    family: str,
    weights_id: str,
    dp_world_size: int,
) -> tuple[int, str]:
    """Merge logical-writer transactions into one ordered pass transaction."""

    if target.exists():
        raise BackendContractError(f"Wan generation pass already exists: {target}")
    expected = {int(case.slot): case for case in cases}
    if len(expected) != len(cases):
        raise BackendContractError("Wan generation merge received duplicate slots")
    manifests: dict[int, Mapping[str, Any]] = {}
    sample_dirs: dict[int, Path] = {}
    for dp_rank in range(dp_world_size):
        partition = parts_root / f"dp-{dp_rank:06d}"
        if not (partition / "COMPLETE.json").is_file():
            raise BackendContractError(f"Wan generation partition {dp_rank} is incomplete")
        for sample_dir in sorted(partition.glob("slot-*")):
            manifest_path = sample_dir / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                slot = int(manifest["case"]["slot"])
            except Exception as exc:
                raise BackendContractError(
                    f"Wan generation partition manifest is invalid: {manifest_path}: {exc}"
                ) from exc
            if slot not in expected or slot in manifests:
                raise BackendContractError(
                    f"Wan generation partition has unexpected/duplicate slot {slot}"
                )
            if (
                str(manifest.get("family")) != family
                or str(manifest.get("weights_id")) != weights_id
                or str(manifest["case"].get("sample_id")) != expected[slot].sample_id
            ):
                raise BackendContractError(
                    f"Wan generation partition identity drifted for slot {slot}"
                )
            manifests[slot] = manifest
            sample_dirs[slot] = sample_dir
    if set(manifests) != set(expected):
        missing = sorted(set(expected).difference(manifests))
        raise BackendContractError(
            f"Wan generation partitions are missing fixed slots {missing[:16]}"
        )
    target.mkdir(parents=True)
    ordered = [manifests[int(case.slot)] for case in cases]
    for case in cases:
        slot = int(case.slot)
        os.replace(sample_dirs[slot], target / f"slot-{slot:06d}")
    ordered_bytes = b"".join(canonical_json(item) for item in ordered)
    ordered_digest = hashlib.blake2s(ordered_bytes).hexdigest()
    _atomic_write(target / "ordered-manifest.jsonl", ordered_bytes)
    _atomic_write(
        target / "COMPLETE.json",
        canonical_json(
            {
                "schema": "solarwm.inference-complete.v1",
                "family": family,
                "weights_id": weights_id,
                "cases": len(cases),
                "ordered_manifest_digest": ordered_digest,
            }
        ),
    )
    shutil.rmtree(parts_root)
    return len(cases), ordered_digest


def _publish_node_local_generation_partition(
    partition: Path | None,
    target: Path,
    *,
    local_cases: Sequence[InferenceCase],
    cases: Sequence[InferenceCase],
    family: str,
    weights_id: str,
    writer: bool,
) -> tuple[int, str]:
    """Publish node-local artifacts while gathering only small manifests globally.

    The output directory is assumed node-local.  Every logical DP writer therefore
        keeps its generated bytes on the node that produced them; manifests are small
        enough to gather through the process group so rank zero can still commit
        one globally ordered transaction. Publishing the disjoint ``slot-*`` trees to
        shared storage is left to the deployment.
    """

    dist, raw_rank, world = _distributed_generation_context()
    if dist is None or world <= 1:
        raise BackendContractError("node-local generation publication requires distributed")
    expected = {int(case.slot): case for case in cases}
    local_records: list[tuple[int, Mapping[str, Any]]] = []
    local_error: str | None = None
    if writer:
        try:
            if partition is None or not (partition / "COMPLETE.json").is_file():
                raise BackendContractError("node-local generation partition is incomplete")
            target.mkdir(parents=True, exist_ok=True)
            for case in local_cases:
                slot = int(case.slot)
                sample_dir = partition / f"slot-{slot:06d}"
                manifest_path = sample_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    int(manifest["case"]["slot"]) != slot
                    or str(manifest.get("family")) != family
                    or str(manifest.get("weights_id")) != weights_id
                    or str(manifest["case"].get("sample_id")) != case.sample_id
                ):
                    raise BackendContractError(
                        f"node-local generation identity drifted for slot {slot}"
                    )
                os.replace(sample_dir, target / f"slot-{slot:06d}")
                local_records.append((slot, manifest))
            shutil.rmtree(partition)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(local_error, phase="node-local partition publication")

    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, tuple(local_records))
    record: tuple[int, str] | None = None
    aggregate_error: str | None = None
    if raw_rank == 0:
        try:
            manifests: dict[int, Mapping[str, Any]] = {}
            for partition_records in gathered:
                if not isinstance(partition_records, tuple):
                    raise BackendContractError(
                        "node-local generation gathered an invalid manifest partition"
                    )
                for slot, manifest in partition_records:
                    slot = int(slot)
                    if slot not in expected or slot in manifests:
                        raise BackendContractError(
                            f"node-local generation has unexpected/duplicate slot {slot}"
                        )
                    manifests[slot] = manifest
            if set(manifests) != set(expected):
                missing = sorted(set(expected).difference(manifests))
                raise BackendContractError(
                    f"node-local generation is missing fixed slots {missing[:16]}"
                )
            ordered = [manifests[int(case.slot)] for case in cases]
            ordered_bytes = b"".join(canonical_json(item) for item in ordered)
            ordered_digest = hashlib.blake2s(ordered_bytes).hexdigest()
            _atomic_write(target / "ordered-manifest.jsonl", ordered_bytes)
            _atomic_write(
                target / "COMPLETE.json",
                canonical_json(
                    {
                        "schema": "solarwm.inference-complete.v1",
                        "family": family,
                        "weights_id": weights_id,
                        "cases": len(cases),
                        "ordered_manifest_digest": ordered_digest,
                        "node_local_artifacts": True,
                    }
                ),
            )
            record = (len(cases), ordered_digest)
        except Exception as exc:
            aggregate_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(
        aggregate_error,
        phase="node-local manifest aggregation",
    )
    record = _broadcast_generation_object(record)
    if not isinstance(record, tuple) or len(record) != 2:
        raise BackendContractError("node-local generation returned no aggregate record")
    return int(record[0]), str(record[1])


def _comparison_validation_step(target: Path) -> int | None:
    validation_parent = target.parent
    if validation_parent.name == ".staging":
        validation_parent = validation_parent.parent
    if validation_parent.name != "validation" or not target.name.startswith("step-"):
        return None
    suffix = target.name.removeprefix("step-")
    return int(suffix) if suffix.isdigit() else None


def _comparison_view_name(step: int, generation_pass: GenerationPass) -> str:
    """Use the stable public comparison directory convention for every Wan pass."""

    prefix = f"step_{int(step):06d}"
    if generation_pass.solver == "flowmap":
        prefix += f"_flowmap_nfe{int(generation_pass.num_inference_steps):04d}"
    return f"{prefix}_{generation_pass.name}"


def _publish_comparison_views(
    target: Path,
    *,
    plan: GenerationPlan,
    cases: int,
    dp_world_size: int,
    sp_size: int,
) -> tuple[str, ...]:
    step = _comparison_validation_step(target)
    if step is None:
        return ()
    published = []
    validation_root = target.parent.parent if target.parent.name == ".staging" else target.parent
    run_root = validation_root.parent
    for generation_pass in plan.passes:
        destination = public_validation_dir(
            run_root,
            step=step,
            pass_name=_comparison_view_name(step, generation_pass).removeprefix(
                f"step_{int(step):06d}_"
            ),
        )
        records = publish_comparison_partition(
            target / generation_pass.name,
            destination,
            step=step,
            pass_name=generation_pass.name,
            cases=cases,
            dp_world_size=dp_world_size,
            sp_size=sp_size,
            run_root=run_root,
        )
        publish_comparison_complete(
            destination,
            step=step,
            pass_name=generation_pass.name,
            local_slots=len(records),
            global_slots=cases,
        )
        published.append(str(destination))
    return tuple(published)


def run_wan_generation(
    config: Mapping[str, Any],
    *,
    provider: Any | None = None,
    cases: Sequence[InferenceCase] | None = None,
    weights_ids: Mapping[str, str] | None = None,
    output_dir: str | Path | None = None,
) -> WanGenerationSummary:
    """Run every configured pass through the common inference engine.

    ``provider`` only needs ``family`` and ``generate(case, weights_id=...)``;
    Stage2 can therefore expose its sampler as the same adapter.  When cases or
    weight identities are not supplied, the provider must expose
    ``build_cases(plan)`` and ``weight_id(role)`` respectively.
    """

    plan = resolve_generation_plan(config)
    publication_layout = camera_inference_output_layout(config)
    if publication_layout is not None and output_dir is not None:
        raise BackendContractError(
            "camera dataset publication uses runtime.output_dir and does not accept an "
            "explicit generation output_dir"
        )
    family = str(config.get("model", {}).get("family", ""))
    owns_provider = provider is None
    provider_error: str | None = None
    if provider is None:
        try:
            provider = CudaWanGenerationAdapter(config, plan)
        except Exception as exc:
            provider_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(provider_error, phase="provider setup")
    if provider is None:
        raise BackendContractError("Wan generation provider setup returned no adapter")
    if str(getattr(provider, "family", "")) != family:
        raise BackendContractError("Wan generation adapter family differs from config")

    topology_error: str | None = None
    topology: tuple[int, int, int, int, int] | None = None
    try:
        topology = _generation_topology(provider)
    except Exception as exc:
        topology_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(topology_error, phase="topology")
    if topology is None:
        raise BackendContractError("Wan generation topology resolution failed")
    raw_rank, raw_world, dp_rank, dp_world_size, sp_rank = topology
    node_local_artifacts = raw_world > 1 and int(os.environ.get("NNODES", "1")) > 1
    local_rank = int(
        getattr(getattr(provider, "topology", None), "local_rank", os.environ.get("LOCAL_RANK", 0))
    )
    if publication_layout is not None and node_local_artifacts:
        raise BackendContractError(
            "camera dataset publication currently requires one shared-filesystem node; "
            "node-local multi-node output is unsupported"
        )

    case_error: str | None = None
    supplied_cases: tuple[InferenceCase, ...] | None = None
    builder_returned_partition = False
    if cases is None:
        try:
            builder = getattr(provider, "build_cases", None)
            if not callable(builder):
                raise BackendContractError(
                    "an injected Wan adapter must receive explicit fixed inference cases"
                )
            supplied_cases = tuple(builder(plan))
            builder_returned_partition = bool(
                getattr(provider, "build_cases_returns_partition", False)
            )
        except Exception as exc:
            case_error = f"{type(exc).__name__}: {exc}"
    else:
        try:
            supplied_cases = tuple(cases)
        except Exception as exc:
            case_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(case_error, phase="fixed-case materialization")
    if supplied_cases is None:
        raise BackendContractError("Wan generation materialized no case collection")

    partition_error: str | None = None
    all_cases: tuple[InferenceCase, ...] | None = None
    local_cases: tuple[InferenceCase, ...] | None = None
    try:
        if builder_returned_partition:
            local_cases = supplied_cases
            all_cases = _gather_partitioned_cases(local_cases)
            expected_local = _partition_generation_cases(
                all_cases,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
            )
            if local_cases != expected_local:
                raise BackendContractError(
                    "Wan case builder partition differs from logical-DP ownership"
                )
        else:
            all_cases = supplied_cases
            local_cases = _partition_generation_cases(
                all_cases,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
            )
        if not all_cases:
            raise BackendContractError("Wan generation plan resolved no cases")
    except Exception as exc:
        partition_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(partition_error, phase="logical-DP partition")
    if all_cases is None or local_cases is None:
        raise BackendContractError("Wan generation case partition returned no cases")

    input_policy_error: str | None = None
    try:
        invalid_cases = [
            {
                "slot": int(case.slot),
                "sample_id": str(case.sample_id),
            }
            for case in all_cases
            if case.metadata.get("artifact_valid", True) is False
        ]
        if invalid_cases:
            raise BackendContractError(
                f"Wan validation requires materialized inputs; first={invalid_cases[:8]}"
            )
    except Exception as exc:
        input_policy_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(
        input_policy_error,
        phase="validation input policy",
    )

    publication_preflight_error: str | None = None
    if publication_layout is not None and raw_rank == 0:
        try:
            from .publication import preflight_camera_publication

            preflight_camera_publication(publication_layout, all_cases)
        except Exception as exc:
            publication_preflight_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(
        publication_preflight_error,
        phase="camera publication preflight",
    )

    identity_error: str | None = None
    identities: dict[str, str] | None = None
    try:
        identities = _weight_ids(provider, plan, weights_ids)
    except Exception as exc:
        identity_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(identity_error, phase="weights identity")
    if identities is None:
        raise BackendContractError("Wan generation lacks weights identities")

    configured_runtime = str(config.get("runtime", {}).get("output_dir", ""))
    configured_output = str(output_dir) if output_dir is not None else configured_runtime
    if not configured_output.startswith("/"):
        raise BackendContractError("Wan generation output_dir must be absolute")
    if publication_layout is not None:
        target = publication_layout.run_root / "generation"
    else:
        target = (
            Path(configured_output).resolve()
            if output_dir is not None
            else Path(configured_output).resolve() / "generation"
        )
    temporary_sink: tempfile.TemporaryDirectory[str] | None = None
    staging: Path | None = None
    setup_error: str | None = None
    try:
        if node_local_artifacts:
            staging_id = _broadcast_generation_object(uuid.uuid4().hex if raw_rank == 0 else None)
            if not isinstance(staging_id, str) or not staging_id:
                raise BackendContractError("Wan generation returned no staging identity")
            staging = target.with_name(f".{target.name}.{staging_id}.partial")
            if local_rank == 0:
                if target.exists():
                    raise BackendContractError(f"Wan generation output already exists: {target}")
                if staging.exists():
                    raise BackendContractError(
                        f"Wan generation staging output already exists: {staging}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                staging.mkdir()
        elif raw_rank == 0:
            if target.exists():
                raise BackendContractError(f"Wan generation output already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
            staging.mkdir()
    except Exception as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(setup_error, phase="output setup")
    if node_local_artifacts:
        if staging is None or not staging.is_absolute():
            raise BackendContractError("Wan generation output setup returned no local staging path")
    else:
        staging_value = _broadcast_generation_object(
            str(staging) if raw_rank == 0 and staging is not None else None
        )
        if not isinstance(staging_value, str) or not staging_value.startswith("/"):
            raise BackendContractError("Wan generation output setup returned no staging path")
        staging = Path(staging_value)
    peer_root: Path | None = None
    peer_error: str | None = None
    if sp_rank != 0:
        try:
            temporary_sink = tempfile.TemporaryDirectory(prefix="solarwm-wan-sp-peer-")
            peer_root = Path(temporary_sink.name) / "output"
            peer_root.mkdir()
        except Exception as exc:
            peer_error = f"{type(exc).__name__}: {exc}"
    _collective_generation_failure(peer_error, phase="SP peer output setup")

    adapter = _FamilyAdapter(provider, family)
    pass_records: list[dict[str, Any]] = []
    complete_digest = ""
    try:
        for generation_pass in plan.passes:
            pass_cases = tuple(_pass_case(case, generation_pass) for case in local_cases)
            global_pass_cases = tuple(_pass_case(case, generation_pass) for case in all_cases)
            parts_root = staging / ".parts" / generation_pass.name
            pass_output = (
                parts_root / f"dp-{dp_rank:06d}"
                if sp_rank == 0
                else peer_root / generation_pass.name
            )
            summary = None
            pass_error: str | None = None
            try:
                summary = InferenceEngine(adapter).run(
                    pass_cases,
                    weights_id=identities[generation_pass.weights],
                    output_dir=pass_output,
                    collective_case_waves=(len(all_cases) + dp_world_size - 1) // dp_world_size,
                    collective_error=lambda error, phase, pass_name=(generation_pass.name): (
                        _collective_generation_failure(
                            error,
                            phase=f"pass {pass_name} {phase}",
                        )
                    ),
                )
            except Exception as exc:
                pass_error = f"{type(exc).__name__}: {exc}"
            _collective_generation_failure(
                pass_error,
                phase=f"pass {generation_pass.name}",
            )
            if summary is None:
                raise BackendContractError(
                    f"Wan generation pass {generation_pass.name} returned no summary"
                )

            record: dict[str, Any] | None = None
            merge_error: str | None = None
            if node_local_artifacts:
                try:
                    merged_cases, ordered_digest = _publish_node_local_generation_partition(
                        parts_root / f"dp-{dp_rank:06d}" if sp_rank == 0 else None,
                        staging / generation_pass.name,
                        local_cases=pass_cases,
                        cases=global_pass_cases,
                        family=family,
                        weights_id=identities[generation_pass.weights],
                        writer=sp_rank == 0,
                    )
                    record = {
                        "pass": asdict(generation_pass),
                        "weights_id": identities[generation_pass.weights],
                        "cases": merged_cases,
                        "ordered_manifest_digest": ordered_digest,
                    }
                except Exception as exc:
                    merge_error = f"{type(exc).__name__}: {exc}"
                cleanup_error: str | None = None
                if local_rank == 0:
                    try:
                        if parts_root.exists():
                            (parts_root / ".solarwm-directory-publication.lock").unlink(
                                missing_ok=True
                            )
                            parts_root.rmdir()
                    except Exception as exc:
                        cleanup_error = f"{type(exc).__name__}: {exc}"
                _collective_generation_failure(
                    cleanup_error,
                    phase=f"pass {generation_pass.name} node-local cleanup",
                )
            elif raw_rank == 0:
                try:
                    merged_cases, ordered_digest = _merge_generation_partitions(
                        parts_root,
                        staging / generation_pass.name,
                        cases=global_pass_cases,
                        family=family,
                        weights_id=identities[generation_pass.weights],
                        dp_world_size=dp_world_size,
                    )
                    record = {
                        "pass": asdict(generation_pass),
                        "weights_id": identities[generation_pass.weights],
                        "cases": merged_cases,
                        "ordered_manifest_digest": ordered_digest,
                    }
                except Exception as exc:
                    merge_error = f"{type(exc).__name__}: {exc}"
            _collective_generation_failure(
                merge_error,
                phase=f"pass {generation_pass.name} merge",
            )
            record = _broadcast_generation_object(record)
            if not isinstance(record, dict):
                raise BackendContractError(
                    f"Wan generation pass {generation_pass.name} lacks aggregate record"
                )
            pass_records.append(record)

        commit_error: str | None = None
        if raw_rank == 0:
            try:
                complete = {
                    "schema": "solarwm.wan22-generation-complete.v1",
                    "family": family,
                    "test_index": plan.index,
                    "selection_seed": plan.selection_seed,
                    "noise_seed": plan.noise_seed,
                    "cases": len(all_cases),
                    "passes": pass_records,
                    "weights_ids": identities,
                    "shared_inference_validation_implementation": True,
                    "logical_dp_partitioned": True,
                    "node_local_artifacts": node_local_artifacts,
                }
                complete_digest = _atomic_write(staging / "COMPLETE.json", canonical_json(complete))
            except Exception as exc:
                commit_error = f"{type(exc).__name__}: {exc}"
        _collective_generation_failure(commit_error, phase="aggregate commit")
        publish_error: str | None = None
        if node_local_artifacts:
            if local_rank == 0:
                try:
                    parts_parent = staging / ".parts"
                    if parts_parent.exists():
                        parts_parent.rmdir()
                    os.replace(staging, target)
                except Exception as exc:
                    publish_error = f"{type(exc).__name__}: {exc}"
        elif raw_rank == 0:
            try:
                parts_parent = staging / ".parts"
                if parts_parent.exists():
                    parts_parent.rmdir()
                if publication_layout is not None:
                    publish_directory_no_replace(
                        staging,
                        target,
                        error_type=BackendContractError,
                        label="camera inference generation transaction",
                    )
                else:
                    os.replace(staging, target)
            except Exception as exc:
                publish_error = f"{type(exc).__name__}: {exc}"
        _collective_generation_failure(publish_error, phase="output publication")
        publication_record: dict[str, Any] | None = None
        publication_error: str | None = None
        if publication_layout is not None and raw_rank == 0:
            try:
                from .publication import publish_camera_triplets

                generation_pass = plan.passes[0]
                published = publish_camera_triplets(
                    target,
                    publication_layout,
                    generation_pass=generation_pass.name,
                    cases=all_cases,
                    family=family,
                    weights_id=identities[generation_pass.weights],
                    generation_complete_digest=complete_digest,
                )
                publication_record = {
                    "layout": publication_layout.layout,
                    "complete_path": str(published.complete_path),
                    "complete_digest": published.complete_digest,
                }
            except Exception as exc:
                publication_error = f"{type(exc).__name__}: {exc}"
        _collective_generation_failure(
            publication_error,
            phase="camera dataset publication",
        )
        if publication_layout is not None:
            publication_record = _broadcast_generation_object(publication_record)
            if not isinstance(publication_record, Mapping):
                raise BackendContractError("camera publication returned no completion record")
        comparison_error: str | None = None
        if (node_local_artifacts and local_rank == 0) or (
            not node_local_artifacts and raw_rank == 0
        ):
            try:
                topology = getattr(provider, "topology", None)
                sp_size = int(getattr(topology, "sp_size", 1))
                _publish_comparison_views(
                    target,
                    plan=plan,
                    cases=len(all_cases),
                    dp_world_size=dp_world_size,
                    sp_size=sp_size,
                )
            except Exception as exc:
                comparison_error = f"{type(exc).__name__}: {exc}"
        _collective_generation_failure(
            comparison_error,
            phase="comparison publication",
        )
        validation_staging = target.parent == validation_staging_root(
            config["runtime"]["output_dir"]
        )
        cleanup_error: str | None = None
        if validation_staging and (
            (node_local_artifacts and local_rank == 0)
            or (not node_local_artifacts and raw_rank == 0)
        ):
            try:
                cleanup_validation_staging(
                    target,
                    output_dir=config["runtime"]["output_dir"],
                )
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
        _collective_generation_failure(
            cleanup_error,
            phase="validation staging cleanup",
        )
        complete_digest = str(_broadcast_generation_object(complete_digest))
        if not complete_digest:
            raise BackendContractError("Wan generation aggregate identity is empty")
        return WanGenerationSummary(
            output_dir=(target.parent.parent if validation_staging else target),
            family=family,
            cases=len(all_cases),
            passes=tuple(item.name for item in plan.passes),
            weights_ids=identities,
            complete_digest=complete_digest,
            validation_plan_source=str(getattr(provider, "validation_plan_source", "")),
            validation_plan_path=str(getattr(provider, "validation_plan_path", "")),
            publication_layout=(
                str(publication_record["layout"]) if isinstance(publication_record, Mapping) else ""
            ),
            publication_complete_path=(
                str(publication_record["complete_path"])
                if isinstance(publication_record, Mapping)
                else ""
            ),
            publication_complete_digest=(
                str(publication_record["complete_digest"])
                if isinstance(publication_record, Mapping)
                else ""
            ),
        )
    finally:
        if temporary_sink is not None:
            temporary_sink.cleanup()
        if owns_provider:
            closer = getattr(provider, "close", None)
            if callable(closer):
                closer()


def run_wan_inference(
    config: Mapping[str, Any],
    **kwargs: Any,
) -> WanGenerationSummary:
    """Standalone inference is a named call to the unified implementation."""

    return run_wan_generation(config, **kwargs)


def run_wan_validation(
    config: Mapping[str, Any],
    **kwargs: Any,
) -> WanGenerationSummary:
    """Training validation is a named call to the unified implementation."""

    provider = kwargs.get("provider")
    if provider is not None and getattr(provider, "validation_plan_path", None) is None:
        output_dir = Path(str(config["runtime"]["output_dir"])).expanduser().resolve()
        provider.validation_plan_path = output_dir / "validation" / "frozen-plan.json"
    return run_wan_generation(config, **kwargs)


__all__ = [
    "CudaWanGenerationAdapter",
    "TrainingWanGenerationAdapter",
    "WanGenerationSummary",
    "run_wan_generation",
    "run_wan_inference",
    "run_wan_validation",
]
