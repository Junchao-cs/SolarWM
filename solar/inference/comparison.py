"""Comparison-video visualization and publication contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from solar.errors import BackendContractError


@dataclass(frozen=True)
class ComparisonValidationRecord:
    slot: int
    compare_path: Path
    manifest_path: Path


def _sample_slug(value: object, *, max_length: int = 160) -> str:
    text = str(value).strip()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    if not slug:
        slug = hashlib.blake2s(text.encode()).hexdigest()[:16]
    if len(slug) <= max_length:
        return slug
    digest = hashlib.blake2s(text.encode()).hexdigest()[:16]
    return f"{slug[: max_length - len(digest) - 2]}__{digest}"


def encode_compare_mp4(
    reference: Any,
    generated: Any,
    *,
    fps: float,
    layout: Literal["btchw", "bcthw"],
    value_range: Literal["zero_one", "minus_one_one"],
) -> bytes:
    """Encode the left-GT/right-GEN H.264 visualization."""

    import torch

    def canonical(value: Any, name: str) -> Any:
        if value.ndim != 5 or int(value.shape[0]) != 1:
            raise BackendContractError(
                f"validation compare {name} has unexpected shape {tuple(value.shape)}"
            )
        result = value if layout == "btchw" else value.permute(0, 2, 1, 3, 4)
        if int(result.shape[2]) != 3:
            raise BackendContractError(
                f"validation compare {name} is not RGB: {tuple(result.shape)}"
            )
        # Backends keep source pixels on CPU while decoded generations normally
        # remain on the local CUDA device. The comparison is encoded by
        # ffmpeg on the host, so normalize both sides on CPU before joining.
        result = result.detach().to(device="cpu", dtype=torch.float32)
        if value_range == "minus_one_one":
            result = result.mul(0.5).add(0.5)
        return torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)

    gt = canonical(reference, "GT")
    gen = canonical(generated, "GEN")
    frames = min(int(gt.shape[1]), int(gen.shape[1]))
    if frames < 1 or tuple(gt.shape[2:]) != tuple(gen.shape[2:]):
        raise BackendContractError(
            f"validation compare GT/GEN shapes differ: gt={tuple(gt.shape)} gen={tuple(gen.shape)}"
        )
    joined = torch.cat((gt[:, :frames], gen[:, :frames]), dim=-1)
    array = joined[0].permute(0, 2, 3, 1).mul(255.0).to(torch.uint8).cpu().numpy()
    height, width = int(array.shape[1]), int(array.shape[2])
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise BackendContractError("validation compare requires ffmpeg")
    with tempfile.TemporaryDirectory(prefix="solarwm-validation-compare-") as directory:
        target = Path(directory) / "compare.mp4"
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
        result = subprocess.run(command, input=array.tobytes(), capture_output=True, check=False)
        if result.returncode or not target.is_file():
            detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
            raise BackendContractError(f"ffmpeg failed to encode validation compare: {detail}")
        return target.read_bytes()


def _comparison_manifest(
    manifest: Mapping[str, Any],
    *,
    step: int,
    pass_name: str,
    compare_path: str,
    cases: int,
    dp_world_size: int,
    sp_size: int,
    logical_world_size_per_round: int,
) -> dict[str, Any]:
    case = manifest["case"]
    metadata = case.get("metadata", {})
    generation_pass = metadata.get("generation_pass", {})
    slot = int(case["slot"])
    dp_rank = slot % int(dp_world_size)
    round_index = slot // int(logical_world_size_per_round)
    shape = manifest.get("shape", ())
    output_pixel_frames = int(shape[1]) if len(shape) >= 2 else None
    if metadata.get("output_pixel_frames") is not None:
        output_pixel_frames = int(metadata["output_pixel_frames"])
    rollout_latent_frames = generation_pass.get(
        "output_rollout_latent_frames",
        generation_pass.get("rollout_latent_frames", metadata.get("rollout_latent_frames")),
    )
    comparison = {
        "rank": dp_rank * int(sp_size),
        "global_rank": dp_rank * int(sp_size),
        "dp_rank": dp_rank,
        "sp_rank": 0,
        "sp_size": int(sp_size),
        "logical_validation_rank": slot,
        "logical_rank_within_round": slot % int(logical_world_size_per_round),
        "logical_validation_world_size": int(cases),
        "logical_world_size_per_round": int(logical_world_size_per_round),
        "validation_round_index": round_index,
        "validation_num_rounds": int(cases) // int(logical_world_size_per_round),
        "source_pixel_frames": metadata.get("source_pixel_frames"),
        "source_num_frames": metadata.get("source_num_frames"),
        "validation_noise_seed": int(case["noise_seed"]),
        "step": int(step),
        "video_id": metadata.get("key", case["sample_id"]),
        "dataset": metadata.get("dataset"),
        "scene": metadata.get("scene"),
        "start_frame": int(case["start_frame"]),
        "T_pix": output_pixel_frames,
        "T_lat": rollout_latent_frames,
        "train_T_lat": metadata.get("train_latent_frames"),
        "rollout_T_lat": rollout_latent_frames,
        "collective_rollout_T_lat": generation_pass.get("rollout_latent_frames"),
        "variable_rollout_by_source": bool(
            generation_pass.get("variable_rollout_by_source", False)
        ),
        "variable_rollout_start_adjusted": bool(
            metadata.get("variable_rollout_start_adjusted", False)
        ),
        "generation_mode": generation_pass.get("mode", metadata.get("generation_mode")),
        "active_pass_name": pass_name,
        "inference_mode": generation_pass.get("solver", metadata.get("inference_mode")),
        "camera_translation_transform": metadata.get("camera_translation_transform"),
        "sample_solver": generation_pass.get("solver", metadata.get("sample_solver")),
        "num_inference_steps": generation_pass.get(
            "num_inference_steps", metadata.get("num_inference_steps")
        ),
        "weights_source": generation_pass.get("weights", metadata.get("weights_source")),
        "requested_video_id": metadata.get("requested_key"),
        "requested_start_frame": metadata.get("requested_start_frame"),
        "validation_input_status": metadata.get("validation_input_status", "requested"),
        "artifact_valid": metadata.get("artifact_valid", True) is not False,
        "compare_mp4": compare_path,
        "video_stats": manifest.get("metrics", {}),
        "generated_latents_finite": manifest.get("metrics", {}).get("finite_fraction") == 1.0,
        "error": metadata.get("recoverable_input_error"),
        "solarwm_manifest": manifest,
    }
    denoising_step_list = manifest.get("provenance", {}).get("denoising_step_list")
    if denoising_step_list is not None:
        comparison["denoising_step_list"] = [int(value) for value in denoising_step_list]
    return comparison


def publish_comparison_partition(
    source: Path,
    destination: Path,
    *,
    step: int,
    pass_name: str,
    cases: int,
    dp_world_size: int,
    sp_size: int,
    run_root: Path,
    logical_world_size_per_round: int | None = None,
) -> tuple[ComparisonValidationRecord, ...]:
    """Merge one node/rank partition into the common comparison directory."""

    cases_per_round = (
        int(cases) if logical_world_size_per_round is None else int(logical_world_size_per_round)
    )
    if int(cases) < 1 or cases_per_round < 1 or int(cases) % cases_per_round:
        raise BackendContractError("comparison validation cases must form complete logical rounds")

    compare_dir = destination / "compare"
    manifest_dir = destination / "manifests"
    compare_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_dir in sorted(source.glob("slot-*")):
        manifest = json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8"))
        case = manifest["case"]
        metadata = case.get("metadata", {})
        slot = int(case["slot"])
        key = _sample_slug(metadata.get("key") or case["sample_id"])
        start = int(case["start_frame"])
        dp_rank = slot % int(dp_world_size)
        round_index = slot // cases_per_round
        filename = (
            f"rank{slot:03d}_round{round_index:02d}_dp{dp_rank:03d}_{key}_start{start:06d}.mp4"
        )
        source_compare = sample_dir / "compare.mp4"
        if not source_compare.is_file():
            raise BackendContractError(f"validation slot {slot} lacks compare.mp4")
        compare_path = compare_dir / filename
        try:
            os.link(source_compare, compare_path)
        except FileExistsError as exc:
            raise BackendContractError(
                f"comparison validation compare already exists: {compare_path}"
            ) from exc
        relative = compare_path.relative_to(run_root).as_posix()
        comparison_manifest = _comparison_manifest(
            manifest,
            step=step,
            pass_name=pass_name,
            compare_path=relative,
            cases=cases,
            dp_world_size=dp_world_size,
            sp_size=sp_size,
            logical_world_size_per_round=cases_per_round,
        )
        manifest_path = manifest_dir / f"rank_{slot:03d}.json"
        payload = (json.dumps(comparison_manifest, ensure_ascii=False, indent=2) + "\n").encode()
        descriptor = os.open(
            manifest_path,
            os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY,
            0o444,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            compare_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise
        records.append(ComparisonValidationRecord(slot, compare_path, manifest_path))
    if not records:
        raise BackendContractError(f"comparison validation source has no slots: {source}")
    return tuple(records)


def publish_comparison_complete(
    destination: Path,
    *,
    step: int,
    pass_name: str,
    local_slots: int,
    global_slots: int,
) -> None:
    payload = (
        json.dumps(
            {
                "schema": "solarwm.validation-comparison.v1",
                "step": int(step),
                "pass": pass_name,
                "local_slots": int(local_slots),
                "global_slots": int(global_slots),
                "compare_layout": "ground_truth_left_generated_right",
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    target = destination / "COMPLETE.json"
    if target.exists():
        return
    temporary = target.with_name(
        f".{target.name}.{hashlib.blake2s(payload).hexdigest()[:16]}.partial"
    )
    temporary.write_bytes(payload)
    try:
        os.link(temporary, target)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ComparisonValidationRecord",
    "encode_compare_mp4",
    "publish_comparison_complete",
    "publish_comparison_partition",
]
