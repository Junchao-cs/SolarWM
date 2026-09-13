"""Portable fixed H3 validation plans over preencoded conditions and cameras.

Paths in each row may be absolute or relative to the plan. Stage1 supplies
170 consecutive source cameras; Stage2 supplies the exact 47 latent cameras.
The train reader is independent of these small, immutable validation inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from solarwm.errors import DataContractError

from .artifacts import H3ArtifactBatch, read_encoder_contract
from .camera import validate_absolute_c2w, validate_normalized_intrinsics
from .codec import _validate_video_text_tensors


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_cameras(path: Path, row: dict[str, Any], *, stage: str) -> tuple[Any, Any]:
    from safetensors.torch import load_file

    if path.suffix == ".safetensors":
        values = load_file(str(path), device="cpu")
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
    else:
        raise DataContractError("H3 validation cameras must be safetensors or NPZ")
    return validation_cameras(values, row, stage=stage)


def validation_cameras(values: Any, row: Any, *, stage: str) -> tuple[Any, Any]:
    """Align complete camera trajectories to the causal validation rollout."""
    import torch

    convention = row.get("camera_convention")
    if stage == "stage2":
        if convention != "relative_w2c+normalized_K" or row.get("camera_alignment") != "latent":
            raise DataContractError("H3 SGF validation needs declared relative latent cameras")
        from .geometry import latent_aligned_pixel_indices

        expected = torch.tensor(latent_aligned_pixel_indices(158)) + int(
            row["validation_start_frame"]
        )
        if not torch.equal(torch.as_tensor(values["source_latent_frame_indices"]), expected):
            raise DataContractError("H3 SGF camera indices differ from the 158-frame encode")
        views, intrinsics = values["viewmats"], values["K"]
        count = 47
    else:
        expected = torch.arange(
            int(row["validation_start_frame"]), int(row["validation_start_frame"]) + 170
        )
        if not torch.equal(torch.as_tensor(values["source_frame_indices"]), expected):
            raise DataContractError("H3 Stage1 needs 170 consecutive source camera frames")
        intrinsics = values.get("camera_K", values.get("K", values.get("intrinsics")))
        if convention == "absolute_c2w+normalized_K":
            from .torch_prope import _invert_se3

            poses = torch.as_tensor(values.get("camera_c2w", values.get("c2w"))).float()
            if poses.shape != (170, 4, 4):
                raise DataContractError("H3 Stage1 absolute camera shape differs")
            relative = torch.matmul(_invert_se3(poses[:1]), poses)
            relative[0] = torch.eye(4)
            views = _invert_se3(relative)
        elif convention == "relative_w2c+normalized_K":
            views = values["viewmats"]
        else:
            raise DataContractError("H3 validation camera convention is missing or unsupported")
        views, intrinsics = torch.as_tensor(views).float(), torch.as_tensor(intrinsics).float()
        if views.shape != (170, 4, 4) or intrinsics.shape != (170, 3, 3):
            raise DataContractError("H3 Stage1 camera must cover the full 170-frame trajectory")
        indices = [0]
        for index in range(49):
            indices.append(indices[-1] + (1, 4, 4, 4, 4)[index % 5])
        views, intrinsics = views[indices], intrinsics[indices]
        count = 50
    views, intrinsics = torch.as_tensor(views).float(), torch.as_tensor(intrinsics).float()
    if views.shape != (count, 4, 4) or intrinsics.shape != (count, 3, 3):
        raise DataContractError("H3 validation camera tensor shape differs")
    if not bool(torch.isfinite(views).all() and torch.isfinite(intrinsics).all()):
        raise DataContractError("H3 validation cameras contain non-finite values")
    if not torch.allclose(views[0], torch.eye(4), atol=1e-4, rtol=0):
        raise DataContractError("H3 relative camera must anchor the first frame at identity")
    try:
        validate_absolute_c2w(views.numpy())
        validate_normalized_intrinsics(intrinsics.numpy())
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"invalid H3 validation camera: {exc}") from exc
    return views.contiguous(), intrinsics.contiguous()


class H3PreparedValidationStream:
    """Read a fixed plan in complete logical-DP waves, with no sample substitution."""

    def __init__(self, config: Any, topology: Any) -> None:
        self.encoder_contract, self.encoder_profile = read_encoder_contract(
            config["data"]["encoder_contract_path"]
        )
        self.path = Path(config["validation"]["prepared_plan"])
        payload = self.path.read_bytes()
        document = json.loads(payload)
        self.digest = hashlib.sha256(payload).hexdigest()
        self.rows = document["rows"]
        self.stage = config["train"]["stage"]
        self.topology = topology
        self.cursor = topology.dp_rank
        expected_count = int(config["validation"]["sample_count"])
        if len(self.rows) != expected_count or [
            row["validation_slot"] for row in self.rows
        ] != list(range(expected_count)):
            raise DataContractError(
                "H3 prepared validation plan must have exactly the ordered configured slots"
            )
        if len({row["sample_id"] for row in self.rows}) != expected_count:
            raise DataContractError("H3 prepared validation plan repeats a sample")
        if config["validation"].get("balance_by_dataset", False):
            counts = Counter(
                row.get("dataset_source", row["sample_id"].split("/")[0]) for row in self.rows
            )
            if max(counts.values()) - min(counts.values()) > 1:
                raise DataContractError(f"H3 validation datasets are unbalanced: {dict(counts)}")

    def next(self) -> H3ArtifactBatch:
        import torch
        from safetensors.torch import load_file

        if self.cursor >= len(self.rows):
            raise StopIteration
        row = self.rows[self.cursor]
        self.cursor += self.topology.dp_world_size
        paths = {
            key: _path(self.path.parent, row[key]) for key in ("condition_safetensors", "camera")
        }
        for key, path in paths.items():
            expected = row.get(key + "_sha256")
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise DataContractError(f"H3 prepared validation artifact changed: {path}")
        values = load_file(str(paths["condition_safetensors"]), device="cpu")
        start = int(row["validation_start_frame"])
        if not torch.equal(values["source_frame_indices"], torch.arange(start, start + 158)):
            raise DataContractError("H3 validation condition source indices differ from its plan")
        for key in ("target_latents", "anchor_latents"):
            if values[key].ndim == 5 and values[key].shape[0] == 1:
                values[key] = values[key][0]
        prompt, tags = values["prompt_embeds"], values["text_token_tags"]
        if prompt.ndim == 3 and prompt.shape[0] == 1:
            prompt = prompt[0]
        tags = tags.reshape(-1)
        try:
            _validate_video_text_tensors(
                values["target_latents"], values["anchor_latents"], prompt, tags
            )
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"invalid H3 validation condition: {exc}") from exc
        if not all(
            bool(tensor.isfinite().all())
            for tensor in (values["target_latents"], values["anchor_latents"], prompt)
        ):
            raise DataContractError("H3 validation condition contains non-finite values")
        views, intrinsics = _load_cameras(paths["camera"], row, stage=self.stage)
        return H3ArtifactBatch(
            sample_id=str(row["sample_id"]),
            start_frame=start,
            plan_fingerprint=self.digest,
            target_latents=values["target_latents"],
            anchor_latents=values["anchor_latents"],
            prompt_embeds=prompt,
            text_token_tags=tags,
            source_frame_indices=values["source_frame_indices"],
            camera_viewmats=views[:47],
            camera_K=intrinsics[:47],
            source_fps=None,
            validation_slot=int(row["validation_slot"]),
            validation_noise_seed=int(row["validation_noise_seed"]),
            rollout_camera_viewmats=views if self.stage == "stage1" else None,
            rollout_camera_K=intrinsics if self.stage == "stage1" else None,
            dataset_source=str(row.get("dataset_source", row["sample_id"].split("/")[0])),
        )

    def close(self) -> None:
        """The prepared stream owns no background workers or open descriptors."""
