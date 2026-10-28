"""Reader for standalone safetensors samples produced by ``solar.renderer.preencode``.

A sample carries its target latents (any ``5n+2`` length, landscape or portrait), the released
``ref2va`` prompt embeddings, any number of untyped visual observations and, optionally, the known
degree of every target row. It has no first-frame anchor and no camera unless the member provides
them. The reader is deliberately small next to the release reader: an index of members, a
deterministic shuffle per epoch, a logical data-parallel partition, and a cursor that survives a
checkpoint. Members live wherever they were written, a directory or a remote object-storage prefix.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from solar.backends.minimax_h3.artifacts import (
    H3ArtifactBatch,
    H3VisualObservation,
    read_encoder_contract,
    validate_absolute_c2w,
    validate_normalized_intrinsics,
)
from solar.errors import DataContractError
from solar.renderer import storage


def is_remote(path: str) -> bool:
    """Whether a location uses an object-storage scheme."""
    return storage.is_remote(path)


@contextlib.contextmanager
def _local(path: str):
    """Yield a local file for ``path``; a remote object goes through scratch and is removed."""
    if not is_remote(path):
        yield path
        return
    handle, scratch = tempfile.mkstemp(
        suffix=Path(path).suffix, dir=os.environ.get("OMNI_SCRATCH", "/tmp")
    )
    os.close(handle)
    try:
        storage.get(path, scratch)
        yield scratch
    finally:
        Path(scratch).unlink(missing_ok=True)


def coarse(item: H3VisualObservation, spec: Mapping[str, Any]) -> H3VisualObservation:
    """An aligned video observation read coarser than it was encoded: its latent grid averaged over
    ``pool`` x ``pool`` cells and every ``stride``-th latent kept, each on its own target frame.

    The layout lays an aligned observation over the whole canvas whatever its grid, so a half-grid
    structure video still covers the picture; it costs a quarter of the tokens, and a third of them
    again when temporal stride is three. H3's front
    end resizes every video reference to its own size, so the coarse grid is made here, from the
    member already encoded. A single frame (the first frame) and a free observation are left alone.
    """
    import torch

    pool, stride = (int(spec.get("pool") or 1), int(spec.get("stride") or 1))
    if item.frames is None or len(item.frames) < 2 or (pool <= 1 and stride <= 1):
        return item
    latents, frames, boxes = (item.latents, item.frames, item.boxes)
    if stride > 1:
        latents, frames = (latents[:, ::stride], frames[::stride])
        if boxes is not None and len(boxes) == len(item.frames):
            boxes = boxes[::stride]
    if pool > 1:
        height, width = (int(latents.shape[-2]), int(latents.shape[-1]))
        size = tuple(max(2, 2 * int(side / pool / 2 + 0.5)) for side in (height, width))
        if size == (height // pool, width // pool) and (not height % pool) and (not width % pool):
            latents = torch.nn.functional.avg_pool2d(latents.float(), pool)
        else:
            latents = torch.nn.functional.adaptive_avg_pool2d(latents.float(), size)
        latents = latents.to(item.latents.dtype)
    return replace(item, latents=latents.contiguous(), frames=tuple(frames), boxes=boxes)


class H3OmniReader:
    """Read standalone encoded SolarRenderer samples."""

    def __init__(self, root: str, plan_fingerprint: str = "render") -> None:
        self.root = str(root).rstrip("/") or "/"
        self.current_plan_fingerprint = plan_fingerprint

    def camera(self, tensors: Mapping[str, Any], latent_frames: int) -> tuple[Any, Any]:
        """The clip's camera in the model's frame, or nothing when the member carries none.

        A member stores absolute camera-to-world poses, one per latent, exactly as the released
        artifacts do; the first camera of the clip becomes the origin here, as it does there.
        """
        import torch

        c2w, intrinsics = (tensors.get("camera_c2w"), tensors.get("camera_K"))
        if c2w is None and intrinsics is None:
            return (None, None)
        if c2w is None or intrinsics is None:
            raise DataContractError("an omni camera needs both the poses and the intrinsics")
        if tuple(c2w.shape) != (latent_frames, 4, 4):
            raise DataContractError(
                f"omni camera holds {tuple(c2w.shape)}, not [{latent_frames},4,4]"
            )
        if tuple(intrinsics.shape) == (3, 3):
            intrinsics = intrinsics.unsqueeze(0).expand(latent_frames, -1, -1).clone()
        if tuple(intrinsics.shape) != (latent_frames, 3, 3):
            raise DataContractError(
                f"omni intrinsics hold {tuple(intrinsics.shape)}, not [{latent_frames},3,3]"
            )
        c2w, intrinsics = (c2w.float(), intrinsics.float())
        if not bool(torch.isfinite(c2w).all()) or not bool(torch.isfinite(intrinsics).all()):
            raise DataContractError("omni camera holds non-finite values")
        validate_absolute_c2w(c2w.numpy())
        validate_normalized_intrinsics(intrinsics.numpy())
        views = views_from_c2w(c2w)
        if float(views.abs().max()) > 20.0:
            raise DataContractError("omni camera exceeds the magnitude guard")
        return (views, intrinsics.contiguous())

    def first_frame(self, path: str, target: Any) -> H3VisualObservation:
        """The clip's first frame, encoded apart from its member, as an observation on latent 0.

        An index row may name a file holding ``first_frame_latents`` [24,1,h,w] (the frame held as
        a still and encoded on the target's grid, as the preencoder handles an aligned
        image). It joins the member's observations as given; the member's prompt is kept, so the
        frame reaches the model through its latent alone and dropping it leaves exactly the sample
        without one - the same input a render without a first frame gives.
        """
        import torch
        from safetensors import safe_open

        if not (is_remote(path) or path.startswith("/")):
            path = f"{self.root}/{path}"
        with _local(path) as local, safe_open(local, framework="pt") as handle:
            latents = handle.get_tensor("first_frame_latents")
        if tuple(latents.shape) != (24, 1, *target.shape[2:]):
            raise DataContractError(
                f"first frame latents {tuple(latents.shape)} "
                f"are not [24,1,{','.join(str(v) for v in target.shape[2:])}]"
            )
        if not bool(torch.isfinite(latents.float()).all()):
            raise DataContractError("first frame latents hold non-finite values")
        return H3VisualObservation(latents=latents.contiguous(), frames=(0,), degree=0.999)

    def read(self, row: dict[str, Any]) -> H3ArtifactBatch:
        import torch
        from safetensors import safe_open

        member = row["member"]
        if not (is_remote(member) or member.startswith("/")):
            member = f"{self.root}/{member}"
        with _local(member) as path, safe_open(path, framework="pt") as handle:
            meta = json.loads((handle.metadata() or {}).get("omni", "{}"))
            keys = handle.keys()
            tensors = {key: handle.get_tensor(key) for key in keys}
        target, prompt, tags = (
            tensors["target_latents"],
            tensors["prompt_embeds"],
            tensors["text_token_tags"],
        )
        if target.ndim != 4 or target.shape[0] != 24 or (target.shape[1] - 2) % 5:
            raise DataContractError(
                f"omni target latents {tuple(target.shape)} are not [24,5n+2,h,w]"
            )
        if tuple(target.shape[2:]) not in {(48, 84), (84, 48)}:
            raise DataContractError(
                f"omni target grid {tuple(target.shape[2:])} is not 48x84 or 84x48"
            )
        if prompt.ndim != 2 or prompt.shape[-1] != 5120 or tags.shape != prompt.shape[:1]:
            raise DataContractError("omni prompt embeddings differ")
        if tags.dtype not in (torch.int32, torch.int64) or not bool(
            ((tags >= 0) & (tags <= 2)).all()
        ):
            raise DataContractError("omni token tags must be integer modality IDs in [0, 2]")
        if not bool(torch.isfinite(target.float()).all()) or not bool(
            torch.isfinite(prompt.float()).all()
        ):
            raise DataContractError("omni member holds non-finite values")
        observations = []
        for index, item in enumerate(meta.get("observations", ())):
            latents = tensors[f"observation_{index}_latents"]
            frames = item.get("frames")
            if (
                latents.ndim != 4
                or latents.shape[0] != 24
                or min(latents.shape[1:]) < 1
                or any(size % 2 for size in latents.shape[2:])
                or not bool(torch.isfinite(latents).all())
            ):
                raise DataContractError("visual observations must be finite [24,F,2H,2W] tensors")
            if frames is not None and (
                len(frames) != latents.shape[1]
                or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames)
                or any(frame < 0 or frame >= target.shape[1] for frame in frames)
            ):
                raise DataContractError(
                    "observation frames must align with the target latent range"
                )
            if not 0 < float(item.get("degree", 0.999)) <= 1:
                raise DataContractError("observation degree must lie in (0, 1]")
            observations.append(
                H3VisualObservation(
                    latents=latents.contiguous(),
                    frames=None if frames is None else tuple(int(f) for f in frames),
                    degree=float(item.get("degree", 0.999)),
                    boxes=None
                    if not item.get("boxes")
                    else tuple(tuple(float(v) for v in box) for box in item["boxes"]),
                )
            )
        if row.get("coarse_observations"):
            observations = [coarse(item, row["coarse_observations"]) for item in observations]
        if row.get("free_observations"):
            observations = [
                replace(item, frames=None)
                if item.frames is not None and len(item.frames) > 1
                else item
                for item in observations
            ]
        if row.get("first_frame"):
            observations.append(self.first_frame(row["first_frame"], target))
        views, intrinsics = self.camera(tensors, int(target.shape[1]))
        known = tensors.get("known_degrees")
        expected_rows = int(target.shape[1] * target.shape[2] * target.shape[3] // 4)
        if known is not None and (
            known.numel() != expected_rows
            or not bool(torch.isfinite(known).all())
            or not bool(((known >= 0) & (known <= 1)).all())
        ):
            raise DataContractError(
                "known_degrees must cover each target row with a value in [0, 1]"
            )
        batch = H3ArtifactBatch(
            sample_id=str(row["id"]),
            start_frame=0,
            plan_fingerprint=self.current_plan_fingerprint,
            target_latents=target.contiguous(),
            anchor_latents=None,
            prompt_embeds=prompt.contiguous(),
            text_token_tags=tags.contiguous(),
            source_frame_indices=None,
            camera_viewmats=views,
            camera_K=intrinsics,
            source_fps=24.0,
            dataset_source=str(meta.get("source", row.get("source", ""))),
            text_prompt_embeds=prompt.contiguous(),
            text_prompt_tags=tags.contiguous(),
            visual_observations=tuple(observations) or None,
            known_degrees=tensors.get("known_degrees"),
        )
        return batch


class H3OmniStream(H3OmniReader):
    def __init__(
        self,
        *,
        root: str,
        index: str,
        topology: Any,
        seed: int,
        encoder_contract_path: str,
        shuffle: bool = True,
        length_buckets: bool = False,
    ) -> None:
        self.root = root.rstrip("/") if is_remote(root) else str(Path(root).resolve())
        with _local(index) as path, open(path) as lines:
            rows = [json.loads(line) for line in lines if line.strip()]
        if not rows:
            raise DataContractError(f"omni index {index} is empty")
        self.rows = sorted(rows, key=lambda row: row["id"])
        if any(not isinstance(row.get("id"), str) or not row["id"] for row in self.rows):
            raise DataContractError("omni index IDs must be nonempty strings")
        if len({row["id"] for row in self.rows}) != len(self.rows):
            raise DataContractError("omni index contains duplicate sample IDs")
        self.rank, self.world = (int(topology.dp_rank), int(topology.dp_world_size))
        if self.world < 1 or not 0 <= self.rank < self.world:
            raise DataContractError("invalid data-parallel rank or world size")
        if len(self.rows) < self.world:
            raise DataContractError("omni index holds fewer samples than data-parallel readers")
        self.seed, self.shuffle = (int(seed), bool(shuffle))
        self.length_buckets = bool(length_buckets)
        _, self.encoder_profile = read_encoder_contract(encoder_contract_path)
        self.epoch, self.cursor = (0, 0)
        identity = {
            "schema": "solarwm.renderer-data-plan.v1",
            "root": self.root,
            "rows": self.rows,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "length_buckets": self.length_buckets,
            "world_size": self.world,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.current_plan_fingerprint = hashlib.sha256(encoded.encode()).hexdigest()

    def _order(self) -> np.ndarray:
        """This reader's rows for the epoch, computed once per epoch (an index runs to millions)."""
        cached = getattr(self, "_cached_order", None)
        if cached is not None and cached[0] == self.epoch:
            return cached[1]
        order = None
        if getattr(self, "length_buckets", False) and self.shuffle:
            order = self._bucketed_order()
            order = order if len(order) else None
        if order is None:
            order = np.arange(len(self.rows))
            if self.shuffle:
                np.random.default_rng([self.seed, self.epoch]).shuffle(order)
            usable = len(order) - len(order) % self.world
            order = order[:usable]
        mine = order[self.rank :: self.world]
        self._cached_order = (self.epoch, mine)
        return mine

    def _bucketed_order(self) -> np.ndarray:
        """The epoch's global order in runs of ``world`` rows that share a clip length.

        Rows are grouped by target length and number of observations (what sets the packed
        sequence), each group shuffled and cut into runs of one row per reader, and the runs
        shuffled together. A group's last partial run waits for a later epoch. Readers take every
        ``world``-th row as before, so at each step they all hold the same length.
        """
        rng = np.random.default_rng([self.seed, self.epoch])
        groups: dict[tuple[int, int], list[int]] = {}
        for index, row in enumerate(self.rows):
            key = (int(row.get("latent_frames") or 47), int(row.get("observations") or 0))
            groups.setdefault(key, []).append(index)
        runs = []
        for key in sorted(groups):
            members = np.asarray(groups[key], dtype=np.int64)
            rng.shuffle(members)
            usable = len(members) - len(members) % self.world
            runs.extend(members[:usable].reshape(-1, self.world))
        if not runs:
            return np.zeros(0, dtype=np.int64)
        runs = np.stack(runs)
        rng.shuffle(runs)
        return runs.reshape(-1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "plan_fingerprint": self.current_plan_fingerprint,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("plan_fingerprint") != self.current_plan_fingerprint:
            raise DataContractError("renderer data plan fingerprint changed since the checkpoint")
        epoch, cursor = state.get("epoch"), state.get("cursor")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (epoch, cursor)
        ):
            raise DataContractError("renderer data epoch and cursor must be nonnegative integers")
        previous = self.epoch, self.cursor
        self.epoch, self.cursor = epoch, cursor
        if cursor > len(self._order()):
            self.epoch, self.cursor = previous
            raise DataContractError("renderer data cursor exceeds the epoch length")

    def close(self) -> None:
        return None

    def __iter__(self):
        while True:
            yield self.next()

    def next(self) -> H3ArtifactBatch:
        previous = self.epoch, self.cursor
        order = self._order()
        if self.cursor == len(order):
            self.epoch, self.cursor = self.epoch + 1, 0
            order = self._order()
        row = self.rows[int(order[self.cursor])]
        try:
            batch = self.read(row)
        except Exception as exc:
            self.epoch, self.cursor = previous
            raise DataContractError(f"cannot read renderer sample {row['id']!r}: {exc}") from exc
        self.cursor += 1
        return batch


__all__ = ["H3OmniStream"]


def invert_se3_torch(matrices: Any) -> Any:
    """Invert a stack of rigid transforms, on the Torch side of the house."""
    import torch

    rotation = matrices[..., :3, :3]
    rotation_inverse = rotation.transpose(-1, -2)
    output = torch.zeros_like(matrices)
    output[..., :3, :3] = rotation_inverse
    output[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_inverse, matrices[..., :3, 3])
    output[..., 3, 3] = 1.0
    return output


def views_from_c2w(c2w: Any) -> Any:
    """World-to-camera views in the model's frame: the clip's first camera is the origin.

    The Torch arithmetic order is preserved exactly; the NumPy equivalent differs by a few FP32
    ULPs, and that changes the fused-PRoPE loss.
    """
    import torch

    first_w2c = invert_se3_torch(c2w[:1])[0]
    relative_c2w = torch.matmul(first_w2c.unsqueeze(0), c2w)
    relative_c2w[0] = torch.eye(4, dtype=relative_c2w.dtype)
    return invert_se3_torch(relative_c2w).contiguous()
