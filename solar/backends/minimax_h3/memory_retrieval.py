"""Retrieve generated latent frames outside the recent context window."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POSE_OFFSETS = (0, 1, 5, 9, 13)
REVISION = "native17-top5"


def retrieval_scores(query_c2w, candidate_c2w, intrinsics, *, width=1344, height=768):
    """Score camera overlap, translation distance and viewing angle.

    The frustum uses 8x8 pixels and four logarithmic depth-bin centres, with
    near/far 2/60 in original camera units. Coordinates/depths are both /4.
    """
    q = np.asarray(query_c2w, dtype=np.float64).copy()
    c = np.asarray(candidate_c2w, dtype=np.float64).copy().reshape(-1, 4, 4)
    if not len(c):
        return np.empty(0, dtype=np.float64)
    q[:3, 3] /= 4.0
    c[:, :3, 3] /= 4.0
    K = np.asarray(intrinsics, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # Convert normalized intrinsics to pixel coordinates.
    fx, cx = fx * width, cx * width
    fy, cy = fy * height, cy * height
    u = (np.arange(8) + 0.5) / 8 * width
    v = (np.arange(8) + 0.5) / 8 * height
    uu, vv = np.meshgrid(u, v, indexing="ij")
    rays = np.stack(((uu.ravel() - cx) / fx, (vv.ravel() - cy) / fy, np.ones(64)))
    edges = np.linspace(np.log(2.0 / 4), np.log(60.0 / 4), 5)
    points = (rays[:, :, None] * np.exp(0.5 * (edges[:-1] + edges[1:]))[None, None, :]).reshape(
        3, -1
    )
    world = q[:3, :3] @ points + q[:3, 3:4]
    rt = c[:, :3, :3].transpose(0, 2, 1)
    pc = np.einsum("mij,jp->mip", rt, world) - np.einsum("mij,mj->mi", rt, c[:, :3, 3])[:, :, None]
    z = pc[:, 2]
    zs = np.clip(z, 1e-9, None)
    pu = pc[:, 0] * fx / zs + cx
    pv = pc[:, 1] * fy / zs + cy
    overlap = ((z > 0.5) & (z < 15) & (pu >= 0) & (pu <= width) & (pv >= 0) & (pv <= height)).mean(
        1
    )
    distance = np.linalg.norm(c[:, :3, 3] - q[:3, 3], axis=1)
    angle = np.arccos(np.clip(c[:, :3, 2] @ q[:3, 2], -1.0, 1.0))
    return overlap - 0.2 * (distance + 4 * angle)


def select_memory(c2w, K, chunk_index, *, top_k=5, recent_chunks=5):
    """Select up to top_k historical latents, excluding the recent context window."""
    poses = np.asarray(c2w)
    stop = max(0, chunk_index - recent_chunks) * 5
    candidates = np.arange(stop, dtype=np.int64)
    if not len(candidates):
        return candidates, np.empty(0)
    query = 5 * chunk_index + 4
    scores = retrieval_scores(poses[query], poses[candidates], np.asarray(K)[query])
    order = np.argsort(-scores, kind="stable")[:top_k]
    return candidates[order], scores[order]


@dataclass
class LatentMemoryBank:
    """Store completed latent chunks for retrieval."""

    chunks: list = field(default_factory=list)

    def commit(self, chunk, index):
        import torch

        if index != len(self.chunks) or chunk.shape[2] != 5:
            raise ValueError("latent memory must commit completed five-latent chunks in order")
        self.chunks.append(chunk.detach().to(dtype=torch.bfloat16).clone())

    def gather(self, indices, reference):
        import torch

        if not len(indices):
            return reference[:, :, :0]
        return torch.cat(
            [self.chunks[int(i) // 5][:, :, int(i) % 5 : int(i) % 5 + 1] for i in indices], 2
        ).float()
