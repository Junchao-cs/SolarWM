"""Append-only CPU retrieval cameras from actual FP32 inference view matrices.

Do not invert high-precision navigation poses instead: retrieval must retain the
native FP64 inverse of the FP32 view matrix. Only the five newly appended rows
are copied from the inference device. Existing rows are cooperatively immutable
outside the separately restored local-camera scope. Not yet GPU-qualified.
"""
import numpy as np


class RetrievalCamera:
    def __init__(self):
        self.owner = None
        self.count = self.copied_rows = self.calls = 0
        self.c2w = self.K = None

    def append(self, owner, views, intrinsics, start, needed):
        import torch
        if type(start) is not int or type(needed) is not int or needed != start + 6:
            raise ValueError('exact five-new-latent camera extent required')
        if start == 0:
            if self.count:
                raise RuntimeError('reset camera cache before a new request')
            self.owner = owner
        elif self.owner is not owner or start != self.count:
            raise RuntimeError('camera request owner or append order changed')
        if views.dtype != torch.float32 or views.ndim != 4 or views.shape[0] != 1 or views.shape[-2:] != (4,4):
            raise ValueError('actual FP32 view matrices required')
        if intrinsics.ndim != 4 or intrinsics.shape[0] != 1 or intrinsics.shape[-2:] != (3,3):
            raise ValueError('actual intrinsics required')
        if views.shape[1] < needed or intrinsics.shape[1] < needed:
            raise ValueError('incomplete camera rows')
        actual = views[0, 1+start:needed].cpu().numpy().astype(np.float64)
        new_c2w = np.linalg.inv(actual)
        new_K = intrinsics[0, 1+start:needed].cpu().numpy().copy()
        self.c2w = new_c2w if self.c2w is None else np.concatenate((self.c2w, new_c2w), axis=0)
        self.K = new_K if self.K is None else np.concatenate((self.K, new_K), axis=0)
        self.count += 5
        self.copied_rows += 5
        self.calls += 1
        return self.c2w, self.K
