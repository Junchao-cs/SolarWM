"""Deployed translation-only window reference, not a new camera policy.

The image anchor and the retained history share the same origin. Persistent
poses are restored afterwards. The trained nonlinear logd4 mapping is retained;
this is not a claim of equivalence to the original global-origin conditioning.
"""
from contextlib import contextmanager


@contextmanager
def local_window(views, index):
    import torch
    if index <= 5:
        yield
        return
    start, stop = 1 + max(0, index - 5) * 5, 1 + (index + 1) * 5
    if (views.ndim != 4 or views.shape[0] != 1 or tuple(views.shape[-2:]) != (4, 4)
            or views.dtype != torch.float32 or views.shape[1] < stop or views.requires_grad):
        raise ValueError("complete B1 FP32 inference camera table required")
    first = views[:, start]
    origin = -(first[:, :3, :3].transpose(-1, -2) @ first[:, :3, 3:4])
    spans = (views[:, :1], views[:, start:stop])
    backups = [span[..., :3, 3].clone() for span in spans]
    try:
        for span in spans:
            span[..., :3, 3].add_((span[..., :3, :3] @ origin[:, None]).squeeze(-1))
        yield
    finally:
        for span, original in zip(spans, backups, strict=True):
            span[..., :3, 3].copy_(original)
