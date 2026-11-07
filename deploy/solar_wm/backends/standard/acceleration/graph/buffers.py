"""Pointer-stable camera and KV buffers with steady-window copy-in mappings."""

from __future__ import annotations
from dataclasses import dataclass
import torch


CAPTURE_CHUNK = 5
HISTORY = 5
FRAMES_PER_CHUNK = 5
FIRST_STEADY_CHUNK = 5
FULL_MODEL_LAYERS = 50


@dataclass(frozen=True)
class SteadyCopyIn:
    actual_chunk: int
    capture_chunk: int
    history_actual: tuple[int, ...]
    history_capture: tuple[int, ...]
    camera_source: tuple[int, ...]
    camera_destination: tuple[int, ...]
    current_camera_source: tuple[int, ...]
    current_camera_destination: tuple[int, ...]


def plan(actual_chunk: int, *, layers: int = FULL_MODEL_LAYERS) -> SteadyCopyIn:
    if type(actual_chunk) is not int or actual_chunk < FIRST_STEADY_CHUNK:
        raise ValueError("only steady denoise after chunk 5 is in scope")
    if layers != FULL_MODEL_LAYERS:
        raise ValueError("Graph mapping must cover all 50 H3 layers")
    first = actual_chunk - HISTORY
    camera_source = tuple(range(1 + first * FRAMES_PER_CHUNK,
                                1 + (actual_chunk + 1) * FRAMES_PER_CHUNK))
    camera_destination = tuple(range(1, 1 + (HISTORY + 1) * FRAMES_PER_CHUNK))
    return SteadyCopyIn(
        actual_chunk=actual_chunk,
        capture_chunk=CAPTURE_CHUNK,
        history_actual=tuple(range(first, actual_chunk)),
        history_capture=tuple(range(HISTORY)),
        camera_source=camera_source,
        camera_destination=camera_destination,
        current_camera_source=camera_source[-FRAMES_PER_CHUNK:],
        current_camera_destination=camera_destination[-FRAMES_PER_CHUNK:],
    )


class StableSteadyInputs:
    """Own pointer-stable camera and per-layer packed raw K/V tensor buffers."""

    def __init__(self, source_cache, chunk_index, viewmats, intrinsics):
        self.ready = False
        self.poisoned = False
        self.next_chunk = chunk_index
        self.buffers = {}
        self.viewmats = None
        self.intrinsics = None
        mapping = plan(chunk_index)
        pairs = self._source_pairs(source_cache, mapping)
        self._check_camera(viewmats, intrinsics, mapping.camera_source[-1] + 1)
        self.buffers = {
            layer: (torch.empty_like(key), torch.empty_like(value))
            for layer, (key, value) in pairs.items()
        }
        self.viewmats = torch.empty((1, 291, 4, 4), device=viewmats.device,
                                    dtype=viewmats.dtype)
        self.intrinsics = torch.empty((1, 291, 3, 3), device=intrinsics.device,
                                      dtype=intrinsics.dtype)
        self.load(source_cache, chunk_index, viewmats, intrinsics)

    @staticmethod
    def _check_camera(viewmats, intrinsics, required_frames):
        if (not isinstance(viewmats, torch.Tensor)
                or not isinstance(intrinsics, torch.Tensor)
                or viewmats.ndim != 4 or intrinsics.ndim != 4
                or viewmats.shape[0] != 1 or intrinsics.shape[0] != 1
                or tuple(viewmats.shape[2:]) != (4, 4)
                or tuple(intrinsics.shape[2:]) != (3, 3)
                or viewmats.shape[1] != intrinsics.shape[1]
                or viewmats.shape[1] < max(291, required_frames)
                or viewmats.device != intrinsics.device
                or viewmats.requires_grad or intrinsics.requires_grad):
            raise ValueError("full native expanding camera inputs differ")

    @staticmethod
    def _source_pairs(source_cache, mapping):
        sidecars = getattr(source_cache, "packed_history_by_layer", None)
        if type(sidecars) is not dict or set(sidecars) != set(range(FULL_MODEL_LAYERS)):
            raise ValueError("Operator packed sidecar must cover all 50 layers")
        pairs = {}
        for layer in range(FULL_MODEL_LAYERS):
            entries = source_cache.history(layer, mapping.actual_chunk)
            if (len(entries) != 5
                    or tuple(row[0] for row in entries) != mapping.history_actual):
                raise ValueError(f"layer {layer} raw history differs")
            prepared = sidecars[layer]
            if (type(prepared) is not tuple or len(prepared) != 2
                    or prepared[0] != mapping.actual_chunk
                    or type(prepared[1]) is not tuple or len(prepared[1]) != 2):
                raise ValueError(f"layer {layer} packed sidecar is stale")
            key, value = prepared[1]
            if (not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor)
                    or key.ndim < 2 or value.ndim < 2
                    or key.shape[1] != sum(row[1].shape[1] for row in entries)
                    or value.shape[1] != sum(row[2].shape[1] for row in entries)
                    or key.device != value.device or key.dtype != value.dtype
                    or key.requires_grad or value.requires_grad):
                raise ValueError(f"layer {layer} packed tensor metadata differs")
            pairs[layer] = (key, value)
        return pairs

    def load(self, source_cache, chunk_index, viewmats, intrinsics):
        if self.poisoned or chunk_index != self.next_chunk:
            raise ValueError("steady copy-in cannot skip, replay, or resume after failure")
        self.ready = False
        mapping = plan(chunk_index)
        pairs = self._source_pairs(source_cache, mapping)
        self._check_camera(viewmats, intrinsics, mapping.camera_source[-1] + 1)
        if (viewmats.device != self.viewmats.device
                or intrinsics.device != self.intrinsics.device
                or viewmats.dtype != self.viewmats.dtype
                or intrinsics.dtype != self.intrinsics.dtype):
            raise ValueError("camera buffer specialization changed")
        for layer, pair in pairs.items():
            owned = self.buffers[layer]
            if any(source.shape != target.shape or source.dtype != target.dtype
                   or source.device != target.device
                   for source, target in zip(pair, owned)):
                raise ValueError(f"layer {layer} KV buffer specialization changed")
        try:
            with torch.no_grad():
                self.viewmats[:, 0].copy_(viewmats[:, 0])
                self.intrinsics[:, 0].copy_(intrinsics[:, 0])
                for destination, source in zip(mapping.camera_destination,
                                               mapping.camera_source):
                    self.viewmats[:, destination].copy_(viewmats[:, source])
                    self.intrinsics[:, destination].copy_(intrinsics[:, source])
                for layer, pair in pairs.items():
                    for target, source in zip(self.buffers[layer], pair):
                        target.copy_(source)
        except BaseException:
            self.poisoned = True
            raise
        self.ready = True
        self.next_chunk += 1

    def get_packed_history(self, layer, chunk_index, empty_key, empty_value):
        if not self.ready or self.poisoned or chunk_index != 5 or layer not in self.buffers:
            raise ValueError("captured H3 steady cache read is not ready")
        return self.buffers[layer]
