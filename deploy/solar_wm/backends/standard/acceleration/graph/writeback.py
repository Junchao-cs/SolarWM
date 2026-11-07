"""Publish captured refresh writes into the authoritative raw and packed KV cache."""

from __future__ import annotations
import torch


LAYERS = 50
FIXED_CHUNK = 5


def _stamp(tensor):
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype, tensor.device)


class RefreshWriteback:
    """Fixed captured K/V destinations plus one-use ordered host publication."""

    def __init__(self, history_owner, global_cache, *, actual_chunk=FIXED_CHUNK):
        if type(actual_chunk) is not int or actual_chunk < FIXED_CHUNK:
            raise ValueError("refresh Graph must start at steady chunk 5")
        if not callable(getattr(history_owner, "get_packed_history", None)):
            raise TypeError("pointer-stable packed history owner required")
        if not all(callable(getattr(global_cache, name, None))
                   for name in ("history", "commit")):
            raise TypeError("authoritative raw cache required")
        self.history_owner = history_owner
        self.global_cache = global_cache
        self.slots = {}
        for layer in range(LAYERS):
            entries = global_cache.history(layer, actual_chunk)
            if len(entries) != 5 or tuple(row[0] for row in entries) != tuple(
                    range(actual_chunk - 5, actual_chunk)):
                raise ValueError(f"layer {layer} lacks five authoritative entries")
            key, value = entries[-1][1:]
            if (not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor)
                    or key.requires_grad or value.requires_grad or key.device != value.device
                    or key.dtype != value.dtype or key.ndim < 2 or value.ndim < 2
                    or key.shape[1] != value.shape[1]):
                raise ValueError(f"layer {layer} raw KV specialization differs")
            self.slots[layer] = (torch.empty_like(key), torch.empty_like(value))
        self.stamps = {layer: tuple(map(_stamp, pair))
                       for layer, pair in self.slots.items()}
        self.next_chunk = actual_chunk
        self.phase = "ready"
        self.next_layer = 0
        self.failed = False

    def get_packed_history(self, layer, chunk_index, empty_key, empty_value):
        if self.failed or self.phase != "capturing" or chunk_index != FIXED_CHUNK:
            raise RuntimeError("refresh Graph history read outside fixed capture")
        return self.history_owner.get_packed_history(
            layer, FIXED_CHUNK, empty_key, empty_value)

    def begin_capture_pass(self):
        if self.failed or self.phase not in ("ready", "primed"):
            raise RuntimeError("refresh capture pass cannot start")
        self.phase = "capturing"
        self.next_layer = 0

    def commit(self, layer, chunk_index, key, value):
        """Record one device copy per layer during an actual captured call."""
        if (self.failed or self.phase != "capturing" or layer != self.next_layer
                or chunk_index != FIXED_CHUNK or torch.is_grad_enabled()):
            self.failed = True
            raise RuntimeError("refresh capture lost ordered no-grad layer commits")
        targets = self.slots[layer]
        for source, target in zip((key, value), targets):
            if (not isinstance(source, torch.Tensor) or source.shape != target.shape
                    or source.dtype != target.dtype or source.device != target.device
                    or source.requires_grad):
                self.failed = True
                raise ValueError(f"refresh layer {layer} KV shape or device changed")
        try:
            targets[0].copy_(key.detach())
            targets[1].copy_(value.detach())
        except BaseException:
            self.failed = True
            raise
        self.next_layer += 1

    def end_capture_pass(self, *, final=False):
        if self.failed or self.phase != "capturing" or self.next_layer != LAYERS:
            self.failed = True
            raise RuntimeError("refresh capture omitted a KV layer")
        self.phase = "captured" if final else "primed"
        if final:
            self.stamps = {layer: tuple(map(_stamp, pair))
                           for layer, pair in self.slots.items()}

    def mark_replayed(self, actual_chunk):
        """Caller calls only after CUDA Graph capture/replay returned on its stream."""
        if (self.failed or self.phase != "captured" or actual_chunk != self.next_chunk
                or any(tuple(map(_stamp, self.slots[layer])) != self.stamps[layer]
                       for layer in range(LAYERS))):
            self.failed = True
            raise RuntimeError("refresh Graph replay or slot ownership differs")
        self.phase = "replayed"

    def publish(self, actual_chunk):
        """Clone captured writes into native cache, then build next sidecar.

        A partially failed publication poisons this owner and the enclosing
        request must be aborted/cleared.  It is not an atomic cache commit.
        """
        if (self.failed or self.phase != "replayed" or actual_chunk != self.next_chunk):
            self.failed = True
            raise RuntimeError("refresh Graph publish lacks current replay")
        cache = self.global_cache
        following = actual_chunk + 1
        try:
            if any(tuple(map(_stamp, self.slots[layer])) != self.stamps[layer]
                   for layer in range(LAYERS)):
                raise RuntimeError("refresh Graph static KV slots changed")
            with torch.no_grad():
                for layer in range(LAYERS):
                    key, value = self.slots[layer]
                    cache.commit(layer, actual_chunk, key, value)
                prepared = {}
                for layer in range(LAYERS):
                    entries = cache.history(layer, following)
                    if len(entries) != 5:
                        raise RuntimeError(f"layer {layer} next history incomplete")
                    keys = [entry[1] for entry in entries]
                    values = [entry[2] for entry in entries]
                    prepared[layer] = (
                        following,
                        (torch.cat(keys, dim=1), torch.cat(values, dim=1)),
                    )
            if type(getattr(cache, "packed_history_by_layer", None)) is not dict:
                raise TypeError("global cache lacks packed sidecar")
            cache.packed_history_by_layer = prepared
        except BaseException:
            self.failed = True
            self.phase = "failed"
            raise
        self.next_chunk = following
        self.phase = "captured"
        return {"actual_chunk": actual_chunk, "layers_published": LAYERS,
                "next_packed_chunk": following, "native_cache_cloned": True}
