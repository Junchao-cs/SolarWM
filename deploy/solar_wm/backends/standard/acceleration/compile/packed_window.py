"""Private packed-KV windows and scoped singleton-stride normalization."""

from __future__ import annotations
from dataclasses import replace
import torch
from .window import SingletonStrideWindowCall
from ..attention.history import PackedHistoryCache
from .forward import tensor_stamp
from .window import singleton_stride_view


class PackedWindowCall(SingletonStrideWindowCall):
    def __init__(self, control, *, layer_count=50):
        if type(control.cache) is not PackedHistoryCache:
            raise RuntimeError("global raw-KV cache is not owned packed cache")
        super().__init__(control, layer_count=layer_count)
        native_private = self.control.cache
        private = PackedHistoryCache()
        private.layers = native_private.layers
        self.borrowed_pairs = ()
        if self.index:
            expected = control.cache.packed_history_by_layer
            if set(expected) != set(range(layer_count)):
                raise RuntimeError("global packed sidecar lacks a layer")
            for layer in range(layer_count):
                global_index, tensors = expected[layer]
                if global_index != self.index or len(tensors) != 2:
                    raise RuntimeError("global packed sidecar is stale")
                private.packed_history_by_layer[layer] = (self.local_index, tensors)
            self.borrowed_pairs = tuple(
                tensor for layer in range(layer_count)
                for tensor in expected[layer][1]
            )
        elif expected := control.cache.packed_history_by_layer:
            raise RuntimeError("first chunk has preexisting packed sidecar")
        self.control = replace(self.control, cache=private)
        self.borrowed_stamps = tuple(tensor_stamp(t) for t in self.borrowed_pairs)

    def finish(self):
        if tuple(tensor_stamp(t) for t in self.borrowed_pairs) != self.borrowed_stamps:
            raise RuntimeError("borrowed packed KV tensor changed before finish")
        result = super().finish()
        if tuple(tensor_stamp(t) for t in self.borrowed_pairs) != self.borrowed_stamps:
            raise RuntimeError("borrowed packed KV tensor changed during forward")
        if self.source.commit_cache:
            global_cache = self.source.cache
            if type(global_cache) is not PackedHistoryCache:
                raise RuntimeError("global raw-KV cache changed before sidecar publish")
            following = self.index + 1
            prepared = {}
            for layer in range(self.layer_count):
                entries = global_cache.history(layer, following)
                if not entries:
                    raise RuntimeError("validated refresh left empty global raw KV")
                if len(entries) == 1:
                    pair = (entries[0][1], entries[0][2])
                else:
                    pair = (torch.cat([entry[1] for entry in entries], dim=1),
                            torch.cat([entry[2] for entry in entries], dim=1))
                if (pair[0].shape[1] != sum(entry[1].shape[1] for entry in entries)
                        or pair[1].shape[1] != sum(entry[2].shape[1] for entry in entries)):
                    raise RuntimeError("packed KV length differs from raw history")
                prepared[layer] = (following, pair)
            global_cache.packed_history_by_layer = prepared
        return result


OriginalWindow=PackedWindowCall


def normalize_private_sidecars(cache):
    count=0
    for layer,(index,pair) in tuple(cache.packed_history_by_layer.items()):
        key,value=pair
        new_key,new_value=singleton_stride_view(key),singleton_stride_view(value)
        cache.packed_history_by_layer[layer]=(index,(new_key,new_value))
        count+=int(new_key is not key)+int(new_value is not value)
    return count


class SidecarCanonicalWindow(OriginalWindow):
    def __init__(self,control,*,layer_count=50):
        super().__init__(control,layer_count=layer_count)
        private=self.control.cache
        if private is control.cache or private.packed_history_by_layer is control.cache.packed_history_by_layer:
            raise RuntimeError('sidecar normalization requires a distinct owned private map')
        self.packed_singleton_views=normalize_private_sidecars(private)
        # borrowed_pairs/stamps intentionally still refer to ORIGINAL global
        # tensors. Inherited finish() checks them before and after the forward.


class Installed:
    def __init__(self):
        global PackedWindowCall
        if PackedWindowCall is not OriginalWindow:
            raise RuntimeError('packed-window port already owned')
        PackedWindowCall=SidecarCanonicalWindow
        self.closed=False

    def close(self):
        global PackedWindowCall
        if self.closed:return
        changed=PackedWindowCall is not SidecarCanonicalWindow
        PackedWindowCall=OriginalWindow
        self.closed=True
        if changed:raise RuntimeError('packed-window port changed while owned')


def install():return Installed()
