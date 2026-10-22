"""Bind stable graph inputs to existing dynamic-compile entries without retracing."""

from contextlib import contextmanager
from dataclasses import replace
import sys


def canonical_rows(entries, actual_chunk):
    """Exact five real source indices become read-only local indices 0..4."""
    if type(actual_chunk) is not int or actual_chunk < 5:
        raise ValueError('steady chunk required')
    if tuple(row[0] for row in entries) != tuple(range(actual_chunk-5, actual_chunk)):
        raise ValueError('actual five-chunk history required')
    return [(i, row[1], row[2]) for i, row in enumerate(entries)]


class StableDynamicEntry:
    def __init__(self, compiler, history, source_cache, actual_chunk):
        import torch
        from ..compile.compiler import SharedCompiler
        from ..attention.history import PackedHistoryCache
        from ..compile.window import singleton_stride_view
        from .buffers import StableSteadyInputs
        if type(history) is not StableSteadyInputs or type(source_cache) is not PackedHistoryCache:
            raise TypeError('actual stable owner and global packed cache required')
        if torch.is_grad_enabled() or (5, False) not in compiler.entries:
            raise RuntimeError('restore and verify the dynamic steady entry first')
        if set(history.buffers) != set(range(50)) or history.next_chunk != actual_chunk+1:
            raise RuntimeError('current full-model stable history required')
        self.history, self.compiler = history, compiler
        # Old stable ownership reserves 291 camera entries. The current
        # dynamic Compile contract is the exact 31-entry canonical window.
        # Retain only an aliasing view of the populated prefix, not a new
        # camera shape or reads of the uninitialized unused tail.
        self.views = singleton_stride_view(history.viewmats[:, :31])
        self.intrinsics = singleton_stride_view(history.intrinsics[:, :31])
        self.private = PackedHistoryCache()
        self.sources = []
        for layer in range(50):
            entries = source_cache.history(layer, actual_chunk)
            self.private.layers[layer] = [(i, singleton_stride_view(k), singleton_stride_view(v))
                for i,k,v in canonical_rows(entries, actual_chunk)]
            pair = tuple(singleton_stride_view(t) for t in history.buffers[layer])
            self.private.packed_history_by_layer[layer] = (5, pair)
            self.sources.extend(t for row in entries for t in row[1:])
        # The base method returns the existing compiled callable directly,
        # not the per-chunk direct-replay adapter around its backend Graph.
        self.fn = SharedCompiler.entry(compiler, (5, False))
        self.backend_counts = dict(compiler.backend_calls)
        self.calls = 0

    def __call__(self, **kwargs):
        import torch
        from solarwm.backends.minimax_h3 import sgf_attention
        from ..compile.compiler import dynamic_metadata
        from ..compile.compiler import mark_prompt_dimensions
        control = kwargs['sgf_attention']
        if (control.cache is not self.history or control.chunk_index != 5 or control.commit_cache
                or control.viewmats is not self.history.viewmats
                or control.intrinsics is not self.history.intrinsics):
            raise RuntimeError('full-student stable call binding changed')
        private = replace(control, cache=self.private, viewmats=self.views, intrinsics=self.intrinsics)
        packet = dict(kwargs, sgf_attention=private)
        rotaries = getattr(sgf_attention.prepare_sgf_rotaries, '__self__', None)
        mark_prompt_dimensions(packet, rotaries)
        previous = self.compiler.allow_compile
        self.compiler.allow_compile = False
        try:
            with dynamic_metadata(), torch._dynamo.config.patch(error_on_recompile=True):
                result = self.fn(**packet)
        finally:
            self.compiler.allow_compile = previous
        if self.compiler.backend_calls != self.backend_counts:
            raise RuntimeError('full-student bridge retraced the dynamic entry')
        if any(len(v) != 5 for v in self.private.layers.values()):
            raise RuntimeError('read-only full-student changed private raw history')
        self.calls += 1
        return result


@contextmanager
def installed_full_student_bridge(base, backend_owner, bridge):
    """Only during prime/capture: use compiled FX, not a nested backend Graph."""
    previous_forward, previous_mode = base.forward, backend_owner.mode
    base.forward = bridge
    backend_owner.mode = 'plain'
    try:
        yield
    finally:
        primary = sys.exc_info()[1]
        altered = base.forward is not bridge or backend_owner.mode != 'plain'
        base.forward = previous_forward
        backend_owner.mode = previous_mode
        if altered:
            secondary = RuntimeError('full-student bridge ownership changed')
            if primary is not None:
                primary.full_student_bridge_cleanup_secondary = secondary
            else:
                raise secondary
