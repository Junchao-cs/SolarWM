"""Reuse old full-student refresh Graph with the existing dynamic FX entry.

No new compile entry. Fixed private packed inputs feed the restored refresh
callable; its 100 returned K/V values are copied into the old RefreshWriteback
slots during capture. Native publication remains outside replay. Not deployed."""

from dataclasses import replace


class StableDynamicRefreshEntry:
    def __init__(self, compiler, denoise_bundle, writeback):
        import torch
        from ..compile.compiler import SharedCompiler
        from ..attention.history import PackedHistoryCache
        from .writeback import RefreshWriteback
        if type(writeback) is not RefreshWriteback or writeback.history_owner is not denoise_bundle['history']:
            raise TypeError('actual shared refresh writeback/history required')
        if torch.is_grad_enabled() or (5, True) not in compiler.entries:
            raise RuntimeError('restore dynamic refresh before Graph capture')
        self.compiler = compiler
        self.history = denoise_bundle['history']
        self.denoise_bridge = denoise_bundle['bridge']
        self.writeback = writeback
        self.private = PackedHistoryCache()
        self.private.packed_history_by_layer = dict(self.denoise_bridge.private.packed_history_by_layer)
        self.fn = SharedCompiler.entry(compiler, (5, True))
        self.counts = dict(compiler.backend_calls)
        self.calls = 0

    def __call__(self, **kwargs):
        import torch
        from solar.backends.minimax_h3 import sgf_attention
        from ..compile.compiler import dynamic_metadata
        from ..compile.compiler import mark_prompt_dimensions
        control = kwargs['sgf_attention']
        if (control.cache is not self.writeback or control.chunk_index != 5
                or control.commit_cache is not True or self.writeback.phase != 'capturing'
                or control.viewmats is not self.history.viewmats
                or control.intrinsics is not self.history.intrinsics):
            raise RuntimeError('full refresh capture binding changed')
        if torch.is_grad_enabled() or not self.history.ready or self.history.poisoned:
            raise RuntimeError('ready no-grad history required')
        # Each prime/capture starts from the same five real canonical inputs.
        # Python dictionary updates run only during prime/capture, not replay.
        original = self.denoise_bridge.private.layers
        if set(original) != set(range(50)) or any(tuple(r[0] for r in v) != tuple(range(5)) for v in original.values()):
            raise RuntimeError('five canonical raw-history entries required')
        self.private.layers = {layer: list(rows) for layer, rows in original.items()}
        borrowed = [(t, t._version) for rows in original.values() for row in rows for t in row[1:]]
        private_control = replace(control, cache=self.private,
            viewmats=self.denoise_bridge.views, intrinsics=self.denoise_bridge.intrinsics)
        packet = dict(kwargs, sgf_attention=private_control)
        rotary = getattr(sgf_attention.prepare_sgf_rotaries, '__self__', None)
        mark_prompt_dimensions(packet, rotary)
        previous = self.compiler.allow_compile
        self.compiler.allow_compile = False
        try:
            with dynamic_metadata(), torch._dynamo.config.patch(error_on_recompile=True):
                result = self.fn(**packet)
        finally:
            self.compiler.allow_compile = previous
        if self.compiler.backend_calls != self.counts:
            raise RuntimeError('refresh bridge retraced the dynamic entry')
        if any(t._version != version for t, version in borrowed):
            raise RuntimeError('refresh bridge mutated borrowed history')
        for layer in range(50):
            rows = self.private.layers[layer]
            if tuple(r[0] for r in rows) != (1, 2, 3, 4, 5):
                raise RuntimeError('dynamic refresh did not return current KV for every layer')
            self.writeback.commit(layer, 5, rows[-1][1], rows[-1][2])
        self.calls += 1
        return result


def capture(*, engine, compiler, backend_owner, denoise_bundle, cache, noisy, timestep):
    from .refresh import RefreshStudentCall, ManualRefreshGraph
    from .bridge import installed_full_student_bridge
    call = RefreshStudentCall(student=engine.model, inputs=engine.inputs,
        history_owner=denoise_bundle['history'], global_cache=cache, noisy=noisy, timestep=timestep)
    bridge = StableDynamicRefreshEntry(compiler, denoise_bundle, call.writeback)
    graph = ManualRefreshGraph(call)
    with installed_full_student_bridge(engine.base, backend_owner, bridge):
        first = graph.capture()
    if graph.reference_max_abs != 0 or graph.kv_reference_max_abs != 0:
        raise RuntimeError('full refresh Graph differs from pinned compiled reference')
    call.writeback.publish(5)
    return dict(graph=graph, call=call, bridge=bridge, first_output=first,
                initial_published_cache=cache)


def replay_and_publish(bundle, *, index, noisy, timestep):
    from .denoise import _read_token
    graph, call = bundle['graph'], bundle['call']
    call.load(actual_chunk=index, noisy=noisy, timestep=timestep)
    # Same chunk-5 resident replay path used by the old web service. The old
    # ManualRefreshGraph.replay public entry permits only index >= 6.
    if index == 5:
        graph._check()
        if graph.capture_count != 1 or graph.graph is None:
            raise RuntimeError('resident refresh capture missing')
        before = _read_token(call)
        result = graph._replay_once()
        if _read_token(call) != before:
            raise RuntimeError('full refresh replay mutated stable inputs')
        call.writeback.mark_replayed(index)
    else:
        result = graph.replay()
    call.writeback.publish(index)
    return result
