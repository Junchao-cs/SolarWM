"""Resident Graph input slots refreshed for every camera-window chunk.

Graph stores addresses, never the assumption that a translated prefix remains
numerically invariant. A shape owner is separate from a request owner.
"""
from dataclasses import replace, fields
import os
import threading
import torch
from deploy.solar_wm.backends.memory.acceleration.graph.tensor_slots import clone
from deploy.solar_wm.backends.memory.acceleration.graph.tensor_slots import copied
from deploy.solar_wm.backends.memory.acceleration.graph.tensor_slots import identity

def ordered_resident_kwargs(source, tensors, constants, control):
    """Preserve Dynamo's observed dictionary ABI, including insertion order."""
    if set(source) != set(tensors) | set(constants) | {'sgf_attention'}:
        raise RuntimeError('native full-fill argument set changed')
    return {key: control if key == 'sgf_attention' else tensors[key] if key in tensors else constants[key] for key in source}

class ControlSlots:

    def __init__(self, source):
        if source.chunk_index != 5 or len(source.cache.layers) != 50:
            raise RuntimeError('normalized full50 W5 required')
        (self.layout, self.ids) = (source.layout, source.bank_indices)
        (self.views, self.K) = (clone(source.viewmats), clone(source.intrinsics))
        self.rotaries = {n: tuple((clone(t) for t in getattr(source, n))) for n in ('prefix_rotary', 'window_rotary')}
        self.cache = type(source.cache)()
        self.cache.layers = {i: [(j, clone(k), clone(v)) for (j, k, v) in source.cache.history(i, 5)] for i in range(50)}
        self.controls = {}

    def load(self, source):
        if source.layout is not self.layout or source.bank_indices != self.ids or source.chunk_index != 5:
            raise RuntimeError('camera Graph shape owner changed')
        pairs = [(self.views, source.viewmats), (self.K, source.intrinsics)]
        for (n, targets) in self.rotaries.items():
            pairs.extend(zip(targets, getattr(source, n), strict=True))
        for i in range(50):
            rows = source.cache.history(i, 5)
            if len(rows) != 5:
                raise RuntimeError('exactly five history chunks required')
            for ((j, k, v), (sj, sk, sv)) in zip(self.cache.layers[i], rows, strict=True):
                if j != sj:
                    raise RuntimeError('history order changed')
                pairs.extend(((k, sk), (v, sv)))
        return copied(pairs)

    def control(self, source):
        commit = bool(source.commit_cache)
        if commit not in self.controls:
            self.controls[commit] = replace(source, cache=self.cache, viewmats=self.views, intrinsics=self.K, prefix_rotary=self.rotaries['prefix_rotary'], window_rotary=self.rotaries['window_rotary'])
        return self.controls[commit]

class StaticControlSlots(ControlSlots):
    """Own layout tensors so query mark_dynamic cannot mutate D0's metadata."""

    def __init__(self, source):
        super().__init__(source)
        self.bound_layout = replace(source.layout, **{f.name: clone(getattr(source.layout, f.name)) for f in fields(source.layout) if isinstance(getattr(source.layout, f.name), torch.Tensor)})

    def control(self, source):
        commit = bool(source.commit_cache)
        if commit not in self.controls:
            self.controls[commit] = replace(super().control(source), layout=self.bound_layout)
        return self.controls[commit]

def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in tensor_leaves(item)]
    if value is None:
        return []
    raise TypeError('unexpected graph output leaf')

class Captured:

    def __init__(self, fn, count, args, group):
        import torch.distributed as dist
        if torch.is_grad_enabled() or torch.is_autocast_enabled('cuda') or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('inference context required')
        before = count()
        for _ in range(3):
            fn(*args)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        torch.cuda.synchronize()
        self.stream = torch.cuda.Stream()
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                fn(*args)
        caller.wait_stream(self.stream)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        reference = [t.clone() for t in tensor_leaves(fn(*args))]
        torch.cuda.synchronize()
        dist.barrier(group=group)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.outputs = fn(*args)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        if count() != before:
            raise RuntimeError('capture must not trigger compilation')
        self.replays = 0
        actual = tensor_leaves(self.replay())
        local_exact = len(actual) == len(reference) and all((a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b) for (a, b) in zip(actual, reference)))
        verdicts = [None] * dist.get_world_size(group)
        dist.all_gather_object(verdicts, bool(local_exact), group=group)
        if not all(verdicts):
            raise RuntimeError('Graph replay differs from same-input compiled call')
        self.capture_reference_exact = True
        self.compared_tensors = len(reference)
        del reference, actual
        self.replays = 0

    def replay(self):
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        caller.wait_stream(self.stream)
        self.replays += 1
        return self.outputs

    def close(self):
        torch.cuda.synchronize()
        self.graph.reset()
        self.outputs = None

class FullFill:

    def __init__(self, compiler, group):
        (self.compiler, self.group) = (compiler, group)
        self.control_slots = None
        self.tensors = {}
        self.constants = {}
        self.phase = None
        self.loads = 0
        self.bytes = 0
        self.closed = False
        self.failed = False
        self.owner = (os.getpid(), threading.get_ident())

    def __call__(self, kwargs, *, graph):
        if self.closed or self.failed or self.owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError('full refill Graph owner unavailable')
        try:
            source = kwargs['sgf_attention']
            if source.commit_cache:
                raise RuntimeError('D0 cannot commit refresh')
            if self.control_slots is None:
                self.control_slots = StaticControlSlots(source)
                self.tensors = {k: clone(v) for (k, v) in kwargs.items() if isinstance(v, torch.Tensor)}
                self.constants = {k: v for (k, v) in kwargs.items() if k != 'sgf_attention' and k not in self.tensors}
                self.pointers = tuple((identity(t) for t in self.tensors.values()))
            if set(kwargs) != set(self.tensors) | set(self.constants) | {'sgf_attention'}:
                raise RuntimeError('native full-fill argument set changed')
            if any((kwargs[k] != v for (k, v) in self.constants.items())):
                raise RuntimeError('native full-fill non-tensor argument changed')
            self.bytes += self.control_slots.load(source)
            self.bytes += copied([(t, kwargs[k]) for (k, t) in self.tensors.items()])
            if tuple((identity(t) for t in self.tensors.values())) != self.pointers:
                raise RuntimeError('native full-fill input address changed')
            resident = ordered_resident_kwargs(kwargs, self.tensors, self.constants, self.control_slots.control(source))
            if not graph:
                return self.compiler(resident)
            if self.phase is None:
                if self.compiler.calls != 1:
                    raise RuntimeError('warm full-fill compile first')
                self.phase = Captured(self.compiler, lambda : self.compiler.calls, (resident,), self.group)
            self.loads += 1
            return self.phase.replay()
        except BaseException:
            self.failed = True
            raise

    def report(self):
        return dict(captures=int(self.phase is not None), replays=0 if self.phase is None else self.phase.replays, input_loads=self.loads, copied_bytes=self.bytes, failed=self.failed, compile_calls=self.compiler.calls, capture_reference_exact=None if self.phase is None else self.phase.capture_reference_exact)

    def close(self):
        if self.closed:
            return
        if self.phase is not None:
            self.phase.close()
        self.closed = True

class Query:

    def __init__(self, compiler, group):
        (self.compiler, self.group) = (compiler, group)
        self.control_slots = None
        self.common = {}
        self.times = {}
        self.prefix = None
        self.phases = {}
        self.epoch = None
        self.ordinal = 0
        self.closed = False
        self.loads = self.bytes = 0

    def __call__(self, phase, args, epoch):
        if self.closed:
            raise RuntimeError('closed query Graph')
        expected = ('derived-fill', 'reuse', 'reuse', 'refresh')
        if epoch != self.epoch:
            if self.epoch is not None and self.ordinal != 4:
                raise RuntimeError('unfinished query epoch')
            self.epoch = epoch
            self.ordinal = 0
        if self.ordinal >= 4 or phase != expected[self.ordinal]:
            raise RuntimeError('query Graph phase sequence')
        source = args[4]
        if bool(source.commit_cache) != (phase == 'refresh'):
            raise RuntimeError('query refresh mismatch')
        if self.control_slots is None:
            self.control_slots = ControlSlots(source)
            self.common = {i: clone(args[i]) for i in (0, 2, 3, 7, 8)}
            self.prefix = tuple((tuple((clone(t) for t in row)) for row in args[5]))
        if phase not in self.times:
            self.times[phase] = clone(args[1])
        pairs = [(self.common[i], args[i]) for i in (0, 2, 3)] + [(self.times[phase], args[1])]
        if self.ordinal == 0:
            self.bytes += self.control_slots.load(source)
            pairs.extend(((self.common[i], args[i]) for i in (7, 8)))
            pairs.extend(((a, b) for (dst, src) in zip(self.prefix, args[5], strict=True) for (a, b) in zip(dst, src, strict=True)))
        self.bytes += copied(pairs)
        if phase == 'derived-fill':
            if args[6] is not None or args[9] is not None:
                raise RuntimeError('unexpected initial derived cache')
            derived = geometry = None
        else:
            fill = self.phases['derived-fill'].outputs
            (derived, geometry) = (fill[2], fill[4])
            if any((a is not b for (x, y) in zip(args[6], derived, strict=True) for (a, b) in zip(x, y, strict=True))):
                raise RuntimeError('query must consume this Graph D1 producer')
            if any((a is not b for (a, b) in zip(args[9], geometry, strict=True))):
                raise RuntimeError('query geometry producer changed')
        resident = (self.common[0], self.times[phase], self.common[2], self.common[3], self.control_slots.control(source), self.prefix, derived, self.common[7], self.common[8], geometry)
        if phase not in self.phases:
            if self.compiler.calls.get(phase) != 1:
                raise RuntimeError('warm query compile before capture')
            self.phases[phase] = Captured(self.compiler.entry(phase), lambda : dict(self.compiler.calls), resident, self.group)
        result = self.phases[phase].replay()
        self.ordinal += 1
        self.loads += 1
        return result

    def report(self):
        return dict(phases={k: dict(captures=1, replays=v.replays, capture_reference_exact=v.capture_reference_exact) for (k, v) in self.phases.items()}, input_loads=self.loads, copied_bytes=self.bytes)

    def close(self):
        if self.closed:
            return
        for phase in self.phases.values():
            phase.close()
        self.closed = True
