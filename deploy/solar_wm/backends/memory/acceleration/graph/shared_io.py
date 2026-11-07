"""Direct prefix and shared history with exact addressed-view ownership.

No attention, precision, retrieval, camera or refresh arithmetic changes.
The producer owns its outputs until all four query passes complete. Each shape
owns a separate pair of graphs; references must never cross shape owners.
The resident implementation is bundled locally; no experiment-directory imports
or external deployment monkey patches are required.
"""
from deploy.solar_wm.backends.memory.acceleration.graph import memory_resident as resident_core

def tensor_stamp(tensor):
    return (resident_core.identity(tensor), tensor._version)

def history_binding(source):
    rows = []
    for layer in range(50):
        history = source.cache.history(layer, 5)
        if tuple((j for (j, _, _) in history)) != (0, 1, 2, 3, 4):
            raise RuntimeError('normalized W5 history required')
        for (j, key, value) in history:
            rows.append((layer, j, key, tensor_stamp(key), value, tensor_stamp(value)))
    return tuple(rows)

def verify_history(expected, source):
    actual = history_binding(source)
    if len(actual) != len(expected):
        raise RuntimeError('shared history extent changed')
    for (old, new) in zip(expected, actual, strict=True):
        (a, j, k, ks, v, vs) = old
        (b, q, nk, nks, nv, nvs) = new
        if (a, j, ks, vs) != (b, q, nks, nvs) or tensor_stamp(k) != ks or tensor_stamp(v) != vs:
            raise RuntimeError('query history differs from current D0 producer')

class BorrowPrefixMixin:
    """Use with the unchanged PrefixOwner checks and finish/reset methods."""

    def fill(self, layer, key, value, prefix):
        if self.ready or layer in self.seen:
            raise RuntimeError('prefix fill ownership/order')
        if not 0 <= layer < 50 or key.shape != value.shape or key.shape[1] != prefix:
            raise RuntimeError('D0 must export exactly the complete prefix')
        self.seen.add(layer)
        self.kv[layer] = (key, value)

class FullFill(resident_core.FullFill):

    def __init__(self, compiler, group, *, shared_history):
        super().__init__(compiler, group)
        self.shared_history = shared_history
        self.query_owner = None
        self.generation = 0
        self.source_history = None
        self.current_prefix = None

    def __call__(self, kwargs, *, graph):
        query = self.query_owner
        if query is not None and query.inflight:
            raise RuntimeError('cannot overwrite D0 outputs before refresh completes')
        result = super().__call__(kwargs, graph=graph)
        if len(result) != 3 or len(result[2]) != 50:
            raise RuntimeError('D0 full50 prefix output ABI changed')
        self.source_history = history_binding(kwargs['sgf_attention']) if self.shared_history else None
        self.current_prefix = result[2]
        self.generation += 1
        return result

    def close(self):
        if self.query_owner is not None and self.query_owner.inflight:
            raise RuntimeError('cannot close a borrowed D0 during query execution')
        super().close()

class SharedHistorySlots(resident_core.ControlSlots):
    """Share only raw history. Cameras/rotaries retain their original copies."""

    def __init__(self, source, producer):
        self.producer = producer
        (self.layout, self.ids) = (source.layout, source.bank_indices)
        (self.views, self.K) = (resident_core.clone(source.viewmats), resident_core.clone(source.intrinsics))
        self.rotaries = {name: tuple((resident_core.clone(t) for t in getattr(source, name))) for name in ('prefix_rotary', 'window_rotary')}
        self.cache = producer.control_slots.cache
        self.controls = {}
        verify_history(producer.source_history, source)

    def load(self, source):
        if source.layout is not self.layout or source.bank_indices != self.ids or source.chunk_index != 5:
            raise RuntimeError('camera Graph shape owner changed')
        if self.producer.closed or self.cache is not self.producer.control_slots.cache:
            raise RuntimeError('shared history producer unavailable')
        verify_history(self.producer.source_history, source)
        pairs = [(self.views, source.viewmats), (self.K, source.intrinsics)]
        for (name, targets) in self.rotaries.items():
            pairs.extend(zip(targets, getattr(source, name), strict=True))
        return resident_core.copied(pairs)

class Query(resident_core.Query):
    """Keep phase ordering and capture equivalence; borrow prefix storage."""

    def __init__(self, compiler, group, producer):
        super().__init__(compiler, group)
        if producer.query_owner is not None:
            raise RuntimeError('D0 already has a query owner')
        self.producer = producer
        producer.query_owner = self
        self.inflight = False
        self.failed = False
        self.producer_generation = None
        self.prefix_identity = None
        self.saved_prefix_copy_bytes = 0
        self.saved_history_copy_bytes = 0

    def __call__(self, phase, args, epoch):
        if self.closed or self.failed or self.producer.closed or self.producer.failed:
            raise RuntimeError('linked Graph owner unavailable')
        try:
            return self._run(phase, args, epoch)
        except BaseException:
            self.failed = True
            raise

    def _run(self, phase, args, epoch):
        expected = ('derived-fill', 'reuse', 'reuse', 'refresh')
        if epoch != self.epoch:
            if self.epoch is not None and self.ordinal != 4:
                raise RuntimeError('unfinished query epoch')
            if self.producer_generation == self.producer.generation:
                raise RuntimeError('new query epoch requires a new D0 producer')
            (self.epoch, self.ordinal) = (epoch, 0)
            self.producer_generation = self.producer.generation
        if self.ordinal >= 4 or phase != expected[self.ordinal]:
            raise RuntimeError('query Graph phase sequence')
        if self.producer_generation != self.producer.generation:
            raise RuntimeError('D0 changed inside query epoch')
        if self.producer.phase is None:
            raise RuntimeError('query requires resident captured D0 outputs')
        source = args[4]
        if bool(source.commit_cache) != (phase == 'refresh'):
            raise RuntimeError('query refresh mismatch')
        prefix = args[5]
        if len(prefix) != 50 or any((len(row) != 2 for row in prefix)):
            raise RuntimeError('query prefix ABI changed')
        if any((a is not b for (left, right) in zip(prefix, self.producer.current_prefix, strict=True) for (a, b) in zip(left, right, strict=True))):
            raise RuntimeError('query must borrow this D0 output, not a clone or another owner')
        pointers = tuple((resident_core.identity(t) for row in prefix for t in row))
        if self.control_slots is None:
            self.control_slots = SharedHistorySlots(source, self.producer) if self.producer.shared_history else resident_core.ControlSlots(source)
            self.common = {i: resident_core.clone(args[i]) for i in (0, 2, 3, 7, 8)}
            self.prefix = prefix
            self.prefix_identity = pointers
        if pointers != self.prefix_identity:
            raise RuntimeError('D0 output address changed; new shape owner required')
        if phase not in self.times:
            self.times[phase] = resident_core.clone(args[1])
        pairs = [(self.common[i], args[i]) for i in (0, 2, 3)] + [(self.times[phase], args[1])]
        if self.ordinal == 0:
            self.bytes += self.control_slots.load(source)
            pairs.extend(((self.common[i], args[i]) for i in (7, 8)))
            self.saved_prefix_copy_bytes += sum((t.numel() * t.element_size() for row in prefix for t in row))
            if self.producer.shared_history:
                self.saved_history_copy_bytes += sum((t.numel() * t.element_size() for (_, _, k, _, v, _) in self.producer.source_history for t in (k, v)))
        self.bytes += resident_core.copied(pairs)
        if phase == 'derived-fill':
            if args[6] is not None or args[9] is not None:
                raise RuntimeError('unexpected initial derived cache')
            derived = geometry = None
        else:
            fill = self.phases['derived-fill'].outputs
            (derived, geometry) = (fill[2], fill[4])
            if any((a is not b for (left, right) in zip(args[6], derived, strict=True) for (a, b) in zip(left, right, strict=True))):
                raise RuntimeError('query must consume current D1 producer')
            if any((a is not b for (a, b) in zip(args[9], geometry, strict=True))):
                raise RuntimeError('query geometry producer changed')
        resident = (self.common[0], self.times[phase], self.common[2], self.common[3], self.control_slots.control(source), self.prefix, derived, self.common[7], self.common[8], geometry)
        self.inflight = True
        if phase not in self.phases:
            if self.compiler.calls.get(phase) != 1:
                raise RuntimeError('warm query compile before capture')
            self.phases[phase] = resident_core.Captured(self.compiler.entry(phase), lambda : dict(self.compiler.calls), resident, self.group)
        result = self.phases[phase].replay()
        self.ordinal += 1
        self.loads += 1
        if self.ordinal == 4:
            self.inflight = False
        return result

    def report(self):
        return dict(super().report(), direct_prefix=True, shared_history=self.producer.shared_history, saved_prefix_input_copy_bytes=self.saved_prefix_copy_bytes, saved_history_input_copy_bytes=self.saved_history_copy_bytes, failed=self.failed)

    def close(self):
        super().close()
        self.inflight = False
