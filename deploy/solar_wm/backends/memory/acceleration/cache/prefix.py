def stamp(t):
    return (tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t._version)

class PrefixOwner:

    def __init__(self, *, request):
        if not request:
            raise ValueError('explicit request owner')
        self.request = request
        self.chunk = None
        self.layout = None
        self.ids = None
        self.cache = None
        self.kv = {}
        self.seen = set()
        self.history = {}
        self.camera = None
        self.video = None
        self.audio = None
        self.ready = False
        self.fills = 0
        self.reuses = 0
        self.commits = 0

    def begin_fill(self, source):
        if source.commit_cache or source.chunk_index < 6 or len(source.bank_indices) != 5:
            raise RuntimeError('candidate covers mature bank5 d0 only')
        if self.chunk == source.chunk_index:
            raise RuntimeError('duplicate prefix fill in request')
        self.ready = False
        self.chunk = source.chunk_index
        self.layout = source.layout
        self.ids = source.bank_indices
        self.cache = source.cache
        self.seen = set()
        self.camera = (source.viewmats, stamp(source.viewmats), source.intrinsics, stamp(source.intrinsics))
        self.history = {layer: tuple(((i, k, stamp(k), v, stamp(v)) for (i, k, v) in source.cache.history(layer, self.chunk))) for layer in range(50)}

    def fill(self, layer, k, v, prefix):
        if self.ready or layer in self.seen:
            raise RuntimeError('prefix fill ownership/order')
        self.seen.add(layer)
        current = (k[:, :prefix], v[:, :prefix])
        if layer not in self.kv:
            self.kv[layer] = tuple((t.clone() for t in current))
        else:
            for (target, t) in zip(self.kv[layer], current):
                if (target.shape, target.dtype, target.device) != (t.shape, t.dtype, t.device):
                    raise RuntimeError('prefix domain changed; new owner required')
                target.copy_(t)

    def finish_fill(self, result, video_prefix):
        if self.seen != set(range(50)):
            raise RuntimeError('not all 50 prefix layers were produced')
        self.video = result[0][:, :video_prefix].clone()
        self.audio = result[1].clone()
        self.saved_stamps = {i: tuple((stamp(t) for t in kv)) for (i, kv) in self.kv.items()}
        self.output_stamps = (stamp(self.video), stamp(self.audio))
        self.ready = True
        self.fills += 1

    def verify(self, source):
        if not self.ready or source.chunk_index != self.chunk or source.layout is not self.layout or (source.bank_indices != self.ids) or (source.cache is not self.cache):
            raise RuntimeError('prefix semantic owner changed')
        (a, sa, b, sb) = self.camera
        if source.viewmats is not a or source.intrinsics is not b or stamp(a) != sa or (stamp(b) != sb):
            raise RuntimeError('camera changed after prefix fill')
        if (stamp(self.video), stamp(self.audio)) != self.output_stamps:
            raise RuntimeError('cached output mutated')
        for layer in range(50):
            previous = self.history[layer]
            actual = source.cache.history(layer, self.chunk)
            if len(actual) != len(previous):
                raise RuntimeError('history length changed')
            if any((i != j or k is not a or v is not b or (stamp(k) != sa) or (stamp(v) != sb) for ((i, k, v), (j, a, sa, b, sb)) in zip(actual, previous))):
                raise RuntimeError('history changed after prefix fill')
            if tuple((stamp(t) for t in self.kv[layer])) != self.saved_stamps[layer]:
                raise RuntimeError('cached prefix KV mutated')

    def reset(self):
        self.ready = False
        self.chunk = None
        self.kv.clear()
        self.history.clear()
        self.video = None
        self.audio = None
