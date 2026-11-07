"""Reuse the bounded current-chunk preparation, not committed future state.

The native Window still performs the first preparation and final real commit.
Only repeated same-chunk camera bounding, history list assembly and singleton
normalization are skipped. A completed refresh cannot be replayed as a writer.
"""
from dataclasses import replace

def stamp(t):
    return (t.untyped_storage()._cdata, t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t._version)

class ChunkWindow:

    def __init__(self, factory):
        self.factory = factory
        self.window = None
        self.source = None
        self.bindings = None
        self.committed = False
        self.pending = False
        self.rope = None
        self.fills = 0
        self.hits = 0
        self.finishes = 0
        self.commits = 0

    def _bindings(self, source):
        return (tuple(((t, stamp(t)) for t in (source.viewmats, source.intrinsics))), tuple(((layer, j, k, stamp(k), v, stamp(v)) for layer in range(50) for (j, k, v) in source.cache.history(layer, source.chunk_index))))

    def prepare(self, source, rope):
        if self.pending:
            raise RuntimeError('previous query phase not finished')
        if self.window is not None and rope is not self.rope:
            raise RuntimeError('rotary producer changed')
        if self.window is None or source.chunk_index != self.source.chunk_index:
            if self.window is not None and (not self.committed):
                raise RuntimeError('new chunk before final refresh')
            if source.commit_cache:
                raise RuntimeError('first query pass cannot be refresh')
            self.window = self.factory(source, rope)
            self.source = source
            self.rope = rope
            self.bindings = self._bindings(source)
            self.committed = False
            self.fills += 1
        else:
            if self.committed:
                raise RuntimeError('window reused after refresh commit')
            if source.layout is not self.source.layout or source.cache is not self.source.cache or source.bank_indices != self.source.bank_indices:
                raise RuntimeError('chunk window semantic owner changed')
            (cameras, history) = self._bindings(source)
            (oldcams, oldhistory) = self.bindings
            if any((t is not old or s != os for ((t, s), (old, os)) in zip(cameras, oldcams, strict=True))):
                raise RuntimeError('camera changed within query chunk')
            if len(history) != len(oldhistory) or any(((l, j, s, u) != (ol, oj, os, ou) or k is not ok or v is not ov for ((l, j, k, s, v, u), (ol, oj, ok, os, ov, ou)) in zip(history, oldhistory, strict=True))):
                raise RuntimeError('history changed within query chunk')
            self.window.source = source
            self.window.control = replace(self.window.control, commit_cache=bool(source.commit_cache))
            self.source = source
            self.hits += 1
        self.pending = True
        return self.window

    def finish(self):
        if self.window is None or self.committed or (not self.pending):
            raise RuntimeError('window finish without pending phase')
        self.window.finish()
        self.finishes += 1
        self.pending = False
        if self.source.commit_cache:
            self.committed = True
            self.commits += 1

    def reset(self):
        self.window = None
        self.source = None
        self.bindings = None
        self.committed = False
        self.pending = False
        self.rope = None
