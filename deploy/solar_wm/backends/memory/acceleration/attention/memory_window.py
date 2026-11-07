import torch
from solar.backends.minimax_h3.memory_kv_attention import MemoryKVAttention
from solar.backends.minimax_h3.sgf_attention import H3RawKVCache

class PreparedMemoryAttention(MemoryKVAttention):

    def prepare_rotaries(self, rope):
        if self.prefix_rotary is None or self.window_rotary is None:
            raise RuntimeError('outer boundary must prepare native rotaries')
        return self

class Window:

    def __init__(self, source, rope):
        if type(source) is not MemoryKVAttention or source.chunk_index < 6 or len(source.bank_indices) != 5:
            raise ValueError('nonempty Memory steady regime required')
        self.source = source
        j = source.chunk_index
        self.private = H3RawKVCache()
        self.stamps = []
        for layer in range(50):
            history = source.cache.history(layer, j)
            self.private.layers[layer] = [(i, k, v) for (i, (_, k, v)) in enumerate(history)]
            self.stamps.extend(((t, t._version) for (_, k, v) in history for t in (k, v)))
        ids = torch.tensor(source.bank_indices, device=source.viewmats.device) + 1

        def bounded(value):
            return torch.cat((value[:, :1], value[:, 1 + (j - 5) * 5:1 + (j + 1) * 5], value.index_select(1, ids)), 1)
        (views, K) = (bounded(source.viewmats), bounded(source.intrinsics))
        if views.shape[1] != 36 or K.shape[1] != 36:
            raise RuntimeError('bounded camera shape differs')
        control = PreparedMemoryAttention(source.layout, views, K, tuple(range(30, 35)), 5, self.private, source.commit_cache)
        prepared = MemoryKVAttention.prepare_rotaries(control, rope)
        self.control = prepared

    def finish(self):
        if any((t._version != v for (t, v) in self.stamps)):
            raise RuntimeError('borrowed history mutated')
        if self.source.commit_cache:
            for layer in range(50):
                entries = self.private.history(layer, 6)
                if len(entries) != 5 or entries[-1][0] != 5:
                    raise RuntimeError('private commit missing')
                (_, k, v) = entries[-1]
                self.source.cache.commit(layer, self.source.chunk_index, k, v)
