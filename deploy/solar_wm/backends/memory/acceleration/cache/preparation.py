"""Owned bank gather/patchification and Memory rotary preparation caches.

These sit outside the full-model compiled boundary. No future control is read.
Version stamps require cooperative writers; alias writes bypassing counters are
not authenticated by this cache. Reset explicitly on request/model replacement.
"""
from contextlib import contextmanager
from dataclasses import replace

def stamp(t):
    return (tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.dtype, t.device, t._version)

class BankPreparation:

    def __init__(self, bank, patchify):
        self.bank = bank
        self.native = bank.gather
        self.patchify_native = patchify
        self.chunk = None
        self.ordinal = 0
        self.value = None
        self.packed = None
        self.selected = None
        self.parents = None
        self.value_stamp = None
        self.packed_stamp = None
        self.stats = dict(gather_calls=0, gather_hits=0, patchify_calls=0, patchify_hits=0, retained_logical_bytes=0)

    def begin(self, chunk, commit=False):
        if self.chunk != chunk:
            if self.chunk is not None and self.ordinal != 5:
                raise RuntimeError('previous chunk incomplete')
            self.chunk = chunk
            self.ordinal = 0
            self.value = None
            self.packed = None
            self.parents = None
            self.selected = None
        if self.ordinal >= 5 or commit != (self.ordinal == 4):
            raise RuntimeError('bank cache phase/commit order changed')
        self.ordinal += 1

    def reset(self):
        self.chunk = None
        self.ordinal = 0
        self.value = None
        self.packed = None
        self.parents = None
        self.selected = None

    def gather(self, indices, reference):
        if self.chunk is None or not self.ordinal:
            raise RuntimeError('begin chunk before gathering')
        selected = tuple((int(i) for i in indices))
        if any((i < 0 or i // 5 >= len(self.bank.chunks) for i in selected)):
            raise ValueError('unavailable bank latent')
        parents = tuple(((self.bank.chunks[i // 5], stamp(self.bank.chunks[i // 5])) for i in selected))
        signature = (selected, tuple(reference.shape[:2]), tuple(reference.shape[3:]), reference.dtype, reference.device)
        if self.value is not None:
            if signature != self.selected or len(parents) != len(self.parents) or any((a is not b or sa != sb for ((a, sa), (b, sb)) in zip(parents, self.parents))):
                raise RuntimeError('bank dependency changed within chunk')
            if stamp(self.value) != self.value_stamp:
                raise RuntimeError('cached bank payload mutated')
            self.stats['gather_hits'] += 1
            return self.value
        result = self.native(indices, reference)
        self.stats['gather_calls'] += 1
        if selected:
            self.selected = signature
            self.parents = parents
            self.value = result
            self.value_stamp = stamp(result)
            self.stats['retained_logical_bytes'] = result.numel() * result.element_size()
        return result

    def patchify(self, value):
        if value is not self.value:
            return self.patchify_native(value)
        if stamp(value) != self.value_stamp:
            raise RuntimeError('bank changed before patchify')
        if self.packed is not None:
            if stamp(self.packed) != self.packed_stamp:
                raise RuntimeError('cached packed bank mutated')
            self.stats['patchify_hits'] += 1
            return self.packed
        self.packed = self.patchify_native(value)
        self.packed_stamp = stamp(self.packed)
        self.stats['patchify_calls'] += 1
        self.stats['retained_logical_bytes'] += self.packed.numel() * self.packed.element_size()
        return self.packed

    @contextmanager
    def installed(self, forward_module):
        prior = forward_module.patchify_video
        if prior is not self.patchify_native:
            raise RuntimeError('patchify route not owned')
        had = 'gather' in vars(self.bank)
        old = vars(self.bank).get('gather')
        gather = self.gather
        patchify = self.patchify
        self.bank.gather = gather
        forward_module.patchify_video = patchify
        try:
            yield self
        finally:
            corrupted = self.bank.gather is not gather or forward_module.patchify_video is not patchify
            if self.bank.gather is gather:
                if had:
                    self.bank.gather = old
                else:
                    del self.bank.gather
            if forward_module.patchify_video is patchify:
                forward_module.patchify_video = prior
            self.reset()
            if corrupted:
                raise RuntimeError('bank preparation route ownership changed')

    @contextmanager
    def installed_forward(self, forward_module, forward_owner):
        if forward_owner.bank is not self.bank:
            raise RuntimeError('different latent bank owner')
        kind = type(forward_owner)
        original = kind.__call__

        def call(instance, student, inputs, noisy, timestep, *, chunk_index=0, cache=None, commit_cache=False):
            if instance is forward_owner:
                self.begin(chunk_index, commit_cache)
            return original(instance, student, inputs, noisy, timestep, chunk_index=chunk_index, cache=cache, commit_cache=commit_cache)
        with self.installed(forward_module):
            kind.__call__ = call
            try:
                yield self
            finally:
                corrupted = kind.__call__ is not call
                if not corrupted:
                    kind.__call__ = original
                if corrupted:
                    raise RuntimeError('Memory forward ownership changed')

class RotaryPreparation:

    def __init__(self, *, max_domains=8):
        self.max_domains = max_domains
        self.entries = {}
        self.stats = dict(calls=0, hits=0, retained_bytes=0)

    def reset(self):
        self.entries.clear()

    def prepare(self, original, control, rope):
        position = control.layout.position_ids
        frequency = rope.inv_freq
        key = (id(control.layout), stamp(position), id(rope), stamp(frequency), rope.rope_freq_dim, int(control.layout.sequence_length), int(control.layout.rows_per_video_frame), int(control.layout.latent_height), int(control.layout.latent_width), int(control.layout.text_indices.numel()), min(int(control.chunk_index), 5) * 5 + 5)
        if key in self.entries:
            (layout, owned_rope, old_pos, old_freq, prefix, window, stamps) = self.entries[key]
            if layout is not control.layout or owned_rope is not rope or old_pos is not position or (old_freq is not frequency):
                raise RuntimeError('rotary owner changed')
            values = tuple(prefix) + tuple(window)
            if tuple((stamp(t) for t in values)) != stamps:
                raise RuntimeError('cached rotary output mutated')
            self.stats['hits'] += 1
            return replace(control, prefix_rotary=prefix, window_rotary=window)
        if len(self.entries) >= self.max_domains:
            raise RuntimeError('rotary domain bound exceeded; reset explicitly')
        result = original(control, rope)
        (prefix, window) = (result.prefix_rotary, result.window_rotary)
        values = tuple(prefix) + tuple(window)
        self.entries[key] = (control.layout, rope, position, frequency, prefix, window, tuple((stamp(t) for t in values)))
        self.stats['calls'] += 1
        self.stats['retained_bytes'] += sum((t.numel() * t.element_size() for t in values))
        return result

    @contextmanager
    def installed(self, attention_class):
        original = attention_class.prepare_rotaries

        def prepare(control, rope):
            return self.prepare(original, control, rope)
        attention_class.prepare_rotaries = prepare
        try:
            yield self
        finally:
            corrupted = attention_class.prepare_rotaries is not prepare
            if not corrupted:
                attention_class.prepare_rotaries = original
            self.reset()
            if corrupted:
                raise RuntimeError('rotary preparation route ownership changed')
