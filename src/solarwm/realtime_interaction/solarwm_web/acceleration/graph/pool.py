"""Same compiled program, single argument pass and chunk-owned KV copy pool.

Only the exact 100 packed-history FX inputs may skip repeated copies. Sharing
is confined to an explicit serialized chunk, with strong source/root refs and
observable version counters. Non-KV inputs are copied on every replay. This
does not protect against out-of-band CUDA writes that bypass tensor versions."""


def layout(t):
    return tuple(t.shape), tuple(t.stride()), t.dtype, t.device


def root_and_version(t):
    root = t
    while root._base is not None:
        root = root._base
    try:
        version = root._version
    except RuntimeError:
        version = None
    return root, version


def copy_views(target, source):
    # Avoid constructing two all-slice views for every ordinary tensor copy.
    selectors = tuple(0 if stride == 0 and size > 1 else slice(None)
                      for size, stride in zip(source.shape, source.stride()))
    if any(isinstance(x, int) for x in selectors):
        return target[selectors], source[selectors]
    return target, source


class ChunkCopyPool:
    def __init__(self):
        self.epoch = 0
        self.active = False
        self.slots = {}
        self.copies = self.skips = 0

    def begin_chunk(self):
        if self.active:
            raise RuntimeError('nested chunk copy transaction')
        self.epoch += 1
        self.active = True

    def end_chunk(self):
        self.active = False

    def target(self, role, source):
        import torch
        if not self.active:
            raise RuntimeError('KV buffer access outside serialized chunk')
        key = role, layout(source)
        if key not in self.slots:
            target = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype, device=source.device)
            self.slots[key] = dict(target=target, pointer=target.data_ptr(), epoch=-1, root=None, version=None)
        return self.slots[key]['target']

    def stage(self, role, source, target, jobs):
        if not self.active:
            raise RuntimeError('KV copy outside serialized chunk')
        slot = self.slots[(role, layout(source))]
        if target is not slot['target'] or target.data_ptr() != slot['pointer'] or layout(target) != layout(source):
            raise RuntimeError('shared KV storage changed')
        root, version = root_and_version(source)
        stamp = source.data_ptr(), layout(source)
        unchanged = (version is not None and slot['epoch'] == self.epoch
                     and slot['root'] is root and slot['version'] == version and slot.get('stamp') == stamp)
        if unchanged:
            self.skips += 1
            return
        jobs.append(copy_views(target, source))
        # Strong refs prevent allocator reuse being mistaken for the old source.
        slot.update(epoch=self.epoch, root=root, source=source, version=version, stamp=stamp)
        self.copies += 1

    def report(self):
        return dict(epochs=self.epoch, kv_copies=self.copies, kv_copy_skips=self.skips, buffers=len(self.slots))

    def close(self):
        self.slots.clear()
        self.active = False
