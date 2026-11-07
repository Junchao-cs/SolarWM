from deploy.solar_wm.backends.memory.acceleration.cache.prefix import stamp

class Dependencies:

    def __init__(self, base, inputs, bank):
        (self.base, self.inputs, self.bank) = (base, inputs, bank)
        self.saved = None

    def observe(self, source):
        tensors = list(self.base.named_parameters()) + [('buffer:' + n, t) for (n, t) in self.base.named_buffers()]
        tensors += [('input:' + n, getattr(self.inputs, n)) for n in ('prompt', 'anchor_rows', 'audio_rows', 'audio_timestep')]
        tensors += [('layout.position_ids', source.layout.position_ids)]
        tensors += [('bank:' + str(i), self.bank.chunks[int(i) // 5]) for i in source.bank_indices]
        rows = tuple(((n, t, stamp(t)) for (n, t) in tensors))
        metadata = []
        for (n, module) in self.base.named_modules():
            if module.training:
                raise RuntimeError('training module inside inference cache')
            if hasattr(module, 'scaling'):
                scaling = module.scaling
                if not isinstance(scaling, dict) or any((type(k) is not str or type(v) not in (int, float) for (k, v) in scaling.items())):
                    raise RuntimeError('unknown adapter scale contract')
                metadata.append((n, tuple(sorted(scaling.items())), tuple(module.active_adapters), bool(module.disable_adapters), tuple(module.merged_adapters)))
        return (rows, tuple(metadata))

    def check(self, source, *, fill):
        current = self.observe(source)
        if fill:
            self.saved = current
            return
        if self.saved is None:
            raise RuntimeError('dependency reuse without fill')
        (old, metadata) = self.saved
        (rows, new_metadata) = current
        if metadata != new_metadata or len(rows) != len(old) or any((n != m or a is not b or sa != sb for ((n, a, sa), (m, b, sb)) in zip(rows, old))):
            raise RuntimeError('model/adapter/condition/bank/layout dependency changed')

    def reset(self):
        self.saved = None
