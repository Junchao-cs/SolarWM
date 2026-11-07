"""Cache named traversal, not values or mutation checks. Cooperative writers only.

Every observation still verifies the complete registered module/parameter/buffer
topology, all parameter/buffer stamps, training flags and LoRA metadata. Changes
to registry structure fail closed until reset, rather than trusting stale refs.
No CUDA, payload copies, model edits, relaxed precision or permanent KV claim.
"""
from deploy.solar_wm.backends.memory.acceleration.cache.dependency_base import Dependencies as Parent
from deploy.solar_wm.backends.memory.acceleration.cache.prefix import stamp

class Dependencies(Parent):

    def __init__(self, base, inputs, bank):
        super().__init__(base, inputs, bank)
        self.registry = None
        self.model_rows = None
        self.modules = None
        self.builds = 0
        self.observations = 0

    def _build(self):
        modules = tuple(self.base.named_modules())
        model_rows = tuple(self.base.named_parameters()) + tuple((('buffer:' + n, t) for (n, t) in self.base.named_buffers()))
        registry = []
        for (_, module) in modules:
            for attribute in ('_modules', '_parameters', '_buffers'):
                mapping = getattr(module, attribute)
                if not isinstance(mapping, dict):
                    raise RuntimeError('unsupported model registry')
                registry.append((module, attribute, mapping, tuple(mapping.items())))
        self.modules = modules
        self.model_rows = model_rows
        self.registry = tuple(registry)
        self.builds += 1

    def observe(self, source):
        if self.registry is None:
            self._build()
        for (module, attribute, mapping, items) in self.registry:
            actual = getattr(module, attribute)
            if actual is not mapping or len(actual) != len(items) or any((key != old_key or value is not old_value for ((key, value), (old_key, old_value)) in zip(actual.items(), items))):
                raise RuntimeError('model registry changed; reset cache owner')
        dynamic = [('input:' + n, getattr(self.inputs, n)) for n in ('prompt', 'anchor_rows', 'audio_rows', 'audio_timestep')]
        dynamic.append(('layout.position_ids', source.layout.position_ids))
        dynamic.extend((('bank:' + str(i), self.bank.chunks[int(i) // 5]) for i in source.bank_indices))
        rows = tuple(((n, t, stamp(t)) for (n, t) in self.model_rows)) + tuple(((n, t, stamp(t)) for (n, t) in dynamic))
        metadata = []
        for (name, module) in self.modules:
            if module.training:
                raise RuntimeError('training module inside inference cache')
            if hasattr(module, 'scaling'):
                scaling = module.scaling
                if not isinstance(scaling, dict) or any((type(k) is not str or type(v) not in (int, float) for (k, v) in scaling.items())):
                    raise RuntimeError('unknown adapter scale contract')
                metadata.append((name, tuple(sorted(scaling.items())), tuple(module.active_adapters), bool(module.disable_adapters), tuple(module.merged_adapters)))
        self.observations += 1
        return (rows, tuple(metadata))

    def reset(self):
        super().reset()
        self.registry = self.model_rows = self.modules = None

    def report(self):
        return dict(builds=self.builds, observations=self.observations, cached_naming_only=True, parameter_versions_checked_every_observation=True, arbitrary_alias_payload_authenticated=False)
