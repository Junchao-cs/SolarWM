"""QKV overlap with scoped installation and restoration."""

import hashlib
import json
from pathlib import Path
from types import MethodType


class QkvOverlapRuntime:
    def __init__(self, manifest, rank, emit):
        self.manifest_path = Path(manifest)
        self.manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        document = json.loads(self.manifest_path.read_text())
        self.record = document['ranks'][rank]
        assert self.record['rank'] == rank
        assert self.record['layers'] == 50 and self.record['moved_gemms'] == 300
        self.candidate = Path(self.record['candidate'])
        assert hashlib.sha256(self.candidate.read_bytes()).hexdigest() == self.record['candidate_function_sha256']
        self.mode = 'baseline'
        self.emit = emit
        self.modules = []
        self.counts = dict(baseline=0, overlap=0)

    def set_mode(self, mode):
        if mode not in self.counts:
            raise ValueError(mode)
        self.mode = mode

    def install_module(self, module, path):
        path = Path(path)
        if path.name != Path(self.record['original']).name:
            return module
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.record['original_source_sha256']:
            raise RuntimeError('observed denoise wrapper source changed')
        if any(m is module for m, _, _ in self.modules):
            return module
        original = module.call
        if not isinstance(original, MethodType) or type(original.__self__) is not module.Runner or original.__func__ is not module.Runner.call:
            raise RuntimeError('unexpected generated Runner binding')
        namespace = {}
        source = self.candidate.read_text()
        if hashlib.sha256(source.encode()).hexdigest() != self.record['candidate_function_sha256']:
            raise RuntimeError('candidate function changed')
        # Defines only Runner.call in a separate locals dictionary. Global
        # kernel objects are the original, actual cache-loaded module objects.
        exec(compile(source, str(self.candidate), 'exec'), module.__dict__, namespace)
        if set(namespace) != {'call'}:
            raise RuntimeError('candidate must define only one call function')
        optimized = MethodType(namespace['call'], original.__self__)
        def dispatch(*args, **kwargs):
            self.counts[self.mode] += 1
            return (original if self.mode == 'baseline' else optimized)(*args, **kwargs)
        if hasattr(original, '_boxed_call'):
            dispatch._boxed_call = original._boxed_call
        module.call = dispatch
        self.modules.append((module, original, dispatch))
        self.emit('qkv_overlap_wrapper_bound', original=str(path), manifest_sha256=self.manifest_sha,
                  candidate=str(self.candidate), candidate_sha256=self.record['candidate_function_sha256'],
                  moved_gemms=300, new_storage_sites=100, same_kernel_globals=True)
        return module

    def __enter__(self):
        from torch._inductor.codecache import PyCodeCache
        self.cache = PyCodeCache
        self.original_descriptor = PyCodeCache.__dict__['load_by_key_path']
        original = PyCodeCache.load_by_key_path
        def load(cls, key, path, *args, **kwargs):
            module = original(key, path, *args, **kwargs)
            return self.install_module(module, path)
        self.installed_descriptor = classmethod(load)
        PyCodeCache.load_by_key_path = self.installed_descriptor
        return self

    def report(self):
        return dict(manifest_sha256=self.manifest_sha, bound_modules=len(self.modules),
                    calls=dict(self.counts), mode=self.mode, new_gpu_kernel_codegen=0,
                    candidate_is_runtime_validated=False)

    def __exit__(self, typ, error, tb):
        changed = self.cache.__dict__['load_by_key_path'] is not self.installed_descriptor
        self.cache.load_by_key_path = self.original_descriptor
        for module, original, dispatch in reversed(self.modules):
            changed = changed or module.call is not dispatch
            module.call = original
        if changed and error is None:
            raise RuntimeError('wrapper dispatch hook changed before restoration')


class ScopedQkvDispatch(QkvOverlapRuntime):
    """Reinstall the same owned module dispatch at each DiT-only scope."""
    def __enter__(self):
        for module,original,dispatch in self.modules:
            if module.call is not original:
                raise RuntimeError('resident QKV module changed outside its scope')
        super().__enter__()
        for module,original,dispatch in self.modules:
            module.call=dispatch
        return self
