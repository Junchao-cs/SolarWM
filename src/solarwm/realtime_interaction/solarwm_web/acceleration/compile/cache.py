"""Rebuild compiler metadata using source-bound cached kernels, without autotuning."""

from contextlib import contextmanager
from pathlib import Path
import hashlib
from ..operators.pins import PinnedRuntime


class MetadataRebuildPins(PinnedRuntime):
    def __enter__(self):
        from torch._inductor.runtime import triton_heuristics as th
        native_precompile=th.CachingAutotuner.precompile
        super().__enter__()
        parent_precompile=th.CachingAutotuner.precompile
        def precompile(autotuner,*args,**kwargs):
            if autotuner.compile_results:
                return parent_precompile(autotuner,*args,**kwargs)
            if kwargs.get('warm_cache_only'):
                raise RuntimeError('metadata rebuild must remain in the owning process')
            name=Path(autotuner.filename).name
            bound=self.seed/f'inductor-rank{self.rank}'/name[1:3]/name
            actual=Path(autotuner.filename)
            if not bound.is_file() or hashlib.sha256(bound.read_bytes()).digest()!=hashlib.sha256(actual.read_bytes()).digest():
                raise RuntimeError('metadata rebuild generated an unbound kernel source: '+name)
            # Actual binary cache hits only: inherited make_backend refuses
            # add_stages on a Triton cache miss. No autotuning is permitted.
            result=native_precompile(autotuner,*args,**kwargs)
            self.register(autotuner,kwargs.get('reload_kernel'))
            return result
        self._patch(th.CachingAutotuner,'precompile',precompile)
        return self

class MetadataRebuildBudget:
    def __init__(self,maximum):self.maximum=maximum;self.calls=0
    @contextmanager
    def scope(self,module,emit):
        import torch._inductor.config as config
        if config.unsafe_skip_cache_dynamic_shape_guards:
            raise RuntimeError('unsafe cache guard bypass prohibited')
        original=module.fx_codegen_and_compile
        def rebuild(*args,**kwargs):
            if self.calls>=self.maximum:raise RuntimeError('metadata rebuild budget exhausted')
            self.calls+=1
            emit('fresh_cache_metadata_lowering',ordinal=self.calls,maximum=self.maximum)
            return original(*args,**kwargs)
        module.fx_codegen_and_compile=rebuild
        try:
            with config.patch(compile_threads=1):yield
        finally:
            altered=module.fx_codegen_and_compile is not rebuild
            module.fx_codegen_and_compile=original
            if altered:raise RuntimeError('metadata rebuild ownership changed')
