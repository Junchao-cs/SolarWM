"""Restore source-bound Inductor/Triton launch choices; reject binary cache misses."""

from ..compatibility import pinned_path, kernel_source
import copy
import hashlib
import json
from pathlib import Path


TARGET = kernel_source('matmul')
REL = pinned_path("matmul_configuration")


def config_record(config):
    return dict(config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages)


def matches(config, record):
    actual = config_record(config)
    ignored = {'configs_hash', 'found_by_coordesc', 'time_taken_ms', 'triton_cache_hash'}
    return actual == {k: v for k, v in record.items() if k not in ignored}


def choose(results, record):
    found = [r for r in results if matches(r.config, record)]
    if len(found) != 1:
        raise RuntimeError('required pinned config is missing or ambiguous')
    return found[0]


class PinnedRuntime:
    def __init__(self, rank, seed, alternative, emit):
        self.rank, self.seed, self.alternative, self.emit = rank, Path(seed), Path(alternative), emit
        self.variant = 'A'
        self.registry = {}
        self.calls = {}
        self.manifest = {}
        self.originals = []

    def _patch(self, obj, name, value):
        self.originals.append((obj, name, getattr(obj, name), value))
        setattr(obj, name, value)

    def __enter__(self):
        self.originals = []
        from torch._inductor.runtime import triton_heuristics as th
        from torch._inductor.runtime.autotune_cache import AutotuneCache
        import triton.compiler.compiler as tc
        self.th, self.cache_class = th, AutotuneCache
        original_make_backend = tc.make_backend

        def cached_backend(target):
            backend = original_make_backend(target)
            def refuse_stages(*args, **kwargs):
                raise RuntimeError('pinned runtime refuses Triton GPU codegen/cache miss')
            # Triton calls add_stages only AFTER failing its actual binary lookup.
            backend.add_stages = refuse_stages
            return backend

        def recheck(autotuner, reload_kernel_from_src):
            self.register(autotuner, reload_kernel_from_src)

        original_precompile = th.CachingAutotuner.precompile
        def precompile(autotuner, *args, **kwargs):
            if kwargs.get('warm_cache_only'):
                raise RuntimeError('unexpected compile-worker path in pinned cache-only run')
            self.register(autotuner, kwargs.get('reload_kernel'))
            return original_precompile(autotuner, *args, **kwargs)

        original_run = th.CachingAutotuner.run
        def run(autotuner, *args, **kwargs):
            if not autotuner.launchers:
                autotuner.precompile()
            entry = self.register(autotuner)
            variant = self.variant if entry['target'] else 'A'
            if variant not in entry['launchers']:
                result = self.result_for(autotuner, entry, variant)
                entry['launchers'][variant] = result.make_launcher()
            launcher = entry['launchers'][variant]
            expected = entry['records'][variant]
            if not matches(launcher.config, expected):
                raise RuntimeError('runtime launcher config diverged from pin')
            binary_hash = getattr(launcher, 'cache_hash', None)
            if expected.get('triton_cache_hash') and binary_hash != expected['triton_cache_hash']:
                raise RuntimeError('runtime binary differs from pinned artifact')
            autotuner.launchers = [launcher]
            key = entry['name'] + ':' + variant
            self.calls[key] = self.calls.get(key, 0) + 1
            if key not in self.manifest:
                self.manifest[key] = dict(source=entry['source'], source_sha=entry['source_sha'],
                    config=config_record(launcher.config), binary_hash=binary_hash,
                    best_config=entry['paths'].get(variant), target=entry['target'])
                self.emit('pinned_kernel_first_launch', **self.manifest[key])
            return original_run(autotuner, *args, **kwargs)

        def denied(autotuner, *args, **kwargs):
            raise RuntimeError('runtime autotuning forbidden by pinned manifest')

        self._patch(tc, 'make_backend', cached_backend)
        self._patch(th.CachingAutotuner, 'recheck_autotune_cache', recheck)
        self._patch(th.CachingAutotuner, 'precompile', precompile)
        self._patch(th.CachingAutotuner, '_dynamic_scale_rblock', lambda autotuner: None)
        self._patch(th.CachingAutotuner, 'autotune_to_one_config', denied)
        self._patch(th.CachingAutotuner, 'coordinate_descent_tuning', denied)
        self._patch(th.CachingAutotuner, 'run', run)
        return self

    def register(self, auto, reload_kernel=None):
        if id(auto) in self.registry:
            entry = self.registry[id(auto)]
            if reload_kernel is not None:
                entry['reload'] = reload_kernel
            return entry
        if not auto.filename or not auto.compile_results:
            raise RuntimeError('pin requires an actual cached compiled autotuner')
        name = Path(auto.filename).name
        source = self.seed / f'inductor-rank{self.rank}' / name[1:3] / name
        if not source.is_file():
            raise RuntimeError('unbound cached kernel source: '+name)
        from torch._inductor.codecache import torch_key
        key = hashlib.sha256(self.cache_class._prepare_key(str(source)).encode()+torch_key()).hexdigest()
        best = source.parent / (key+'.best_config')
        records, paths = {}, {}
        if best.is_file():
            records['A'] = json.loads(best.read_text())
            paths['A'] = str(best)
        elif len(auto.compile_results) == 1:
            records['A'] = config_record(auto.compile_results[0].config)
        else:
            raise RuntimeError('multiple cached configurations without a frozen selection')
        target = self.rank == 5 and name == TARGET
        if target:
            alternate = self.alternative / 'inductor-rank5' / REL
            records['B'] = json.loads(alternate.read_text())
            paths['B'] = str(alternate)
            assert records['A']['R0_BLOCK'] == 2048 and records['B']['R0_BLOCK'] == 4096
        entry = dict(name=name, source=str(source), source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
            target=target, records=records, paths=paths, results=list(auto.compile_results),
            launchers={}, reload=reload_kernel, auto=auto)
        self.registry[id(auto)] = entry
        actual_configs = [config_record(r.config) for r in entry['results']]
        self.emit('pin_registered', kernel=name, target=target, cached_configs=actual_configs,
            prior_configs_hash=records['A'].get('configs_hash'),
            reloaded_configs_hash=self.th.hash_configs([r.config for r in entry['results']]))
        auto.compile_results = [self.result_for(auto, entry, 'A')]
        auto.launchers = []
        auto.save_cache_hook = None
        return entry

    def result_for(self, auto, entry, variant):
        record = entry['records'][variant]
        candidates = [r for r in entry['results'] if matches(r.config, record)]
        if len(candidates) > 1:
            raise RuntimeError('ambiguous cached pinned result')
        if candidates:
            return candidates[0]
        # Dynamic R-block variants may not have been serialized into the FX
        # bundle. Reload ONLY an already compiled binary, under the miss guard.
        if auto.fn.fn is None:
            if entry['reload'] is None:
                raise RuntimeError('no source loader for required existing binary')
            auto.fn = entry['reload']().fn
        config = copy.deepcopy(entry['results'][0].config)
        for key in config.kwargs:
            config.kwargs[key] = record[key]
        config.num_warps, config.num_stages = record['num_warps'], record['num_stages']
        assert matches(config, record)
        result = auto._precompile_config(config)
        entry['results'].append(result)
        self.emit('existing_binary_rehydrated', kernel=entry['name'], variant=variant, config=config_record(config))
        return result

    def set_variant(self, variant):
        if variant not in ('A','B'):
            raise ValueError(variant)
        self.variant = variant

    def report(self):
        return dict(manifest=self.manifest, calls=self.calls, variant=self.variant,
            runtime_autotuning_allowed=False, cold_gpu_codegen_allowed=False)

    def __exit__(self, typ, error, tb):
        altered = []
        for obj, name, prior, installed in reversed(self.originals):
            if getattr(obj, name) is not installed:
                altered.append(name)
            setattr(obj, name, prior)
        if altered and error is None:
            raise RuntimeError('pin hooks changed: '+str(altered))
