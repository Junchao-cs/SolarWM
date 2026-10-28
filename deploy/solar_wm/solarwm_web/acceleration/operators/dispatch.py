"""Bind rebuilt wrappers to the same source-checked operator dispatch."""

import ast
import hashlib
from pathlib import Path
from types import MethodType
from .qkv import ScopedQkvDispatch
from .resident import ResidentFmaLayoutDispatch
from .resident import GraphPrefixCallsite


def fingerprint(source):
    return ast.dump(ast.parse(source), include_attributes=True)


class WrapperBindings:
    def __init__(self, cache, records):
        self.cache = Path(cache).resolve(strict=True)
        self.records = {}
        self.aliases = {}
        for phase, (path, expected) in records.items():
            path = Path(path)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('original wrapper hash changed')
            self.records[phase] = dict(path=str(path), sha256=expected, ast=fingerprint(data))

    def classify(self, path):
        path = Path(path)
        resolved = path.resolve(strict=True)
        if resolved != path or not resolved.is_relative_to(self.cache):
            raise RuntimeError('rebuilt wrapper escaped owned cache')
        if path.suffix != '.py' or path.parent.parent != self.cache or path.parent.name != path.name[1:3]:
            raise RuntimeError('unexpected generated wrapper path')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if str(path) in self.aliases:
            result = self.aliases[str(path)]
            if digest != result['sha256']:
                raise RuntimeError('bound wrapper bytes changed')
            return result
        tree = fingerprint(data)
        matches = [phase for phase, record in self.records.items() if tree == record['ast']]
        if len(matches) != 1:
            raise RuntimeError('rebuilt steady wrapper is not source-and-position identical')
        phase = matches[0]
        result = dict(phase=phase, path=str(path), sha256=digest,
                      original=self.records[phase]['path'], original_sha256=self.records[phase]['sha256'],
                      whole_module_ast_and_positions_equal=True)
        self.aliases[str(path)] = result
        return result

    def verify_bound(self):
        for path, record in self.aliases.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != record['sha256']:
                raise RuntimeError('bound wrapper bytes changed between scopes')


class BoundPrefix(GraphPrefixCallsite):
    def __init__(self, *args):
        super().__init__(*args)
        self.source_aliases = {self.wrapper}

    def matches_site(self, frame):
        return (self.binding is not None and frame.f_code.co_filename in self.source_aliases
                and frame.f_lineno == self.binding['line'])


class BoundCombined(ResidentFmaLayoutDispatch):
    def __init__(self, denoise, refresh, layout, qkv_record, cache):
        super().__init__(denoise, refresh, layout, qkv_record, cache)
        if self.prefix.calls or self.prefix.launcher is not None:
            raise RuntimeError('binding must precede first operator use')
        self.prefix = BoundPrefix(refresh, cache, refresh.emit)

    def bind_alias(self, record):
        owner = self.denoise if record['phase'] == 'denoise' else self.refresh
        pair = (owner, owner.plans)
        prior = self.sites.get(record['path'])
        if prior is not None and (prior[0] is not owner or prior[1] is not owner.plans):
            raise RuntimeError('wrapper alias ownership collision')
        self.sites[record['path']] = pair
        if record['phase'] == 'refresh':
            self.prefix.source_aliases.add(record['path'])


class BoundQkv(ScopedQkvDispatch):
    def __init__(self, manifest, rank, emit, bindings, combined):
        super().__init__(manifest, rank, emit)
        self.bindings, self.combined = bindings, combined

    def __enter__(self):
        self.bindings.verify_bound()
        return super().__enter__()

    def install_module(self, module, path):
        # Kernel helper modules do not contain a generated Runner.call. They
        # retain their existing independent source/binary pin checks.
        if not hasattr(module, 'Runner'):
            return super().install_module(module, path)
        original = module.call
        if (not isinstance(original, MethodType) or type(original.__self__) is not module.Runner
                or original.__func__ is not module.Runner.call):
            if any(m is module for m, _, _ in self.modules):
                self.bindings.classify(path)
                return module
            raise RuntimeError('actual generated Runner binding differs')
        if Path(original.__func__.__code__.co_filename) != Path(path):
            raise RuntimeError('actual wrapper code filename differs')
        record = self.bindings.classify(path)
        self.combined.bind_alias(record)
        self.emit('rebuilt_operator_wrapper_bound', **record,
                  preserved_kernel_globals=True, new_gpu_codegen=0)
        if record['phase'] == 'refresh':
            return module
        # The existing installer still verifies the actual source hash and
        # installs exactly the frozen candidate into this module's globals.
        previous = self.record
        self.record = dict(previous, original=record['path'], original_source_sha256=record['sha256'])
        try:
            return super().install_module(module, path)
        finally:
            self.record = previous

    def report(self):
        report = super().report()
        report['rebuilt_wrapper_aliases'] = list(self.bindings.aliases.values())
        return report


def install(optim, *, cache, triton_cache, rank, emit):
    if optim.active or optim.qkv.modules or optim.combined.active:
        raise RuntimeError('install only before first steady forward')
    records = {name: (owner.record['wrapper'], owner.record['wrapper_sha256'])
               for name, owner in (('denoise', optim.denoise), ('refresh', optim.refresh))}
    bindings = WrapperBindings(cache, records)
    combined = BoundCombined(optim.denoise, optim.refresh, optim.layout, optim.qkv.record, triton_cache)
    qkv = BoundQkv(optim.qkv.manifest_path, rank, emit, bindings, combined)
    qkv.set_mode('overlap')
    optim.qkv, optim.combined = qkv, combined
    return bindings
