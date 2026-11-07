"""Build and apply the shipped operator recipe in the current compiler cache.

Preparation may compile the small replacement kernels. Graph replay never
compiles, and no historical experiment directory or GPU binary is required.
"""
import ast
import copy
import hashlib
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

from .build import build, recipe, sha, signature
from .fma import PhaseFma
from .fma_layout import FmaLayoutDispatch, remap_triplets
from .layout import SpLayout
from .pins import config_record, matches
from .resident import BytecodeSites


class RuntimePins:
    def __init__(self, emit):
        self.emit = emit
        self.registry = {}

    def bind(self, auto, source, config):
        if Path(auto.filename).resolve() != Path(source).resolve():
            raise RuntimeError('actual kernel does not match the generated source')
        if not auto.compile_results:
            auto.precompile()
        if not auto.compile_results:
            raise RuntimeError('no compiled kernel configurations')
        found = [r for r in auto.compile_results if matches(r.config, config)]
        if len(found) > 1:
            raise RuntimeError('ambiguous operator configuration')
        if not found:
            if auto.fn.fn is None:
                if not callable(getattr(auto, '_reload_kernel', None)):
                    raise RuntimeError('actual kernel source loader unavailable')
                auto.fn = auto._reload_kernel().fn
            selected = copy.deepcopy(auto.compile_results[0].config)
            if set(selected.kwargs) != set(config) - {'num_warps', 'num_stages'}:
                raise RuntimeError('operator configuration ABI changed')
            selected.kwargs.update({k: config[k] for k in selected.kwargs})
            selected.num_warps, selected.num_stages = config['num_warps'], config['num_stages']
            found = [auto._precompile_config(selected)]
            self.emit('operator_configuration_prepared', source=str(source), config=config)
        entry = dict(auto=auto, source=str(source), source_sha=sha(source), target=False,
                     result=found[0], launcher=found[0].make_launcher())
        prior = self.registry.get(id(auto))
        if prior and not matches(prior['result'].config, config):
            raise RuntimeError('conflicting shared kernel defaults')
        self.registry[id(auto)] = entry
        return entry

    def register(self, auto):
        entry = self.registry[id(auto)]
        if entry['auto'] is not auto:
            raise RuntimeError('operator kernel identity changed')
        return entry

    def result_for(self, auto, entry, variant):
        if variant != 'A' or entry['auto'] is not auto:
            raise RuntimeError('unexpected operator configuration selection')
        return entry['result']


def prepare_fusion(owner):
    """Compile the same three unrounded FMA kernels for this phase/device."""
    if not owner.plans:
        return
    import torch
    owner.load_kernels()
    for kind in ('global', 'ffn', 'local'):
        x = torch.ones((2, 5376), device=owner.device, dtype=torch.bfloat16)
        branch = torch.full_like(x, .25)
        ids = torch.zeros(2, device=owner.device, dtype=torch.int64)
        mod = torch.full((owner.index_bound, 6, 5376), .125, device=owner.device, dtype=torch.bfloat16)
        gamma, out = torch.ones(5376, device=owner.device, dtype=torch.bfloat16), torch.empty_like(x)
        if kind == 'global':
            pointers = (x, branch, ids, ids, mod, branch, gamma, out)
        elif kind == 'ffn':
            pointers = (x, ids, ids, mod, branch, branch, gamma, mod, out)
        else:
            pointers = (x, ids, ids, mod, branch, branch, gamma, out)
        binary = owner.kernels[kind][(2,)](*pointers, 0, 2, 5376, XBLOCK=1, R0_BLOCK=2048,
            ROUND_RESIDUAL=False, num_warps=16, num_stages=1, enable_fp_fusion=True)
        torch.cuda.synchronize(owner.device)
        if not bool(torch.isfinite(x).all() and torch.isfinite(out).all()):
            raise FloatingPointError('nonfinite operator kernel preparation')
        owner.binaries[kind + ':False'] = dict(hash=binary.hash,
            ptx_sha256=hashlib.sha256(binary.asm['ptx'].encode()).hexdigest())


class PortableOperators(FmaLayoutDispatch):
    """Lazy phase binding before resident Graph capture, then cached replay."""

    def __init__(self, *, rank, device, cache, emit, control_group):
        self.rank, self.device, self.directory, self.emit = rank, device, Path(cache), emit
        self.control_group = control_group
        self.pins = RuntimePins(emit)
        self.sites, self.modules, self.phases, self.norms, self.prefixes = {}, {}, {}, {}, {}
        self.dispatch_counts = dict(fma_triplets=0, layout_calls=0, passthrough=0)
        self.pending, self.active, self.current_phase, self.request = None, False, None, None
        self.bytecode_sites = BytecodeSites()
        self.prefix_calls = 0
        self.history_modules = set()
        self.binding_error = None

    def bind_request(self, inputs):
        if self.active:
            raise RuntimeError('rebind only between requests')
        layout = inputs.layout('rollout')
        self.request = (int(inputs.prompt.shape[1]), int(layout.condition_indices.numel()), layout.sequence_length)
        for phase in self.phases.values():
            phase['layout'].bind_request(*self.request)

    @contextmanager
    def phase(self, phase):
        prior = self.current_phase
        self.current_phase = phase
        try:
            yield
        except BaseException:
            if self.binding_error is not None:
                raise self.binding_error
            raise
        finally:
            self.current_phase = prior

    def _bind_module(self, module, path):
        if not hasattr(module, 'Runner') or self.current_phase is None:
            return module
        if self.current_phase[0] < 5:
            return self._bind_history(module)
        if id(module) in self.modules:
            return module
        phase = 'refresh' if self.current_phase[1] else 'denoise'
        if phase in self.phases:
            raise RuntimeError('second full-model wrapper for one operator phase')
        original = module.call
        if (not isinstance(original, MethodType) or type(original.__self__) is not module.Runner
                or original.__func__ is not module.Runner.call
                or Path(original.__func__.__code__.co_filename) != Path(path)):
            raise RuntimeError('unexpected generated Runner ownership')
        helpers = {name: Path(value.filename) for name, value in vars(module).items()
                   if name.startswith('triton_') and getattr(value, 'filename', None)}
        bindings = build(path, helpers, phase=phase, rank=self.rank, directory=self.directory)
        fma = PhaseFma(bindings['fma'], self.rank, self.device, self.emit)
        fma.set_rounding(False)
        fma.set_mode('replace')
        layout = SpLayout(bindings['layout'], self.rank, self.device, self.emit)
        layout.set_mode('replace')
        if self.request is None:
            raise RuntimeError('operator request must be bound before compilation')
        layout.bind_request(*self.request)
        for name, record in fma.record['records'].items():
            self.pins.bind(vars(module)[name], record['source'], record['config'])
        for record in layout.records.values():
            self.pins.bind(vars(module)[record['name']], record['source'], record['config'])
        prepare_fusion(fma)
        if self.phases:
            first = next(iter(self.phases.values()))['layout']
            layout.kernels, layout.binaries = first.kernels, copy.deepcopy(first.binaries)
        else:
            layout.prepare()
        self.sites[str(path)] = (fma, fma.plans)
        optimized = original
        qkv = None
        if phase == 'denoise':
            import json
            qkv = json.loads(bindings['qkv'].read_text())['ranks'][self.rank]
            candidate = Path(qkv['candidate'])
            namespace = {}
            exec(compile(candidate.read_text(), str(candidate), 'exec'), vars(module), namespace)
            if set(namespace) != {'call'}:
                raise RuntimeError('scheduled wrapper may define only call')
            optimized = MethodType(namespace['call'], original.__self__)
            self.sites[str(candidate)] = (fma, remap_triplets(Path(path).read_text(), candidate.read_text(), fma.plans))
        for rule in bindings['norms']:
            auto = vars(module)[rule['helper']]
            if rule['first_refresh_call_only']:
                # Prepare a private alternate without changing the shared
                # default used by the subsequent fused reductions.
                private = RuntimePins(self.emit)
                selected = private.bind(auto, rule['source'], rule['config'])
                nodes = sorted((n for n in ast.walk(ast.parse(Path(path).read_text()))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run'
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == rule['helper']), key=lambda n:n.lineno)
                reductions = {p['lines'][1] for p in fma.plans.values() if p['functions'][1] == rule['helper']}
                if not nodes or {n.lineno for n in nodes[1:]} != reductions:
                    raise RuntimeError('refresh prefix/shared reduction classification changed')
                first = nodes[0]
                if len(first.args) not in (4,5) or ast.unparse(first.args[-1]) != '5376':
                    raise RuntimeError('refresh prefix tensor ABI changed')
                self.prefixes[(str(path), first.lineno)] = (auto, selected['launcher'])
            else:
                selected = self.pins.bind(auto, rule['source'], rule['config'])
                self.norms[id(auto)] = selected
        def dispatch(*args, **kwargs):
            if not self.active:
                raise RuntimeError('optimized wrapper executed without its operator owner')
            return optimized(*args, **kwargs)
        if hasattr(original, '_boxed_call'):
            dispatch._boxed_call = original._boxed_call
        module.call = dispatch
        self.modules[id(module)] = (module, original, dispatch)
        self.phases[phase] = dict(fma=fma, layout=layout, qkv=qkv, norms=bindings['norms'])
        self.emit('portable_operator_phase_ready', phase=phase, fusion_sites=len(fma.plans),
            layout_helpers=len(layout.records), qkv_moved_gemms=0 if qkv is None else qkv['moved_gemms'],
            norm_rules=[r['name'] for r in bindings['norms']], historical_build_required=False)
        return module

    def _bind_history(self, module):
        if id(module) in self.history_modules:
            return module
        if self.rank == 5 and not self.current_phase[1]:
            rule, = [r for r in recipe()['norms'] if r['name'] == 'rank_fma_alignment']
            auto = vars(module).get(rule['function'])
            if auto is None or signature(Path(auto.filename).read_text()) != rule['signature']:
                raise RuntimeError('history normalization kernel changed')
            self.norms[id(auto)] = self.pins.bind(auto, auto.filename, rule['config'])
            self.emit('portable_history_normalization_ready', phase=self.current_phase,
                      config=rule['config'])
        self.history_modules.add(id(module))
        return module

    def _run(self, auto, args, kwargs, frame, original):
        frame = self.bytecode_sites.bind(frame)
        prefix = self.prefixes.get((frame.f_code.co_filename, frame.f_lineno))
        selected = prefix or (self.norms.get(id(auto)) and (auto, self.norms[id(auto)]['launcher']))
        if selected:
            if selected[0] is not auto or self.pending is not None:
                raise RuntimeError('normalization owner/order changed')
            if prefix:
                import torch
                x, y = args[:2]
                if (set(kwargs) != {'stream'} or len(args) not in (4, 5)
                        or kwargs['stream'] != torch.cuda.current_stream(self.device).cuda_stream
                        or x.device != self.device or y.device != self.device
                        or x.dtype != torch.bfloat16 or y.dtype != torch.float32
                        or not x.is_contiguous() or not y.is_contiguous()
                        or type(args[-2]) is not int or args[-2] <= 0 or args[-1] != 5376
                        or y.numel() != args[-2] or x.numel() < args[-2] * 5376):
                    raise RuntimeError('refresh prefix tensor/stream contract changed')
            previous = auto.launchers
            auto.launchers = [selected[1]]
            try:
                result = original(auto, *args, **kwargs)
            finally:
                auto.launchers = previous
            if prefix:
                self.prefix_calls += 1
            return result
        pair = self.sites.get(frame.f_code.co_filename)
        if pair:
            self.layout = self.phases[pair[0].doc['phase']]['layout']
        elif self.pending is None:
            return original(auto, *args, **kwargs)
        return super().dispatch(auto, args, kwargs, frame, self.pins, original)

    @contextmanager
    def installed(self):
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        if self.active or self.pending is not None:
            raise RuntimeError('non-reentrant operator scope required')
        descriptor = PyCodeCache.__dict__['load_by_key_path']
        load = PyCodeCache.load_by_key_path
        original = CachingAutotuner.run
        def load_bound(cls, key, path, *args, **kwargs):
            module = load(key, path, *args, **kwargs)
            if (not hasattr(module, 'Runner') or self.current_phase is None
                    or id(module) in self.modules or id(module) in self.history_modules):
                return module
            error = None
            try:
                self._bind_module(module, path)
            except BaseException as caught:
                error = caught
                self.emit('portable_operator_binding_failed', phase=self.current_phase,
                          error=repr(caught), traceback=traceback.format_exc()[-12000:])
            # Nobody enters the model's NCCL work until every rank has bound
            # this wrapper, including a peer that rejected an operator recipe.
            import torch.distributed as dist
            rows = [None] * 8
            dist.all_gather_object(rows, dict(rank=self.rank, phase=self.current_phase,
                error=None if error is None else repr(error)), group=self.control_group)
            failures = [r for r in rows if r['error'] is not None]
            if failures:
                self.binding_error = error or RuntimeError('peer operator binding failed: ' + str(failures))
                raise self.binding_error
            if {r['rank'] for r in rows} != set(range(8)) or any(tuple(r['phase']) != self.current_phase for r in rows):
                raise RuntimeError('operator preparation rank/phase mismatch')
            return module
        installed = classmethod(load_bound)
        def run(auto, *args, **kwargs):
            return self._run(auto, args, kwargs, sys._getframe(1), original)
        PyCodeCache.load_by_key_path, CachingAutotuner.run = installed, run
        self.active = True
        try:
            for module, before, after in self.modules.values():
                if module.call is not before:
                    raise RuntimeError('generated module changed outside operator scope')
                module.call = after
            yield self
            if self.pending is not None:
                raise RuntimeError('unfinished fused operation')
        finally:
            changed = PyCodeCache.__dict__['load_by_key_path'] is not installed or CachingAutotuner.run is not run
            PyCodeCache.load_by_key_path, CachingAutotuner.run = descriptor, original
            for module, before, after in self.modules.values():
                changed |= module.call is not after
                module.call = before
            self.active = False
            if changed:
                raise RuntimeError('operator hook ownership changed')

    def report(self):
        def fusion_report(owner):
            row = owner.report()
            row.pop('denoise_actual_binaries_reused', None)
            row.pop('new_targeted_variants', None)
            row.update(prepared_from_shipped_kernels=True, prepared_variants=len(owner.binaries))
            return row
        return dict(phases={name:dict(fma=fusion_report(part['fma']), layout=part['layout'].report(),
            qkv_moved_gemms=0 if part['qkv'] is None else part['qkv']['moved_gemms'],
            norm_rules=[r['name'] for r in part['norms']]) for name, part in self.phases.items()},
            calls=dict(self.dispatch_counts), prefix_calls=self.prefix_calls,
            historical_build_required=False, history_wrappers=len(self.history_modules), active=self.active)
