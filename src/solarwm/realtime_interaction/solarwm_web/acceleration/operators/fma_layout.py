"""One source-bound launch hook for FMA normalization plus SP layout copies.

The original and QKV-reordered denoise wrappers share actual kernel globals.
Their line numbers differ. Bind both exact sources, pass the real caller frame
once, and never nest the two old frame-sensitive CachingAutotuner.run hooks.
This source is not a claim of combined CUDA Graph or performance acceptance."""

import ast
import copy
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch_calls(source):
    tree = ast.parse(source)
    return sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == 'run'
                   and isinstance(n.func.value, ast.Name)), key=lambda n: n.lineno)


def remap_triplets(original_source, candidate_source, plans):
    """Exact run-call AST mapping, not a function-name-only match."""
    original = {n.lineno: n for n in launch_calls(original_source)}
    candidate = launch_calls(candidate_source)
    dump = lambda n: ast.dump(n, include_attributes=False)
    fingerprints = [dump(n) for n in candidate]
    triples = {}
    for i in range(len(candidate)-2):
        triples.setdefault(tuple(fingerprints[i:i+3]), []).append(i)
    result = {}
    used = set()
    for key, plan in plans.items():
        if int(key) != plan['lines'][0] or len(plan['lines']) != 3:
            raise RuntimeError('invalid original triplet')
        nodes = [original[line] for line in plan['lines']]
        if [n.func.value.id for n in nodes] != plan['functions']:
            raise RuntimeError('original triplet names changed')
        wanted = [dump(n) for n in nodes]
        positions = triples.get(tuple(wanted), [])
        if len(positions) != 1:
            raise RuntimeError('candidate triplet missing, reordered, or ambiguous')
        lines = [n.lineno for n in candidate[positions[0]:positions[0]+3]]
        if used.intersection(lines):
            raise RuntimeError('candidate triplets overlap')
        used.update(lines)
        updated = copy.deepcopy(plan)
        updated['lines'] = lines
        result[lines[0]] = updated
    return result


def scheduled_sites(fma, qkv_record):
    original = Path(qkv_record['original'])
    candidate = Path(qkv_record['candidate'])
    if (sha(original) != qkv_record['original_source_sha256']
            or sha(candidate) != qkv_record['candidate_function_sha256']
            or sha(original) != fma.record['wrapper_sha256']):
        raise RuntimeError('QKV/FMA wrapper source binding mismatch')
    if fma.index_bound != 12 or qkv_record['rank'] != fma.rank:
        raise RuntimeError('scheduled mapping is denoise/rank specific')
    return str(candidate), remap_triplets(original.read_text(), candidate.read_text(), fma.plans)


def restore_refresh_binaries(fma, prior_result):
    """Only reuse the actual successful phase-9 FMA variants, never prepare."""
    if fma.index_bound != 9 or fma.binaries:
        raise RuntimeError('single-use refresh binary restore required')
    doc = json.loads(Path(prior_result).read_text())
    if doc['status'] != 'SHORT_FMA_REMAINING_COMPLETE_NOT_ACCEPTANCE':
        raise RuntimeError('completed actual remaining-step evidence required')
    row = doc['ranks'][fma.rank]['phase_reports']['refresh']
    if (row['index_bound'] != 9 or not row['explicit_square_accumulation_fma']
            or row['replacement_round_residual'] is not False
            or row['planned_triplets'] != len(fma.plans)):
        raise RuntimeError('refresh phase semantics changed')
    expected = {kind+':False' for kind in ('global', 'ffn', 'local')} if fma.plans else set()
    if set(row['binaries']) != expected:
        raise RuntimeError('actual refresh binary set mismatch')
    if fma.plans:
        fma.load_kernels()
    fma.binaries = copy.deepcopy(row['binaries'])


class FmaLayoutDispatch:
    def __init__(self, denoise, refresh, layout, qkv_record):
        if denoise.index_bound != 12 or refresh.index_bound != 9:
            raise RuntimeError('distinct actual denoise/refresh phases required')
        if len({denoise.rank, refresh.rank, layout.rank, qkv_record['rank']}) != 1:
            raise RuntimeError('combined rank mismatch')
        self.denoise, self.refresh, self.layout = denoise, refresh, layout
        candidate, plans = scheduled_sites(denoise, qkv_record)
        self.sites = {
            denoise.record['wrapper']: (denoise, denoise.plans),
            refresh.record['wrapper']: (refresh, refresh.plans),
            candidate: (denoise, plans),
        }
        if len(self.sites) != 3:
            raise RuntimeError('three distinct source-bound wrapper sites required')
        for fma in (denoise, refresh):
            if sha(fma.record['wrapper']) != fma.record['wrapper_sha256']:
                raise RuntimeError('actual cache wrapper source changed')
        self.pending = None
        self.dispatch_counts = dict(fma_triplets=0, layout_calls=0, passthrough=0)
        self.active = False

    def _layout(self, auto, args, kwargs, frame, pins):
        owner = self.layout
        key = Path(getattr(auto, 'filename', '') or '').name
        record = owner.records[key]
        if frame.f_globals.get(record['name']) is not auto:
            raise RuntimeError('layout helper is not the actual wrapper global')
        if set(kwargs) != {'stream'}:
            raise RuntimeError('layout stream keyword mismatch')
        if id(auto) not in owner.checked:
            if not auto.launchers:
                auto.precompile()
            bound = pins.register(auto)
            selected = pins.result_for(auto, bound, 'A')
            config = dict(selected.config.kwargs, num_warps=selected.config.num_warps,
                          num_stages=selected.config.num_stages)
            if bound['source_sha'] != record['source_sha256'] or bound['target'] or config != record['config']:
                raise RuntimeError('layout pinned source/config mismatch')
            owner.checked[id(auto)] = auto
        length, n, count = owner._check(record, args, kwargs['stream'])
        x, y = args[:2]
        before = x._version
        owner._launch(record['kind'], x, y, n, count)
        if x._version != before:
            raise RuntimeError('layout source version changed')
        owner.calls['replace'] += 1
        self.dispatch_counts['layout_calls'] += 1
        return None

    def dispatch(self, auto, args, kwargs, frame, pins, original):
        # Called with the wrapper frame, not with the frame of another hook.
        filename, line = frame.f_code.co_filename, frame.f_lineno
        if self.pending is not None:
            item = self.pending
            owner, plan, stage = item['owner'], item['plan'], item['stage']
            if (filename != item['wrapper'] or line != plan['lines'][stage]
                    or kwargs != item['kwargs'] or owner.mode != 'replace'):
                raise RuntimeError('combined FMA triplet site/order/stream changed')
            owner._bind_auto(auto, plan['functions'][stage], frame, pins)
            if stage == 1:
                item['reduction'], item['stage'] = args, 2
                return None
            kind, rows, offset = owner._check_inputs(item['residual'], item['reduction'], args, plan, kwargs['stream'])
            owner._launch(item['residual'], args, kind, rows, offset, False, args[6])
            owner.calls['replace'] += 1
            self.dispatch_counts['fma_triplets'] += 1
            self.pending = None
            return None
        selected = self.sites.get(filename)
        if selected is not None:
            owner, plans = selected
            plan = plans.get(line)
            if owner.mode == 'replace' and plan is not None:
                if owner.rounded is not False or set(kwargs) != {'stream'}:
                    raise RuntimeError('explicit unrounded FMA/owned stream required')
                owner._bind_auto(auto, plan['functions'][0], frame, pins)
                self.pending = dict(owner=owner, plan=plan, stage=1, wrapper=filename,
                                    residual=args, kwargs=dict(kwargs))
                return None
        key = Path(getattr(auto, 'filename', '') or '').name
        if self.layout.mode == 'replace' and key in self.layout.records:
            if selected is None:
                raise RuntimeError('layout helper executed outside source-bound wrappers')
            return self._layout(auto, args, kwargs, frame, pins)
        self.dispatch_counts['passthrough'] += 1
        return original(auto, *args, **kwargs)

    @contextmanager
    def installed(self, pins):
        from torch._inductor.runtime import triton_heuristics as th
        if self.active or self.pending is not None:
            raise RuntimeError('single non-reentrant combined launch owner required')
        if any(f.mode not in ('off', 'replace') or f.pending is not None for f in (self.denoise, self.refresh)):
            raise RuntimeError('combined path does not permit shadow or nested hooks')
        if self.layout.mode not in ('off', 'replace'):
            raise RuntimeError('combined layout only supports off/replace')
        original = th.CachingAutotuner.run
        def run(auto, *args, **kwargs):
            return self.dispatch(auto, args, kwargs, sys._getframe(1), pins, original)
        self.active = True
        th.CachingAutotuner.run = run
        try:
            yield self
            if self.pending is not None:
                raise RuntimeError('unfinished combined FMA triplet')
        finally:
            changed = th.CachingAutotuner.run is not run
            th.CachingAutotuner.run = original
            self.active = False
            if changed:
                raise RuntimeError('combined launch hook ownership changed')

    def report(self):
        return dict(calls=dict(self.dispatch_counts), pending=self.pending is not None,
                    active=self.active, wrappers=list(self.sites),
                    graph_runtime_validated=False, new_gpu_codegen=0)
