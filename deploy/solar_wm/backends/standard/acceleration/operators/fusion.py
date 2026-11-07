"""Source-bound residual/RMS fusion, with shadow checks and explicit dispatch modes."""

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
from ..compatibility import kernel_module
import json
from pathlib import Path
import sys


def arith(node, values):
    """Only generated integer pointer-offset arithmetic; never eval source."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp):
        a, b = arith(node.left, values), arith(node.right, values)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.FloorDiv) and b > 0: return a // b
    raise ValueError('unsupported generated pointer arithmetic')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def offset(expression, values):
    return arith(ast.parse(expression, mode='eval').body, values)


class ResidualRmsFusion:
    def __init__(self, manifest, rank, device, emit):
        self.path = Path(manifest)
        self.manifest_sha256 = sha(self.path)
        doc = json.loads(self.path.read_text())
        for p, digest in doc['evidence_sha256'].items():
            if sha(p) != digest:
                raise RuntimeError('fusion source changed: ' + p)
        self.doc, self.rank, self.device, self.emit = doc, rank, device, emit
        self.record = doc['ranks'][rank]
        self.plans = {int(k): v for k, v in self.record['plans'].items()}
        self.mode, self.pending = 'off', None
        self.kernels, self.binaries = {}, {}
        self.shadow, self.calls = [], {'shadow': 0, 'replace': 0}
        self.checked = {}

    def set_mode(self, mode):
        if self.pending is not None or mode not in ('off', 'shadow', 'replace'):
            raise RuntimeError('invalid fusion mode transition')
        self.mode = mode

    def prepare(self):
        if not self.plans:
            return
        import torch
        path = Path(self.doc['kernels'])
        if sha(path) != self.doc['kernels_sha256']:
            raise RuntimeError('generated kernel source changed')
        spec = importlib.util.spec_from_file_location(kernel_module('denoise'), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Exactly three bodies x two rounding choices. No full-model lowering.
        for kind in ('global', 'ffn', 'local'):
            fn = getattr(module, 'fused_' + kind)
            self.kernels[kind] = fn
            for rounded in (True, False):
                x = torch.ones((2, 5376), device=self.device, dtype=torch.bfloat16)
                branch = torch.full_like(x, .25)
                ids = torch.zeros(2, device=self.device, dtype=torch.int64)
                mod = torch.full((12, 6, 5376), .125, device=self.device, dtype=torch.bfloat16)
                gamma = torch.ones(5376, device=self.device, dtype=torch.bfloat16)
                out = torch.empty_like(x)
                if kind == 'global':
                    pointers = (x, branch, ids, ids, mod, branch, gamma, out)
                elif kind == 'ffn':
                    pointers = (x, ids, ids, mod, branch, branch, gamma, mod, out)
                else:
                    pointers = (x, ids, ids, mod, branch, branch, gamma, out)
                binary = fn[(2,)](*pointers, 0, 2, 5376, XBLOCK=1, R0_BLOCK=2048,
                                  ROUND_RESIDUAL=rounded, num_warps=16, num_stages=1, enable_fp_fusion=True)
                torch.cuda.synchronize(self.device)
                if not bool(torch.isfinite(x).all() and torch.isfinite(out).all()):
                    raise RuntimeError('new kernel preparation produced nonfinite output')
                self.binaries[kind + ':' + str(rounded)] = dict(hash=binary.hash,
                    ptx_sha256=hashlib.sha256(binary.asm['ptx'].encode()).hexdigest())
        self.emit('residual_rms_kernels_prepared', binaries=self.binaries,
                  new_full_model_lowerings=0, targeted_kernel_variants=6)

    def _bind_auto(self, auto, name, frame, pins):
        if frame.f_globals.get(name) is not auto:
            raise RuntimeError('fusion call-site kernel object changed')
        record = self.record['records'][name]
        if Path(getattr(auto, 'filename', '')).name != Path(record['source']).name:
            raise RuntimeError('fusion helper filename mismatch')
        if id(auto) not in self.checked:
            if not auto.launchers:
                auto.precompile()
            bound = pins.register(auto)
            selected = pins.result_for(auto, bound, 'A')
            config = dict(selected.config.kwargs, num_warps=selected.config.num_warps, num_stages=selected.config.num_stages)
            if bound['source_sha'] != record['source_sha256'] or bound['target'] or config != record['config']:
                raise RuntimeError('fusion helper source/config mismatch')
            self.checked[id(auto)] = auto

    def _check_inputs(self, residual, reduction, pointwise, plan, stream):
        import torch
        rc, red, pc = plan['residual'], plan['reduction'], plan['pointwise']
        r, d, p = residual, reduction, pointwise
        if len(r) != (8 if rc['has_symbol'] else 7) or len(d) != (5 if red.get('has_symbol') else 4) or len(p) != (9 if pc['has_symbol'] else 8):
            raise RuntimeError('fusion runtime argument count changed')
        if d[-1] != 5376 or type(d[-2]) is not int or d[-2] <= 0 or r[-1] != p[-1] or r[-1] != d[-2]*5376:
            raise RuntimeError('fusion runtime row extent mismatch')
        rows = d[-2]
        roff = offset(rc['offset'], {'ks0': r[6] if rc['has_symbol'] else 0})
        xoff, tagoff, timeoff = [offset(v, {'ks0': p[7] if pc['has_symbol'] else 0, 'x1': 0, 'x2': 0}) for v in pc['offsets']]
        red_offset = offset(red.get('offset', '0'), {'ks0': d[2] if red.get('has_symbol') else 0, 'r0_1': 0, 'x0': 0})
        if min(roff, xoff, red_offset) < 0 or tagoff != timeoff or tagoff != roff or pc['bound'] != 12:
            raise RuntimeError('fusion rank/index offsets mismatch')
        if p[0].data_ptr()+2*xoff != r[0].data_ptr() or d[0].data_ptr()+2*red_offset != r[0].data_ptr() or d[1].data_ptr() != p[1].data_ptr():
            raise RuntimeError('fusion residual/reduction/modulation dataflow mismatch')
        kind = rc['kind']
        ids = (r[2], r[3]) if kind == 'global' else (r[1], r[2])
        previous_mod = r[4] if kind == 'global' else r[3]
        if any(a.data_ptr() != b.data_ptr() for a, b in zip(ids, p[3:5])):
            raise RuntimeError('fusion index object binding mismatch')
        if pc['shift'] != (0 if kind == 'ffn' else 16128):
            raise RuntimeError('fusion modulation slice mismatch')
        if kind != 'ffn' and previous_mod.data_ptr() != p[5].data_ptr():
            raise RuntimeError('fusion within-layer modulation object mismatch')
        bf16 = [r[0], previous_mod, r[5], p[2], p[5], p[6]]
        bf16.append(r[1] if kind == 'global' else r[4])
        for tensor in bf16:
            if tensor.device != self.device or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise RuntimeError('fusion BF16 tensor layout mismatch')
        for tensor in ids:
            if tensor.device != self.device or tensor.dtype != torch.int64 or not tensor.is_contiguous() or tensor.numel() < roff+rows:
                raise RuntimeError('fusion index tensor extent/type mismatch')
        if d[1].dtype != torch.float32 or d[1].numel() < rows or r[0].numel() < rows*5376 or p[6].numel() != rows*5376 or p[2].numel() != 5376:
            raise RuntimeError('fusion output/reduction extent mismatch')
        if previous_mod.numel() != 12*32256 or p[5].numel() != 12*32256 or r[5].numel() < rows*5376:
            raise RuntimeError('fusion modulation/branch extent mismatch')
        branch = r[1] if kind == 'global' else r[4]
        if branch.numel() < (rows + (roff if kind == 'global' else 0))*5376:
            raise RuntimeError('fusion branch slice out of bounds')
        if stream != torch.cuda.current_stream(self.device).cuda_stream:
            raise RuntimeError('fusion stream mismatch')
        out_range = (p[6].data_ptr(), p[6].data_ptr()+p[6].numel()*2)
        def overlap(tensor):
            a, b = tensor.data_ptr(), tensor.data_ptr()+tensor.numel()*tensor.element_size()
            return max(a, out_range[0]) < min(b, out_range[1])
        if any(overlap(t) for t in (r[0], *ids, previous_mod, p[2], p[5])):
            raise RuntimeError('fusion output overlaps persistent input')
        # Branch reuse is safe only at the identical row-relative range: first
        # loop consumes the row completely before its second loop stores out.
        for tensor in (branch, r[5]):
            start = tensor.data_ptr() + (2*5376*roff if kind == 'global' and tensor is branch else 0)
            if overlap(tensor) and start != out_range[0]:
                raise RuntimeError('fusion output has shifted branch overlap')
        return kind, rows, roff

    def _launch(self, r, p, kind, rows, roff, rounded, out):
        if kind == 'ffn':
            pointers = (*r[:6], p[2], p[5], out)
        else:
            pointers = (*r[:6], p[2], out)
        result = self.kernels[kind][(rows,)](*pointers, roff, rows, 5376, XBLOCK=1, R0_BLOCK=2048,
            ROUND_RESIDUAL=rounded, num_warps=16, num_stages=1, enable_fp_fusion=True)
        if result.hash != self.binaries[kind + ':' + str(rounded)]['hash']:
            raise RuntimeError('unexpected fused kernel specialization')

    @contextmanager
    def installed(self, pins):
        import torch
        from torch._inductor.runtime import triton_heuristics as th
        original = th.CachingAutotuner.run
        def run(auto, *args, **kwargs):
            if self.mode == 'off' or not self.plans:
                return original(auto, *args, **kwargs)
            frame = sys._getframe(1)
            if self.pending is None:
                if frame.f_code.co_filename != self.record['wrapper']:
                    return original(auto, *args, **kwargs)
                plan = self.plans.get(frame.f_lineno)
                if plan is None:
                    return original(auto, *args, **kwargs)
                self._bind_auto(auto, plan['functions'][0], frame, pins)
                if set(kwargs) != {'stream'}:
                    raise RuntimeError('fusion launch keywords changed')
                shadow = tuple(x.detach().clone() for x in args[:6]) if self.mode == 'shadow' else None
                self.pending = dict(plan=plan, residual=args, kwargs=kwargs, stage=1, shadow=shadow)
                return original(auto, *args, **kwargs) if self.mode == 'shadow' else None
            pending = self.pending
            plan, stage = pending['plan'], pending['stage']
            if (frame.f_code.co_filename != self.record['wrapper'] or frame.f_lineno != plan['lines'][stage] or kwargs != pending['kwargs']):
                raise RuntimeError('fusion triplet order/site/stream changed')
            self._bind_auto(auto, plan['functions'][stage], frame, pins)
            if stage == 1:
                pending['reduction'], pending['stage'] = args, 2
                return original(auto, *args, **kwargs) if self.mode == 'shadow' else None
            r = pending['residual']
            kind, rows, roff = self._check_inputs(r, pending['reduction'], args, plan, kwargs['stream'])
            if self.mode == 'replace':
                self._launch(r, args, kind, rows, roff, True, args[6])
                result = None
            else:
                result = original(auto, *args, **kwargs)
                # Copies were made before residual and before reused branches
                # could be overwritten by the original pointwise output.
                saved = pending['shadow']
                rounded_r = (*saved, *r[6:])
                unrounded_r = (saved[0].clone(), *saved[1:], *r[6:])
                rounded_out, unrounded_out = torch.empty_like(args[6]), torch.empty_like(args[6])
                self._launch(rounded_r, args, kind, rows, roff, True, rounded_out)
                self._launch(unrounded_r, args, kind, rows, roff, False, unrounded_out)
                residual_exact = torch.equal(rounded_r[0], r[0])
                output_exact = torch.equal(rounded_out, args[6])
                delta = unrounded_out.float()-rounded_out.float()
                diagnostic = dict(kind=kind, rows=rows, offset=roff, line=plan['lines'][0],
                    residual_exact=residual_exact, rounded_output_exact=output_exact,
                    unrounded_exact=torch.equal(unrounded_out, rounded_out),
                    unrounded_relative_l2=float(delta.norm()/rounded_out.float().norm().clamp_min(1e-30)),
                    unrounded_maxabs=float(delta.abs().max()),
                    finite=bool(torch.isfinite(rounded_out).all() and torch.isfinite(unrounded_out).all()))
                self.shadow.append(diagnostic)
                if not residual_exact or not output_exact or not diagnostic['finite']:
                    self.emit('residual_rms_shadow_mismatch', **diagnostic)
                    raise RuntimeError('rounded fusion differs on actual model tensors: '+str(diagnostic))
            self.calls[self.mode] += 1
            self.pending = None
            return result
        th.CachingAutotuner.run = run
        try:
            yield self
            if self.pending is not None:
                raise RuntimeError('unfinished residual/RMS triplet')
        finally:
            changed = th.CachingAutotuner.run is not run
            th.CachingAutotuner.run = original
            if changed:
                raise RuntimeError('fusion runtime hook ownership changed')

    def report(self):
        return dict(manifest_sha256=self.manifest_sha256, planned_triplets=len(self.plans),
                    calls=dict(self.calls), shadow=list(self.shadow), binaries=dict(self.binaries),
                    mode=self.mode, pending=self.pending is not None, replacement_round_residual=True)
