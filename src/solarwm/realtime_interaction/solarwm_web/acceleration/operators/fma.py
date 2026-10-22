"""Phase-specific fused RMS with explicit residual rounding and cached binaries."""

import hashlib
import importlib.util
import json
from pathlib import Path
from .fusion import ResidualRmsFusion, sha, offset
from ..compatibility import kernel_module


class PhaseFma(ResidualRmsFusion):
    def __init__(self, manifest, rank, device, emit):
        super().__init__(manifest, rank, device, emit)
        self.index_bound = self.doc['index_bound']
        if (self.doc['phase'], self.index_bound) not in (('denoise', 12), ('refresh', 9)):
            raise ValueError('phase-specific real modulation extent required')
        self.rounded = True

    def set_rounding(self, rounded):
        if type(rounded) is not bool or self.pending is not None:
            raise ValueError('explicit bool outside a pending triplet required')
        self.rounded = rounded

    def prepare(self):
        raise RuntimeError('use restore_actual_denoise or prepare_refresh_fma explicitly')

    def load_kernels(self):
        path = Path(self.doc['kernels'])
        if sha(path) != self.doc['kernels_sha256']:
            raise RuntimeError('phase kernel source changed')
        name = kernel_module('denoise' if self.index_bound == 12 else 'refresh')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.kernels = {kind: getattr(module, 'fused_' + kind) for kind in ('global','ffn','local')}

    def bind_wrapper(self, cache):
        source = Path(self.record['wrapper'])
        target = Path(cache) / source.parent.name / source.name
        if sha(target) != self.record['wrapper_sha256']:
            raise RuntimeError('actual relocated dynamic wrapper changed')
        self.record = dict(self.record, wrapper=str(target))

    def _launch(self, r, p, kind, rows, roff, rounded, out):
        # Shadow compares both existing semantics independently. Replacement
        # selects the configured residual rounding; parity is checked separately.
        selected = self.rounded if self.mode == 'replace' else rounded
        return super()._launch(r,p,kind,rows,roff,selected,out)

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
        if min(roff, xoff, red_offset) < 0 or tagoff != timeoff or tagoff != roff or pc['bound'] != self.index_bound:
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
        if previous_mod.numel() != self.index_bound*32256 or p[5].numel() != self.index_bound*32256 or r[5].numel() < rows*5376:
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

    def restore_denoise(self,*args):
        raise RuntimeError('old non-FMA binaries cannot stand in for this intervention')

    def restore_actual_denoise(self,prior):
        if self.index_bound!=12 or self.binaries:
            raise RuntimeError('single-use real denoise binary restore required')
        row=json.loads(Path(prior).read_text())['ranks'][self.rank]['denoise_counterfactual']
        if not row['explicit_square_accumulation_fma'] or row['replacement_round_residual'] is not False:
            raise RuntimeError('prior denoise FMA evidence missing')
        if not self.plans:
            if row['binaries']:raise RuntimeError('unexpected denoise binaries')
            return
        self.load_kernels();self.binaries=dict(row['binaries'])
        if set(self.binaries)!={k+':False' for k in ('global','ffn','local')}:
            raise RuntimeError('three actual denoise variants required')

    def prepare_refresh_fma(self):
        if self.index_bound!=9 or self.binaries:
            raise RuntimeError('single-use real refresh preparation required')
        if not self.plans:return
        import torch
        self.load_kernels()
        for kind in ('global','ffn','local'):
            x=torch.ones((2,5376),device=self.device,dtype=torch.bfloat16)
            branch=torch.full_like(x,.25);ids=torch.zeros(2,device=self.device,dtype=torch.int64)
            mod=torch.full((9,6,5376),.125,device=self.device,dtype=torch.bfloat16)
            gamma=torch.ones(5376,device=self.device,dtype=torch.bfloat16);out=torch.empty_like(x)
            if kind=='global':pointers=(x,branch,ids,ids,mod,branch,gamma,out)
            elif kind=='ffn':pointers=(x,ids,ids,mod,branch,branch,gamma,mod,out)
            else:pointers=(x,ids,ids,mod,branch,branch,gamma,out)
            binary=self.kernels[kind][(2,)](*pointers,0,2,5376,XBLOCK=1,R0_BLOCK=2048,
                ROUND_RESIDUAL=False,num_warps=16,num_stages=1,enable_fp_fusion=True)
            torch.cuda.synchronize(self.device)
            if not bool(torch.isfinite(x).all() and torch.isfinite(out).all()):raise RuntimeError('nonfinite preparation')
            self.binaries[kind+':False']=dict(hash=binary.hash,ptx_sha256=hashlib.sha256(binary.asm['ptx'].encode()).hexdigest())
        self.emit('refresh_explicit_fma_prepared',binaries=self.binaries,new_targeted_variants=3,new_full_model_lowerings=0)

    def report(self):
        row = super().report()
        row.update(phase=self.doc['phase'], index_bound=self.index_bound,
                   replacement_round_residual=self.rounded, production_approved=False,
                   explicit_square_accumulation_fma=True,
                   new_targeted_variants=3 if self.plans and self.index_bound == 9 else 0,
                   denoise_actual_binaries_reused=self.index_bound == 12)
        return row
