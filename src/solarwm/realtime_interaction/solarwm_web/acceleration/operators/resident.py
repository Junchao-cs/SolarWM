"""Resident first-refresh callsite plus existing FMA/QKV/layout dispatch.

Only Python warm/capture calls execute these hooks; replays must not. Existing
binary identities are retained and all cache-miss guards stay active."""

import copy,json
from pathlib import Path
from .refresh_callsite import RefreshPrefixCallsite, alternate_launcher, launch_with_restore, TARGETS, FUNCTION, sha
from .fma_layout import FmaLayoutDispatch
from bisect import bisect_right


class FrameSite:
    __slots__=('f_code','f_globals','f_lineno')
    def __init__(self,frame,line):
        self.f_code=frame.f_code
        self.f_globals=frame.f_globals
        self.f_lineno=line


class BytecodeSites:
    def __init__(self):
        self.cache={}
        self.lookups=0
    def bind(self,frame):
        code=frame.f_code
        key=id(code)
        entry=self.cache.get(key)
        if entry is None:
            rows=[(start,end,line) for start,end,line in code.co_lines() if end>start]
            if not rows or rows[0][0]!=0 or any(a[1]!=b[0] for a,b in zip(rows,rows[1:])):
                raise RuntimeError('bytecode line table is incomplete')
            entry=(code,tuple(x[0] for x in rows),tuple(x[1] for x in rows),tuple(x[2] for x in rows))
            self.cache[key]=entry
        if entry[0] is not code:raise RuntimeError('code identity changed')
        offset=frame.f_lasti
        index=bisect_right(entry[1],offset)-1
        if index<0 or offset>=entry[2][index] or type(entry[3][index]) is not int:
            raise RuntimeError('active call has no exact source line')
        self.lookups+=1
        return FrameSite(frame,entry[3][index])
    def report(self):
        return dict(code_objects=len(self.cache),lookups=self.lookups,
            frame_lineno_reads=0,source_line_checks_retained=True)


class GraphPrefixCallsite(RefreshPrefixCallsite):
    def launch(self,auto,args,kwargs,frame,pins):
        import torch
        if not self.matches_site(frame) or frame.f_globals.get(FUNCTION) is not auto:
            raise RuntimeError('prefix actual wrapper callsite mismatch')
        if set(kwargs)!={'stream'} or len(args)!=self.binding['argument_count']:
            raise RuntimeError('prefix refresh signature changed')
        if kwargs['stream']!=torch.cuda.current_stream(self.rank).cuda_stream:
            raise RuntimeError('prefix current stream mismatch')
        x,y=args[:2]
        if (x.device!=torch.device('cuda',self.rank) or y.device!=x.device
                or x.dtype!=torch.bfloat16 or y.dtype!=torch.float32 or not x.is_contiguous()
                or not y.is_contiguous() or type(args[-2]) is not int or args[-1]!=5376
                or args[-2]<=0 or y.numel()!=args[-2] or x.numel()<args[-2]*5376):
            raise RuntimeError('prefix actual tensor contract changed')
        if self.launcher is None:
            self.launcher=alternate_launcher(pins,auto,self.proof)
            self.autotuner=auto
        elif self.autotuner is not auto:
            raise RuntimeError('resident prefix autotuner identity changed')
        from torch._inductor.runtime import triton_heuristics as th
        bases=[prior for obj,name,prior,installed in pins.originals if obj is th.CachingAutotuner and name=='run']
        if len(bases)!=1:raise RuntimeError('one original framework runner required')
        result=launch_with_restore(auto,self.launcher,bases[0],args,kwargs)
        self.calls+=1
        self.emit('refresh_prefix_callsite_launched',line=self.binding['line'],binary_hash=self.proof['binary_hash'],
            shared_default_unchanged=True,new_gpu_codegen=0)
        return result

    def report(self):
        if bool(self.calls) != (self.rank in TARGETS):
            raise RuntimeError('resident prefix invocation incomplete')
        return dict(rank=self.rank,calls=self.calls,binding=self.binding,
            binary_hash=None if self.proof is None else self.proof['binary_hash'],
            shared_default_unchanged=True,new_gpu_codegen=0,new_preparation=False)


class ResidentFmaLayoutDispatch(FmaLayoutDispatch):
    def __init__(self,denoise,refresh,layout,qkv_record,cache):
        super().__init__(denoise,refresh,layout,qkv_record)
        self.prefix=GraphPrefixCallsite(refresh,cache,refresh.emit)
        self.bytecode_sites=BytecodeSites()

    def dispatch(self,auto,args,kwargs,frame,pins,original):
        frame=self.bytecode_sites.bind(frame)
        if self.prefix.matches_site(frame):
            if self.pending is not None:
                raise RuntimeError('prefix interrupts a combined triplet')
            return self.prefix.launch(auto,args,kwargs,frame,pins)
        return super().dispatch(auto,args,kwargs,frame,pins,original)

    def report(self):
        row=super().report()
        row['prefix']=self.prefix.report()
        row['bytecode_sites']=self.bytecode_sites.report()
        return row


def restore_layout_binaries(layout,prior_result):
    if layout.binaries or layout.kernels:
        raise RuntimeError('layout restore must precede first use')
    doc=json.loads(Path(prior_result).read_text())
    row=doc['ranks'][layout.rank]['fusion']
    if set(row['binaries'])!={'gather','pack','copy'}:
        raise RuntimeError('three actual layout binaries required')
    if row['manifest_sha256']!=layout.manifest_sha256:
        raise RuntimeError('layout manifest differs from observed prior kernels')
    from .layout_kernels import gather_heads, pack_heads, copy_heads
    layout.kernels=dict(gather=gather_heads,pack=pack_heads,copy=copy_heads)
    layout.binaries=copy.deepcopy(row['binaries'])
    layout.emit('layout_existing_binaries_restored',binaries=layout.binaries,primitive_tests_rerun=False,new_gpu_codegen=0)
