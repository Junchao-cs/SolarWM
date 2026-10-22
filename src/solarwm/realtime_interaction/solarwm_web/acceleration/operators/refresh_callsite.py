"""Reuse existing4096 binary at exactly the first refresh norm, never global pins.

Other calls of the same autotuner retain2048 and the unchanged FMA guard.
This owner never compiles a new GPU binary. A cache miss remains an error."""

import ast,copy,json
from pathlib import Path
from .refresh_prefix import TARGETS, BEFORE, AFTER, FUNCTION, sha
from .pins import PinnedRuntime, matches
from .norm import plain


def select_prefix(source,plans):
    calls=sorted((n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='run'
        and isinstance(n.func.value,ast.Name) and n.func.value.id==FUNCTION),key=lambda n:n.lineno)
    if not calls:raise RuntimeError('refresh prefix call absent')
    reductions={p['lines'][1] for p in plans.values() if p['functions'][1]==FUNCTION}
    if set(n.lineno for n in calls[1:])!=reductions:
        raise RuntimeError('shared norm has unclassified or missing callsites')
    first=calls[0]
    if first.lineno in {line for p in plans.values() for line in p['lines']}:
        raise RuntimeError('first norm overlaps an FMA triplet')
    if len(first.args) not in (4,5) or ast.unparse(first.args[0])!='buf78' or ast.unparse(first.args[1])!='buf84' or ast.unparse(first.args[-1])!='5376':
        raise RuntimeError('first normalization argument dataflow changed')
    if [x.arg for x in first.keywords]!=['stream']:
        raise RuntimeError('prefix stream call changed')
    return dict(line=first.lineno,call_ast=ast.dump(first,include_attributes=False),
        argument_count=len(first.args),shared_calls=len(calls),skipped_reductions=len(reductions))


def alternate_launcher(pins,auto,proof):
    """Private alternate selection, leaving original entry records/results intact."""
    if not auto.launchers:auto.precompile()
    entry=pins.register(auto)
    if entry['name']!=TARGETS[pins.rank][0] or entry['source_sha']!=TARGETS[pins.rank][1] or entry['target']:
        raise RuntimeError('prefix source binding mismatch')
    if plain(entry['records']['A'])!=BEFORE:
        raise RuntimeError('shared default must remain2048')
    alternate=dict(entry,records={'A':dict(AFTER)},paths={'A':'first-refresh-callsite-only'},
        results=list(entry['results']),launchers={})
    result=PinnedRuntime.result_for(pins,auto,alternate,'A')
    launcher=result.make_launcher()
    if not matches(launcher.config,AFTER) or launcher.cache_hash!=proof['binary_hash']:
        raise RuntimeError('prefix existing4096 binary mismatch')
    if plain(entry['records']['A'])!=BEFORE:
        raise RuntimeError('prefix changed shared default')
    return launcher


def launch_with_restore(auto,launcher,base_run,args,kwargs):
    previous=auto.launchers
    auto.launchers=[launcher]
    try:return base_run(auto,*args,**kwargs)
    finally:auto.launchers=previous


class RefreshPrefixCallsite:
    def __init__(self,fma,cache,emit):
        self.rank,self.emit=fma.rank,emit
        if fma.index_bound!=9:raise RuntimeError('refresh-only prefix owner')
        self.wrapper=fma.record['wrapper'];self.wrapper_sha=fma.record['wrapper_sha256']
        self.binding=None;self.proof=None;self.launcher=None;self.autotuner=None;self.calls=0
        if self.rank not in TARGETS:return
        if sha(self.wrapper)!=self.wrapper_sha:raise RuntimeError('prefix wrapper changed')
        self.binding=select_prefix(Path(self.wrapper).read_text(),fma.plans)
        prior=json.loads(Path(fma.prefix_binary_provenance).read_text())
        if prior['status']!='IMPORTED_BINARY_PROVENANCE_NOT_ACCEPTANCE':
            raise RuntimeError('explicit source-bound binary provenance required')
        found=[x for x in prior['prepared_binaries'] if x['rank']==self.rank]
        if len(found)!=1:raise RuntimeError('one actual prior prepared binary required')
        self.proof=found[0]
        if self.proof['source_sha256']!=TARGETS[self.rank][1] or self.proof['before']!=BEFORE or self.proof['after']!=AFTER:
            raise RuntimeError('prior reduction semantics changed')
        if sha(self.proof['source'])!=TARGETS[self.rank][1]:raise RuntimeError('prior reduction source changed')
        ptx=Path(cache)/self.proof['binary_hash']/Path(self.proof['ptx_path']).name
        if sha(ptx)!=self.proof['ptx_sha256']:raise RuntimeError('prepared binary not copied intact')
        self.emit('refresh_prefix_callsite_bound',binding=self.binding,wrapper_sha256=self.wrapper_sha,
            binary_hash=self.proof['binary_hash'],imported_binary_provenance_only=True,new_gpu_codegen=0)

    def matches_site(self,frame):
        return self.binding is not None and frame.f_code.co_filename==self.wrapper and frame.f_lineno==self.binding['line']

    def launch(self,auto,args,kwargs,frame,pins):
        import torch
        if not self.matches_site(frame) or frame.f_globals.get(FUNCTION) is not auto:
            raise RuntimeError('prefix actual wrapper callsite mismatch')
        if self.calls or set(kwargs)!={'stream'} or len(args)!=self.binding['argument_count']:
            raise RuntimeError('prefix single refresh invocation/signature changed')
        if kwargs['stream']!=torch.cuda.current_stream(self.rank).cuda_stream:
            raise RuntimeError('prefix current stream mismatch')
        x,y=args[:2]
        if (x.device!=torch.device('cuda',self.rank) or y.device!=x.device
                or x.dtype!=torch.bfloat16 or y.dtype!=torch.float32 or not x.is_contiguous()
                or not y.is_contiguous() or type(args[-2]) is not int or args[-1]!=5376
                or args[-2]<=0 or y.numel()!=args[-2] or x.numel()<args[-2]*5376):
            raise RuntimeError('prefix actual tensor contract changed')
        self.launcher=alternate_launcher(pins,auto,self.proof)
        self.autotuner=auto
        from torch._inductor.runtime import triton_heuristics as th
        bases=[prior for obj,name,prior,installed in pins.originals if obj is th.CachingAutotuner and name=='run']
        if len(bases)!=1:raise RuntimeError('one original framework runner required')
        result=launch_with_restore(auto,self.launcher,bases[0],args,kwargs)
        self.calls+=1
        self.emit('refresh_prefix_callsite_launched',line=self.binding['line'],binary_hash=self.proof['binary_hash'],
            shared_default_unchanged=True,new_gpu_codegen=0)
        return result

    def report(self):
        if self.calls!=(1 if self.rank in TARGETS else 0):raise RuntimeError('prefix invocation incomplete')
        return dict(rank=self.rank,calls=self.calls,binding=self.binding,
            binary_hash=None if self.proof is None else self.proof['binary_hash'],
            shared_default_unchanged=True,new_gpu_codegen=0,new_preparation=False)
