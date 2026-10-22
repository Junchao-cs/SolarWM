"""Sequence-parallel layout expressions and fused kernel dispatch."""

import ast
from contextlib import contextmanager
import hashlib,json,sys
from pathlib import Path


def scalar(node, names):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -scalar(node.operand, names)
    if isinstance(node, ast.BinOp):
        a, b = scalar(node.left, names), scalar(node.right, names)
        if isinstance(node.op, ast.Add): return a+b
        if isinstance(node.op, ast.Sub): return a-b
        if isinstance(node.op, ast.Mult): return a*b
        if isinstance(node.op, ast.FloorDiv): return a//b
        if isinstance(node.op, ast.Mod): return a%b
    raise ValueError('not an integer layout expression: ' + ast.dump(node))


def partition(length):
    n=5566+length
    return [(p*(n//8)+min(p,n%8),n//8+(p<n%8)) for p in range(8)]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dimensions(record, rank, args):
    if len(args)!=2+len(record['scalar_expressions']): raise RuntimeError('layout ABI changed')
    kind=record['kind']
    prefix=args[3] if kind in ('gather','copy') else args[2]
    if type(prefix) is not int:
        raise RuntimeError('condition prefix must be int, got '+type(prefix).__name__)
    length=prefix-1008
    if not 1000<=length<=1300:
        raise RuntimeError('prompt/prefix out of domain: '+str((length,prefix)))
    expected=tuple(scalar(ast.parse(x,mode='eval').body,{'L':prefix}) for x in record['scalar_expressions'])
    if any(type(x) is not int for x in args[2:]) or tuple(args[2:])!=expected:
        raise RuntimeError('layout runtime scalar dataflow changed')
    n=5566+prefix
    count=7168*partition(prefix)[rank][1] if kind=='copy' else 896*n
    if args[-1]!=count: raise RuntimeError('layout output extent changed')
    return length,n,count


class SpLayout:
    def __init__(self,manifest,rank,device,emit):
        self.path=Path(manifest);self.manifest_sha256=sha(self.path)
        doc=json.loads(self.path.read_text())
        for p,d in doc['evidence_sha256'].items():
            if sha(p)!=d: raise RuntimeError('layout evidence changed: '+p)
        self.rank,self.device,self.emit=rank,device,emit
        self.records=doc['ranks'][rank]['records'];self.mode='off'
        self.kernels={};self.binaries={};self.checked={};self.shadow=[];self.prepared=[]
        self.calls={'shadow':0,'replace':0}
        self.request=None

    def bind_request(self, prompt, condition_prefix, total):
        if any(type(x) is not int for x in (prompt,condition_prefix,total)):
            raise RuntimeError('native request lengths must be actual integers')
        if not 1000<=prompt<=1300 or condition_prefix!=prompt+1008 or total!=condition_prefix+5566:
            raise RuntimeError('native request prompt/anchor/audio/video layout changed')
        self.request=(prompt,condition_prefix,total)
        self.emit('sp_layout_request_bound',prompt_tokens=prompt,condition_prefix_tokens=condition_prefix,total_tokens=total)

    def set_mode(self,mode):
        if mode not in ('off','shadow','replace'):raise ValueError(mode)
        self.mode=mode

    def _launch(self,kind,x,y,n,count):
        grid=((count+1023)//1024,) if kind=='copy' else ((((n+7)//8)*128+511)//512,8,7)
        binary=self.kernels[kind][grid](x,y,count if kind=='copy' else n,
            BLOCK=1024 if kind=='copy' else 512,num_warps=4,num_stages=1)
        digest=dict(hash=binary.hash,ptx_sha256=hashlib.sha256(binary.asm['ptx'].encode()).hexdigest())
        if kind in self.binaries and self.binaries[kind]!=digest:
            raise RuntimeError('unexpected layout binary specialization')
        self.binaries[kind]=digest

    def prepare(self):
        import torch
        from .layout_kernels import gather_heads, pack_heads, copy_heads
        self.kernels=dict(gather=gather_heads,pack=pack_heads,copy=copy_heads)
        generator=torch.Generator(device=self.device).manual_seed(9041+self.rank)
        # All eight partition residues, plus the two actual scene lengths.
        # These are real CUDA data-layout checks, not full-model benchmarks.
        for length in (*range(1000,1008),1115,1157):
            prefix=length+1008;n=5566+prefix;parts=partition(prefix)
            x=torch.randn((n,7,128),device=self.device,dtype=torch.bfloat16,generator=generator)
            expected=torch.cat([x[s:s+l].permute(1,0,2).contiguous().reshape(-1) for s,l in parts])
            packed=torch.empty_like(expected)
            self._launch('pack',x,packed,n,896*n)
            gathered=torch.empty((7,n,128),device=self.device,dtype=torch.bfloat16)
            self._launch('gather',packed,gathered,n,896*n)
            count=7168*parts[self.rank][1]
            source=torch.randn(count,device=self.device,dtype=torch.bfloat16,generator=generator)
            copied=torch.empty_like(source)
            self._launch('copy',source,copied,n,count)
            torch.cuda.synchronize(self.device)
            if not (torch.equal(packed,expected) and torch.equal(gathered,x.permute(1,0,2)) and torch.equal(copied,source)):
                raise RuntimeError('new layout primitive differs from native indexing')
            self.prepared.append(dict(length=length,condition_prefix=prefix,pack_exact=True,gather_exact=True,copy_exact=True))
        self.emit('sp_layout_prepared',checks=self.prepared,binaries=self.binaries,new_full_model_lowerings=0)

    def _check(self,record,args,stream):
        import torch
        length,n,count=dimensions(record,self.rank,args)
        if self.request is None or self.request!=(length,length+1008,n):
            raise RuntimeError('helper scalars differ from bound native request')
        x,y=args[:2]
        for t in (x,y):
            if t.device!=self.device or t.dtype!=torch.bfloat16 or not t.is_contiguous() or t.numel()!=count:
                raise RuntimeError('layout BF16/device/contiguous/extent changed')
        if max(x.data_ptr(),y.data_ptr())<min(x.data_ptr()+2*count,y.data_ptr()+2*count):
            raise RuntimeError('layout input/output overlap')
        if stream!=torch.cuda.current_stream(self.device).cuda_stream: raise RuntimeError('layout stream changed')
        return length,n,count

    @contextmanager
    def installed(self,pins):
        import torch
        from torch._inductor.runtime import triton_heuristics as th
        original=th.CachingAutotuner.run
        def run(auto,*args,**kwargs):
            key=Path(getattr(auto,'filename','') or '').name
            if self.mode=='off' or key not in self.records:return original(auto,*args,**kwargs)
            record=self.records[key];frame=sys._getframe(1)
            if frame.f_globals.get(record['name']) is not auto: raise RuntimeError('layout helper object binding changed')
            if set(kwargs)!={'stream'}: raise RuntimeError('layout launch keywords changed')
            if id(auto) not in self.checked:
                if not auto.launchers:auto.precompile()
                bound=pins.register(auto);selected=pins.result_for(auto,bound,'A')
                config=dict(selected.config.kwargs,num_warps=selected.config.num_warps,num_stages=selected.config.num_stages)
                if bound['source_sha']!=record['source_sha256'] or bound['target'] or config!=record['config']:
                    raise RuntimeError('layout source/config mismatch')
                self.checked[id(auto)]=auto
            length,n,count=self._check(record,args,kwargs['stream'])
            x,y=args[:2];before=x._version
            if self.mode=='shadow':
                result=original(auto,*args,**kwargs)
                candidate=torch.empty_like(y)
                self._launch(record['kind'],x,candidate,n,count)
                exact=torch.equal(candidate,y);finite=bool(torch.isfinite(candidate).all())
                row=dict(helper=record['name'],kind=record['kind'],length=length,condition_prefix=length+1008,count=count,exact=exact,finite=finite)
                self.shadow.append(row)
                if not exact or not finite:raise RuntimeError('actual layout shadow mismatch: '+str(row))
            else:
                self._launch(record['kind'],x,y,n,count);result=None
            if x._version!=before:raise RuntimeError('layout input version changed')
            self.calls[self.mode]+=1
            return result
        th.CachingAutotuner.run=run
        try:yield self
        finally:
            changed=th.CachingAutotuner.run is not run
            th.CachingAutotuner.run=original
            if changed:raise RuntimeError('layout hook ownership changed')

    def report(self):
        return dict(manifest_sha256=self.manifest_sha256,mode=self.mode,calls=dict(self.calls),
            prepared=list(self.prepared),shadow=list(self.shadow),binaries=dict(self.binaries),request=self.request,
            bound_helper_objects=len(self.checked),new_full_model_lowerings=0)
