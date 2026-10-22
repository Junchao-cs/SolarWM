"""Source-bound denoise and refresh normalization configurations."""

from ..compatibility import expected_digest, kernel_source, kernel_binaries
import copy
from .pins import PinnedRuntime, matches


NAME=kernel_source('first_norm')
SOURCE_SHA=expected_digest("first_norm")
BEFORE=dict(XBLOCK=1,R0_BLOCK=4096,num_warps=16,num_stages=1)
AFTER=dict(BEFORE,R0_BLOCK=2048)
BINARY_ALLOWLIST=set(kernel_binaries('first_norm'))


def plain(record):
    return {k:record[k] for k in BEFORE}


class FirstNormPinnedRuntime(PinnedRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.aligned={}

    def result_for(self,auto,entry,variant):
        selected=self.rank==5 and entry['name']==NAME
        if selected:
            if entry['source_sha']!=SOURCE_SHA or entry['target'] or variant!='A':
                raise RuntimeError('first-norm exact source/variant binding failed')
            if 'firstnorm_original_record' not in entry:
                if plain(entry['records']['A'])!=BEFORE:
                    raise RuntimeError('first-norm baseline configuration changed')
                entry['firstnorm_original_record']=copy.deepcopy(entry['records']['A'])
                entry['records']['A']=dict(AFTER)
                entry['paths']['A']='explicit:denoise-normalization'
            elif plain(entry['records']['A'])!=AFTER:
                raise RuntimeError('first-norm aligned configuration changed')
        # Inherited rehydration still executes beneath the cache-miss guard.
        result=super().result_for(auto,entry,variant)
        if selected:
            launcher=result.make_launcher()
            binary=getattr(launcher,'cache_hash',None)
            if not matches(launcher.config,AFTER) or binary not in BINARY_ALLOWLIST:
                raise RuntimeError('first-norm cached binary/config not bound')
            entry['records']['A']['triton_cache_hash']=binary
            proof=dict(source_sha256=SOURCE_SHA,before=plain(entry['firstnorm_original_record']),
                after=dict(AFTER),binary_hash=binary,new_gpu_codegen=False)
            if NAME in self.aligned and self.aligned[NAME]!=proof:
                raise RuntimeError('first-norm binary drift')
            if NAME not in self.aligned:self.emit('firstnorm_config_aligned',**proof)
            self.aligned[NAME]=proof
        return result

    def report(self):
        report=super().report()
        if set(self.aligned)!=({NAME} if self.rank==5 else set()):
            raise RuntimeError('missing or additional first-norm intervention')
        report['firstnorm_alignment']=self.aligned
        return report


REFRESH=kernel_source('refresh_norm')
REFRESH_SHA=expected_digest("refresh_norm")
REFRESH_BINARY, = kernel_binaries('refresh_norm')


class PhaseNormPins(FirstNormPinnedRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.refresh_alignment={}

    def result_for(self,auto,entry,variant):
        if self.rank!=5 or entry['name']!=REFRESH:
            return super().result_for(auto,entry,variant)
        if entry['source_sha']!=REFRESH_SHA or entry['target'] or variant!='A':
            raise RuntimeError('refresh first-norm exact binding failed')
        if 'refreshnorm_original_record' not in entry:
            if plain(entry['records']['A'])!=BEFORE:
                raise RuntimeError('refresh first-norm baseline config changed')
            entry['refreshnorm_original_record']=copy.deepcopy(entry['records']['A'])
            entry['records']['A']=dict(AFTER)
            entry['paths']['A']='explicit:refresh-normalization'
        elif plain(entry['records']['A'])!=AFTER:
            raise RuntimeError('refresh first-norm aligned config changed')
        result=PinnedRuntime.result_for(self,auto,entry,variant)
        launcher=result.make_launcher();binary=getattr(launcher,'cache_hash',None)
        if not matches(launcher.config,AFTER) or binary!=REFRESH_BINARY:
            raise RuntimeError('refresh existing binary/config not bound')
        entry['records']['A']['triton_cache_hash']=binary
        proof=dict(source_sha256=REFRESH_SHA,before=plain(entry['refreshnorm_original_record']),
                   after=dict(AFTER),binary_hash=binary,new_gpu_codegen=False)
        if REFRESH in self.refresh_alignment and self.refresh_alignment[REFRESH]!=proof:
            raise RuntimeError('refresh first-norm binary drift')
        if REFRESH not in self.refresh_alignment:self.emit('refresh_firstnorm_config_aligned',**proof)
        self.refresh_alignment[REFRESH]=proof
        return result

    def report(self):
        report=super().report()
        if set(self.refresh_alignment)!=({REFRESH} if self.rank==5 else set()):
            raise RuntimeError('missing or additional refresh first-norm intervention')
        report['refresh_firstnorm_alignment']=self.refresh_alignment
        return report
