"""Bounded compiler inputs and storage-preserving singleton-stride views."""

from dataclasses import replace
import torch
from solar.backends.minimax_h3.sgf_attention import H3RawKVCache, H3SGFAttention
from .forward import tensor_references, tensor_stamp


class LocalWindowCall:
    def __init__(self, control, *, layer_count=50):
        if torch.is_grad_enabled():
            raise RuntimeError("inference no-grad owner required")
        if not isinstance(control,H3SGFAttention) or control.mode!="rollout" or not isinstance(control.cache,H3RawKVCache):
            raise ValueError("native rollout cache required")
        index=control.chunk_index
        if type(index) is not int or index<0 or type(control.commit_cache) is not bool:
            raise ValueError("explicit nonnegative global chunk and boolean phase required")
        if type(layer_count) is not int or layer_count<1:raise ValueError("invalid layer count")
        if control.viewmats.shape[1]<1+(index+1)*5 or control.intrinsics.shape[1]!=control.viewmats.shape[1]:
            raise ValueError("global camera coverage incomplete")
        expected=list(range(max(0,index-5),index))
        if index==0:
            if control.cache.layers:raise ValueError("first chunk cache not empty")
        elif set(control.cache.layers)!=set(range(layer_count)) or any([e[0] for e in h]!=expected for h in control.cache.layers.values()):
            raise ValueError("global causal history mismatch")
        self.source=control;self.index=index;self.first=max(0,index-5)
        self.local_index=min(index,5);self.layer_count=layer_count;self.finished=False
        self.history={k:list(v) for k,v in control.cache.layers.items()}
        self.tensors=tensor_references(control)
        self.stamps=tuple(tensor_stamp(t) for t in self.tensors)
        private=H3RawKVCache({layer:[(i,k,v) for i,(_,k,v) in enumerate(h)]
                              for layer,h in self.history.items()})
        # Preserve global anchor 0 and select exactly the native current window.
        start=1+self.first*5;stop=1+(index+1)*5
        views=torch.cat((control.viewmats[:,:1],control.viewmats[:,start:stop]),dim=1)
        intrinsics=torch.cat((control.intrinsics[:,:1],control.intrinsics[:,start:stop]),dim=1)
        self.control=replace(control,chunk_index=self.local_index,cache=private,
                             viewmats=views,intrinsics=intrinsics)
        self.private_before=self.identities(private.layers)

    @staticmethod
    def identities(layers):
        return {k:[(i,id(a),id(b)) for i,a,b in h] for k,h in layers.items()}

    def finish(self):
        if self.finished:raise RuntimeError("call already committed/finished")
        if self.identities(self.source.cache.layers)!=self.identities(self.history):
            raise RuntimeError("global cache changed during private execution")
        if tuple(tensor_stamp(t) for t in self.tensors)!=self.stamps:
            raise RuntimeError("borrowed control or KV tensor mutated")
        layers=self.control.cache.layers
        if not self.source.commit_cache:
            if self.identities(layers)!=self.private_before:raise RuntimeError("denoise committed private cache")
        else:
            if set(layers)!=set(range(self.layer_count)):raise RuntimeError("refresh incomplete")
            expected=list(range(max(0,self.local_index-4),self.local_index+1))
            old_ptrs={t.data_ptr() for t in self.tensors}
            updated={}
            for layer,entries in layers.items():
                if [e[0] for e in entries]!=expected:raise RuntimeError("wrong local refresh history")
                retained=self.private_before.get(layer,[])[-4:]
                if self.identities({layer:entries[:-1]})[layer]!=retained:
                    raise RuntimeError("refresh replaced prior raw KV")
                _,key,value=entries[-1]
                if key.requires_grad or value.requires_grad or key.data_ptr() in old_ptrs or value.data_ptr() in old_ptrs:
                    raise RuntimeError("new KV not detached private clones")
                updated[layer]=[*self.history.get(layer,[]),(self.index,key,value)][-5:]
            # Publish only after every layer and borrowed source passes validation.
            self.source.cache.layers.update(updated)
        self.finished=True
        self.control.cache.clear()
        return dict(global_chunk=self.index,local_chunk=self.local_index,
                    camera_source_latent_interval=[self.first*5,(self.index+1)*5],
                    phase="refresh" if self.source.commit_cache else "denoise",
                    global_history_preserved=True,private_cache_cleared=True)


def singleton_stride_view(t):
    if torch.is_grad_enabled():raise RuntimeError("inference no-grad required")
    if not isinstance(t,torch.Tensor) or t.layout!=torch.strided:
        raise ValueError("strided raw KV tensor required")
    if t.ndim!=4 or any(s<=0 for s in t.shape):raise ValueError("nonempty 4D raw KV required")
    stride=list(t.stride())
    for i in range(t.ndim-1,-1,-1):
        if t.shape[i]==1:stride[i]=1 if i==t.ndim-1 else stride[i+1]*t.shape[i+1]
    if tuple(stride)==t.stride():return t
    # A size-one coordinate is always zero; changing only its stride leaves
    # every addressed storage element and storage offset unchanged.
    out=t.as_strided(t.shape,stride,t.storage_offset())
    if out.data_ptr()!=t.data_ptr() or out.storage_offset()!=t.storage_offset():
        raise RuntimeError("singleton view changed storage ownership")
    return out

class SingletonStrideWindowCall(LocalWindowCall):
    def __init__(self,control,*,layer_count=50):
        super().__init__(control,layer_count=layer_count)
        self.singleton_views=0
        for layer,history in self.control.cache.layers.items():
            entries=[]
            for i,k,v in history:
                key,value=singleton_stride_view(k),singleton_stride_view(v)
                self.singleton_views+=int(key is not k)+int(value is not v)
                entries.append((i,key,value))
            self.control.cache.layers[layer]=entries
        # Parent finish validates retained private identities against this new
        # private representation, while global history and borrowed stamps stay original.
        self.private_before=self.identities(self.control.cache.layers)
    def finish(self):
        result=super().finish()
        result['singleton_stride_views']=self.singleton_views
        result['global_kv_representation_unchanged']=True
        return result
