"""Pack raw KV history once after refresh; preserve the native attention math."""

from __future__ import annotations
from ..compatibility import expected_digest
import hashlib
from contextlib import contextmanager
from pathlib import Path
import torch
from solar.backends.minimax_h3 import sgf_attention as native_attention
from solar.backends.minimax_h3 import sgf_rollout as native_rollout
from solar.backends.minimax_h3.torch_prope import prope_qkv_separate


NATIVE_SOURCE=Path(native_attention.__file__)
NATIVE_SHA=expected_digest("sgf_attention")
ROLLOUT_SOURCE=Path(native_rollout.__file__)
ROLLOUT_SHA=expected_digest("sgf_rollout")
BaseCache=native_attention.H3RawKVCache
H3SGFAttention=native_attention.H3SGFAttention
_slice_rotary=native_attention._slice_rotary
_cat_rotary=native_attention._cat_rotary


class PackedHistoryCache(BaseCache):
    """Native raw cache plus a derived next-chunk K/V concat sidecar."""

    def __init__(self):
        super().__init__()
        self.packed_history_by_layer={}

    def commit(self,layer,chunk_index,key,value):
        # Pack only in the private-window publication step, not twice during refresh.
        super().commit(layer,chunk_index,key,value)

    def get_packed_history(self,layer,chunk_index,empty_key,empty_value):
        entries=self.history(layer,chunk_index)
        if not entries:
            if layer in self.packed_history_by_layer:
                raise RuntimeError('packed KV exists for empty history')
            return empty_key,empty_value
        prepared=self.packed_history_by_layer.get(layer)
        if prepared is None or prepared[0]!=chunk_index:
            raise RuntimeError('packed KV is missing or stale for current chunk')
        key,value=prepared[1]
        return key,value

    def clear(self):
        super().clear()
        self.packed_history_by_layer.clear()


def candidate_sgf_attention(q,k,v,*,control:H3SGFAttention,layer:int,apply_rotary,attend):
    """Native SGF attention with only history K/V concat replaced by sidecar."""
    layout=control.layout
    condition_rows=int(layout.condition_indices.numel())
    prefix=condition_rows+int(layout.audio_indices.numel())
    rows=int(layout.rows_per_video_frame)
    chunk_rows=5*rows
    expected_video=100 if control.mode=='replay' else 5
    if q.shape[1]!=prefix+expected_video*rows:
        raise ValueError('H3 SGF packed query shape differs from its rollout/replay layout')
    if control.prefix_rotary is None:
        raise ValueError('H3 SGF rotaries must be prepared before attention')
    if control.mode not in {'rollout','replay'}:
        raise ValueError('H3 SGF attention mode must be rollout or replay')
    batch=q.shape[0]

    prefix_view=torch.eye(4,device=q.device,dtype=torch.float32).repeat(batch,prefix,1,1)
    prefix_K=torch.eye(3,device=q.device,dtype=torch.float32).repeat(batch,prefix,1,1)
    anchor_indices=layout.condition_video_indices
    prefix_view[:,anchor_indices]=control.viewmats[:,:1]
    prefix_K[:,anchor_indices]=control.intrinsics[:,:1]

    def camera_for_frames(start,stop):
        return (control.viewmats[:,1+start:1+stop].repeat_interleave(rows,dim=1),
                control.intrinsics[:,1+start:1+stop].repeat_interleave(rows,dim=1))

    def calculate(query,key,value,qr,kr,qcam,kcam):
        query=apply_rotary(query,*qr)
        key=apply_rotary(key,*kr)
        query,key,value,output_basis=prope_qkv_separate(
            query.transpose(1,2),key.transpose(1,2),value.transpose(1,2),
            query_viewmats=qcam[0],query_K=qcam[1],
            key_viewmats=kcam[0],key_K=kcam[1],
        )
        output=attend(query.transpose(1,2),key.transpose(1,2),value.transpose(1,2))
        return output_basis(output.transpose(1,2)).transpose(1,2)

    prefix_cam=(prefix_view,prefix_K)
    outputs=[calculate(q[:,:condition_rows],k[:,:condition_rows],v[:,:condition_rows],
                       _slice_rotary(control.prefix_rotary,0,condition_rows),
                       _slice_rotary(control.prefix_rotary,0,condition_rows),
                       (prefix_view[:,:condition_rows],prefix_K[:,:condition_rows]),
                       (prefix_view[:,:condition_rows],prefix_K[:,:condition_rows]))]
    if prefix>condition_rows:
        outputs.append(calculate(
            q[:,condition_rows:prefix],k[:,:prefix],v[:,:prefix],
            _slice_rotary(control.prefix_rotary,condition_rows,prefix),
            control.prefix_rotary,
            (prefix_view[:,condition_rows:prefix],prefix_K[:,condition_rows:prefix]),
            prefix_cam,
        ))
    chunk_specs=(
        [(False,i) for i in range(10)]+[(True,i) for i in range(10)]
        if control.mode=='replay' else [(True,control.chunk_index)]
    )
    for noisy,index in chunk_specs:
        first=max(0,index-5)
        local_frames=(index-first+1)*5
        if control.mode=='replay':
            qstart=prefix+(50*rows if noisy else 0)+index*chunk_rows
            clean_start=prefix+first*chunk_rows
            clean_stop=prefix+index*chunk_rows
            history_keys,history_values=(k[:,clean_start:clean_stop],v[:,clean_start:clean_stop])
        else:
            if control.cache is None:
                raise ValueError('H3 rollout requires an explicit raw KV cache')
            qstart=prefix
            history_keys,history_values=control.cache.get_packed_history(
                layer,index,k[:,:0],v[:,:0])
        qstop=qstart+chunk_rows
        key=torch.cat((k[:,:prefix],history_keys,k[:,qstart:qstop]),dim=1)
        value=torch.cat((v[:,:prefix],history_values,v[:,qstart:qstop]),dim=1)
        rotary=control.window_rotaries[local_frames]
        qrotary=_slice_rotary(rotary,(local_frames-5)*rows,local_frames*rows)
        krotary=_cat_rotary(control.prefix_rotary,rotary)
        current_camera=camera_for_frames(index*5,(index+1)*5)
        window_camera=camera_for_frames(first*5,(index+1)*5)
        key_camera=tuple(torch.cat((p,w),dim=1) for p,w in zip(prefix_cam,window_camera,strict=True))
        outputs.append(calculate(q[:,qstart:qstop],key,value,qrotary,krotary,
                                 current_camera,key_camera))
        if control.mode=='rollout' and control.commit_cache:
            control.cache.commit(layer,index,k[:,qstart:qstop],v[:,qstart:qstop])
    return torch.cat(outputs,dim=1)


@contextmanager
def installed_candidate():
    """Own and restore exactly the two H3 source-level effect ports."""
    if hashlib.sha256(NATIVE_SOURCE.read_bytes()).hexdigest()!=NATIVE_SHA:
        raise RuntimeError('native SGF attention source changed')
    if hashlib.sha256(ROLLOUT_SOURCE.read_bytes()).hexdigest()!=ROLLOUT_SHA:
        raise RuntimeError('native rollout source changed')
    old_attention=native_attention.sgf_attention
    old_cache=native_rollout.H3RawKVCache
    if old_cache is not BaseCache or old_attention.__module__!=native_attention.__name__:
        raise RuntimeError('SGF candidate ports already modified')
    native_attention.sgf_attention=candidate_sgf_attention
    native_rollout.H3RawKVCache=PackedHistoryCache
    try:
        yield
    finally:
        changed=(native_attention.sgf_attention is not candidate_sgf_attention
                 or native_rollout.H3RawKVCache is not PackedHistoryCache)
        native_attention.sgf_attention=old_attention
        native_rollout.H3RawKVCache=old_cache
        if changed:
            raise RuntimeError('candidate SGF ports were modified by another owner')
