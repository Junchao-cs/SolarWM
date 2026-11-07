"""Pure full-50-block query compute boundary with explicit cache tensors.

No compilation/Graph is performed by this module. Host owners must validate
request/weights/history/cameras and publish refresh exactly once. All derived
outputs are explicit return tensors, not captured Python cache side effects.
"""
from dataclasses import dataclass
import torch
from solar.backends.minimax_h3.memory_kv_attention import MemoryKVAttention
from solar.backends.minimax_h3.sgf_attention import _slice_rotary, _cat_rotary
from solar.backends.minimax_h3.torch_prope import logd4_relative_viewmats, _prepare_suffix_transforms
from deploy.solar_wm.backends.memory.acceleration.attention.memory_suffix import suffix

def matrices(c, reference):
    layout = c.layout
    rows = layout.rows_per_video_frame
    shared = int(layout.text_indices.numel()) + rows
    bank_end = shared + len(c.bank_indices) * rows
    prefix = bank_end + int(layout.audio_indices.numel())
    cameras = torch.eye(4, device=reference.device).repeat(reference.shape[0], prefix, 1, 1)
    intrinsics = torch.eye(3, device=reference.device).repeat(reference.shape[0], prefix, 1, 1)
    text = int(layout.text_indices.numel())
    cameras[:, text:shared] = c.viewmats[:, :1]
    intrinsics[:, text:shared] = c.intrinsics[:, :1]
    ids = torch.tensor(c.bank_indices, device=reference.device) + 1
    cameras[:, shared:bank_end] = c.viewmats.index_select(1, ids).repeat_interleave(rows, 1)
    intrinsics[:, shared:bank_end] = c.intrinsics.index_select(1, ids).repeat_interleave(rows, 1)
    first = max(0, c.chunk_index - 5) * 5
    stop = (c.chunk_index + 1) * 5
    wv = c.viewmats[:, 1 + first:1 + stop].repeat_interleave(rows, 1)
    wk = c.intrinsics[:, 1 + first:1 + stop].repeat_interleave(rows, 1)
    (aq, _, ao) = _prepare_suffix_transforms(viewmats=logd4_relative_viewmats(wv[:, -5 * rows:].to(reference.dtype)), Ks=wk[:, -5 * rows:].to(reference.dtype))
    (_, ak, _) = _prepare_suffix_transforms(viewmats=logd4_relative_viewmats(torch.cat((cameras, wv), 1).to(reference.dtype)), Ks=torch.cat((intrinsics, wk), 1).to(reference.dtype))
    return (aq.keywords['matrix'], ak.keywords['matrix'], ao.keywords['matrix'])

class Computation:

    def __init__(self, prefix, fixed, geometry):
        self.prefix = prefix
        self.fixed = fixed
        self.geometry = geometry
        self.derived = []
        self.raw = []

@dataclass(frozen=True)
class FunctionalQueryControl(MemoryKVAttention):
    computation: object = None

    def prepare_rotaries(self, rope):
        if self.prefix_rotary is None or self.window_rotary is None:
            raise RuntimeError('prepare rotaries outside core')
        return self

    def attend(self, q, k, v, *, layer, apply_rotary, attend):
        s = self.computation
        (qm, km, om) = s.geometry
        (pk, pv) = s.prefix[layer]
        history = self.cache.history(layer, self.chunk_index)
        hrows = sum((x[1].shape[1] for x in history))
        qrows = k.shape[1]
        cr = _slice_rotary(self.window_rotary, hrows, hrows + qrows)
        fixed_rows = pk.shape[1] + hrows
        if s.fixed is None:
            hk = torch.cat([x[1] for x in history], 1)
            hv = torch.cat([x[2] for x in history], 1)
            keys = torch.cat((pk, hk, k), 1)
            values = torch.cat((pv, hv, v), 1)
            kr = apply_rotary(keys, *_cat_rotary(self.prefix_rotary, _slice_rotary(self.window_rotary, 0, hrows), cr))
            kr = suffix(kr.transpose(1, 2), km)
            values = suffix(values.transpose(1, 2), km)
            s.derived.append((kr[:, :, :fixed_rows].clone(), values[:, :, :fixed_rows].clone()))
        else:
            (fk, fv) = s.fixed[layer]
            kr = suffix(apply_rotary(k, *cr).transpose(1, 2), km[:, fixed_rows:])
            values = suffix(v.transpose(1, 2), km[:, fixed_rows:])
            kr = torch.cat((fk, kr), 2)
            values = torch.cat((fv, values), 2)
        qr = suffix(apply_rotary(q, *cr).transpose(1, 2), qm)
        output = attend(qr.transpose(1, 2), kr.transpose(1, 2), values.transpose(1, 2))
        if self.commit_cache:
            s.raw.append((k, v))
        return suffix(output.transpose(1, 2), om).transpose(1, 2)

def make_control(source, computation):
    return FunctionalQueryControl(source.layout, source.viewmats, source.intrinsics, source.bank_indices, source.chunk_index, source.cache, source.commit_cache, source.prefix_rotary, source.window_rotary, computation)

class FullQuery:

    def __init__(self, base, rank, packed_dtype):
        if len(base.transformer_blocks) != 50 or not 0 <= rank < 8:
            raise RuntimeError('full50/SP8 required')
        self.base = base
        self.rank = rank
        self.packed_dtype = packed_dtype

    def __call__(self, hidden_states, timestep, timestep_indices, token_tags, source, prefix_kv, fixed_kv, prefix_video, prefix_audio, geometry):
        from solar.backends.minimax_h3 import distributed as d
        from solar.backends.minimax_h3.model import H3AttentionControl, get_parameter_dtype, MINIMAX_H3_MODALITY_NUM
        rows = 5 * source.layout.rows_per_video_frame
        (lo, hi) = d.contiguous_packed_bounds(rows, sp_size=8, sp_rank=self.rank)
        lengths = (rows // 8,) * 8
        hidden = self.base.proj_in(hidden_states[:, -rows:].to(get_parameter_dtype(self.base.proj_in)))[:, lo:hi].contiguous().to(self.packed_dtype)
        temb = self.base._time_condition(timestep, None, parameter_dtype=get_parameter_dtype)
        prefix = source.layout.sequence_length - rows
        indices = timestep_indices[prefix + lo:prefix + hi].contiguous()
        tags = token_tags[prefix + lo:prefix + hi].contiguous()
        adaln = indices * MINIMAX_H3_MODALITY_NUM + tags
        if geometry is None:
            geometry = matrices(source, prefix_kv[0][0])
        state = Computation(prefix_kv, fixed_kv, geometry)
        control = make_control(source, state)
        attention = H3AttentionControl(None, True, None, None, None, None, None, control, lengths)
        for block in self.base.transformer_blocks:
            hidden = block(hidden, temb, adaln, source.window_rotary, attention)
        hidden = self.base.norm_out(hidden, temb, indices).to(get_parameter_dtype(self.base.proj_out))
        video = d.sequence_all_gather(self.base.proj_out(hidden), dim=1)
        return (torch.cat((prefix_video, video), 1), prefix_audio.clone(), tuple(state.derived), tuple(state.raw), geometry)
