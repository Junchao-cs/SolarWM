"""Inference-only caches and FA4 for the native four-step causal schedule."""

import numpy as np
import torch
from solarrenderer_sampling import SolarCausalRun
from torch import nn

from solarwm.renderer.causal import (
    CLEAN_LEVEL,
    STATIC_TIME,
    H3SegmentPlan,
    _merge,
    _remap,
    select_rows,
)


class ExactInputCache(nn.Module):
    """One-entry cache, invalidated whenever the actual input changes."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.saved_input = None
        self.saved_output = None

    def forward(self, value):
        if value.shape[1] == 0:
            return value
        if torch.is_grad_enabled():
            return self.module(value)
        if self.saved_input is None or not torch.equal(value, self.saved_input):
            self.saved_output = self.module(value)
            self.saved_input = value.detach().clone()
        return self.saved_output


def _transformer(core):
    """Support merged native weights and the earlier PEFT-wrapped renderer."""
    unwrap = getattr(core.model, "get_base_model", None)
    return unwrap() if callable(unwrap) else core.model


def cache_text_refiner(core):
    model = _transformer(core)
    model.token_refiner = ExactInputCache(model.token_refiner)


class ModulationTable(nn.Module):
    def __init__(self, tables, cursor):
        super().__init__()
        self.tables = tables
        self.cursor = cursor

    def forward(self, temb):
        if torch.is_grad_enabled():
            raise RuntimeError("Precomputed modulation is inference-only")
        return self.tables[self.cursor[0]]


@torch.no_grad()
def precompute_adaln(core):
    from solarwm.backends.minimax_h3.flow import make_shifted_schedule
    from solarwm.backends.minimax_h3.model import get_parameter_dtype
    from solarwm.renderer.causal import CLEAN_LEVEL, STATIC_TIME

    model = _transformer(core)
    schedule = make_shifted_schedule(5, shift=12.0)
    # A reference-free static prefill contains text rows only (all STATIC_TIME).
    keys = [(float(np.float32(STATIC_TIME)),)]
    for level in [CLEAN_LEVEL, *(1 - float(s) for s in schedule.sigmas[:4])]:
        key = tuple(
            np.unique(np.asarray([STATIC_TIME, CLEAN_LEVEL, level], dtype=np.float32)).tolist()
        )
        if key not in keys:
            keys.append(key)
        # Static-prefix reuse removes text rows from later forwards.
        key = tuple(np.unique(np.asarray([CLEAN_LEVEL, level], dtype=np.float32)).tolist())
        if key not in keys:
            keys.append(key)
    embeddings = [
        model._time_condition(
            torch.tensor(k, device=core.device, dtype=torch.float32),
            None,
            parameter_dtype=get_parameter_dtype,
        )
        for k in keys
    ]
    cursor = [0]
    freed = 0
    for block in model.transformer_blocks:
        projection = block.adaln_proj
        # Each call has the native shape and preserves the original precision boundaries.
        tables = [tuple(x.detach().clone() for x in projection(e)) for e in embeddings]
        freed += sum(p.numel() * p.element_size() for p in projection.parameters())
        block.adaln_proj = ModulationTable(tables, cursor)
    original = core._run

    def run(**kw):
        key = tuple(np.unique(kw["times"]).tolist())
        if key not in keys:
            raise ValueError("Unprepared timestep set; refuse stale modulation")
        cursor[0] = keys.index(key)
        return original(**kw)

    core._run = run
    del projection, embeddings
    torch.cuda.empty_cache()
    return {"adaln_weight_bytes_removed": freed, "time_sets": keys}


def install_fa4():
    """Swap only the causal segmented attention kernel; keep original plan/cache."""
    from flash_attn.cute.interface import flash_attn_func

    from solarwm.backends.minimax_h3 import model as h3_model

    def attention(query, key, value, plan, layer):
        if int(query.shape[1]) != plan.sequence_length:
            raise ValueError("Segment sequence length mismatch")
        outputs = []
        for start, stop, spans, cached, commit in plan.segments:
            keys = [key[:, a:b] for a, b in spans]
            values = [value[:, a:b] for a, b in spans]
            for chunk in cached:
                k, v = plan.cache.get(layer, chunk)
                if chunk[0] == "prefix":
                    keys.insert(0, k)
                    values.insert(0, v)
                else:
                    keys.append(k)
                    values.append(v)
            output = flash_attn_func(
                query[:, start:stop].contiguous(),
                torch.cat(keys, dim=1).contiguous(),
                torch.cat(values, dim=1).contiguous(),
                causal=False,
                num_splits=1,
            )
            outputs.append(output[0] if isinstance(output, tuple) else output)
            if commit is not None:
                plan.cache.put(layer, commit, key[:, start:stop], value[:, start:stop])
        return torch.cat(outputs, dim=1)

    h3_model.segment_attention = attention


class StaticPrefixRun(SolarCausalRun):
    """Cache text/reference KV once; they attend only to each other in native plan."""

    @torch.no_grad()
    def __init__(self, core, batch, generator, **kwargs):
        super().__init__(core, batch, generator, **kwargs)
        keep = np.concatenate([np.arange(a, b) for a, b in sorted(self.rows.static)])
        arrays = select_rows(self.layout, keep)
        cond_rows = self.layout.source.video_indices[
            : int(self.layout.source.num_condition_video_rows)
        ]
        kept = cond_rows[np.isin(cond_rows, keep)]
        pick = torch.from_numpy(np.searchsorted(cond_rows, kept)).to(core.device)
        times = np.full(len(keep), CLEAN_LEVEL, dtype=np.float32)
        times[: self.rows.text[1] - self.rows.text[0]] = STATIC_TIME
        plan = H3SegmentPlan(
            ((0, len(keep), ((0, len(keep)),), (), ("prefix", 0)),),
            len(keep),
            self.cache,
        )
        core._run(
            video_rows=self.observations.index_select(1, pick),
            audio_rows=self.audio[:, :0],
            prompt=self.prompt,
            arrays=arrays,
            times=times,
            plan=plan,
            model=None,
        )
        self.prompt = self.prompt[:, :0]

    def _document(self, target, extra):
        keep = np.concatenate(
            [np.arange(a, b) for a, b in sorted([self.rows.audio, *extra, target])]
        )
        where = np.full(self.rows.sequence_length, -1, dtype=np.int64)
        where[keep] = np.arange(len(keep))
        source = self.layout.source
        video_rows = source.video_indices[: int(source.num_condition_video_rows)]
        kept = video_rows[where[video_rows] >= 0]
        pick = torch.from_numpy(np.searchsorted(video_rows, kept)).to(self.core.device)
        times = np.full(len(keep), CLEAN_LEVEL, dtype=np.float32)
        return select_rows(self.layout, keep), where, pick, _remap(target, where), times

    def _plan(self, where, mine, size, keys, cached, commit, extra):
        audio = _remap(self.rows.audio, where)
        pieces = [(audio, [audio])] + [(_remap(e, where), [_remap(e, where)]) for e in extra]
        segments = [
            (a, b, _merge(k), (("prefix", 0),), None) for (a, b), k in sorted(pieces) if b > a
        ]
        segments.append((mine[0], mine[1], _merge([*keys, mine]), (("prefix", 0), *cached), commit))
        return H3SegmentPlan(tuple(sorted(segments)), size, self.cache)
