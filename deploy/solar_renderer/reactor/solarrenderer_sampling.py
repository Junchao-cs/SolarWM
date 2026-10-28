"""Incremental form of SolarRenderer's released four-evaluation causal sampler.

Adapted from Enderfga/SolarRenderer, Apache-2.0, omni_causal.py at
a2371eaecff9a8ab378b5ec36199c5863d138074. Keep the native random draws,
six-block structure window and five-block clean history unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from solar.backends.minimax_h3.flow import make_shifted_schedule
from solar.backends.minimax_h3.layout import patchify_video, unpatchify_video
from solar.renderer.causal import (
    CLEAN_LEVEL,
    STATIC_TIME,
    H3ChunkKVCache,
    H3SegmentPlan,
    _merge,
    _remap,
    _window,
    select_rows,
)


class SolarCausalRun:
    """One finite native rollout; each step samples and commits one latent block."""

    def __init__(self, core, batch, generator, *, cache_structure=False):
        self.core = core
        self.generator = generator
        self.cache_structure = cache_structure
        self.schedule = make_shifted_schedule(5, shift=12.0)
        self.window = int(core.causal.get("window", 6))
        (
            self.prompt,
            self.layout,
            self.observations,
            self.rows,
            (self.frames, self.grid),
        ) = core._document(batch, generator)
        self.audio = core._audio(self.frames, generator)
        self.cache = H3ChunkKVCache(window=self.window)
        self.index = 0

    @property
    def complete(self):
        return self.index == len(self.rows.frames)

    def _draw(self, shape):
        value = torch.randn(
            shape,
            generator=self.generator,
            device=self.core.device,
            dtype=torch.float32,
        )
        self.core._broadcast_logical_sample(value)
        return value

    def _document(self, target, extra):
        rows = self.rows
        source = self.layout.source
        video_rows = source.video_indices[: int(source.num_condition_video_rows)]
        keep = np.concatenate(
            [np.arange(a, b) for a, b in sorted([*rows.static, rows.audio, *extra, target])]
        )
        where = np.full(rows.sequence_length, -1, dtype=np.int64)
        where[keep] = np.arange(keep.size)
        kept_video = np.asarray([r for r in video_rows if where[r] >= 0], dtype=np.int64)
        pick = torch.from_numpy(np.searchsorted(video_rows, kept_video)).to(self.core.device)
        times = np.full(int(keep.size), CLEAN_LEVEL, dtype=np.float32)
        text = _remap(rows.text, where)
        times[text[0] : text[1]] = STATIC_TIME
        return select_rows(self.layout, keep), where, pick, _remap(target, where), times

    def _plan(self, where, mine, size, keys, cached, commit, extra):
        static = [_remap(s, where) for s in self.rows.static]
        audio = _remap(self.rows.audio, where)
        pieces = [(span, static) for span in static] + [(audio, [*static, audio])]
        pieces += [(_remap(e, where), [*static, _remap(e, where)]) for e in extra]
        segments = [(a, b, _merge(k), (), None) for (a, b), k in sorted(pieces) if b > a]
        segments.append((mine[0], mine[1], _merge([*static, *keys, mine]), cached, commit))
        segments.sort()
        return H3SegmentPlan(tuple(segments), size, self.cache)

    @torch.no_grad()
    def step(self):
        if self.complete:
            raise RuntimeError("The native rollout is complete; start a new run.")
        c = self.index
        f0, f1 = self.rows.frames[c]
        history = tuple(
            ("structure", j)
            for j in _window(c, self.window)
            if self.rows.structure and self.cache_structure
        ) + tuple(("clean", j) for j in _window(c, self.window) if j < c)
        extra = (
            [self.rows.structure[j] for j in _window(c, self.window)]
            if self.rows.structure and not self.cache_structure
            else []
        )
        if self.rows.structure and self.cache_structure:
            # Native structure rows attend only to static rows and themselves.
            # Commit once without changing the six-block window or RNG draws.
            arrays, where, pick, mine, times = self._document(self.rows.structure[c], [])
            self.core._run(
                video_rows=self.observations.index_select(1, pick),
                audio_rows=self.audio,
                prompt=self.prompt,
                arrays=arrays,
                times=times,
                plan=self._plan(where, mine, len(times), [], (), ("structure", c), []),
                model=None,
            )
        arrays, where, pick, mine, times = self._document(self.rows.target[c], extra)
        condition = self.observations.index_select(1, pick)
        audio_span = [
            _remap(self.rows.audio, where),
            *(_remap(e, where) for e in extra),
        ]
        shape = (1, 24, f1 - f0, *self.grid)
        current = self._draw(shape)
        for index in range(4):
            sigma, sigma_next = (
                float(self.schedule.sigmas[index]),
                float(self.schedule.sigmas[index + 1]),
            )
            times[mine[0] : mine[1]] = 1.0 - sigma
            output = self.core._run(
                video_rows=torch.cat((condition, patchify_video(current)), dim=1),
                audio_rows=self.audio,
                prompt=self.prompt,
                arrays=arrays,
                times=times.copy(),
                plan=self._plan(where, mine, len(times), audio_span, history, None, extra),
                model=None,
            )
            velocity = unpatchify_video(
                output[:, int(condition.shape[1]) :], f1 - f0, *self.grid, channels=24
            ).float()
            if index == 3:
                current = current + sigma * velocity
                break
            current = (1.0 - sigma_next) * (current + sigma * velocity) + sigma_next * self._draw(
                shape
            )
        jitter = self._draw(shape)
        times[mine[0] : mine[1]] = CLEAN_LEVEL
        self.core._run(
            video_rows=torch.cat(
                (
                    condition,
                    patchify_video(CLEAN_LEVEL * current + (1 - CLEAN_LEVEL) * jitter),
                ),
                dim=1,
            ),
            audio_rows=self.audio,
            prompt=self.prompt,
            arrays=arrays,
            times=times.copy(),
            plan=self._plan(where, mine, len(times), audio_span, history, ("clean", c), extra),
            model=None,
        )
        self.index += 1
        return current
