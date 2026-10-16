"""Block-causal omni rendering: the structure-conditioned renderer generated chunk by chunk.

A clip of ``5n+2`` latent frames is cut into chunks of at most ``chunk`` latents (5: 0-4, 5-9, ...,
45-46). The
packed document is the ordinary omni one - ``[text | first frame | structure | audio | target]``,
the structure laid
over the target frames on their rotary times - and a training document appends a clean copy of the
target (teacher
forcing). Who sees whom:

* text and the first frame (the static rows) see the static rows; the audio (silence) sees them and
itself;
* structure chunk ``c`` sees the static rows and itself - its rows never depend on another chunk,
so they are the
  same in training and in a stream that holds only the last ``window`` chunks;
* noisy chunk ``c`` sees the static rows, the audio, the structure of chunks ``c-window+1 .. c``,
the clean frames of
  chunks ``c-window+1 .. c-1`` and itself;
* clean chunk ``c`` sees what noisy chunk ``c`` sees, its own clean frames in place of its noisy
ones - exactly what a
  stream computes when it writes chunk ``c`` to its key/value cache.

Nothing sees a later chunk, of video or of structure. Diffusion forcing drops the clean copy: noisy
chunk ``c`` sees
the noisy chunks ``c-window+1 .. c``, each at its own noise level.

Attention runs per query segment over exactly its visible keys (flash attention, no token-level
mask): the plan is a
list of ``(query rows, key spans, cached chunks, chunk to commit)``. A streaming sampler keeps
every layer's keys and
values of committed clean chunks in :class:`H3ChunkKVCache` and runs one chunk at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from solarwm.backends.minimax_h3.layout import H3PackedLayout
from solarwm.backends.minimax_h3.segmented_attention import (
    H3ChunkKVCache,
    H3SegmentPlan,
    segment_attention,
)

STATIC_TIME = 1.0
CLEAN_LEVEL = 0.999


def _merge(spans: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    out: list[list[int]] = []
    for start, stop in sorted(s for s in spans if s[1] > s[0]):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], stop)
        else:
            out.append([start, stop])
    return tuple(((a, b) for a, b in out))


def chunk_frames(frames: int, chunk: int) -> list[tuple[int, int]]:
    """Latent frame ranges of the chunks of a ``frames``-latent clip, ``chunk`` at most each."""
    if not 0 < int(chunk) <= 5:
        raise ValueError(f"a chunk holds 1 to 5 latents, got {chunk}")
    return [(f, min(f + int(chunk), int(frames))) for f in range(0, int(frames), int(chunk))]


@dataclass(frozen=True)
class H3CausalRows:
    """Row ranges of an omni document (``[text | condition video | audio | target]``) by role
    and chunk."""

    sequence_length: int
    text: tuple[int, int]
    static: tuple[tuple[int, int], ...]
    structure: tuple[tuple[int, int], ...]
    audio: tuple[int, int]
    target: tuple[tuple[int, int], ...]
    frames: tuple[tuple[int, int], ...]
    rows_per_frame: int

    @property
    def chunks(self) -> int:
        return len(self.target)


def causal_rows(layout: H3PackedLayout, structure_index: int | None, chunk: int) -> H3CausalRows:
    """Rows of ``layout`` (a Stage0.5 omni layout) by chunk; ``structure_index`` is the
    structure video's place among
    the layout's visual observations (None: no structure video)."""
    if layout.stage != "stage0p5" or layout.num_clean_video_rows:
        raise ValueError("causal rows read an ordinary omni (Stage0.5) layout")
    rows = int(layout.rows_per_video_frame)
    frames = int(layout.num_noisy_video_rows) // rows
    ranges = chunk_frames(frames, chunk)
    text = (0, int(layout.text_indices.size))
    condition = (text[1], text[1] + int(layout.num_condition_video_rows))
    audio = (condition[1], condition[1] + int(layout.audio_indices.size))
    target_start = audio[1]
    if (
        int(layout.target_video_indices[0]) != target_start
        or int(layout.target_video_indices[-1]) != layout.sequence_length - 1
    ):
        raise ValueError("the omni layout's target rows must close the document")
    observation_start = condition[1] - int(layout.num_observation_rows)
    static = [text, (condition[0], observation_start)]
    structure: list[tuple[int, int]] = []
    offset = observation_start
    for index, count in enumerate(layout.observation_rows):
        if index == structure_index:
            if count % frames:
                raise ValueError(
                    "the structure video must hold one equal slab of rows per target frame"
                )
            per = count // frames
            structure = [(offset + a * per, offset + b * per) for a, b in ranges]
        else:
            static.append((offset, offset + count))
        offset += count
    target = tuple(((target_start + a * rows, target_start + b * rows) for a, b in ranges))
    return H3CausalRows(
        sequence_length=int(layout.sequence_length),
        text=text,
        static=_merge(static),
        structure=tuple(structure),
        audio=audio,
        target=target,
        frames=tuple(ranges),
        rows_per_frame=rows,
    )


def _window(c: int, window: int) -> range:
    return range(max(0, c - int(window) + 1), c + 1)


def training_plan(
    rows: H3CausalRows, *, window: int, teacher: bool
) -> tuple[H3SegmentPlan, tuple[int, int]]:
    """The plan of a training document: teacher forcing appends a clean copy of the target
    after it (returned as its
    row range), diffusion forcing has none."""
    n = rows.sequence_length
    target_rows = rows.target[-1][1] - rows.target[0][0]
    clean = tuple(((a + target_rows, b + target_rows) for a, b in rows.target)) if teacher else ()
    static = list(rows.static)
    segments = []

    def add(span: tuple[int, int], keys: list[tuple[int, int]]) -> None:
        if span[1] > span[0]:
            segments.append((span[0], span[1], _merge(keys), (), None))

    pieces = [(s, "static", None) for s in rows.static] + [
        (s, "structure", c) for c, s in enumerate(rows.structure)
    ]
    for span, kind, _chunk in sorted(pieces):
        add(span, static if kind == "static" else [*static, span])
    add(rows.audio, [*static, rows.audio])
    for c, span in enumerate(rows.target):
        seen = [
            *static,
            rows.audio,
            *(rows.structure[j] for j in _window(c, window) if rows.structure),
        ]
        if teacher:
            add(span, [*seen, *(clean[j] for j in _window(c, window) if j < c), span])
        else:
            add(span, [*seen, *(rows.target[j] for j in _window(c, window))])
    for c, span in enumerate(clean):
        seen = [
            *static,
            rows.audio,
            *(rows.structure[j] for j in _window(c, window) if rows.structure),
        ]
        add(span, [*seen, *(clean[j] for j in _window(c, window))])
    end = n + (target_rows if teacher else 0)
    return (H3SegmentPlan(tuple(segments), end), (n, end) if teacher else (0, 0))


def with_clean_copy(layout: Any) -> dict[str, Any]:
    """The model's index arrays for ``layout`` followed by a copy of its target rows (same
    positions and tags): video
    rows ordered condition, target, clean copy."""
    import torch

    target = layout.source.target_video_indices
    n = int(layout.source.sequence_length)
    copy = torch.arange(n, n + int(target.size), device=layout.position_ids.device)
    target_t = torch.from_numpy(target).to(layout.position_ids.device)
    return {
        "position_ids": torch.cat((layout.position_ids, layout.position_ids[target_t])),
        "token_tags": torch.cat((layout.token_tags, layout.token_tags[target_t])),
        "video_indices": torch.cat((layout.video_indices, copy)),
        "audio_indices": layout.audio_indices,
        "text_indices": layout.text_indices,
    }


def select_rows(layout: Any, keep: np.ndarray) -> dict[str, Any]:
    """The model's index arrays for the rows ``keep`` (ascending) of ``layout``'s document,
    renumbered; video rows keep
    their document order, which is the order the caller packs them in."""
    import torch

    device = layout.position_ids.device
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size == 0 or np.any(np.diff(keep) <= 0):
        raise ValueError("rows to keep must be strictly ascending")
    where = np.full(int(layout.source.sequence_length), -1, dtype=np.int64)
    where[keep] = np.arange(keep.size)

    def renumber(indices: np.ndarray) -> Any:
        mapped = where[indices]
        return torch.from_numpy(mapped[mapped >= 0]).to(device)

    keep_t = torch.from_numpy(keep).to(device)
    out = {
        "position_ids": layout.position_ids[keep_t],
        "token_tags": layout.token_tags[keep_t],
        "video_indices": renumber(layout.source.video_indices),
        "audio_indices": renumber(layout.source.audio_indices),
        "text_indices": renumber(layout.source.text_indices),
    }
    if sum(int(out[k].numel()) for k in ("video_indices", "audio_indices", "text_indices")) != int(
        keep.size
    ):
        raise RuntimeError("selected rows lost their text/video/audio bookkeeping")
    return out


def row_times(
    length: int,
    rows: H3CausalRows,
    *,
    target_times: list[float],
    audio: float,
    clean: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Per-row data-ward times of a document of ``length`` rows: static text fixed,
    observations and the clean copy
    at the condition level, audio at ``audio``, target chunk ``c`` at ``target_times[c]``."""
    times = np.full(length, CLEAN_LEVEL, dtype=np.float32)
    times[rows.text[0] : rows.text[1]] = STATIC_TIME
    times[rows.audio[0] : rows.audio[1]] = audio
    for (a, b), t in zip(rows.target, target_times, strict=True):
        times[a:b] = t
    times[clean[0] : clean[1]] = CLEAN_LEVEL
    return times


def unique_times(times: np.ndarray, device: Any) -> tuple[Any, Any]:
    import torch

    values, inverse = np.unique(times, return_inverse=True)
    return (
        torch.from_numpy(values.astype(np.float32)).to(device),
        torch.from_numpy(inverse.reshape(-1).astype(np.int64)).to(device),
    )


__all__ = [
    "H3CausalRows",
    "H3ChunkKVCache",
    "H3SegmentPlan",
    "causal_rows",
    "chunk_frames",
    "row_times",
    "segment_attention",
    "select_rows",
    "training_plan",
    "unique_times",
    "with_clean_copy",
]


def _structure_index(batch: Any) -> int | None:
    """The structure video's place among the batch's observations: the one laid over every
    target frame."""
    frames = int(batch.target_latents.shape[1]) if batch.target_latents is not None else None
    found = [
        i
        for i, item in enumerate(batch.visual_observations or ())
        if item.frames is not None
        and frames is not None
        and (int(item.latents.shape[1]) == len(item.frames))
        and (set(range(frames)) <= {int(f) for f in item.frames})
    ]
    if len(found) > 1:
        raise ValueError("an omni sample carries one structure video")
    return found[0] if found else None


def _remap(span: tuple[int, int], where: np.ndarray) -> tuple[int, int]:
    return (int(where[span[0]]), int(where[span[1] - 1]) + 1)


class H3OmniCausalMixin:
    """Block-causal omni training and streaming on an :class:`~.stage0p5.H3Stage0p5Core`
    (``train.causal_omni``)."""

    causal: dict

    def _document(self, batch: Any, generator: Any) -> tuple[Any, Any, Any, H3CausalRows, Any]:
        import torch

        prompt, tags = self.prompt(batch, use_anchor=False)
        for item in batch.visual_observations or ():
            if abs(float(item.degree) - CLEAN_LEVEL) > 1e-06:
                raise ValueError("block-causal omni reads observations at the condition level")
        specs, _degrees, observation_rows = self.observation_rows(batch, generator)
        frames, grid = (self.latent_frames(batch), self.latent_grid(batch))
        layout = self.layout(tags, anchors=(), observations=specs, grid=grid, frames=frames)
        rows = causal_rows(layout.source, _structure_index(batch), int(self.causal.get("chunk", 5)))
        if observation_rows is None:
            observation_rows = torch.zeros((1, 0, 96), device=self.device)
        return (prompt, layout, observation_rows, rows, (frames, grid))

    def _audio(self, frames: int, generator: Any) -> Any:
        import torch

        clean = self.audio_rows(frames)
        noise = torch.randn(
            clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        self._broadcast_logical_sample(noise)
        return CLEAN_LEVEL * clean + (1.0 - CLEAN_LEVEL) * noise

    def _run(
        self, *, video_rows, audio_rows, prompt, arrays, times, plan, model: Any = None
    ) -> Any:
        from solarwm.backends.minimax_h3.distributed import is_sequence_parallel_enabled

        values, inverse = unique_times(times, self.device)
        video, _audio = (self.model if model is None else model)(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=prompt,
            timestep=values,
            timestep_indices=inverse,
            token_tags=arrays["token_tags"],
            position_ids=arrays["position_ids"],
            video_indices=arrays["video_indices"],
            audio_indices=arrays["audio_indices"],
            text_indices=arrays["text_indices"],
            attention_mask=plan,
            fused_prope=False,
            stage0p5_sequence_parallel=is_sequence_parallel_enabled(),
            return_dict=False,
        )
        return video

    def forward_loss(self, batch: Any, *, noise_seed: int | None) -> Any:
        """Flow matching on every chunk, each at its own noise level, under the
        block-causal plan: teacher forcing
        (``mode: teacher``, the clean target as history, optionally corrupted) or diffusion
        forcing (``diffusion``)."""
        import torch
        import torch.nn.functional as F

        from solarwm.backends.minimax_h3.layout import patchify_video

        generator = (
            None
            if noise_seed is None
            else torch.Generator(device=self.device).manual_seed(int(noise_seed))
        )
        teacher = str(self.causal.get("mode", "teacher")) == "teacher"
        document = self._document(batch, generator)
        _prompt, _layout, _observations, rows, (frames, _grid) = document
        clean = batch.target_latents.to(self.device, dtype=torch.float32).unsqueeze(0)
        noise = torch.randn(
            clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        draws = [
            self.shifted_timestep(generator, shift=12.0, device=self.device) for _ in rows.frames
        ]
        self._broadcast_logical_sample(clean, noise, *draws)
        level = torch.ones((1, 1, frames, 1, 1), device=self.device)
        for (a, b), t in zip(rows.frames, draws, strict=True):
            level[:, :, a:b] = t
        noisy = level * clean + (1.0 - level) * noise
        history = None
        if teacher:
            history = clean
            corrupt = float(self.causal.get("corrupt", 0.0))
            if corrupt > 0:
                mix = (
                    torch.rand(len(rows.frames), generator=generator, device=self.device) * corrupt
                )
                pick = torch.rand(len(rows.frames), generator=generator, device=self.device) < 0.5
                extra = torch.randn(
                    clean.shape, generator=generator, device=self.device, dtype=torch.float32
                )
                self._broadcast_logical_sample(mix, pick, extra)
                amount = torch.zeros((1, 1, frames, 1, 1), device=self.device)
                for (a, b), m, p in zip(rows.frames, mix, pick, strict=True):
                    amount[:, :, a:b] = m * p.float()
                history = (1.0 - amount) * clean + amount * extra
        prediction = self.training_prediction(
            document, noisy, [float(t) for t in draws], history, generator=generator
        )
        target = patchify_video(clean - noise)
        return F.mse_loss(prediction.float(), target.float())

    def training_prediction(
        self,
        document: Any,
        noisy: Any,
        times: list[float],
        history: Any,
        *,
        generator: Any = None,
        jitter: Any = None,
        model: Any = None,
    ) -> Any:
        """The velocity rows of every target chunk of a training document: ``noisy`` the
        target with chunk ``c`` at
        time ``times[c]``, ``history`` the clean copy's content (teacher forcing) or None
        (diffusion forcing)."""
        import torch

        from solarwm.backends.minimax_h3.layout import patchify_video

        prompt, layout, observation_rows, rows, (frames, _grid) = document
        teacher = history is not None
        parts = [observation_rows, patchify_video(noisy)]
        plan, clean_span = training_plan(
            rows, window=int(self.causal.get("window", 6)), teacher=teacher
        )
        if teacher:
            if jitter is None:
                jitter = torch.randn(
                    history.shape, generator=generator, device=self.device, dtype=torch.float32
                )
                self._broadcast_logical_sample(jitter)
            parts.append(patchify_video(CLEAN_LEVEL * history + (1.0 - CLEAN_LEVEL) * jitter))
            arrays = with_clean_copy(layout)
        else:
            arrays = {
                "position_ids": layout.position_ids,
                "token_tags": layout.token_tags,
                "video_indices": layout.video_indices,
                "audio_indices": layout.audio_indices,
                "text_indices": layout.text_indices,
            }
        row_t = row_times(
            plan.sequence_length, rows, target_times=times, audio=CLEAN_LEVEL, clean=clean_span
        )
        video = self._run(
            video_rows=torch.cat(parts, dim=1),
            audio_rows=self._audio(frames, generator),
            prompt=prompt,
            arrays=arrays,
            times=row_t,
            plan=plan,
            model=model,
        )
        start = int(observation_rows.shape[1])
        return video[:, start : start + int(patchify_video(noisy).shape[1])]

    def generate_causal(
        self,
        batch: Any,
        *,
        noise_seed: int,
        steps: int,
        window: int | None = None,
        record: list | None = None,
        cache_structure: bool = False,
    ) -> Any:
        """The clip chunk by chunk, as a stream renders it: each chunk denoised over
        ``steps`` schedule points (the
        shifted grid, ``steps - 1`` Euler forwards) against the cached keys and values of
        the chunks before it, then
        written to the cache at the condition level. Returns ``[1, 24, frames, h, w]`` latents."""
        import torch

        from solarwm.backends.minimax_h3.flow import make_shifted_schedule

        generator = torch.Generator(device=self.device).manual_seed(int(noise_seed))
        clip, _noisy, _times, _jitter = self.stream(
            batch,
            generator=generator,
            schedule=make_shifted_schedule(int(steps), shift=12.0),
            window=window,
            record=record,
            cache_structure=cache_structure,
        )
        return clip

    def stream(
        self,
        batch: Any,
        *,
        generator: Any,
        schedule: Any,
        window: int | None = None,
        exit_index: int | None = None,
        model: Any = None,
        record: list | None = None,
        cache_structure: bool = True,
    ) -> tuple[Any, Any, list[float], Any]:
        """Render ``batch`` chunk by chunk through a key/value cache. Without
        ``exit_index``: Euler over ``schedule``.
        With it (a few-step student, as DMD trains one): every chunk's clean estimate is
        re-noised to the next level
        between evaluations and taken at evaluation ``exit_index``. Returns the clean clip,
        the input each chunk had at
        its last evaluation, that evaluation's time per chunk, and the noise each committed
        chunk was written with -
        everything a replay of the same chunks needs (:meth:`training_prediction`)."""
        import torch

        from solarwm.backends.minimax_h3.layout import patchify_video, unpatchify_video

        window = int(window or self.causal.get("window", 6))
        prompt, layout, observation_rows, rows, (frames, grid) = self._document(batch, generator)
        audio = self._audio(frames, generator)
        cache = H3ChunkKVCache(window=window)
        source = layout.source
        n_cond = int(source.num_condition_video_rows)
        video_doc_rows = source.video_indices
        clip = torch.zeros((1, 24, frames, *grid), device=self.device)
        last_inputs, jitters = (torch.zeros_like(clip), torch.zeros_like(clip))
        last_times: list[float] = []
        evaluations = len(schedule.timesteps) if exit_index is None else int(exit_index) + 1

        def draw(shape: tuple[int, ...]) -> Any:
            value = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
            self._broadcast_logical_sample(value)
            return value

        def sub_document(
            target: tuple[int, int], extra: list = ()
        ) -> tuple[Any, np.ndarray, list, tuple[int, int], Any]:
            """The rows a stream recomputes for one forward: the static rows, the
            audio, ``extra`` (the structure window
            when it is not cached) and ``target`` (a structure or target chunk) -
            everything else comes from the cache."""
            spans = sorted([*rows.static, rows.audio, *extra, target])
            keep = np.concatenate([np.arange(a, b) for a, b in spans])
            where = np.full(rows.sequence_length, -1, dtype=np.int64)
            where[keep] = np.arange(keep.size)
            kept_video = np.asarray(
                [r for r in video_doc_rows[:n_cond] if where[r] >= 0], dtype=np.int64
            )
            pick = torch.from_numpy(np.searchsorted(video_doc_rows[:n_cond], kept_video)).to(
                self.device
            )
            times = np.full(int(keep.size), CLEAN_LEVEL, dtype=np.float32)
            text = _remap(rows.text, where)
            times[text[0] : text[1]] = STATIC_TIME
            return (select_rows(layout, keep), where, pick, _remap(target, where), times)

        def plan_for(
            where: np.ndarray,
            mine: tuple[int, int],
            size: int,
            keys: list,
            cached: tuple,
            commit: Any,
            extra: list = (),
        ) -> H3SegmentPlan:
            static = [_remap(s, where) for s in rows.static]
            audio = _remap(rows.audio, where)
            pieces = [(span, static) for span in static] + [(audio, [*static, audio])]
            pieces += [(_remap(e, where), [*static, _remap(e, where)]) for e in extra]
            segments = [(a, b, _merge(k), (), None) for (a, b), k in sorted(pieces) if b > a]
            segments.append((mine[0], mine[1], _merge([*static, *keys, mine]), cached, commit))
            segments.sort()
            return H3SegmentPlan(tuple(segments), size, cache)

        for c, (f0, f1) in enumerate(rows.frames):
            history = tuple(
                ("structure", j) for j in _window(c, window) if rows.structure and cache_structure
            ) + tuple(("clean", j) for j in _window(c, window) if j < c)
            extra = (
                []
                if cache_structure or not rows.structure
                else [rows.structure[j] for j in _window(c, window)]
            )
            if rows.structure and cache_structure:
                arrays, where, pick, mine, times = sub_document(rows.structure[c])
                with torch.no_grad():
                    self._run(
                        video_rows=observation_rows.index_select(1, pick),
                        audio_rows=audio,
                        prompt=prompt,
                        arrays=arrays,
                        times=times,
                        plan=plan_for(where, mine, len(times), [], (), ("structure", c)),
                        model=model,
                    )
            arrays, where, pick, mine, times = sub_document(rows.target[c], extra)
            condition = observation_rows.index_select(1, pick)
            audio_span = [_remap(rows.audio, where), *(_remap(e, where) for e in extra)]
            shape = (1, 24, f1 - f0, *grid)
            current = draw(shape)
            for index in range(evaluations):
                sigma, sigma_next = (
                    float(schedule.sigmas[index]),
                    float(schedule.sigmas[index + 1]),
                )
                times[mine[0] : mine[1]] = 1.0 - sigma
                rows_out = self._run(
                    video_rows=torch.cat((condition, patchify_video(current)), dim=1),
                    audio_rows=audio,
                    prompt=prompt,
                    arrays=arrays,
                    times=times.copy(),
                    plan=plan_for(where, mine, len(times), audio_span, history, None, extra),
                    model=model,
                )
                velocity = unpatchify_video(
                    rows_out[:, int(condition.shape[1]) :], f1 - f0, *grid, channels=24
                ).float()
                if record is not None:
                    record.append((c, index, current.clone(), velocity.clone(), 1.0 - sigma))
                last_inputs[:, :, f0:f1] = current
                if index == evaluations - 1:
                    last_times.append(1.0 - sigma)
                    current = (
                        current
                        + (sigma if exit_index is not None else sigma - sigma_next) * velocity
                    )
                    break
                if exit_index is None:
                    current = current + (sigma - sigma_next) * velocity
                else:
                    current = (1.0 - sigma_next) * (current + sigma * velocity) + sigma_next * draw(
                        shape
                    )
            clip[:, :, f0:f1] = current
            jitter = draw(shape)
            jitters[:, :, f0:f1] = jitter
            times[mine[0] : mine[1]] = CLEAN_LEVEL
            with torch.no_grad():
                self._run(
                    video_rows=torch.cat(
                        (
                            condition,
                            patchify_video(CLEAN_LEVEL * current + (1 - CLEAN_LEVEL) * jitter),
                        ),
                        dim=1,
                    ),
                    audio_rows=audio,
                    prompt=prompt,
                    arrays=arrays,
                    times=times.copy(),
                    plan=plan_for(
                        where, mine, len(times), audio_span, history, ("clean", c), extra
                    ),
                    model=model,
                )
        return (clip, last_inputs, last_times, jitters)


def _core_class() -> type:
    from .core import H3Stage0p5Core

    class H3OmniCausalCore(H3OmniCausalMixin, H3Stage0p5Core):
        """An omni core whose training loss and sampler are block-causal (``train.causal_omni``)."""

        def __init__(
            self, model: Any, silence_latents: Any, device: Any, causal: dict, **kwargs: Any
        ) -> None:
            H3Stage0p5Core.__init__(self, model, silence_latents, device, **kwargs)
            validate_causal_omni(causal)
            self.causal = dict(causal)

    return H3OmniCausalCore


def validate_causal_omni(block: Any) -> None:
    from collections.abc import Mapping

    from solarwm.errors import ConfigurationError

    if not isinstance(block, Mapping) or set(block) - {"mode", "chunk", "window", "corrupt"}:
        raise ConfigurationError("train.causal_omni holds mode, chunk, window and corrupt")
    if str(block.get("mode", "teacher")) not in {"teacher", "diffusion"}:
        raise ConfigurationError("train.causal_omni.mode must be teacher or diffusion")
    chunk, window = (block.get("chunk", 5), block.get("window", 6))
    if isinstance(chunk, bool) or not isinstance(chunk, int) or (not 1 <= chunk <= 5):
        raise ConfigurationError("train.causal_omni.chunk must be 1 to 5 latents")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ConfigurationError("train.causal_omni.window must be a whole number of chunks >= 1")
    corrupt = block.get("corrupt", 0.0)
    if (
        isinstance(corrupt, bool)
        or not isinstance(corrupt, (int, float))
        or (not 0.0 <= float(corrupt) < 1.0)
    ):
        raise ConfigurationError("train.causal_omni.corrupt must lie in [0, 1)")
    if float(corrupt) > 0 and str(block.get("mode", "teacher")) != "teacher":
        raise ConfigurationError("train.causal_omni.corrupt corrupts a teacher-forced history")


def __getattr__(name: str) -> Any:
    if name == "H3OmniCausalCore":
        return _core_class()
    if name == "H3OmniCausalDMDCore":
        return _dmd_core_class()
    raise AttributeError(name)


def _dmd_core_class() -> type:
    from .dmd import H3OmniDMDCore

    class H3OmniCausalDMDCore(H3OmniCausalMixin, H3OmniDMDCore):
        """The omni DMD core with a block-causal student: the student streams and replays
        (:meth:`stream`,
        :meth:`training_prediction`), the teacher and critic score the whole clip as before."""

        def __init__(
            self,
            model: Any,
            silence_latents: Any,
            device: Any,
            train: Any,
            causal: dict,
            **kwargs: Any,
        ):
            H3OmniDMDCore.__init__(self, model, silence_latents, device, train, **kwargs)
            self.causal = {"mode": "teacher", **dict(causal)}

    return H3OmniCausalDMDCore
