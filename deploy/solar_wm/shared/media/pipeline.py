"""Bounded CPU processing; every frame reaches the resumable encoded stream."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .encoder import FragmentedPreviewWriter
from .archive import Archive
from ..runtime.cleanup import cleanup


class Pipeline:
    def __init__(self, output, *, state, ffmpeg, sequence, resolution):
        self.state, self.resolution = state, resolution
        self.writer = FragmentedPreviewWriter(Path(output), resolution=resolution,
            ffmpeg=ffmpeg, sequence=sequence)
        try:
            self.archive = Archive(Path(output).with_name(Path(output).name.replace("-live.mp4", ".mp4")), ffmpeg)
        except BaseException:
            self.writer.abort()
            raise
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="solarwm-rgb")
        self.pending = deque()
        self.pieces = self.frames = 0
        self.closed = False

    def _encode(self, cpu, first):
        import numpy as np
        from PIL import Image
        from .pixels import rgb
        for offset in range(cpu.shape[2]):
            array = rgb(cpu[0, :, offset])
            self.archive.append(array)
            if self.resolution == "half":
                array = np.asarray(Image.fromarray(array).resize((672, 384), Image.Resampling.LANCZOS))
            self.writer.append_rgb(np.ascontiguousarray(array), first + offset)
            self.state.record_frame(first + offset)

    def push(self, chunk, offset, cpu, stamp):
        if self.closed or chunk * 5 + offset != self.pieces:
            raise RuntimeError("piece order changed")
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()
        # At most two complete chunks. Backpressure is explicit, never a drop.
        while len(self.pending) >= 10:
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(self._encode, cpu, self.frames))
        self.pieces += 1
        self.frames += cpu.shape[2]

    def finish(self):
        try:
            while self.pending:
                self.pending.popleft().result()
            path = self.writer.finish()
            self.archive.finish()
            self.writer.record.done.set()
            return path
        except BaseException:
            self.writer.abort()
            self.archive.abort()
            raise
        finally:
            self.closed = True
            self.executor.shutdown(wait=True)

    def abort(self):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        cleanup([("preview", self.writer.abort), ("archive", self.archive.abort)])
