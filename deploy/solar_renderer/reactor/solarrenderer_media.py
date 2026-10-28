"""Bounded CPU validation; preserve uploaded bytes for native preprocessing."""

import io
import math

import av
from PIL import Image


def validate_structure(data: bytes) -> None:
    if not data or len(data) > 256 * 1024 * 1024:
        raise ValueError("Upload a structure video smaller than 256 MiB.")
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.video:
            raise ValueError("The upload has no video track.")
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 0)
        if not math.isfinite(rate) or not 1 <= rate <= 120:
            raise ValueError("The video must declare a frame rate between 1 and 120 FPS.")
        if not 0 < stream.width * stream.height <= 4096 * 4096:
            raise ValueError("The video frame dimensions exceed the 16-megapixel limit.")
        needed = math.floor(157 * rate / 24 + 0.5) + 1
        for count, frame in enumerate(container.decode(stream), start=1):
            if frame.width * frame.height > 4096 * 4096:
                raise ValueError("Decoded video exceeds the dimension limit.")
            if count >= needed:
                return
        raise ValueError(
            "The video must provide at least 158 frames after native 24 FPS resampling."
        )


def validate_reference(data: bytes) -> None:
    if not data or len(data) > 25 * 1024 * 1024:
        raise ValueError("Upload a reference image smaller than 25 MiB.")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Use a PNG, JPEG or WebP reference image.")
        if image.width * image.height > 4096 * 4096:
            raise ValueError("The reference exceeds the 16-megapixel limit.")
        image.load()
