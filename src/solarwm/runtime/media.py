"""Resolve the same ffmpeg executable for decoding and validation artifacts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from solarwm.errors import DataContractError


def resolve_ffmpeg_executable() -> str:
    """Prefer an explicit override, then PATH, then the imageio bundled binary."""
    override = os.environ.get("SOLARWM_FFMPEG")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise DataContractError(f"SOLARWM_FFMPEG is not an executable file: {candidate}")
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg

        candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, OSError, RuntimeError) as exc:
        raise DataContractError(
            "Media processing requires ffmpeg; install ffmpeg or imageio-ffmpeg, "
            "or set SOLARWM_FFMPEG to an executable"
        ) from exc
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise DataContractError(f"imageio-ffmpeg returned a non-executable file: {candidate}")
    return str(candidate)
