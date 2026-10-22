"""The deployed RGB8 formula, native CPU loop and owned contiguous output."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile

_lib = None


def prepare(cache_root):
    global _lib
    source = Path(__file__).with_suffix(".c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    folder = Path(cache_root) / "native-cpu" / digest
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "pixels.so"
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix="build-", dir=folder) as temporary:
            result = Path(temporary) / "pixels.so"
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-fno-fast-math",
                "-ffp-contract=off", str(source), "-o", str(result), "-lm"], check=True)
            result.replace(target)
    _lib = ctypes.CDLL(str(target))
    _lib.h3_rgb_bf16.argtypes = [ctypes.c_void_p] + [ctypes.c_size_t] * 5 + [ctypes.c_void_p] * 3
    _lib.h3_rgb_bf16.restype = ctypes.c_int
    return dict(source_sha256=digest, library=str(target))


def rgb(frame):
    import numpy as np
    import torch
    if _lib is None:
        raise RuntimeError("prepare native RGB helper before starting model work")
    if frame.device.type != "cpu" or frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("actual CPU CHW RGB output required")
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    out = np.empty((frame.shape[1], frame.shape[2], 3), dtype=np.uint8)
    if frame.dtype == torch.bfloat16 and all(s >= 0 for s in frame.stride()):
        ok = _lib.h3_rgb_bf16(frame.data_ptr(), frame.shape[1], frame.shape[2],
            *frame.stride(), mean.ctypes.data, std.ctypes.data, out.ctypes.data)
        if not ok:
            raise FloatingPointError("nonfinite RGB pixel")
    else:
        value = frame.detach().float().numpy()
        pixels = np.clip(value * std[:, None, None] + mean[:, None, None], 0., 1.)
        if not np.isfinite(pixels).all():
            raise FloatingPointError("nonfinite RGB pixel")
        out[:] = np.rint(pixels * np.float32(255.)).astype(np.uint8).transpose(1, 2, 0)
    out.setflags(write=False)
    return out
