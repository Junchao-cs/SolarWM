"""Runtime environment inspection."""

from __future__ import annotations

import importlib.metadata
import sys
from typing import Any


def probe_python_runtime() -> dict[str, Any]:
    """Return a serializable in-container probe without importing a backend."""

    result: dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
    }
    try:
        import torch
    except ImportError:
        result.update({"torch": None, "cuda": None, "cuda_available": False})
    else:
        result.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "nccl": list(torch.cuda.nccl.version())
                if torch.cuda.is_available() and torch.distributed.is_nccl_available()
                else None,
            }
        )
    packages: dict[str, str] = {}
    for name in ("diffusers", "transformers", "flash-attn", "peft", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    result["packages"] = packages
    return result
