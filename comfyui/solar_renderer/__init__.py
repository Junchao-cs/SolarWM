"""SolarRenderer for ComfyUI (see comfyui/solar_renderer/README.md)."""

import contextlib

from .causal import SolarRendererCausalSampler
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS["SolarRendererCausalSampler"] = SolarRendererCausalSampler
NODE_DISPLAY_NAME_MAPPINGS["SolarRendererCausalSampler"] = (
    "SolarRenderer Causal Sampler (long clips)"
)

with contextlib.suppress(ImportError):  # nodes also load without the server (tests, tooling)
    from . import app  # noqa: F401

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
