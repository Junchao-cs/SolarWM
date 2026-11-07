"""Scoped inference optimizations; importing this package installs no hooks.

compile: unpadded dynamic prompts, phase compilation and private KV windows.
attention: video SDPA, packed history and balanced sequence-parallel exchange.
operators: fused RMS/QKV/layout kernels and source-bound tuning profiles.
graph: resident denoise/refresh capture, scene rebinding and cache writeback.
"""
