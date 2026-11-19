"""Triton block-transpose copy for the SP all-to-all packing (SOLARWM_H3_A2A_COPY=1, see distributed.py)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _block_transpose_kernel(SRC, DST, outer, inner, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # source row index in [outer * inner)
    o = row // inner
    i = row % inner
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    value = tl.load(SRC + row.to(tl.int64) * width + cols, mask=mask)
    dst_row = i.to(tl.int64) * outer + o
    tl.store(DST + dst_row * width + cols, value, mask=mask)


def block_transpose(src: torch.Tensor, outer: int, inner: int, width: int) -> torch.Tensor:
    """``src`` viewed as ``[outer, inner, width]`` -> contiguous ``[inner, outer, width]``."""
    src = src.contiguous()
    if src.numel() != outer * inner * width:
        raise ValueError("block_transpose shape does not match the tensor")
    out = torch.empty((inner, outer, width), device=src.device, dtype=src.dtype)
    block = 1024
    _block_transpose_kernel[(outer * inner, triton.cdiv(width, block))](
        src, out, outer, inner, width, BLOCK=block
    )
    return out
