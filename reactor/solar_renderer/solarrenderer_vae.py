"""Shard independent native VAE spatial tiles across ranks and stitch in native order.

The tested profile batches equally shaped tiles and leaves compilation disabled.
Adapted from the Reactor H3 infrastructure tutorial implementation.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _undo_only(undo):
    """Undo changes when `_decode_clip` was not patched (single rank or no process group)."""

    def uninstall():
        for u in reversed(undo):
            u()

    return uninstall


def install(vae, group=None, batched: bool | None = None, compile_mode: str | None = None):
    """Install spatial tile sharding and return its uninstall function."""
    if batched is None:
        batched = True

    undo = []
    if compile_mode:
        eager_decoder = vae.decoder
        vae.decoder = torch.compile(eager_decoder, mode=compile_mode, dynamic=False)

        def _restore_decoder():
            vae.decoder = eager_decoder
            torch._dynamo.reset()

        undo.append(_restore_decoder)

    if not (dist.is_available() and dist.is_initialized()):
        return _undo_only(undo)

    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world == 1:
        return _undo_only(undo)

    original = vae._decode_clip

    def sharded_decode_clip(z: torch.Tensor) -> torch.Tensor:
        if not vae.use_tiling:
            return original(z)

        height = z.shape[-2] * vae.spatial_compression_ratio
        width = z.shape[-1] * vae.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = vae._split_tiles(
            height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = vae._split_tiles(
            width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
        )
        ratio = vae.spatial_compression_ratio

        coords = [
            (y_indices[i], y_lengths[i], x_indices[j], x_lengths[j])
            for i in range(len(y_indices))
            for j in range(len(x_indices))
        ]
        # Contiguous blocks, not round-robin: `all_gather` concatenates rank 0's block, then rank
        # 1's, and so on, so contiguous assignment is what puts the tiles back in tile order for
        # free. The list is padded up to an equal count per rank so the gather stays a plain
        # fixed-size one; the padding entries are recomputed duplicates of the last tile, at most
        # world-1 of them, and are dropped after the gather.
        per_rank = (len(coords) + world - 1) // world
        padded = coords + [coords[-1]] * (per_rank * world - len(coords))
        mine = padded[rank * per_rank : (rank + 1) * per_rank]

        slices = [
            z[
                ...,
                y // ratio : y // ratio + y_len // ratio,
                x // ratio : x // ratio + x_len // ratio,
            ]
            for y, y_len, x, x_len in mine
        ]
        if batched:
            local_stack = vae.decoder(vae.post_quant_conv(torch.cat(slices, dim=0)))
        else:
            local_stack = torch.stack(
                [vae.decoder(vae.post_quant_conv(s)) for s in slices], dim=0
            ).squeeze(1)

        gathered = [torch.empty_like(local_stack) for _ in range(world)]
        dist.all_gather(gathered, local_stack, group=group)
        flat = torch.cat(gathered, dim=0)[: len(coords)]

        # `_stitch_tiles` only touches dims -2/-1, so a missing batch axis would not raise — it
        # would return `(3, T, H, W)` and `_decode` would then slice the height axis where it means
        # to slice frames. Restore the axis explicitly.
        rows, index = [], 0
        for _ in range(len(y_indices)):
            row = []
            for _ in range(len(x_indices)):
                row.append(flat[index].unsqueeze(0))
                index += 1
            rows.append(row)
        return vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    vae._decode_clip = sharded_decode_clip

    def uninstall():
        vae._decode_clip = original
        for u in reversed(undo):
            u()

    return uninstall
