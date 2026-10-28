"""Direct sparse W6 masks for compiled H3 teacher-forcing attention."""

from __future__ import annotations

import math
from typing import Any

from .layout import AUDIO_ROLE, CLEAN_VIDEO_ROLE, CONDITION_ROLE, NOISY_VIDEO_ROLE, H3PackedLayout
from .mask import stage1_window_allows as stage1_w6_allows


def _validate_layout(layout: H3PackedLayout) -> None:
    if layout.stage != "stage1" or layout.window_chunks is None:
        raise ValueError("W6 masks require a Stage1 layout")
    if not int(layout.clean_video_indices.numel()) or not int(layout.noisy_video_indices.numel()):
        raise ValueError("Stage1 W6 TF requires explicit clean and noisy video copies")


def build_stage1_w6_block_mask(
    layout: H3PackedLayout,
    *,
    batch_size: int | None = None,
    num_heads: int | None = None,
    device: Any = None,
    block_size: int | tuple[int, int] = 128,
) -> Any:
    """Build the exact W6 ``BlockMask`` without a token-level dense mask.

    ``create_block_mask`` evaluates a logical ``[Q,K]`` grid and is unsafe for
    H3's 29k/72k-row Stage1 documents unless its compilation succeeds.  This
    builder instead derives full and partial tiles directly from the small set
    of contiguous role/chunk runs.  Peak construction memory is therefore
    ``O(ceil(Q/128)^2)`` block metadata, never ``O(Q^2)`` token metadata.
    """

    _validate_layout(layout)
    import torch
    from torch.nn.attention.flex_attention import BlockMask

    if batch_size not in (None, 1):
        raise ValueError("H3 W6 currently requires batch_size=1")
    if num_heads is not None:
        raise ValueError("H3 W6 BlockMask must be shared across all attention heads")
    if isinstance(block_size, tuple):
        if tuple(block_size) != (128, 128):
            raise ValueError("H3 W6 requires FlexAttention block_size=(128,128)")
        tile_size = 128
    else:
        tile_size = int(block_size)
        if tile_size != 128:
            raise ValueError("H3 W6 requires FlexAttention block_size=128")

    roles_cpu = layout.row_roles.detach().to(device="cpu", dtype=torch.long)
    chunks_cpu = layout.target_video_chunk_ids.detach().to(device="cpu", dtype=torch.long)
    sequence_length = int(layout.sequence_length)
    if tuple(roles_cpu.shape) != (sequence_length,) or tuple(chunks_cpu.shape) != (
        sequence_length,
    ):
        raise ValueError("H3 W6 layout role/chunk rows do not match sequence_length")
    window = int(layout.window_chunks)

    # Find the handful of contiguous (role, chunk) runs.  A typical seven-
    # chunk Stage1 document has condition, audio, seven clean and seven noisy
    # runs even though it contains tens of thousands of rows.
    changes = (roles_cpu[1:] != roles_cpu[:-1]) | (chunks_cpu[1:] != chunks_cpu[:-1])
    boundaries = [int(value) + 1 for value in torch.nonzero(changes).flatten()]
    starts = [0, *boundaries]
    ends = [*boundaries, sequence_length]
    runs = [
        (
            start,
            end,
            int(roles_cpu[start].item()),
            int(chunks_cpu[start].item()),
        )
        for start, end in zip(starts, ends, strict=True)
    ]

    def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if merged and start == merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    allowed_by_query: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for _start, _end, query_role, query_chunk in runs:
        query_key = (query_role, query_chunk)
        if query_key in allowed_by_query:
            continue
        allowed_by_query[query_key] = merge_intervals(
            [
                (key_start, key_end)
                for key_start, key_end, key_role, key_chunk in runs
                if stage1_w6_allows(
                    query_role,
                    query_chunk,
                    key_role,
                    key_chunk,
                    window_chunks=window,
                )
            ]
        )

    num_blocks = math.ceil(sequence_length / tile_size)
    kv_starts = torch.arange(num_blocks, dtype=torch.long) * tile_size
    kv_ends = torch.clamp(kv_starts + tile_size, max=sequence_length)
    full_tiles = torch.zeros((num_blocks, num_blocks), dtype=torch.bool)
    any_tiles = torch.zeros_like(full_tiles)

    for query_block in range(num_blocks):
        query_start = query_block * tile_size
        query_end = min(query_start + tile_size, sequence_length)
        query_segments = [
            (max(query_start, start), min(query_end, end), role, chunk)
            for start, end, role, chunk in runs
            if start < query_end and query_start < end
        ]
        if not query_segments:
            raise RuntimeError(
                f"H3 W6 direct mask found no query run for [{query_start},{query_end})"
            )
        row_full = torch.ones(num_blocks, dtype=torch.bool)
        row_any = torch.zeros(num_blocks, dtype=torch.bool)
        for _segment_start, _segment_end, query_role, query_chunk in query_segments:
            segment_full = torch.zeros(num_blocks, dtype=torch.bool)
            segment_any = torch.zeros(num_blocks, dtype=torch.bool)
            for allowed_start, allowed_end in allowed_by_query[(query_role, query_chunk)]:
                segment_full |= (allowed_start <= kv_starts) & (kv_ends <= allowed_end)
                segment_any |= (kv_starts < allowed_end) & (allowed_start < kv_ends)
            row_full &= segment_full
            row_any |= segment_any
        if not bool(row_any.any().item()):
            raise RuntimeError(f"H3 W6 query block {query_block} has no visible key block")
        full_tiles[query_block] = row_full
        any_tiles[query_block] = row_any

    partial_tiles = any_tiles & ~full_tiles
    metadata_blocks = 1 << (num_blocks - 1).bit_length()

    def ordered_metadata(tile_mask: Any) -> tuple[Any, Any]:
        # Mirror torch 2.6's internal _dense_to_ordered conversion: its CPU
        # argsort path is defined for int32 metadata, not every bool backend.
        ordered = torch.zeros((metadata_blocks, metadata_blocks), dtype=torch.int32)
        ordered[:num_blocks, :num_blocks].copy_(tile_mask)
        counts = ordered.sum(dim=-1).to(dtype=torch.int32, memory_format=torch.contiguous_format)
        indices = torch.argsort(ordered, dim=-1, descending=True, stable=True).to(
            dtype=torch.int32, memory_format=torch.contiguous_format
        )
        return counts[None, None].to(device), indices[None, None].to(device)

    partial_counts, partial_indices = ordered_metadata(partial_tiles)
    full_counts, full_indices = ordered_metadata(full_tiles)

    # Capture only device scalars. PyTorch 2.6 dynamic Flex lifts the size of
    # a captured lookup table into a symbolic autotuning input it cannot key.
    # Canonical packing permits the identical role/chunk lookup by intervals,
    # without specializing caption lengths or allocating a token-level mask.
    condition_end = int(layout.condition_indices.numel())
    clean_start = condition_end + int(layout.audio_indices.numel())
    clean_rows = int(layout.clean_video_indices.numel())
    noisy_rows = int(layout.noisy_video_indices.numel())
    noisy_start = clean_start + clean_rows
    chunk_rows = int(layout.rows_per_video_frame) * int(layout.chunk_latent_frames or 0)
    if (
        chunk_rows <= 0
        or clean_rows % chunk_rows
        or noisy_rows % chunk_rows
        or noisy_start + noisy_rows != sequence_length
    ):
        raise ValueError("H3 W6 requires canonical whole-chunk [C|A|clean|noisy] packing")
    clean_chunk_start = int(chunks_cpu[clean_start])
    noisy_chunk_start = int(chunks_cpu[noisy_start])
    expected_runs = []
    if condition_end:
        expected_runs.append((0, condition_end, CONDITION_ROLE, -1))
    if clean_start > condition_end:
        expected_runs.append((condition_end, clean_start, AUDIO_ROLE, -1))
    for start, length, role, first_chunk in (
        (clean_start, clean_rows, CLEAN_VIDEO_ROLE, clean_chunk_start),
        (noisy_start, noisy_rows, NOISY_VIDEO_ROLE, noisy_chunk_start),
    ):
        expected_runs.extend(
            (
                start + offset,
                start + offset + chunk_rows,
                role,
                first_chunk + offset // chunk_rows,
            )
            for offset in range(0, length, chunk_rows)
        )
    if runs != expected_runs:
        raise ValueError("H3 W6 requires canonical contiguous role/chunk runs")
    (
        condition_end,
        clean_start,
        noisy_start,
        sequence_end,
        chunk_rows,
        clean_chunk_start,
        noisy_chunk_start,
        window_bound,
    ) = (
        torch.tensor(value, dtype=torch.long, device=device)
        for value in (
            condition_end,
            clean_start,
            noisy_start,
            sequence_length,
            chunk_rows,
            clean_chunk_start,
            noisy_chunk_start,
            window,
        )
    )

    def role_and_chunk(index: Any) -> tuple[Any, Any, Any, Any, Any]:
        valid = (index >= 0) & (index < sequence_end)
        condition = valid & (index < condition_end)
        audio = valid & (index >= condition_end) & (index < clean_start)
        clean = valid & (index >= clean_start) & (index < noisy_start)
        noisy = valid & (index >= noisy_start)
        video_start = torch.where(index >= noisy_start, noisy_start, clean_start)
        first_chunk = torch.where(index >= noisy_start, noisy_chunk_start, clean_chunk_start)
        chunk = torch.div(index - video_start, chunk_rows, rounding_mode="floor") + first_chunk
        return condition, audio, clean, noisy, chunk

    def mask_mod(_batch: Any, _head: Any, query_index: Any, key_index: Any) -> Any:
        condition_query, audio_query, clean_query, noisy_query, query_chunk = role_and_chunk(
            query_index
        )
        condition_key, audio_key, clean_key, noisy_key, key_chunk = role_and_chunk(key_index)
        delta = query_chunk - key_chunk
        common_key = condition_key | audio_key
        condition_visibility = condition_query & condition_key
        audio_visibility = audio_query & common_key
        clean_visibility = clean_query & (common_key | (clean_key & (delta >= 0)))
        noisy_visibility = noisy_query & (
            common_key
            | (clean_key & (delta >= 1) & (delta < window_bound))
            | (noisy_key & (delta == 0))
        )
        return condition_visibility | audio_visibility | clean_visibility | noisy_visibility

    block_mask = BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        BLOCK_SIZE=tile_size,
        mask_mod=mask_mod,
        seq_lengths=(sequence_length, sequence_length),
    )
    # PyTorch 2.6 Flex autotuning requires concrete metadata dimensions.
    # Power-of-two zero-count buckets keep only these sparse tensors static;
    # real token lengths and the visibility of every actual tile are unchanged.
    for metadata in (
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
        block_mask.q_num_blocks,
        block_mask.q_indices,
        block_mask.full_q_num_blocks,
        block_mask.full_q_indices,
    ):
        torch._dynamo.mark_static(metadata)
    return block_mask
