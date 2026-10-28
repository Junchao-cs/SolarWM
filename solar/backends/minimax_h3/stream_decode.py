"""Bounded-memory form of the official H3 temporal decoder.

Keep the official spatial tiles, temporal clips and overlap arithmetic. Only
the collection of decoded clips changes: completed frames are yielded rather
than retaining the whole pixel video on the GPU.
"""

from __future__ import annotations


def decoded_chunks(vae, normalized_latents, *, device):
    import torch

    z = normalized_latents
    size, drop, ratio = vae.tokens_chunk_size, vae.config.token_drop, vae.temporal_compression_ratio
    tokens = z.shape[2] + drop
    padding = (-tokens) % size
    count = (tokens + padding) // size - int(drop > 0)
    if count < 1:
        raise ValueError("H3 streaming decoder needs at least one complete temporal clip")
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    intra_tail = vae.config.clip_length % ratio
    trim = sum(
        intra_tail if intra_tail and (z.shape[2] + k) % size == 0 else ratio for k in range(padding)
    )
    overlap = None
    emitted = 0
    pending_cpu = None

    def completed(chunk):
        # Padding can remove frames from more than the last overlap chunk.
        # Withhold exactly that tail across chunks, like slicing the official
        # concatenated output, while keeping only a bounded CPU frame buffer.
        nonlocal pending_cpu
        chunk = chunk.cpu()
        if pending_cpu is not None:
            chunk = torch.cat((pending_cpu, chunk), dim=2)
        keep = min(trim, chunk.shape[2])
        if chunk.shape[2] > keep:
            yield chunk[:, :, :-keep] if keep else chunk
        pending_cpu = chunk[:, :, -keep:] if keep else None

    for index in range(count):
        start = index * size
        # The official decoder slices after padding only to a chunk boundary;
        # its final clip can be shorter than size + token_overlap.
        stop = min(index * size + size + vae.token_overlap, z.shape[2] + padding)
        part = z[:, :, start : min(stop, z.shape[2])].to(device).float()
        if stop > z.shape[2]:
            part = torch.cat((part, part[:, :, -1:].repeat(1, 1, stop - z.shape[2], 1, 1)), dim=2)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=str(device).split(":")[0],
                dtype=torch.float16,
                enabled=str(device).startswith("cuda"),
            ),
        ):
            clip = vae._decode_clip(part * std + mean)
            for j in range(int(drop > 0) + 1):
                chunk = clip[:, :, j * size * ratio : (j + 1) * size * ratio][
                    :, :, vae.frame_pre_padding :
                ]
                if j == 0:
                    if overlap is not None:
                        chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                    for ready in completed(chunk):
                        emitted += ready.shape[2]
                        yield ready
                else:
                    overlap = chunk
        del clip, part
    if overlap is not None:
        for ready in completed(overlap):
            emitted += ready.shape[2]
            yield ready
    if emitted < 1:
        raise ValueError("H3 temporal decode produced no frames")
