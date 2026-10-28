"""Feed causal blocks into the native H3 temporal decoder without speculative frames.

The overlap and padding arithmetic follows SolarRenderer's Apache-2.0
stream_decode.py. Wait for a complete decoder window before emitting it.
"""

from __future__ import annotations

import torch


class SolarDecoder:
    def __init__(self, vae, latent_frames, device):
        self.vae, self.device = vae, device
        self.total = latent_frames
        self.size = vae.tokens_chunk_size
        self.drop = vae.config.token_drop
        self.ratio = vae.temporal_compression_ratio
        self.padding = (-(latent_frames + self.drop)) % self.size
        self.count = (latent_frames + self.drop + self.padding) // self.size - int(self.drop > 0)
        if self.count < 1:
            raise ValueError("The rollout is too short for the native temporal decoder.")
        tail = vae.config.clip_length % self.ratio
        self.trim = sum(
            tail if tail and (latent_frames + k) % self.size == 0 else self.ratio
            for k in range(self.padding)
        )
        self.mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        self.latents = None
        self.overlap = None
        self.pending = None
        self.index = 0
        self.finished = False

    def _completed(self, chunk):
        chunk = chunk.cpu()
        if self.pending is not None:
            chunk = torch.cat((self.pending, chunk), dim=2)
        keep = min(self.trim, chunk.shape[2])
        ready = chunk[:, :, :-keep] if keep else chunk
        self.pending = chunk[:, :, -keep:] if keep else None
        return ready

    @torch.no_grad()
    def push(self, block):
        if self.finished:
            raise RuntimeError("The decoder has already completed this run.")
        # Only the small normalized latent clip is retained, never the whole pixel video.
        block = block.detach().cpu()
        self.latents = block if self.latents is None else torch.cat((self.latents, block), dim=2)
        known = self.latents.shape[2]
        if known > self.total:
            raise ValueError("Latent input exceeds the native horizon.")
        ready = []
        while self.index < self.count:
            start = self.index * self.size
            stop = min(start + self.size + self.vae.token_overlap, self.total + self.padding)
            if known < min(stop, self.total):
                break
            part = self.latents[:, :, start : min(stop, self.total)].to(self.device).float()
            if stop > self.total:
                part = torch.cat(
                    (part, part[:, :, -1:].repeat(1, 1, stop - self.total, 1, 1)), dim=2
                )
            with torch.autocast(
                device_type=str(self.device).split(":")[0],
                dtype=torch.float16,
                enabled=str(self.device).startswith("cuda"),
            ):
                clip = self.vae._decode_clip(part * self.std + self.mean)
                for j in range(int(self.drop > 0) + 1):
                    chunk = clip[
                        :,
                        :,
                        j * self.size * self.ratio : (j + 1) * self.size * self.ratio,
                    ]
                    chunk = chunk[:, :, self.vae.frame_pre_padding :]
                    if j == 0:
                        if self.overlap is not None:
                            chunk = self.vae._blend(
                                self.overlap, chunk, self.vae.frame_overlap, dim=-3
                            )
                        ready.append(self._completed(chunk))
                    else:
                        self.overlap = chunk
            self.index += 1
        if known == self.total:
            if self.index != self.count:
                raise RuntimeError("Final decoder window was not consumed.")
            if self.overlap is not None:
                ready.append(self._completed(self.overlap))
            self.finished = True
            self.pending = self.overlap = self.latents = None
        ready = [chunk for chunk in ready if chunk.shape[2]]
        return torch.cat(ready, dim=2) if ready else None
