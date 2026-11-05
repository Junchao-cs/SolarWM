"""Causal Memory attention with Stage2's bounded raw-history K/V cache."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .sgf_attention import _cat_rotary, _slice_rotary
from .torch_geometry import native_video_position_grid
from .torch_prope import prope_qkv_separate


@dataclass(frozen=True)
class MemoryKVAttention:
    layout: object
    viewmats: torch.Tensor  # anchor, then all rollout latent cameras
    intrinsics: torch.Tensor
    bank_indices: tuple[int, ...]
    chunk_index: int
    cache: object
    commit_cache: bool = False
    prefix_rotary: object = None
    window_rotary: object = None

    def prepare_rotaries(self, rope):
        prefix = self.layout.sequence_length - 5 * self.layout.rows_per_video_frame
        count = min(self.chunk_index, 5) * 5 + 5
        grid = native_video_position_grid(
            count,
            self.layout.latent_height,
            self.layout.latent_width,
            origin=float(self.layout.text_indices.numel()),
        ).to(self.layout.position_ids.device)
        return replace(
            self, prefix_rotary=rope(self.layout.position_ids[:prefix]), window_rotary=rope(grid)
        )

    def attend(self, q, k, v, *, layer, apply_rotary, attend):
        layout = self.layout
        rows = layout.rows_per_video_frame
        shared = int(layout.text_indices.numel()) + rows
        bank_end = shared + len(self.bank_indices) * rows
        prefix = bank_end + int(layout.audio_indices.numel())
        if q.shape[1] != prefix + 5 * rows:
            raise ValueError("Memory KV packed layout mismatch")
        history = self.cache.history(layer, self.chunk_index)
        hk = torch.cat([e[1] for e in history], 1) if history else k[:, :0]
        hv = torch.cat([e[2] for e in history], 1) if history else v[:, :0]
        hrows = hk.shape[1]
        batch = q.shape[0]
        pv = torch.eye(4, device=q.device).repeat(batch, prefix, 1, 1)
        pk = torch.eye(3, device=q.device).repeat(batch, prefix, 1, 1)
        text = int(layout.text_indices.numel())
        pv[:, text:shared] = self.viewmats[:, :1]
        pk[:, text:shared] = self.intrinsics[:, :1]
        if self.bank_indices:
            ids = torch.tensor(self.bank_indices, device=q.device) + 1
            pv[:, shared:bank_end] = self.viewmats.index_select(1, ids).repeat_interleave(rows, 1)
            pk[:, shared:bank_end] = self.intrinsics.index_select(1, ids).repeat_interleave(rows, 1)
        first = max(0, self.chunk_index - 5) * 5
        stop = (self.chunk_index + 1) * 5
        wv = self.viewmats[:, 1 + first : 1 + stop].repeat_interleave(rows, 1)
        wk = self.intrinsics[:, 1 + first : 1 + stop].repeat_interleave(rows, 1)

        def calculate(qv, kv, vv, qr, kr, qcam, kcam):
            qrq = apply_rotary(qv, *qr)
            krk = apply_rotary(kv, *kr)
            qrq, krk, vv, basis = prope_qkv_separate(
                qrq.transpose(1, 2),
                krk.transpose(1, 2),
                vv.transpose(1, 2),
                query_viewmats=qcam[0],
                query_K=qcam[1],
                key_viewmats=kcam[0],
                key_K=kcam[1],
            )
            out = attend(qrq.transpose(1, 2), krk.transpose(1, 2), vv.transpose(1, 2))
            return basis(out.transpose(1, 2)).transpose(1, 2)

        sr = _slice_rotary(self.prefix_rotary, 0, shared)
        mr = _slice_rotary(self.prefix_rotary, shared, bank_end)
        ar = _slice_rotary(self.prefix_rotary, bank_end, prefix)
        hr = _slice_rotary(self.window_rotary, 0, hrows)
        cr = _slice_rotary(self.window_rotary, hrows, hrows + 5 * rows)
        outputs = [
            calculate(
                q[:, :shared],
                k[:, :shared],
                v[:, :shared],
                sr,
                sr,
                (pv[:, :shared], pk[:, :shared]),
                (pv[:, :shared], pk[:, :shared]),
            )
        ]
        if self.bank_indices:
            outputs.append(
                calculate(
                    q[:, shared:bank_end],
                    torch.cat((k[:, :bank_end], hk), 1),
                    torch.cat((v[:, :bank_end], hv), 1),
                    mr,
                    _cat_rotary(sr, mr, hr),
                    (pv[:, shared:bank_end], pk[:, shared:bank_end]),
                    (
                        torch.cat((pv[:, :bank_end], wv[:, :hrows]), 1),
                        torch.cat((pk[:, :bank_end], wk[:, :hrows]), 1),
                    ),
                )
            )
        if prefix > bank_end:
            outputs.append(
                calculate(
                    q[:, bank_end:prefix],
                    torch.cat((k[:, :shared], k[:, bank_end:prefix]), 1),
                    torch.cat((v[:, :shared], v[:, bank_end:prefix]), 1),
                    ar,
                    _cat_rotary(sr, ar),
                    (pv[:, bank_end:prefix], pk[:, bank_end:prefix]),
                    (
                        torch.cat((pv[:, :shared], pv[:, bank_end:prefix]), 1),
                        torch.cat((pk[:, :shared], pk[:, bank_end:prefix]), 1),
                    ),
                )
            )
        outputs.append(
            calculate(
                q[:, prefix:],
                torch.cat((k[:, :prefix], hk, k[:, prefix:]), 1),
                torch.cat((v[:, :prefix], hv, v[:, prefix:]), 1),
                cr,
                _cat_rotary(self.prefix_rotary, hr, cr),
                (wv[:, hrows:], wk[:, hrows:]),
                (torch.cat((pv, wv), 1), torch.cat((pk, wk), 1)),
            )
        )
        if self.commit_cache:
            self.cache.commit(layer, self.chunk_index, k[:, prefix:], v[:, prefix:])
        return torch.cat(outputs, 1)
