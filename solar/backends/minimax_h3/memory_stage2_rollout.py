"""Memory inference forward for the Stage2 SGF solver."""

from __future__ import annotations

import torch

from .layout import patchify_video, unpatchify_video
from .memory_kv_attention import MemoryKVAttention
from .memory_layout import memory_rollout_layout, memory_row_time_pairs
from .memory_retrieval import LatentMemoryBank, select_memory
from .torch_flow import predict_clean_sample


class Stage2MemoryForward:
    def __init__(self, c2w, K, *, enabled=True):
        self.c2w = c2w
        self.K = K
        self.enabled = enabled
        self.bank = LatentMemoryBank()
        self.layouts = {}
        self.selected = {}
        self.records = []

    def __call__(
        self, student, inputs, noisy, timestep, *, chunk_index=0, cache=None, commit_cache=False
    ):
        j = chunk_index
        if j not in self.selected:
            ids, scores = select_memory(self.c2w, self.K, j)
            if not self.enabled:
                ids = ids[:0]
                scores = scores[:0]
            self.selected[j] = (ids, scores)
        selected, scores = self.selected[j]
        bank = self.bank.gather(selected, noisy)
        n = bank.shape[2]
        if n not in self.layouts:
            self.layouts[n] = memory_rollout_layout(
                inputs.text_tags.cpu(),
                n,
                inputs.audio_rows.shape[1] // 2,
                noisy.device,
                height=inputs.latent_height,
                width=inputs.latent_width,
            )
        layout = self.layouts[n]
        # Anchor .999, bank clean 1, shared text 1; no AnyFlow r_timestep.
        times, _, indices = memory_row_time_pairs(
            layout, timestep, timestep, inputs.audio_timestep, device=noisy.device
        )
        control = MemoryKVAttention(
            layout,
            inputs.camera_viewmats,
            inputs.camera_K,
            tuple(int(i) for i in selected),
            j,
            cache,
            commit_cache,
        )
        prediction, _ = student(
            hidden_states=torch.cat(
                (inputs.anchor_rows, patchify_video(bank), patchify_video(noisy)), 1
            ),
            audio_hidden_states=inputs.audio_rows,
            encoder_hidden_states=inputs.prompt,
            timestep=times,
            timestep_indices=indices,
            fused_prope=True,
            packed_sequence_parallel=inputs.sp_enabled,
            sgf_attention=control,
            return_dict=False,
            **layout.transformer_kwargs(),
        )
        velocity = unpatchify_video(
            prediction[:, layout.noisy_video_output_slice],
            5,
            inputs.latent_height,
            inputs.latent_width,
        )
        if commit_cache:
            self.bank.commit(noisy, j)
            self.records.append(
                dict(
                    chunk=j,
                    query_pose_latent=5 * j + 4,
                    candidate_count=max(0, j - 5) * 5,
                    memory_latent_indices=selected.tolist(),
                    memory_scores=scores.tolist(),
                    history_chunks=list(range(max(0, j - 5), j)),
                    cached_chunks_per_layer=max(len(v) for v in cache.layers.values()),
                    latent_bank_bytes=sum(x.numel() * x.element_size() for x in self.bank.chunks),
                )
            )
        return predict_clean_sample(noisy, velocity, timestep)
