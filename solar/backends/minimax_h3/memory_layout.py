"""Single-chunk packed layout shared by online Memory inference."""

from .layout import build_stage0p5_layout


def memory_rollout_layout(tags, bank_frames, audio_frames, device, *, height=48, width=84):
    layout = build_stage0p5_layout(
        tags, 5, height, width, audio_frames, keyframe_anchors=("first",) * (1 + bank_frames)
    )
    bank = layout.condition_video_indices[layout.rows_per_video_frame :]
    layout.position_ids[bank, 0] = -1.0
    layout.row_roles[bank] = 4
    return layout.to(device)


def memory_row_time_pairs(layout, video_t, video_r, audio_t, *, device, current_chunk=None):
    import torch

    t = torch.ones(layout.sequence_length, device=device, dtype=torch.float32)
    t[layout.condition_video_indices[: layout.rows_per_video_frame]] = 0.999
    t[layout.audio_indices] = audio_t.reshape(())
    noisy = layout.noisy_video_indices
    if current_chunk is not None:
        chunks = layout.target_video_chunk_ids.index_select(0, noisy)
        noisy = noisy[chunks == current_chunk]
    t[noisy] = video_t.reshape(())
    r = t.clone()
    r[noisy] = video_r.reshape(())
    pairs, inverse = torch.unique(torch.stack((t, r), 1), dim=0, return_inverse=True)
    return pairs[:, 0], pairs[:, 1], inverse
