from __future__ import annotations

import numpy as np
import pytest
import torch

from solar.backends.ltx25.adapter import (
    FUSED_PROPE_CONTRACT,
    lora_target_modules,
    pack_stage0p5,
    unpack_video_tokens,
    verify_parameter_free_state_keys,
)
from solar.backends.ltx25.distributed import (
    DistributedContract,
    build_hsdp_topology,
    contiguous_token_bounds,
)
from solar.backends.ltx25.flow import (
    first_frame_excluded_mse,
    predict_clean,
    restore_clean_first_latent,
    sample_shifted_logit_normal,
    scale_noise,
    shifted_logit_normal_mu,
    velocity_target,
)
from solar.backends.ltx25.torch_distributed import clip_replicated_gradient_norm
from solar.errors import BackendContractError


def test_native_flow_equations_reconstruct_clean_sample() -> None:
    clean = np.asarray([1.0, 3.0], dtype=np.float32)
    noise = np.asarray([5.0, -1.0], dtype=np.float32)
    noisy = scale_noise(clean, noise, 0.25)
    velocity = velocity_target(clean, noise)
    assert np.allclose(noisy, [2.0, 2.0])
    assert np.allclose(predict_clean(noisy, velocity, 0.25), clean)


def test_shifted_logit_normal_is_deterministic_and_uses_unclamped_mu() -> None:
    assert shifted_logit_normal_mu() == pytest.approx(10 / 3)
    first = sample_shifted_logit_normal(8, generator=np.random.default_rng(7))
    second = sample_shifted_logit_normal(8, generator=np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert np.all((first >= 0) & (first <= 1))


def test_first_latent_restore_and_loss_exclusion() -> None:
    noisy = np.zeros((1, 128, 20, 16, 24), dtype=np.float32)
    first = np.ones((1, 128, 1, 16, 24), dtype=np.float32)
    restored = restore_clean_first_latent(noisy, first)
    assert np.all(restored[:, :, 0] == 1)
    prediction = np.zeros_like(noisy)
    target = np.zeros_like(noisy)
    prediction[:, :, 0] = 1000
    assert first_frame_excluded_mse(prediction, target) == 0.0


def test_stage0p5_pack_round_trip_and_masks() -> None:
    latent = np.arange(2 * 20 * 16 * 24, dtype=np.float32).reshape(1, 2, 20, 16, 24)
    # Pad channels without allocating a random 128-channel source.
    latent = np.pad(latent, ((0, 0), (0, 126), (0, 0), (0, 0), (0, 0)))
    packed = pack_stage0p5(latent, 0.5)
    assert packed.video_tokens.shape == (1, 7680, 128)
    assert packed.first_frame_mask[:, :384].all()
    assert not packed.first_frame_mask[:, 384:].any()
    assert np.all(packed.token_timesteps[:, :384] == 0)
    assert np.all(packed.token_timesteps[:, 384:] == 0.5)
    assert np.array_equal(unpack_video_tokens(packed.video_tokens), latent)


def test_fused_prope_and_lora_targets_are_exact() -> None:
    assert FUSED_PROPE_CONTRACT.parameter_keys_added == 0
    assert FUSED_PROPE_CONTRACT.applies_to == "video_self_attention_attn1_only"
    targets = lora_target_modules()
    assert len(targets) == 480
    assert targets[0] == "transformer_blocks.0.attn1.to_q"
    assert targets[-1] == "transformer_blocks.47.ff.net.2"
    verify_parameter_free_state_keys(("a", "b"), ("a", "b"))
    with pytest.raises(BackendContractError, match="state keys"):
        verify_parameter_free_state_keys(("a",), ("a", "b"))


def test_sp2_token_shards_are_equal_contiguous_latent_halves() -> None:
    assert contiguous_token_bounds(7680, sp_size=2, sp_rank=0) == (0, 3840)
    assert contiguous_token_bounds(7680, sp_size=2, sp_rank=1) == (3840, 7680)
    with pytest.raises(BackendContractError, match="divisible"):
        contiguous_token_bounds(7681, sp_size=2, sp_rank=0)


def test_sp_aware_hsdp_groups_preserve_sp_columns() -> None:
    topology = build_hsdp_topology(64)
    assert topology.shard_groups[0] == (0, 2, 4, 6)
    assert topology.shard_groups[1] == (1, 3, 5, 7)
    assert topology.replica_groups[0] == (0, 8, 16, 24, 32, 40, 48, 56)
    assert topology.replica_groups[1] == (1, 9, 17, 25, 33, 41, 49, 57)


def test_distributed_contract_checks_global_batch_and_ac48() -> None:
    contract = DistributedContract(
        world_size=256,
        local_world_size=8,
        sp_size=2,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        global_batch_size=256,
        sharding_strategy="HYBRID_SHARD",
        activation_checkpointed_blocks=48,
    )
    assert contract.local_token_bounds == ((0, 3840), (3840, 7680))
    with pytest.raises(BackendContractError, match="global batch"):
        DistributedContract(
            world_size=8,
            local_world_size=8,
            sp_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            global_batch_size=256,
            sharding_strategy="FULL_SHARD",
            activation_checkpointed_blocks=48,
        )


def test_replicated_gradient_clip_avoids_fp32_square_overflow() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    parameter.grad = torch.tensor([2.0e19, -2.0e19], dtype=torch.float32)
    norm = clip_replicated_gradient_norm((parameter,), 1.0)
    assert bool(torch.isfinite(norm))
    assert float(norm) > 2.0e19
    assert float(torch.linalg.vector_norm(parameter.grad)) == pytest.approx(1.0)
