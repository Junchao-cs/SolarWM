"""FULL_SHARD and activation-checkpointing setup for H3."""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from .distributed import get_dp_group, init_sequence_parallel


def initialize_distributed(*, sp_size: int, local_rank: int) -> tuple[int, int]:
    """Initialize NCCL and H3's SP/DP process groups under torchrun."""

    import os

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=1))
    init_sequence_parallel(sp_size)
    return rank, world


def wrap_h3_fsdp(
    model: torch.nn.Module,
    *,
    local_rank: int,
    transformer_block_cls: type[torch.nn.Module],
    fp32_units: Iterable[torch.nn.Module],
    ignored_parameters: Iterable[torch.nn.Parameter],
    activation_checkpointing: bool = True,
    frozen_base_shard_size: int | None = None,
) -> torch.nn.Module:
    """Wrap 50 H3 blocks and the six FP32 leaf owners over logical DP."""

    if not dist.is_initialized() or dist.get_world_size() == 1:
        if activation_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        return model.to(torch.device("cuda", local_rank))
    standalone = tuple(fp32_units)
    standalone_ids = {id(module) for module in standalone}
    if len(standalone_ids) not in {6, 8}:
        raise ValueError("H3 FSDP requires six FP32 leaf owners, plus two for AnyFlow")
    ignored = tuple(ignored_parameters)
    if len({id(parameter) for parameter in ignored}) != len(ignored):
        raise ValueError("H3 FSDP ignored parameters contain duplicates")

    def policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        del nonwrapped_numel
        if recurse:
            return True
        return isinstance(module, transformer_block_cls) or id(module) in standalone_ids

    process_group = get_dp_group()
    strategy = ShardingStrategy.FULL_SHARD
    if frozen_base_shard_size is not None:
        process_group = frozen_base_groups(frozen_base_shard_size)
        strategy = ShardingStrategy.HYBRID_SHARD
    wrapped = FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=strategy,
        mixed_precision=MixedPrecision(
            param_dtype=None, reduce_dtype=torch.float32, buffer_dtype=None
        ),
        cpu_offload=None,
        device_id=local_rank,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sync_module_states=False,
        process_group=process_group,
        limit_all_gathers=True,
        ignored_states=ignored or None,
    )
    if activation_checkpointing:
        wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=wrapper,
            check_fn=lambda module: isinstance(module, transformer_block_cls),
        )
        blocks = sum(isinstance(module, transformer_block_cls) for module in wrapped.modules())
        if blocks != 50:
            raise RuntimeError(f"H3 activation checkpointing expected 50 blocks, got {blocks}")
    return wrapped


@torch.no_grad()
def finite_clip_norm(
    parameters: Iterable[torch.nn.Parameter], max_norm: float
) -> tuple[bool, float]:
    """Clip replicated LoRA gradients with an overflow-safe global 2-norm."""

    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be finite and positive")
    values = tuple(parameter for parameter in parameters if parameter.grad is not None)
    if not values:
        raise RuntimeError("H3 optimizer step has no LoRA gradients")
    device = values[0].grad.device
    squared = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in values:
        gradient = parameter.grad
        if gradient.is_sparse:
            raise RuntimeError("H3 does not support sparse LoRA gradients")
        detached = gradient.detach()
        norm_dtype = torch.float64 if detached.dtype == torch.float64 else torch.float32
        local_scale = detached.abs().amax().double()
        safe_scale = torch.where(local_scale > 0, local_scale, torch.ones_like(local_scale))
        scaled_norm = torch.linalg.vector_norm(
            detached / safe_scale.to(dtype=detached.dtype),
            ord=2,
            dtype=norm_dtype,
        )
        squared.add_((local_scale * scaled_norm.double()).square())
    norm = squared.sqrt()
    finite = bool(torch.isfinite(norm).item())
    if finite:
        coefficient = min(1.0, max_norm / (float(norm.item()) + 1.0e-6))
        if coefficient < 1.0:
            for parameter in values:
                parameter.grad.mul_(coefficient)
    return finite, float(norm.float().item())


__all__ = ["finite_clip_norm", "initialize_distributed", "wrap_h3_fsdp"]


_FROZEN_BASE_GROUPS: dict[tuple[int, int], tuple[object, object]] = {}


def frozen_base_groups(shard_size: int) -> tuple[object, object]:
    """Shard only the frozen base over physical node GPUs; LoRA keeps SP/DP."""
    import os

    world, rank = dist.get_world_size(), dist.get_rank()
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    if shard_size != local_world or world % shard_size:
        raise ValueError("H3 frozen-base shards must be exactly one complete physical node")
    key = (world, shard_size)
    if key not in _FROZEN_BASE_GROUPS:
        shard, replica = None, None
        for first in range(0, world, shard_size):
            ranks = list(range(first, first + shard_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                shard = group
        for offset in range(shard_size):
            ranks = list(range(offset, world, shard_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                replica = group
        if shard is None or replica is None:
            raise RuntimeError("H3 frozen-base process groups are incomplete")
        _FROZEN_BASE_GROUPS[key] = (shard, replica)
    return _FROZEN_BASE_GROUPS[key]
