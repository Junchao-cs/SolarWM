"""Wan shifted rectified-flow scheduler."""

from __future__ import annotations

from typing import Any


def build_wan_flow_unipc_scheduler(
    *,
    num_train_timesteps: int,
    shift: float,
    num_inference_steps: int,
    device: Any,
) -> Any:
    """Build public UniPC while pinning the frozen validation flow grid.

    Diffusers changed the endpoint/rounding behavior of ``use_flow_sigmas``
    after the reference image was frozen.  The solver update remains
    compatible, but its generated grid can drift by several raw timesteps.
    Replacing only the schedule with the original ``0.999 -> 0`` contract
    keeps release inference identical across supported Diffusers versions.
    """

    import numpy as np
    import torch
    from diffusers import UniPCMultistepScheduler

    train_steps = int(num_train_timesteps)
    inference_steps = int(num_inference_steps)
    flow_shift = float(shift)
    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=train_steps,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=flow_shift,
    )
    scheduler.set_timesteps(inference_steps, device=device)
    base = np.linspace(
        1.0 - 1.0 / train_steps,
        0.0,
        inference_steps + 1,
        dtype=np.float64,
    )[:-1]
    shifted = flow_shift * base / (1.0 + (flow_shift - 1.0) * base)
    scheduler.sigmas = torch.from_numpy(
        np.concatenate([shifted, np.asarray([0.0])]).astype(np.float32)
    )
    scheduler.timesteps = torch.from_numpy((shifted * train_steps).astype(np.int64)).to(
        device=device
    )
    scheduler.num_inference_steps = inference_steps
    scheduler.model_outputs = [None] * int(scheduler.config.solver_order)
    scheduler.lower_order_nums = 0
    scheduler.last_sample = None
    scheduler._step_index = None
    scheduler._begin_index = None
    return scheduler


class FlowMatchScheduler:
    """The shifted linear schedule used by Wan training routes."""

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        extra_one_step: bool = True,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.extra_one_step = bool(extra_one_step)
        self.set_timesteps(self.num_train_timesteps, training=True)

    def set_timesteps(self, num_inference_steps: int, *, training: bool = False) -> None:
        import torch

        steps = int(num_inference_steps)
        count = steps + 1 if self.extra_one_step else steps
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, count)
        if self.extra_one_step:
            sigmas = sigmas[:-1]
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            weighting = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
            shifted = weighting - weighting.min()
            self.linear_timesteps_weights = shifted * (steps / shifted.sum())

    def _indices(self, timestep: Any, *, device: Any) -> Any:
        values = timestep.flatten() if timestep.ndim == 2 else timestep
        schedule = self.timesteps.to(device=device)
        return (schedule.unsqueeze(0) - values.unsqueeze(1)).abs().argmin(dim=1)

    def add_noise(self, clean: Any, noise: Any, timestep: Any) -> Any:
        indices = self._indices(timestep, device=noise.device)
        sigma = self.sigmas.to(noise.device)[indices].reshape(-1, 1, 1, 1)
        return ((1 - sigma) * clean + sigma * noise).type_as(noise)

    @staticmethod
    def training_target(clean: Any, noise: Any, timestep: Any) -> Any:
        del timestep
        return noise - clean

    def training_weight(self, timestep: Any) -> Any:
        indices = self._indices(timestep, device=timestep.device)
        return self.linear_timesteps_weights.to(timestep.device)[indices]


__all__ = ["FlowMatchScheduler", "build_wan_flow_unipc_scheduler"]
