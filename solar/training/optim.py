"""Model-keyed AdamW with FP32 master weights and moments."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import torch

FP32_MASTER_ADAMW_SCHEMA = "solarwm.fp32_master_adamw.v1"
_ACCEPTED_SCHEMAS = {FP32_MASTER_ADAMW_SCHEMA}
_SLOT_KEYS = frozenset({"step", "master_param", "exp_avg", "exp_avg_sq"})


class FP32MasterAdamW(torch.optim.Optimizer):
    """Keep FSDP-visible live parameters but perform every AdamW update in FP32."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        self._validate_hparams(lr, betas, eps, weight_decay)
        defaults = {
            "lr": float(lr),
            "betas": tuple(float(value) for value in betas),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "optimizer_state_schema": FP32_MASTER_ADAMW_SCHEMA,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _validate_hparams(
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        learning_rate = float(lr)
        epsilon = float(eps)
        decay = float(weight_decay)
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError(f"invalid learning rate: {learning_rate}")
        if len(betas) != 2:
            raise ValueError(f"betas must contain two values, got {betas!r}")
        beta1, beta2 = (float(value) for value in betas)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError(f"invalid beta parameters: {(beta1, beta2)!r}")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError(f"invalid epsilon: {epsilon}")
        if not math.isfinite(decay) or decay < 0:
            raise ValueError(f"invalid weight_decay: {decay}")

    @classmethod
    def _validate_group(cls, group: Mapping[str, Any]) -> None:
        schema = str(group.get("optimizer_state_schema", ""))
        if schema not in _ACCEPTED_SCHEMAS:
            raise ValueError(f"unsupported FP32-master optimizer state schema: {schema!r}")
        cls._validate_hparams(
            float(group["lr"]),
            tuple(group["betas"]),
            float(group["eps"]),
            float(group["weight_decay"]),
        )

    @staticmethod
    def _initialize_state(parameter: torch.nn.Parameter) -> dict[str, torch.Tensor]:
        if not parameter.is_floating_point():
            raise TypeError(
                f"FP32MasterAdamW supports floating-point parameters only, got {parameter.dtype}"
            )
        return {
            "step": torch.zeros((), dtype=torch.float32, device="cpu"),
            "master_param": parameter.detach().to(dtype=torch.float32, copy=True),
            "exp_avg": torch.zeros_like(parameter, dtype=torch.float32),
            "exp_avg_sq": torch.zeros_like(parameter, dtype=torch.float32),
        }

    @staticmethod
    def _validate_live_state(parameter: torch.nn.Parameter, state: Mapping[str, Any]) -> None:
        keys = set(state)
        if keys != _SLOT_KEYS:
            raise RuntimeError(
                "invalid FP32-master state slots: "
                f"missing={sorted(_SLOT_KEYS - keys)} unexpected={sorted(keys - _SLOT_KEYS)}"
            )
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or step.device.type != "cpu"
            or step.dtype != torch.float32
        ):
            raise RuntimeError("FP32MasterAdamW step must be a CPU float32 scalar")
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                raise RuntimeError(f"FP32MasterAdamW {key} must be a float32 tensor")
            if value.device != parameter.device or value.shape != parameter.shape:
                raise RuntimeError(f"FP32MasterAdamW {key} layout differs from its live parameter")

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate_group(group)
            lr = float(group["lr"])
            beta1, beta2 = (float(value) for value in group["betas"])
            epsilon = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("FP32MasterAdamW does not support sparse gradients")
                if not gradient.is_floating_point():
                    raise TypeError("FP32MasterAdamW requires floating-point gradients")
                state = self.state[parameter]
                if not state:
                    state.update(self._initialize_state(parameter))
                self._validate_live_state(parameter, state)
                step_tensor = state["step"]
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                gradient_fp32 = gradient.detach().to(dtype=torch.float32)
                step_tensor.add_(1)
                step_value = int(step_tensor.item())
                if weight_decay:
                    master.mul_(1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(gradient_fp32, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient_fp32, gradient_fp32, value=1 - beta2)
                bias1 = 1 - beta1**step_value
                bias2 = 1 - beta2**step_value
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias2)).add_(epsilon)
                master.addcdiv_(exp_avg, denominator, value=-(lr / bias1))
                parameter.copy_(master)
        return loss

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore shaped slots without PyTorch casting them to the model dtype."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("optimizer state_dict must be a mapping")
        raw_state = state_dict.get("state")
        raw_groups = state_dict.get("param_groups")
        if not isinstance(raw_state, Mapping):
            raise ValueError("optimizer state_dict['state'] must be a mapping")
        if not isinstance(raw_groups, (list, tuple)):
            raise ValueError("optimizer state_dict['param_groups'] must be a list")
        if len(raw_groups) != len(self.param_groups):
            raise ValueError("optimizer parameter-group count mismatch")

        id_to_parameter: dict[Any, torch.nn.Parameter] = {}
        restored_groups: list[dict[str, Any]] = []
        for group_index, (saved_group, current_group) in enumerate(
            zip(raw_groups, self.param_groups, strict=True)
        ):
            if not isinstance(saved_group, Mapping):
                raise ValueError(f"optimizer parameter group {group_index} is not a mapping")
            saved_ids = saved_group.get("params")
            if not isinstance(saved_ids, (list, tuple)):
                raise ValueError(f"optimizer parameter group {group_index} lacks IDs")
            current_parameters = list(current_group["params"])
            if len(saved_ids) != len(current_parameters):
                raise ValueError(f"optimizer parameter count mismatch in group {group_index}")
            for saved_id, parameter in zip(saved_ids, current_parameters, strict=True):
                if saved_id in id_to_parameter:
                    raise ValueError(f"optimizer checkpoint repeats ID {saved_id!r}")
                id_to_parameter[saved_id] = parameter
            restored = copy.deepcopy(dict(saved_group))
            restored["params"] = current_parameters
            self._validate_group(restored)
            # Preserve the checkpoint format on exact resume. Newly constructed
            # optimizers emit the SolarWM schema.
            restored_groups.append(restored)

        unknown = [saved_id for saved_id in raw_state if saved_id not in id_to_parameter]
        if unknown:
            raise ValueError(f"optimizer state has unknown parameter IDs: {unknown[:8]!r}")
        restored_state: defaultdict[torch.nn.Parameter, dict[str, torch.Tensor]] = defaultdict(dict)
        for saved_id, raw_slots in raw_state.items():
            if not isinstance(raw_slots, Mapping):
                raise ValueError(f"optimizer state for ID {saved_id!r} is not a mapping")
            if not raw_slots:
                continue
            if set(raw_slots) != _SLOT_KEYS:
                raise ValueError(f"invalid FP32-master slots for ID {saved_id!r}")
            parameter = id_to_parameter[saved_id]
            raw_step = raw_slots["step"]
            step_value = (
                float(raw_step.detach().cpu().item())
                if isinstance(raw_step, torch.Tensor) and raw_step.numel() == 1
                else float(raw_step)
            )
            if not math.isfinite(step_value) or step_value < 0 or not step_value.is_integer():
                raise ValueError("optimizer checkpoint step must be a non-negative integer")
            slots: dict[str, torch.Tensor] = {
                "step": torch.tensor(step_value, dtype=torch.float32, device="cpu")
            }
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                value = raw_slots[key]
                if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                    raise ValueError(f"optimizer checkpoint {key} must be floating")
                if value.shape != parameter.shape:
                    raise ValueError(f"optimizer checkpoint {key} shape mismatch")
                slots[key] = value.detach().to(
                    device=parameter.device, dtype=torch.float32, copy=True
                )
            self._validate_live_state(parameter, slots)
            restored_state[parameter] = slots
        self.param_groups = restored_groups
        self.state = restored_state


__all__ = [
    "FP32_MASTER_ADAMW_SCHEMA",
    "FP32MasterAdamW",
]
