"""FP32-master AdamW for BF16 H3 LoRA parameters."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import torch

FP32_MASTER_ADAMW_SCHEMA = "solarwm.fp32-master-adamw.v1"
_SLOTS = frozenset({"step", "master_param", "exp_avg", "exp_avg_sq"})


class FP32MasterAdamW(torch.optim.Optimizer):
    """AdamW keyed by live model params with FP32 master weights and moments."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        self._validate_hparams(lr, betas, eps, weight_decay)
        super().__init__(
            params,
            {
                "lr": float(lr),
                "betas": tuple(float(value) for value in betas),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
                "optimizer_state_schema": FP32_MASTER_ADAMW_SCHEMA,
            },
        )
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _validate_hparams(
        lr: float, betas: tuple[float, float], eps: float, weight_decay: float
    ) -> None:
        if not math.isfinite(float(lr)) or float(lr) < 0:
            raise ValueError("learning rate must be finite and non-negative")
        if len(betas) != 2 or any(not 0 <= float(value) < 1 for value in betas):
            raise ValueError("AdamW betas must be in [0,1)")
        if not math.isfinite(float(eps)) or float(eps) < 0:
            raise ValueError("epsilon must be finite and non-negative")
        if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0:
            raise ValueError("weight decay must be finite and non-negative")

    @classmethod
    def _validate_group(cls, group: Mapping[str, Any]) -> None:
        if group.get("optimizer_state_schema") != FP32_MASTER_ADAMW_SCHEMA:
            raise ValueError("unsupported FP32-master optimizer state schema")
        cls._validate_hparams(
            group["lr"], tuple(group["betas"]), group["eps"], group["weight_decay"]
        )

    @staticmethod
    def _new_state(parameter: torch.nn.Parameter) -> dict[str, torch.Tensor]:
        if not parameter.is_floating_point():
            raise TypeError("FP32MasterAdamW supports floating parameters only")
        return {
            "step": torch.zeros((), dtype=torch.float32, device="cpu"),
            "master_param": parameter.detach().to(dtype=torch.float32, copy=True),
            "exp_avg": torch.zeros_like(parameter, dtype=torch.float32),
            "exp_avg_sq": torch.zeros_like(parameter, dtype=torch.float32),
        }

    @staticmethod
    def _validate_state(parameter: torch.nn.Parameter, values: Mapping[str, Any]) -> None:
        if set(values) != _SLOTS:
            raise RuntimeError("FP32MasterAdamW state slots differ")
        step = values["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or step.device.type != "cpu"
            or step.dtype != torch.float32
        ):
            raise RuntimeError("optimizer step must be a CPU FP32 scalar")
        for name in ("master_param", "exp_avg", "exp_avg_sq"):
            value = values[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.device != parameter.device
                or tuple(value.shape) != tuple(parameter.shape)
            ):
                raise RuntimeError(f"invalid FP32MasterAdamW slot {name!r}")

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate_group(group)
            lr = float(group["lr"])
            beta1, beta2 = (float(value) for value in group["betas"])
            epsilon = float(group["eps"])
            decay = float(group["weight_decay"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("FP32MasterAdamW does not support sparse gradients")
                values = self.state[parameter]
                if not values:
                    values.update(self._new_state(parameter))
                self._validate_state(parameter, values)
                values["step"].add_(1)
                step = int(values["step"].item())
                master = values["master_param"]
                first = values["exp_avg"]
                second = values["exp_avg_sq"]
                gradient_fp32 = gradient.detach().float()
                if decay:
                    master.mul_(1.0 - lr * decay)
                first.mul_(beta1).add_(gradient_fp32, alpha=1.0 - beta1)
                second.mul_(beta2).addcmul_(gradient_fp32, gradient_fp32, value=1.0 - beta2)
                correction1 = 1.0 - beta1**step
                correction2 = 1.0 - beta2**step
                denominator = second.sqrt().div_(math.sqrt(correction2)).add_(epsilon)
                master.addcdiv_(first, denominator, value=-(lr / correction1))
                parameter.copy_(master)
        return loss

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore slots without PyTorch casting them to the live BF16 dtype."""

        raw_state = state_dict.get("state")
        raw_groups = state_dict.get("param_groups")
        if not isinstance(raw_state, Mapping) or not isinstance(raw_groups, (list, tuple)):
            raise ValueError("invalid optimizer state dictionary")
        if len(raw_groups) != len(self.param_groups):
            raise ValueError("optimizer group count differs")
        id_to_parameter: dict[Any, torch.nn.Parameter] = {}
        restored_groups: list[dict[str, Any]] = []
        for saved, current in zip(raw_groups, self.param_groups, strict=True):
            saved_ids = saved.get("params")
            if not isinstance(saved_ids, (list, tuple)):
                raise ValueError("optimizer group has no saved parameter IDs")
            parameters = list(current["params"])
            if len(saved_ids) != len(parameters):
                raise ValueError("optimizer parameter count differs")
            id_to_parameter.update(zip(saved_ids, parameters, strict=True))
            restored = copy.deepcopy(dict(saved))
            restored["params"] = parameters
            self._validate_group(restored)
            restored_groups.append(restored)
        if any(identifier not in id_to_parameter for identifier in raw_state):
            raise ValueError("optimizer state references an unknown parameter")
        restored_state: defaultdict[torch.nn.Parameter, dict[str, torch.Tensor]] = defaultdict(dict)
        for identifier, raw_slots in raw_state.items():
            if not raw_slots:
                continue
            if not isinstance(raw_slots, Mapping) or set(raw_slots) != _SLOTS:
                raise ValueError("optimizer checkpoint slots differ")
            parameter = id_to_parameter[identifier]
            raw_step = raw_slots["step"]
            step = float(
                raw_step.detach().cpu().item() if isinstance(raw_step, torch.Tensor) else raw_step
            )
            if not math.isfinite(step) or step < 0 or not step.is_integer():
                raise ValueError("optimizer checkpoint step is invalid")
            slots = {"step": torch.tensor(step, dtype=torch.float32, device="cpu")}
            for name in ("master_param", "exp_avg", "exp_avg_sq"):
                value = raw_slots[name]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(
                    parameter.shape
                ):
                    raise ValueError(f"optimizer checkpoint slot {name!r} differs")
                slots[name] = value.detach().to(
                    device=parameter.device, dtype=torch.float32, copy=True
                )
            self._validate_state(parameter, slots)
            restored_state[parameter] = slots
        self.param_groups = restored_groups
        self.state = restored_state


__all__ = ["FP32_MASTER_ADAMW_SCHEMA", "FP32MasterAdamW"]
