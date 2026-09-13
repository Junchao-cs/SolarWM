"""Opt-in target-time conditioning for MiniMax-H3 AnyFlow.

Both time inputs use H3's native data-ward coordinate, ``time = 1 - sigma``.
The second MLP is cloned from the loaded base model; enabling this extension
does not initialise random parameters or alter the ordinary FM state dict.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import torch


class H3AnyFlowConditioningMixin:
    """Add AnyFlow only after an upstream H3 checkpoint has strict-loaded.

    The owning model supplies ``time_proj`` and ``time_embedder``. Call
    ``enable_anyflow`` before PEFT/FSDP wrapping so the additional FP32 MLP
    participates in the same freezing and sharding rules as the original.
    """

    @property
    def uses_anyflow(self) -> bool:
        return hasattr(self, "delta_embedding")

    def enable_anyflow(self, gate: float = 0.25) -> bool:
        """Clone the loaded time MLP once, preserving RNG and base parameters.

        Return whether the extension was created. Repeating the call with the
        same gate is harmless and never resets a learned delta embedding.
        The gate is configuration, not trainable or persistent model state.
        """

        gate = float(gate)
        if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
            raise ValueError("H3 AnyFlow gate must be finite and in [0, 1]")
        if self.uses_anyflow:
            if float(self.anyflow_gate_buffer.item()) != float(
                torch.tensor(gate, dtype=self.anyflow_gate_buffer.dtype).item()
            ):
                raise ValueError("H3 AnyFlow is already enabled with a different gate")
            return False
        self.delta_embedding = copy.deepcopy(self.time_embedder)
        self.register_buffer(
            "anyflow_gate_buffer",
            torch.tensor(gate, dtype=torch.float32),
            persistent=False,
        )
        return True

    def initialize_anyflow_delta_from_time(self) -> bool:
        """Reset delta from the currently loaded time MLP before FSDP wrapping."""

        if not self.uses_anyflow:
            return False
        self.delta_embedding.load_state_dict(self.time_embedder.state_dict(), strict=True)
        return True

    def _time_condition(
        self,
        timestep: torch.Tensor,
        r_timestep: torch.Tensor | None = None,
        *,
        parameter_dtype: Callable[[torch.nn.Module], torch.dtype],
    ) -> torch.Tensor:
        """Embed native ``(t, r)`` and mix before block and output modulation.

        The adapter supplies its FSDP-aware dtype lookup. Keeping this lookup
        at the boundary lets the conditioning also run in CPU-only math tests
        without importing the full Diffusers H3 model.
        """

        if self.uses_anyflow:
            if r_timestep is None:
                raise ValueError("H3 AnyFlow model forward requires r_timestep")
            if r_timestep.shape != timestep.shape:
                raise ValueError(
                    f"r_timestep shape {tuple(r_timestep.shape)} != "
                    f"timestep shape {tuple(timestep.shape)}"
                )
        elif r_timestep is not None:
            raise ValueError("r_timestep requires H3 AnyFlow conditioning")

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(parameter_dtype(self.time_embedder)))
        if not self.uses_anyflow:
            return temb
        remb = self.time_proj(r_timestep)
        remb = self.delta_embedding(remb.to(parameter_dtype(self.delta_embedding)))
        gate = self.anyflow_gate_buffer.to(device=temb.device, dtype=temb.dtype)
        return (1.0 - gate) * temb + gate * remb


__all__ = ["H3AnyFlowConditioningMixin"]
