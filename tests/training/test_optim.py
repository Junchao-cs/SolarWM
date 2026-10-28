from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from solar.training.optim import (  # noqa: E402
    FP32_MASTER_ADAMW_SCHEMA,
    FP32MasterAdamW,
)


def test_bfloat16_parameter_uses_fp32_master_and_moments() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    optimizer = FP32MasterAdamW([parameter], lr=1e-2)
    parameter.grad = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["master_param"].dtype == torch.float32
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert optimizer.param_groups[0]["optimizer_state_schema"] == FP32_MASTER_ADAMW_SCHEMA
