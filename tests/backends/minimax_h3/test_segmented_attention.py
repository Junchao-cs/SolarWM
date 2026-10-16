from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from solarwm.backends.minimax_h3.distributed import validate_packed_attention_mask
from solarwm.backends.minimax_h3.segmented_attention import H3SegmentPlan


def test_segment_plan_subclasses_keep_the_global_sequence_contract():
    class SpecializedPlan(H3SegmentPlan):
        pass

    plan = SpecializedPlan(((0, 2, ((0, 2),), (), None),), 2)
    validate_packed_attention_mask(plan, total_tokens=2)
    with pytest.raises(ValueError, match="complete global sequence"):
        validate_packed_attention_mask(plan, total_tokens=3)


def test_unrelated_type_named_segment_plan_is_not_an_attention_plan():
    unrelated = type("H3SegmentPlan", (), {"sequence_length": 2})()
    with pytest.raises(ValueError, match="accepts only"):
        validate_packed_attention_mask(unrelated, total_tokens=2)


def test_shared_attention_contract_imports_without_torch_or_renderer():
    script = """
import importlib.abc
import sys

class RestrictedImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('attention contracts imported torch')
        if fullname == 'solarwm.renderer' or fullname.startswith('solarwm.renderer.'):
            raise AssertionError('H3 backend imported the renderer extension')

sys.meta_path.insert(0, RestrictedImports())
from solarwm.backends.minimax_h3.distributed import validate_packed_attention_mask
from solarwm.backends.minimax_h3.segmented_attention import H3ChunkKVCache, H3SegmentPlan

plan = H3SegmentPlan(((0, 2, ((0, 2),), (), None),), 2, H3ChunkKVCache())
validate_packed_attention_mask(None, total_tokens=2)
validate_packed_attention_mask(plan, total_tokens=2)
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
assert not any(name.startswith('solarwm.renderer') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
