import hashlib
import json
from types import SimpleNamespace

import pytest

from solarwm.backends.minimax_h3.demo_conditions import source_digest, stage_source
from solarwm.backends.minimax_h3.source_length_cache import (
    preparation_identity,
    prepared_case_receipt,
)


def test_demo_sources_are_bound_to_all_three_inputs(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    row = dict(key="case", input_kind="demo")
    for field in ("image", "camera", "prompt"):
        path = root / (field + ".dat")
        path.write_bytes(field.encode())
        row[field] = path.name
        row[field + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    args = SimpleNamespace(
        dataset_root=str(root), work_dir=str(tmp_path / "cache"), config={"model": {}}
    )
    case = stage_source(row, args)
    (case / "condition.safetensors").write_bytes(b"condition")
    receipt = dict(
        preparation_identity=preparation_identity(row, args),
        source_sha256=source_digest(case),
        condition_sha256=hashlib.sha256(b"condition").hexdigest(),
    )
    (case / "READY.json").write_text(json.dumps(receipt))
    assert prepared_case_receipt(case, row, args) == receipt
    (case / "source.prompt.txt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        prepared_case_receipt(case, row, args)
    with pytest.raises(ValueError, match="another input"):
        stage_source(row, args)
