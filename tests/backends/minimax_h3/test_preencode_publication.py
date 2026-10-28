from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from solar.backends.minimax_h3 import preencode_runner
from solar.errors import DataContractError
from solar.runtime.create_only import publish_directory_no_replace


def test_h3_last_moment_target_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".h3.partial"
    staging.mkdir()
    (staging / preencode_runner.H3_COMPLETE_PATH).write_text("complete")
    target = tmp_path / "h3-output"

    def race(source: Path, destination: Path, **kwargs: Any) -> None:
        destination.mkdir()
        (destination / "owner.txt").write_text("other writer")
        publish_directory_no_replace(source, destination, **kwargs)

    monkeypatch.setattr(preencode_runner, "publish_directory_no_replace", race)
    with pytest.raises(DataContractError, match="target appeared during publication"):
        preencode_runner._publish_staging(staging, target)
    assert (target / "owner.txt").read_text() == "other writer"
    assert not (target / preencode_runner.H3_COMPLETE_PATH).exists()
    assert (staging / preencode_runner.H3_COMPLETE_PATH).is_file()
