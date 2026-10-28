from __future__ import annotations

from pathlib import Path

import pytest

from solar.errors import BackendContractError
from solar.runtime.create_only import link_file_create_only, write_file_create_only


def test_create_only_file_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    write_file_create_only(
        target,
        b"first\n",
        error_type=BackendContractError,
        label="artifact",
    )
    with pytest.raises(BackendContractError, match="already exists"):
        write_file_create_only(
            target,
            b"second\n",
            error_type=BackendContractError,
            label="artifact",
        )
    assert target.read_bytes() == b"first\n"
    assert not list(tmp_path.glob("*.partial"))


def test_create_only_link_never_replaces_existing_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable-video")
    destination = tmp_path / "published" / "sample.mp4"

    link_file_create_only(
        source,
        destination,
        error_type=BackendContractError,
        label="test link",
    )

    assert destination.read_bytes() == b"immutable-video"
    assert source.stat().st_ino == destination.stat().st_ino
    with pytest.raises(BackendContractError, match="already exists"):
        link_file_create_only(
            source,
            destination,
            error_type=BackendContractError,
            label="test link",
        )
    assert destination.read_bytes() == b"immutable-video"
