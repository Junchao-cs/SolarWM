"""Deterministic serialization helpers shared by runtime artifacts."""

from __future__ import annotations

import json
from typing import Any


def canonical_json_bytes(value: Any, *, trailing_newline: bool = True) -> bytes:
    """Serialize JSON deterministically and reject non-standard numeric values."""

    suffix = "\n" if trailing_newline else ""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + suffix
    ).encode("utf-8")


__all__ = ["canonical_json_bytes"]
