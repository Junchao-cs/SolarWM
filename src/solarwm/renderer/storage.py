"""Read and write local files or object-storage URLs through fsspec.

Install the backend for the storage scheme in use and configure its credentials.
Transient storage operations are retried; missing objects fail immediately.
"""

from __future__ import annotations

import datetime as _dt
import os
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def is_remote(path: str) -> bool:
    return "://" in str(path)


def _fs(path: str):
    from fsspec.core import url_to_fs

    return url_to_fs(path)


def _retry(call: Callable[[], T], tries: int = 4) -> T:
    for attempt in range(tries):
        try:
            return call()
        except FileNotFoundError:
            raise
        except Exception:
            if attempt + 1 == tries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def get(source: str, destination: str) -> None:
    """Copy one remote object to a local file."""

    fs, path = _fs(source)
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    _retry(lambda: fs.get_file(path, destination))


def put(source: str, destination: str) -> None:
    """Copy one local file to a remote object."""

    fs, path = _fs(destination)
    _retry(lambda: fs.put_file(source, path))


def _mtime(info: dict) -> float | None:
    for key in (
        "mtime",
        "LastModified",
        "last_modified",
        "updated",
        "modified",
        "created",
        "timeCreated",
    ):
        value = info.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, _dt.datetime):
            return (value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)).timestamp()
        try:
            return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return None


def stat(path: str) -> tuple[int, float | None] | None:
    """(size, modification time) of one object, or None when it is not there."""

    fs, key = _fs(path)
    try:
        info = _retry(lambda: fs.info(key))
    except FileNotFoundError:
        return None
    if info.get("type") == "directory":
        return None
    return int(info.get("size") or 0), _mtime(info)
