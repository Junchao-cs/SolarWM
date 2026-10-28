"""Atomic create-only directory publication for local staging trees."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import stat
import uuid
from pathlib import Path


def _raise(error_type: type[Exception], message: str) -> None:
    raise error_type(message)


def _rename_no_replace(
    source: Path,
    destination: Path,
    *,
    error_type: type[Exception],
    label: str,
) -> None:
    """Call Linux renameat2 with RENAME_NOREPLACE and no unsafe fallback."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _raise(
            error_type,
            f"{label} requires Linux renameat2(RENAME_NOREPLACE)",
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        _raise(error_type, f"{label} target appeared during publication: {destination}")
    _raise(error_type, f"{label} create-only publication failed: {os.strerror(error)}")


def publish_directory_no_replace(
    source: str | Path,
    destination: str | Path,
    *,
    error_type: type[Exception],
    label: str,
) -> None:
    """Publish one sibling staging directory without replacing any target.

    The parent advisory lock serializes cooperating writers. Linux
    ``RENAME_NOREPLACE`` is still mandatory so an uncooperative writer racing
    after the check cannot be overwritten.
    """

    raw_staging = Path(source).expanduser()
    raw_target = Path(destination).expanduser()
    staging = raw_staging.parent.resolve() / raw_staging.name
    target = raw_target.parent.resolve() / raw_target.name
    if not staging.is_dir() or staging.is_symlink():
        _raise(error_type, f"{label} staging directory is missing or is a symlink")
    if staging.parent != target.parent:
        _raise(error_type, f"{label} staging and target must be siblings")
    lock_path = target.parent / ".solarwm-directory-publication.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _rename_no_replace(
                staging,
                target,
                error_type=error_type,
                label=label,
            )
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def write_file_create_only(
    path: str | Path,
    payload: bytes,
    *,
    error_type: type[Exception],
    label: str,
    mode: int = 0o644,
) -> None:
    """Atomically create one immutable file without replacing a race winner.

    The payload is fsynced under a private sibling name, then hard-linked to
    the public name. ``link(2)`` is the commit point and fails when the target
    already exists; unlike ``os.replace`` it can never overwrite a prior artifact.
    """

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _raise(error_type, f"{label} made no progress while writing {temporary}")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _raise(error_type, f"{label} already exists: {destination}")
        temporary.unlink()
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def link_file_create_only(
    source: str | Path,
    destination: str | Path,
    *,
    error_type: type[Exception],
    label: str,
) -> None:
    """Atomically publish an existing immutable file without copying its bytes."""

    source_path = Path(source).expanduser()
    target = Path(destination).expanduser()
    try:
        source_stat = source_path.stat(follow_symlinks=False)
    except OSError as exc:
        _raise(error_type, f"{label} source is not readable: {source_path}: {exc}")
    if not stat.S_ISREG(source_stat.st_mode) or source_path.is_symlink():
        _raise(error_type, f"{label} source must be a regular non-symlink file: {source_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, target, follow_symlinks=False)
    except FileExistsError:
        _raise(error_type, f"{label} already exists: {target}")
    except OSError as exc:
        _raise(error_type, f"{label} create-only link failed: {exc}")
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = [
    "link_file_create_only",
    "publish_directory_no_replace",
    "write_file_create_only",
]
