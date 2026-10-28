"""Launch provenance assembled before model allocation or data iteration."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from solar import __version__
from solar.config.loader import canonical_json
from solar.errors import ConfigurationError

_SECRET_KEYS = {"access_token", "api_key", "password", "secret", "token"}
_IMMUTABLE_IMAGE = re.compile(r"[^\s@]+@[a-z0-9][a-z0-9._+-]*:[0-9a-f]{32,}\Z")


@dataclass(frozen=True)
class GitIdentity:
    available: bool
    root: str = ""
    commit: str = ""
    tree: str = ""
    clean: bool = False
    status_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and process.returncode:
        raise ConfigurationError(
            f"git {' '.join(arguments)} failed for {root}: {process.stderr.strip()}"
        )
    return process.stdout


def git_identity(path: str | Path) -> GitIdentity:
    source = Path(path).resolve()
    process = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if process.returncode:
        return GitIdentity(available=False)
    root = Path(process.stdout.strip()).resolve()
    commit = _git(root, "rev-parse", "HEAD").strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").strip()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return GitIdentity(
        available=True,
        root=str(root),
        commit=commit,
        tree=tree,
        clean=not bool(status),
        status_digest=hashlib.blake2s(status.encode()).hexdigest(),
    )


def reject_inline_secrets(value: Any, path: tuple[str, ...] = ()) -> None:
    """Prevent resolved-config artifacts from accidentally persisting credentials."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            child_path = (*path, str(key))
            if normalized in _SECRET_KEYS and child is not None and child != "":
                raise ConfigurationError(
                    f"inline credential at {'.'.join(child_path)} is forbidden; "
                    "use a workload identity or *_file reference"
                )
            reject_inline_secrets(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_inline_secrets(child, (*path, str(index)))


def build_launch_manifest(
    *,
    config: Mapping[str, Any],
    source_config: Path,
    source_digest: str,
    resolved_digest: str,
    route: str,
    repository: str | Path,
) -> dict[str, Any]:
    reject_inline_secrets(config)
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ConfigurationError("runtime must be a mapping")
    declared_image = str(runtime.get("image") or "")
    if bool(runtime.get("enforce_image", False)):
        if not declared_image:
            raise ConfigurationError("runtime.enforce_image=true requires runtime.image")
        if not _IMMUTABLE_IMAGE.fullmatch(declared_image):
            raise ConfigurationError("runtime.enforce_image=true requires a digest-pinned image")

    source = git_identity(repository)
    if bool(runtime.get("require_clean_source", False)) and (
        not source.available or not source.clean
    ):
        raise ConfigurationError("runtime.require_clean_source=true requires a clean Git checkout")

    identity = {
        "schema": "solarwm.launch.v2",
        "version": __version__,
        "action": str(config["action"]),
        "name": str(config["name"]),
        "route": route,
        "source_config": {
            "path": str(source_config),
            "digest": source_digest,
        },
        "resolved_config_digest": resolved_digest,
        "source": source.as_dict(),
        "runtime": {
            "declared_image": declared_image,
            "image_enforced": bool(runtime.get("enforce_image", False)),
        },
        "data": {
            "index": str(config.get("data", {}).get("index", "")),
            "test_index": str(config.get("data", {}).get("test_index", "")),
            "generation": str(config.get("data", {}).get("generation", "")),
        },
        "checkpoint": dict(config.get("checkpoint", {})),
        "metadata": dict(config.get("metadata", {})),
    }
    identity["launch_identity_digest"] = hashlib.blake2s(canonical_json(identity)).hexdigest()
    return identity
