"""Snapshot this checkout's SolarWM package into the Reactor build context."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

UPSTREAM_REVISION = "e018b0f673de865711fe99844d3976865722b8d0"


def stage_source(repository: Path, destination: Path) -> dict:
    """Include local edits and record hashes; never fetch a second renderer core."""
    repository = repository.resolve()
    if destination.is_symlink():
        raise ValueError("The source staging destination must not be a symlink")
    destination = destination.resolve()
    package = repository / "solar"
    if not (package / "renderer/render.py").is_file():
        raise FileNotFoundError(f"Integrated SolarWM renderer not found under {repository}")
    if destination == package or destination in package.parents or package in destination.parents:
        raise ValueError("The staging destination must not overwrite or nest inside source")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        package,
        destination / "solar",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(repository / name, destination / name)
    revision, changes = None, None
    if (repository / ".git").exists() and shutil.which("git"):
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain", "--", "solar"], text=True
        )
    provenance = {
        "solarwm_revision": revision,
        "source_archive": revision is None,
        "solarwm_source_has_local_changes": None if changes is None else bool(changes.strip()),
        "reactor_upstream_revision": UPSTREAM_REVISION,
        "files": {
            str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        },
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    adapter = Path(__file__).resolve().parent
    result = stage_source(adapter.parents[2], adapter / ".solarwm-source")
    print(
        f"Staged {len(result['files'])} files from SolarWM {result['solarwm_revision']} "
        f"(local source changes: {result['solarwm_source_has_local_changes']})."
    )
