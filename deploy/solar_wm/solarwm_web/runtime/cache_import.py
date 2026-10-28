"""Copy an explicitly selected compiler build; never hard-link a live cache.

Only a runtime-compatible build should be supplied. Import is not acceptance:
normal FX guards, operator source hashes and binary pins must still succeed.
"""
import json
from pathlib import Path
import shutil
import tempfile


def import_dit(cache, sources):
    return import_component(cache, "dit", sources)


def import_component(cache, component, sources):
    if component not in ("dit", "vae-current", "vae-streamrender"):
        raise ValueError("unknown compiler component")
    if set(sources) != {"inductor", "triton"}:
        raise ValueError("both compiler cache directories are required")
    dest = cache.root / component
    identity = {name: str(Path(path).resolve(strict=True)) for name, path in sources.items()}
    marker = dest / "import.json"
    if marker.is_file():
        if json.loads(marker.read_text())["sources"] != identity:
            raise RuntimeError("cache import identity changed")
        return
    if dest.exists():
        raise FileExistsError("refuse to merge into an incomplete or populated cache")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=component + "-import-", dir=dest.parent))
    try:
        for name, path in identity.items():
            source = Path(path)
            if not source.is_dir() or source == dest or dest.is_relative_to(source):
                raise ValueError("distinct source cache directory required")
            shutil.copytree(source, temporary / name, symlinks=False)
        (temporary / "import.json").write_text(json.dumps(dict(sources=identity,
            qualified=False, hardlinks=False, action="copy_only")))
        temporary.rename(dest)
    finally:
        # Retain a failed import for diagnostics; never delete user caches.
        pass
