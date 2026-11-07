"""Enumerate numerical sources after backend/common ownership separation."""
from pathlib import Path


def numerical_sources(package):
    package = Path(package).resolve(strict=True)
    common = Path(__file__).resolve().parent
    result = {}
    for directory in (package / "acceleration", package / "runtime"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                result[path.relative_to(package).as_posix()] = path
    for path in sorted(common.glob("*.py")):
        result["shared/runtime/" + path.name] = path
    return result
