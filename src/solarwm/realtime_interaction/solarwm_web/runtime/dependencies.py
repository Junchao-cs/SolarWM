"""Optional explicit wheel locations; no implicit experiment directory search."""
from pathlib import Path
import os
import site


def install(config):
    # Keep auxiliary DSL/JIT writes with deployment data, not a shared /tmp
    # cache or an original experiment checkout. Set before importing QuACK.
    cache = Path(config["cache_directory"]) / "dsl-runtime"
    for variable, child in (("QUACK_CACHE_DIR", "quack"), ("CUTE_DSL_CACHE_DIR", "cute")):
        directory = cache / child
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    for path in config.get("dependency_sites", []):
        path = Path(path).resolve(strict=True)
        if not path.is_dir():
            raise ValueError("dependency site must be a directory")
        site.addsitedir(str(path))
    # FA2 and FA4 share a Python namespace. Separate --target installations
    # require adding the FA4 package directory, not replacing FA2.
    if config.get("flash_attn_extension"):
        import flash_attn
        path = str(Path(config["flash_attn_extension"]).resolve(strict=True))
        if path not in flash_attn.__path__:
            flash_attn.__path__.insert(0, path)
