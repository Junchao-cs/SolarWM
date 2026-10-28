"""Fail before expensive AOT tracing if a seeded deployment cannot reuse it."""
from contextlib import contextmanager
import json


class SeedCacheMiss(RuntimeError):
    pass


@contextmanager
def require_seeded_aot(evidence):
    from torch._functorch._aot_autograd import autograd_cache as module
    owner = module.AOTAutogradCache
    original_descriptor = owner.__dict__["try_load"]
    original_load = owner.try_load
    original_key = module.autograd_cache_key
    observed = {}

    def record_key(*args, **kwargs):
        key, components = original_key(*args, **kwargs)
        observed.clear()
        observed.update(key=key, components=components)
        return key, components

    def load(*args, **kwargs):
        observed.clear()
        compiled = original_load(*args, **kwargs)
        if compiled is None:
            target = evidence.folder / "seed-cache-miss.json"
            with target.open("x") as output:
                json.dump(dict(observed, status="CACHE_MISS_BEFORE_AOT_REBUILD"), output)
            evidence.emit("seed_cache_miss", key=observed.get("key"))
            raise SeedCacheMiss(
                "seeded AOT cache did not load; stopped before AOT rebuild; "
                f"diagnostic: {target}")
        evidence.emit("seed_cache_hit", key=observed.get("key"))
        return compiled

    owner.try_load = staticmethod(load)
    module.autograd_cache_key = record_key
    try:
        yield
    finally:
        changed = owner.try_load is not load or module.autograd_cache_key is not record_key
        owner.try_load = original_descriptor
        module.autograd_cache_key = original_key
        if changed:
            raise RuntimeError("seeded cache scope ownership changed")
