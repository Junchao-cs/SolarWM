"""Explicit compiler cache ownership; no cache deletion or Dynamo reset."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading


_LOCK = threading.RLock()
_ACTIVE = None


def cache_key(*, sources_sha256, runtime, inference):
    """Include actual runtime/config in a reusable disk-cache namespace.

    `runtime` must be observed by the process that will run inference. This
    deterministic function alone does not authenticate a device or runtime.
    Prompt values/lengths and scene identifiers are intentionally absent: actual
    dynamic shape guards and Graph shape ownership remain separate mechanisms.
    """
    required = {"python", "torch", "triton", "cuda", "compute_capability"}
    if set(runtime) != required or any(not isinstance(x, str) or not x for x in runtime.values()):
        raise ValueError("complete observed runtime fingerprint required")
    if not sources_sha256 or any(
        not isinstance(name, str) or not name
        or not isinstance(value, str) or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for name, value in sources_sha256.items()
    ):
        raise ValueError("exact source hashes required")
    required = {"sp_size", "dtype", "latent_height", "latent_width",
                "denoise_steps", "attention_backend", "emulate_precision_casts",
                "prompt_min", "prompt_max"}
    if set(inference) != required:
        raise ValueError("explicit compile-relevant inference configuration required")
    if (
        type(inference["sp_size"]) is not int or inference["sp_size"] != 8
        or inference["dtype"] != "bfloat16"
        or type(inference["latent_height"]) is not int or inference["latent_height"] != 48
        or type(inference["latent_width"]) is not int or inference["latent_width"] != 84
        or type(inference["denoise_steps"]) is not int or inference["denoise_steps"] != 4
        or inference["attention_backend"] != "video_sdpa"
        or type(inference["emulate_precision_casts"]) is not bool
        or type(inference["prompt_min"]) is not int or inference["prompt_min"] != 1000
        or type(inference["prompt_max"]) is not int or inference["prompt_max"] != 1300
    ):
        raise ValueError("configuration lies outside this extracted runtime's tested domain")
    record = {"schema": 1, "sources_sha256": dict(sources_sha256),
              "runtime": dict(runtime), "inference": dict(inference)}
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest(), record


class CompilerCache:
    """Separate rank/component caches, shared across clean process restarts."""

    def __init__(self, root, *, key, rank):
        root = Path(root)
        if not root.is_absolute():
            raise ValueError("cache root must be an explicit absolute directory")
        if type(rank) is not int or not 0 <= rank < 8:
            raise ValueError("SP8 rank required")
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("SHA256 cache namespace required")
        self.root = root / key / f"rank-{rank}"
        self.owner_thread = threading.get_ident()
        self._active = False

    @contextmanager
    def installed(self, component):
        """Scope environment settings around the owner's serial compiler call.

        Not a sandbox for arbitrary library threads: a single model worker must
        serialize DiT and VAE compiler activity. No Graphs are serialized here.
        """
        global _ACTIVE
        if component not in ("dit", "vae", "vae-current", "vae-streamrender"):
            raise ValueError("unknown compiler component")
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("compiler environment belongs to another thread")
        with _LOCK:
            if self._active or _ACTIVE is not None:
                raise RuntimeError("nested compiler cache ownership")
            paths = {
                "TORCHINDUCTOR_CACHE_DIR": self.root / component / "inductor",
                "TRITON_CACHE_DIR": self.root / component / "triton",
            }
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)
            before = {name: os.environ.get(name) for name in paths}
            self._active = True
            _ACTIVE = self
            try:
                os.environ.update({name: str(path) for name, path in paths.items()})
                yield self
            finally:
                for name, value in before.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                self._active = False
                _ACTIVE = None
