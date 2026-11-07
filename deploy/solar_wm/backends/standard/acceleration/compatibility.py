"""Named compatibility pins; keep checks separate from inference code.

These values identify supported model sources and measured kernel selections.
They are not permissions or performance receipts. Missing pins fail closed;
never derive an expected digest from the very file being validated.
"""

import json
import re
from pathlib import Path


_PINS = json.loads(Path(__file__).with_name("compatibility.json").read_text())


def expected_digest(name):
    value = _PINS["digests"][name]
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid compatibility digest: " + name)
    return value


def pinned_path(name):
    value = _PINS["paths"][name]
    path = Path(value)
    if not isinstance(value, str) or path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid relative compatibility path: " + name)
    return value


def kernel_source(name):
    """Exact Inductor cache filename for a measured kernel, not a source alias."""
    value = _PINS["kernel_sources"][name]
    if not isinstance(value, str) or not re.fullmatch(r"c[a-z0-9]{51}\.py", value):
        raise ValueError("invalid kernel source name: " + name)
    return value


def kernel_module(phase):
    """Keep kernel module identities stable for existing compiled profiles."""
    value = _PINS["kernel_modules"][phase]
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError("invalid kernel module name: " + phase)
    return value


def kernel_binaries(name):
    """Allowed Triton binary keys; never regenerate them from unchecked code."""
    values = _PINS["kernel_binaries"][name]
    if (not isinstance(values, list) or not values
            or any(not isinstance(v, str) or not re.fullmatch(r"[A-Z2-7]{52}", v) for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("invalid kernel binary keys: " + name)
    return tuple(values)
