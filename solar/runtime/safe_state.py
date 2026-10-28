"""Safe, versioned serialization for process RNG state.

Checkpoint payloads must remain loadable with ``torch.load(weights_only=True)``.
NumPy's native ``RandomState`` tuple contains an ndarray whose pickle reducer is
not admitted by that loader, so checkpoint writers store only validated scalar
containers and reconstruct the ndarray after loading.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from solar.errors import BackendContractError

_PYTHON_SCHEMA = "solarwm.python-rng-state.v1"
_NUMPY_SCHEMA = "solarwm.numpy-rng-state.v1"
_MT19937_WORDS = 624
_UINT32_MAX = (1 << 32) - 1


def _integer(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BackendContractError(f"{label} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise BackendContractError(f"{label} must be between {minimum} and {maximum}, got {result}")
    return result


def _gaussian(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendContractError(f"{label} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise BackendContractError(f"{label} must be finite")
    return result


def encode_python_rng_state(state: object) -> dict[str, Any]:
    """Convert ``random.getstate()`` into a safe, explicit checkpoint value."""

    if not isinstance(state, tuple) or len(state) != 3:
        raise BackendContractError("Python RNG state must be a three-item tuple")
    version = _integer(state[0], label="Python RNG version", minimum=3, maximum=3)
    raw_words = state[1]
    if not isinstance(raw_words, tuple) or len(raw_words) != _MT19937_WORDS + 1:
        raise BackendContractError("Python MT19937 state must contain 625 integers")
    words = [
        _integer(
            value,
            label=f"Python RNG word {index}",
            minimum=0,
            maximum=_UINT32_MAX,
        )
        for index, value in enumerate(raw_words[:-1])
    ]
    position = _integer(
        raw_words[-1],
        label="Python RNG position",
        minimum=0,
        maximum=_MT19937_WORDS,
    )
    return {
        "schema": _PYTHON_SCHEMA,
        "version": version,
        "words": words,
        "position": position,
        "gaussian": _gaussian(state[2], label="Python RNG Gaussian cache"),
    }


def decode_python_rng_state(value: object) -> tuple[Any, ...]:
    """Validate and reconstruct a state accepted by ``random.setstate``."""

    if not isinstance(value, Mapping) or value.get("schema") != _PYTHON_SCHEMA:
        raise BackendContractError("checkpoint Python RNG state has an unsupported schema")
    version = _integer(value.get("version"), label="Python RNG version", minimum=3, maximum=3)
    raw_words = value.get("words")
    if (
        not isinstance(raw_words, Sequence)
        or isinstance(raw_words, (str, bytes, bytearray))
        or len(raw_words) != _MT19937_WORDS
    ):
        raise BackendContractError("checkpoint Python RNG state must contain 624 words")
    words = tuple(
        _integer(
            item,
            label=f"Python RNG word {index}",
            minimum=0,
            maximum=_UINT32_MAX,
        )
        for index, item in enumerate(raw_words)
    )
    position = _integer(
        value.get("position"),
        label="Python RNG position",
        minimum=0,
        maximum=_MT19937_WORDS,
    )
    gaussian = _gaussian(value.get("gaussian"), label="Python RNG Gaussian cache")
    return (version, (*words, position), gaussian)


def encode_numpy_rng_state(state: object) -> dict[str, Any]:
    """Convert ``numpy.random.get_state()`` without retaining an ndarray reducer."""

    if not isinstance(state, tuple) or len(state) != 5:
        raise BackendContractError("NumPy RNG state must be a five-item tuple")
    if state[0] != "MT19937":
        raise BackendContractError("only NumPy MT19937 global RNG state is supported")
    raw_words = state[1]
    if (
        not hasattr(raw_words, "tolist")
        or getattr(raw_words, "dtype", None) is None
        or getattr(raw_words, "shape", None) != (_MT19937_WORDS,)
    ):
        raise BackendContractError("NumPy MT19937 state must contain 624 uint32 words")
    words = [
        _integer(
            item,
            label=f"NumPy RNG word {index}",
            minimum=0,
            maximum=_UINT32_MAX,
        )
        for index, item in enumerate(raw_words.tolist())
    ]
    position = _integer(
        state[2],
        label="NumPy RNG position",
        minimum=0,
        maximum=_MT19937_WORDS,
    )
    has_gaussian = _integer(state[3], label="NumPy RNG Gaussian flag", minimum=0, maximum=1)
    return {
        "schema": _NUMPY_SCHEMA,
        "bit_generator": "MT19937",
        "words": words,
        "position": position,
        "has_gaussian": has_gaussian,
        "cached_gaussian": _gaussian(state[4], label="NumPy RNG Gaussian cache"),
    }


def decode_numpy_rng_state(value: object) -> tuple[Any, ...]:
    """Validate and reconstruct a state accepted by ``numpy.random.set_state``."""

    if not isinstance(value, Mapping) or value.get("schema") != _NUMPY_SCHEMA:
        raise BackendContractError("checkpoint NumPy RNG state has an unsupported schema")
    if value.get("bit_generator") != "MT19937":
        raise BackendContractError("checkpoint NumPy RNG must use MT19937")
    raw_words = value.get("words")
    if (
        not isinstance(raw_words, Sequence)
        or isinstance(raw_words, (str, bytes, bytearray))
        or len(raw_words) != _MT19937_WORDS
    ):
        raise BackendContractError("checkpoint NumPy RNG state must contain 624 words")
    words = [
        _integer(
            item,
            label=f"NumPy RNG word {index}",
            minimum=0,
            maximum=_UINT32_MAX,
        )
        for index, item in enumerate(raw_words)
    ]
    position = _integer(
        value.get("position"),
        label="NumPy RNG position",
        minimum=0,
        maximum=_MT19937_WORDS,
    )
    has_gaussian = _integer(
        value.get("has_gaussian"),
        label="NumPy RNG Gaussian flag",
        minimum=0,
        maximum=1,
    )
    cached = _gaussian(value.get("cached_gaussian"), label="NumPy RNG Gaussian cache")
    if cached is None:
        raise BackendContractError("NumPy RNG Gaussian cache must be a finite number")

    import numpy as np

    return (
        "MT19937",
        np.asarray(words, dtype=np.uint32),
        position,
        has_gaussian,
        cached,
    )


__all__ = [
    "decode_numpy_rng_state",
    "decode_python_rng_state",
    "encode_numpy_rng_state",
    "encode_python_rng_state",
]
