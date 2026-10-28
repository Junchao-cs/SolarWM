from __future__ import annotations

import io
import random

import numpy as np
import pytest

from solar.errors import BackendContractError
from solar.runtime.safe_state import (
    decode_numpy_rng_state,
    decode_python_rng_state,
    encode_numpy_rng_state,
    encode_python_rng_state,
)


def test_rng_states_round_trip_through_weights_only_loader() -> None:
    torch = pytest.importorskip("torch")
    random.seed(4387)
    np.random.seed(9812)
    python_state = encode_python_rng_state(random.getstate())
    numpy_state = encode_numpy_rng_state(np.random.get_state())

    payload = io.BytesIO()
    torch.save({"python": python_state, "numpy": numpy_state}, payload)
    payload.seek(0)
    loaded = torch.load(payload, map_location="cpu", weights_only=True)

    random.setstate(decode_python_rng_state(loaded["python"]))
    np.random.set_state(decode_numpy_rng_state(loaded["numpy"]))
    actual_python = [random.random() for _ in range(4)]
    actual_numpy = np.random.random(4)

    random.seed(4387)
    np.random.seed(9812)
    expected_python = [random.random() for _ in range(4)]
    expected_numpy = np.random.random(4)
    assert actual_python == expected_python
    np.testing.assert_array_equal(actual_numpy, expected_numpy)


def test_rng_state_decoder_rejects_unversioned_native_tuple() -> None:
    with pytest.raises(BackendContractError, match="unsupported schema"):
        decode_python_rng_state(random.getstate())
    with pytest.raises(BackendContractError, match="unsupported schema"):
        decode_numpy_rng_state(np.random.get_state())


def test_rng_state_decoder_rejects_out_of_range_words() -> None:
    encoded = encode_numpy_rng_state(np.random.get_state())
    encoded["words"][0] = 1 << 32
    with pytest.raises(BackendContractError, match="word 0"):
        decode_numpy_rng_state(encoded)
