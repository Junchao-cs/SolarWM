"""Checkpoint metadata identities used by Memory inference."""

from __future__ import annotations

import hashlib

MEMORY_SCHEMA = "h3-stage1-memory-top5"
RETRIEVAL_REVISION = "native17-top5"

# Accept the published checkpoint schema fingerprint and its canonical name.
_PUBLISHED_IDENTITIES = {
    "schema": (MEMORY_SCHEMA, "291015b4a00db74dfe1177488a307f8f9671a64779b161fa8de6ae5b67aceb4e"),
}


def matches_identity(value, kind):
    if not isinstance(value, str):
        return False
    canonical, published_sha256 = _PUBLISHED_IDENTITIES[kind]
    return value == canonical or hashlib.sha256(value.encode()).hexdigest() == published_sha256
