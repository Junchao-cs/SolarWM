"""Source- and runtime-bound namespace; a cache key is not qualification."""
import hashlib
import json


def cache_key(*, sources_sha256, runtime, inference):
    if inference['attention_backend'] != 'memory-fa2-cudnn':
        raise ValueError('packaged Memory attention namespace required')
    record = dict(schema='memory-accelerated-package-v1', sources_sha256=sources_sha256,
                  runtime=runtime, inference=inference,
                  dit_compile=True, dit_graph=True, qualification='see release evidence')
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(), record
