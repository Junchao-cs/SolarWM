"""Bounded shape owners, separate from process-wide shared compiled programs."""
from collections import OrderedDict
import hashlib

def scene_key(length, tags_bytes, *, height=48, width=84):
    if type(length) is not int or not 1000 <= length <= 2048:
        raise ValueError('prompt must contain 1000..2048 actual tokens; padding is not used')
    if type(tags_bytes) is not bytes or len(tags_bytes) != length * 8:
        raise ValueError('actual int64 text tags required')
    if (height, width) != (48, 84):
        raise ValueError('only native 1344x768 output is currently supported')
    return (length, hashlib.sha256(tags_bytes).hexdigest(), height, width)

class Domains:

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.active = None
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def acquire(self, key, factory):
        if self.active is not None:
            raise RuntimeError('end the previous request first')
        if key in self.entries:
            value = self.entries.pop(key)
            self.entries[key] = value
            self.hits += 1
        else:
            if len(self.entries) >= self.capacity:
                (old, value) = next(iter(self.entries.items()))
                value.close()
                del self.entries[old]
                self.evictions += 1
            value = factory()
            self.entries[key] = value
            self.misses += 1
        self.active = key
        return value

    def release(self):
        if self.active is None:
            raise RuntimeError('no active shape owner')
        self.active = None

    def close(self):
        if self.active is not None:
            raise RuntimeError('active shape request')
        for value in self.entries.values():
            value.close()
        self.entries.clear()

    def report(self):
        return dict(capacity=self.capacity, resident=len(self.entries), hits=self.hits, misses=self.misses, evictions=self.evictions, compiled_programs_owned=False)
