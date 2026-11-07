def spec(t):
    return (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)

def match_singleton(t, expected):
    (shape, strides, dtype, device) = expected
    if tuple(t.shape) != shape or t.dtype != dtype or t.device != device:
        raise RuntimeError('history tensor shape/dtype/device changed')
    if any((a != b and n != 1 for (n, a, b) in zip(shape, t.stride(), strides, strict=True))):
        raise RuntimeError('non-singleton history stride changed')
    if tuple(t.stride()) == strides:
        return t
    out = t.as_strided(shape, strides, t.storage_offset())
    if out.data_ptr() != t.data_ptr() or out.storage_offset() != t.storage_offset():
        raise RuntimeError('history normalization changed addressed storage')
    return out
