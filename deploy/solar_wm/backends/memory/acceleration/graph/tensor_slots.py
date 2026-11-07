import torch

def metadata(t):
    return (tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t.storage_offset())

def identity(t):
    return (t.untyped_storage()._cdata, t.data_ptr(), metadata(t))

def stamp(t):
    return (identity(t), t._version)

def clone(t):
    if t.storage_offset() != 0:
        raise RuntimeError('moving resident nonzero input offset not admitted')
    out = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out

def copied(pairs):
    buckets = {}
    seen = set()
    size = 0
    for (out, src) in pairs:
        if metadata(out) != metadata(src):
            raise RuntimeError('moving resident tensor domain changed')
        key = out.untyped_storage()._cdata
        if key in seen:
            raise RuntimeError('moving resident destinations alias')
        seen.add(key)
        size += out.numel() * out.element_size()
        if out.is_contiguous() and src.is_contiguous():
            (a, b) = buckets.setdefault((out.dtype, out.device), ([], []))
            a.append(out)
            b.append(src)
        else:
            out.copy_(src)
    for (a, b) in buckets.values():
        torch._foreach_copy_(a, b)
    return size
