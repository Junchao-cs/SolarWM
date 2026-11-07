from contextlib import contextmanager

@contextmanager
def helper_binding(module, helper):
    if hasattr(module, '_memory_segment'):
        raise RuntimeError('Memory helper name is already owned')
    module._memory_segment = helper
    try:
        yield
    finally:
        changed = module._memory_segment is not helper
        del module._memory_segment
        if changed:
            raise RuntimeError('Memory helper ownership changed')
