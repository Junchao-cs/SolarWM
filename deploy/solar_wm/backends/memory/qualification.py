"""Bounded deployment checks; not a substitute for visual/control review."""
import hashlib


def smoke_actions(case, chunk):
    if case not in range(3) or chunk < 0:
        raise ValueError("three nonnegative short control schedules")
    key = ("s" if chunk < 6 else "w") if case == 0 else (
        "w" if case == 1 else ("j" if chunk < 8 else "l"))
    return [[key]] * (4 if chunk == 0 else 5)


def compiler_counts(engine):
    return dict(d0=0 if engine.full_compiler is None else engine.full_compiler.calls,
                query=dict(engine.query_compiler.calls))


def check_compilers(engine):
    counts = compiler_counts(engine)
    if counts != dict(d0=1, query={"derived-fill": 1, "reuse": 1, "refresh": 1}):
        raise RuntimeError("expected four process-wide compile entries, without recompilation")
    return counts


def latent_digest(cpu):
    """Cross-rank publication check, not equality across algorithms."""
    import numpy as np
    if tuple(cpu.shape) != (1, 24, 5, 48, 84) or cpu.device.type != "cpu":
        raise ValueError("full native chunk required")
    array = cpu.float().contiguous().numpy()
    if not np.isfinite(array).all():
        raise FloatingPointError("nonfinite smoke latent")
    return hashlib.sha256(array.tobytes()).hexdigest()
