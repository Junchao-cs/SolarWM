"""SP8 layout-only kernels: unpadded lengths, no floating-point arithmetic."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def gather_heads(X, Y, N, BLOCK: tl.constexpr):
    peer = tl.program_id(1)
    head = tl.program_id(2)
    length = N // 8 + (peer < N % 8)
    start = peer * (N // 8) + tl.minimum(peer, N % 8)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + (7 * start + head * length) * 128 + offset,
                    offset < length * 128, other=0)
    tl.store(Y + (head * N + start) * 128 + offset, value, offset < length * 128)


@triton.jit(do_not_specialize=['N'])
def pack_heads(X, Y, N, BLOCK: tl.constexpr):
    peer = tl.program_id(1)
    head = tl.program_id(2)
    length = N // 8 + (peer < N % 8)
    start = peer * (N // 8) + tl.minimum(peer, N % 8)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offset // 128
    channel = offset % 128
    value = tl.load(X + ((start + token) * 7 + head) * 128 + channel,
                    offset < length * 128, other=0)
    tl.store(Y + (7 * start + head * length) * 128 + offset,
             value, offset < length * 128)


@triton.jit(do_not_specialize=['COUNT'])
def copy_heads(X, Y, COUNT, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offset, offset < COUNT, other=0)
    tl.store(Y + offset, value, offset < COUNT)
