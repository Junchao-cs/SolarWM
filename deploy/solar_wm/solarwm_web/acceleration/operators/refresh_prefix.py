"""Source and configuration pins for refresh-prefix reductions."""

from ..compatibility import expected_digest, kernel_source
import hashlib
from pathlib import Path


TARGETS={
    0:(kernel_source('refresh_prefix_rank_zero'),expected_digest("refresh_prefix_rank_zero")),
    1:(kernel_source('refresh_prefix_rank_one'),expected_digest("refresh_prefix_rank_one")),
}
BEFORE=dict(XBLOCK=1,R0_BLOCK=2048,num_warps=16,num_stages=1)
AFTER=dict(BEFORE,R0_BLOCK=4096)
FUNCTION='triton_red_fused__fused_rms_norm_slice_23'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
