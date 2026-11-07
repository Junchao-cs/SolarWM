import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit(do_not_specialize=['row_offset', 'xnumel'])
def fused_global(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, row_offset, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr, ROUND_RESIDUAL: tl.constexpr):
    r0_numel = 5376
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (row_offset + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (row_offset + x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 5376 * x0 + 5376 * row_offset), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_out_ptr0 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.full([1, 1], 3, tl.int64)
        tmp3 = tmp1 * tmp2
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full([1, 1], 9, tl.int32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp5 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp5)
        tl.device_assert((0 <= tmp9) & (tmp9 < 9) | ~xmask, 'index out of bounds: 0 <= tmp9 < 9')
        tmp11 = tl.load(in_ptr3 + (10752 + r0_1 + 32256 * tmp9), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = 1.0
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp0 + tmp17
        tmp19 = tmp18.to(tl.bfloat16).to(tl.float32) if ROUND_RESIDUAL else tmp18.to(tl.float32)
        tmp20 = tmp19 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = tl.fma(tmp19, tmp19, _tmp22)
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
        tl.store(in_out_ptr0 + (r0_1 + 5376 * x0), tmp18, r0_mask & xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp24 = tl.load(in_out_ptr0 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + r0_1, r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 5376.0
        tmp27 = tmp22 / tmp26
        tmp28 = 1e-05
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp31 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tl.full([1, 1], 3, tl.int64)
        tmp37 = tmp1 * tmp36
        tmp38 = tmp37 + tmp4
        tmp39 = tl.full([1, 1], 9, tl.int32)
        tmp40 = tmp38 + tmp39
        tmp41 = tmp38 < 0
        tmp42 = tl.where(tmp41, tmp40, tmp38)
        tl.device_assert((0 <= tmp42) & (tmp42 < 9) | ~xmask, 'index out of bounds: 0 <= tmp42 < 9')
        tmp44 = tl.load(in_ptr3 + (21504 + r0_1 + 32256 * tmp42), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp45 = 1.0
        tmp46 = tmp44 + tmp45
        tmp47 = tmp35 * tmp46
        tmp48 = tl.load(in_ptr3 + (16128 + r0_1 + 32256 * tmp42), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = tmp47 + tmp48
        tl.store(out_ptr1 + (r0_1 + 5376 * x0), tmp49, r0_mask & xmask)

@triton.jit(do_not_specialize=['row_offset', 'xnumel'])
def fused_ffn(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, row_offset, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr, ROUND_RESIDUAL: tl.constexpr):
    r0_numel = 5376
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (row_offset + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (row_offset + x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.full([1, 1], 3, tl.int64)
        tmp3 = tmp1 * tmp2
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full([1, 1], 9, tl.int32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp5 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp5)
        tl.device_assert((0 <= tmp9) & (tmp9 < 9) | ~xmask, 'index out of bounds: 0 <= tmp9 < 9')
        tmp11 = tl.load(in_ptr2 + (26880 + r0_1 + 32256 * tmp9), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = 1.0
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp0 + tmp17
        tmp19 = tmp18.to(tl.bfloat16).to(tl.float32) if ROUND_RESIDUAL else tmp18.to(tl.float32)
        tmp20 = tmp19 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = tl.fma(tmp19, tmp19, _tmp22)
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
        tl.store(in_out_ptr0 + (r0_1 + 5376 * x0), tmp18, r0_mask & xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp24 = tl.load(in_out_ptr0 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + r0_1, r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 5376.0
        tmp27 = tmp22 / tmp26
        tmp28 = 1e-05
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp31 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tl.full([1, 1], 3, tl.int64)
        tmp37 = tmp1 * tmp36
        tmp38 = tmp37 + tmp4
        tmp39 = tl.full([1, 1], 9, tl.int32)
        tmp40 = tmp38 + tmp39
        tmp41 = tmp38 < 0
        tmp42 = tl.where(tmp41, tmp40, tmp38)
        tl.device_assert((0 <= tmp42) & (tmp42 < 9) | ~xmask, 'index out of bounds: 0 <= tmp42 < 9')
        tmp44 = tl.load(in_ptr6 + (5376 + r0_1 + 32256 * tmp42), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp45 = 1.0
        tmp46 = tmp44 + tmp45
        tmp47 = tmp35 * tmp46
        tmp48 = tl.load(in_ptr6 + (r0_1 + 32256 * tmp42), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = tmp47 + tmp48
        tl.store(out_ptr1 + (r0_1 + 5376 * x0), tmp49, r0_mask & xmask)

@triton.jit(do_not_specialize=['row_offset', 'xnumel'])
def fused_local(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, row_offset, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr, ROUND_RESIDUAL: tl.constexpr):
    r0_numel = 5376
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (row_offset + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (row_offset + x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.full([1, 1], 3, tl.int64)
        tmp3 = tmp1 * tmp2
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full([1, 1], 9, tl.int32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp5 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp5)
        tl.device_assert((0 <= tmp9) & (tmp9 < 9) | ~xmask, 'index out of bounds: 0 <= tmp9 < 9')
        tmp11 = tl.load(in_ptr2 + (10752 + r0_1 + 32256 * tmp9), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = 1.0
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp0 + tmp17
        tmp19 = tmp18.to(tl.bfloat16).to(tl.float32) if ROUND_RESIDUAL else tmp18.to(tl.float32)
        tmp20 = tmp19 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = tl.fma(tmp19, tmp19, _tmp22)
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
        tl.store(in_out_ptr0 + (r0_1 + 5376 * x0), tmp18, r0_mask & xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp24 = tl.load(in_out_ptr0 + (r0_1 + 5376 * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + r0_1, r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 5376.0
        tmp27 = tmp22 / tmp26
        tmp28 = 1e-05
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp31 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tl.full([1, 1], 3, tl.int64)
        tmp37 = tmp1 * tmp36
        tmp38 = tmp37 + tmp4
        tmp39 = tl.full([1, 1], 9, tl.int32)
        tmp40 = tmp38 + tmp39
        tmp41 = tmp38 < 0
        tmp42 = tl.where(tmp41, tmp40, tmp38)
        tl.device_assert((0 <= tmp42) & (tmp42 < 9) | ~xmask, 'index out of bounds: 0 <= tmp42 < 9')
        tmp44 = tl.load(in_ptr2 + (21504 + r0_1 + 32256 * tmp42), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp45 = 1.0
        tmp46 = tmp44 + tmp45
        tmp47 = tmp35 * tmp46
        tmp48 = tl.load(in_ptr2 + (16128 + r0_1 + 32256 * tmp42), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = tmp47 + tmp48
        tl.store(out_ptr1 + (r0_1 + 5376 * x0), tmp49, r0_mask & xmask)
