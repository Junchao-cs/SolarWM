"""True head/FFN tensor parallelism for the upstream single-latent decoder.

The global suffix sees the complete token set, as in upstream. No spatial crops,
padding of latent inputs, or eight independent replicas used as a speed metric.
FP32 partial-sum reductions precede the original BF16 residual branch boundary.
This is an experimental adapter, not a qualified production implementation.
"""
import torch
import torch.distributed as dist
from torch import nn
import triton
import triton.language as tl
from quack.gemm_interface import gemm_act
from streamrender_h3.kernels.sr_fused_final import (
    FusedStreamingCausalViT, BF16, flash_attn_varlen_func,
    _suffix_partial, _suffix_merge, resnorm, head_hilo,
)
from streamrender_h3.decoder.causal_vit_decoder import PATCH, PATCH_T, OUT_CH


@triton.jit
def qkv_cache_masked(QKV, COS, SIN, SCOS, SSIN, PERM, WIDX, SIDX, Q, QS, KP, VP,
                     N: tl.constexpr, S: tl.constexpr, NW: tl.constexpr,
                     HEADS: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1) * 16 + tl.arange(0, 16)[:, None]
    j = tl.arange(0, D)[None, :]
    valid = head < HEADS
    is_s = row >= N
    src = tl.load(PERM + row, row < N, 0)
    src = tl.where(is_s, row, src)
    off = src * (HEADS * 3 * D) + head * (3 * D) + j
    q = tl.load(QKV + off, valid, 0).to(tl.float32)
    k = tl.load(QKV + off + D, valid, 0).to(tl.float32)
    v = tl.load(QKV + off + 2 * D, valid, 0)
    qr = (q * tl.rsqrt(tl.sum(q*q, 1)[:, None] / D + EPS)).to(tl.bfloat16).to(tl.float32)
    kr = (k * tl.rsqrt(tl.sum(k*k, 1)[:, None] / D + EPS)).to(tl.bfloat16).to(tl.float32)
    c = tl.load(COS + src*ROT + j, (row < N) & (j < ROT), 0)
    si = tl.load(SIN + src*ROT + j, (row < N) & (j < ROT), 0)
    cs = tl.load(SCOS + (row-N)*ROT + j, is_s & (j < ROT), 0)
    ss = tl.load(SSIN + (row-N)*ROT + j, is_s & (j < ROT), 0)
    c = tl.where(is_s, cs, c).to(tl.bfloat16).to(tl.float32)
    si = tl.where(is_s, ss, si).to(tl.bfloat16).to(tl.float32)
    other = tl.minimum(tl.maximum(tl.where(j < ROT//2, j+ROT//2, j-ROT//2), 0), D-1)
    qo = tl.gather(qr, tl.broadcast_to(other, (16, D)), 1) * tl.where(j < ROT//2, -1., 1.)
    ko = tl.gather(kr, tl.broadcast_to(other, (16, D)), 1) * tl.where(j < ROT//2, -1., 1.)
    qrot = ((qr*c).to(tl.bfloat16).to(tl.float32)+(qo*si).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    krot = ((kr*c).to(tl.bfloat16).to(tl.float32)+(ko*si).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    qr = tl.where(j < ROT, qrot, qr).to(tl.bfloat16)
    kr = tl.where(j < ROT, krot, kr).to(tl.bfloat16)
    tl.store(Q + row*HEADS*D + head*D + j, qr, (row < N) & valid)
    tl.store(QS + (row-N)*HEADS*D + head*D + j, qr, is_s & valid)
    dest = tl.load(WIDX + row, row < N, 0)
    tl.store(KP + dest*HEADS*D + head*D + j, kr, (row < N) & valid)
    tl.store(VP + dest*HEADS*D + head*D + j, v, (row < N) & valid)
    if is_s:
        for win in range(NW):
            dest = tl.load(SIDX + win*S + row-N)
            tl.store(KP + dest*HEADS*D + head*D + j, kr, valid)
            tl.store(VP + dest*HEADS*D + head*D + j, v, valid)


def shard_linear(old, rows=None, cols=None, bias=True):
    w = old.weight.detach()
    b = old.bias.detach() if old.bias is not None and bias else None
    if rows is not None:
        w = w[rows]
        if b is not None:
            b = b[rows]
    if cols is not None:
        w = w[:, cols]
    new = nn.Linear(w.shape[1], w.shape[0], bias=b is not None, device="meta", dtype=w.dtype)
    new.weight = nn.Parameter(w.contiguous().clone(), requires_grad=False)
    if b is not None:
        new.bias = nn.Parameter(b.contiguous().clone(), requires_grad=False)
    return new


class TensorParallelCausalViT(FusedStreamingCausalViT):
    def __init__(self, r, H, W, **kwargs):
        assert dist.is_initialized() and dist.get_world_size() == 8
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        assert r.heads == 32 and r.dh == 64
        # Base streaming constructor uses heads solely for KV/metadata layout;
        # the full 2048-D residual and original model weights are retained.
        r.heads = 4
        super().__init__(r, H, W, **kwargs)
        self.full_dim = r.norm_out.weight.numel()
        head_start, head_stop = self.rank * 4, (self.rank + 1) * 4
        qrows = slice(head_start * 3 * self.dh, head_stop * 3 * self.dh)
        acols = slice(head_start * self.dh, head_stop * self.dh)
        self.attn_bias, self.ff_bias = [], []
        for blk in self.blocks:
            assert blk.ff.w1.out_features == 2 * blk.ff.w2.in_features
            inner = blk.ff.w2.in_features
            width = inner // self.world
            lo, hi = self.rank * width, (self.rank + 1) * width
            frows = torch.cat([torch.arange(lo, hi, device=r.register_tokens.device),
                               torch.arange(inner+lo, inner+hi, device=r.register_tokens.device)])
            self.attn_bias.append(blk.attn.to_out.bias.detach().float().clone())
            self.ff_bias.append(blk.ff.w2.bias.detach().float().clone())
            blk.attn.to_qkv = shard_linear(blk.attn.to_qkv, rows=qrows)
            blk.attn.to_out = shard_linear(blk.attn.to_out, cols=acols, bias=False)
            blk.ff.w1 = shard_linear(blk.ff.w1, rows=frows)
            blk.ff.w2 = shard_linear(blk.ff.w2, cols=slice(lo, hi), bias=False)
        # Fail before long compilation if this installed CUDA/PyTorch lacks FP32 outputs.
        a = torch.ones((16, 16), device=r.register_tokens.device, dtype=BF16)
        assert torch.mm(a, a, out_dtype=torch.float32).dtype == torch.float32

    def reduce_project(self, x, layer, bias):
        partial = torch.mm(x, layer.weight.T, out_dtype=torch.float32)
        dist.all_reduce(partial)
        return (partial + bias).to(BF16)

    @torch.no_grad()
    def _step(self):
        r, C, H, W = self.r, self.C, self.H, self.W
        N, D = C*H*W, self.full_dim
        with torch.autocast("cuda", dtype=BF16):
            raw = self.z_in*r.latents_std.view(1, 1, -1, 1, 1)+r.latents_mean.view(1, 1, -1, 1, 1)
            raw = r.post_quant_conv(raw.transpose(1, 2))
        x = r.x_embedder(raw.permute(0, 2, 3, 4, 1).float()).to(BF16).float().reshape(N, -1)
        suf = torch.cat([r.register_tokens[0], r.register_tokens.new_zeros(1, D)], 0).to(BF16).float()
        x = torch.cat([x, suf], 0)
        for s, g in self.geo.items():
            cos, sin, scos, ssin, idx, used, bias = self._meta(
                r, self.clock, C, H, W, s, self.S, self.banks[s], g["n"], g["row_slot"], g["dup"], self.nslot)
            self.cos[s], self.sin[s], self.scos, self.ssin = cos, sin, scos, ssin
            g["widx_cur"].copy_(idx)
            g["seqused"].copy_(used)
            g["kbias"].copy_(bias)
        xn = self._first_norm(self.blocks[0].norm1, x)
        for li, blk in enumerate(self.blocks):
            s = self.shifts[li]
            g, kp, vp = self.geo[s], self.kpool[li], self.vpool[li]
            qkv = blk.attn.to_qkv(xn)
            Q = torch.empty((N, self.heads, self.dh), device=x.device, dtype=BF16)
            qs = torch.empty((self.S, self.heads, self.dh), device=x.device, dtype=BF16)
            qkv_cache_masked[(N+self.S, triton.cdiv(self.heads, 16))](
                qkv, self.cos[s], self.sin[s], self.scos, self.ssin,
                g["qperm"], g["widx_cur"], g["sidx"], Q, qs, kp, vp,
                N, self.S, g["nW"], self.heads, self.dh, self.cos[s].shape[-1],
                blk.attn.norm_q.eps, num_warps=4, enable_fp_fusion=False)
            O = flash_attn_varlen_func(Q, kp, vp, cu_seqlens_q=g["cu_q"],
                                      cu_seqlens_k=g["cu_k"], max_seqlen_q=g["max_q"],
                                      max_seqlen_k=g["max_k"], seqused_k=g["seqused"])
            if isinstance(O, tuple):
                O = O[0]
            nk, splits = kp.shape[0], triton.cdiv(kp.shape[0], 256)
            pp = torch.empty((self.heads, splits, 5, self.dh), device=x.device, dtype=torch.float32)
            ll = torch.empty((self.heads, splits, 5), device=x.device, dtype=torch.float32)
            os_ = torch.empty((5, self.heads, self.dh), device=x.device, dtype=BF16)
            _suffix_partial[(self.heads, splits)](qs, kp, vp, g["kbias"], pp, ll, nk, self.heads,
                                                self.dh, splits, 256, num_warps=4)
            _suffix_merge[(self.heads, 5)](pp, ll, os_, self.heads, self.dh, splits,
                                          triton.next_power_of_2(splits), num_warps=4)
            attn = torch.cat([O.flatten(1)[g["inv"]], os_.flatten(1)], 0)
            # Autocast must not override explicit FP32 output accumulation.
            with torch.autocast("cuda", enabled=False):
                projected = self.reduce_project(attn, blk.attn.to_out, self.attn_bias[li])
            x, fn = resnorm(x, projected, blk.scale1, blk.norm2)
            _, gate = gemm_act(fn, blk.ff.w1.weight.T, bias=blk.ff.w1.bias,
                               activation="swiglu", store_preact=False, tuned=False,
                               concat_layout=("B", "bias"))
            with torch.autocast("cuda", enabled=False):
                down = self.reduce_project(gate, blk.ff.w2, self.ff_bias[li])
            nxt = self.blocks[li+1].norm1 if li+1 < len(self.blocks) else r.norm_out
            x, xn = resnorm(x, down, blk.scale2, nxt)
        x = r.norm_out(x[:N])
        out = head_hilo(x, r.proj_out)
        out = out.view(C, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(3, 0, 4, 1, 5, 2, 6)
        return out.reshape(1, OUT_CH, C*PATCH_T, H*PATCH, W*PATCH)
