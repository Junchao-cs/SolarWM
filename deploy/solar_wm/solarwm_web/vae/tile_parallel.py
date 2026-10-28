"""Spatial tile parallelism for the existing upgraded streaming decoder.

Only tile ownership, KV allocation and output transport differ. The inherited
transformer, temporal schedule, blend order and compensation are unchanged.
"""
import torch
import torch.distributed as dist
from .streaming import StreamingDecoder, _token_ids


def partitions(n, world):
    if not 1 <= world <= n:
        raise ValueError('requires at least one tile per rank')
    return [(n * r // world, n * (r + 1) // world) for r in range(world)]


class TileParallelDecoder(StreamingDecoder):
    def __init__(self, *args, **kwargs):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        super().__init__(*args, **kwargs)

    def _tile(self, latent, new_stream):
        weight = self.decoder.proj_in.weight
        z = self.vae.post_quant_conv(latent.to(device=weight.device,
                     dtype=self.vae.post_quant_conv.weight.dtype)).to(weight.dtype)
        key = tuple(z.shape[-2:])
        if self._plan is None or self._plan['latent'] != key:
            if not new_stream:
                raise ValueError('latent size changed inside a stream')
            self._plan = self._make_plan(z)
        lo, hi = self._parts[self.rank]
        return torch.cat([z[..., t:t+h, l:l+w]
                          for t, h, l, w in self._plan['tiles'][lo:hi]], 0)

    def _make_plan(self, z):
        ratio = int(self.vae.spatial_compression_ratio)
        if not self.vae.use_tiling:
            raise ValueError('tiling required')
        yi, yl, yo = self.vae._split_tiles(z.shape[-2] * ratio,
                self.vae.tile_sample_min_height, self.vae.tile_sample_min_overlap_height)
        xi, xl, xo = self.vae._split_tiles(z.shape[-1] * ratio,
                self.vae.tile_sample_min_width, self.vae.tile_sample_min_overlap_width)
        tiles = [(t//ratio, h//ratio, l//ratio, w//ratio)
                 for t,h in zip(yi,yl) for l,w in zip(xi,xl)]
        shapes = {(h,w) for _,h,_,w in tiles}
        if len(shapes) != 1:
            raise ValueError('unequal tile shapes')
        h,w = shapes.pop()
        self._parts = partitions(len(tiles), self.world)
        dtype = self.decoder.proj_in.weight.dtype
        ids = _token_ids((self.window,h,w), z.device, dtype)
        suffix = torch.zeros((1,self.suffix,3), device=z.device, dtype=dtype)
        cos,sin = self.decoder.rope(torch.cat([ids,suffix],1))
        attn = self.decoder.transformer_blocks[0].attn
        lo,hi = self._parts[self.rank]
        shape = (hi-lo, self.suffix+self.window*h*w, attn.heads, attn.dim_head)
        self._k = [torch.empty(shape,device=z.device,dtype=dtype)
                   for _ in self.decoder.transformer_blocks]
        self._v = [torch.empty_like(k) for k in self._k]
        self._rope_cache = {}
        return dict(latent=tuple(z.shape[-2:]),tiles=tiles,rows=len(yi),cols=len(xi),
                    y_overlap=yo,x_overlap=xo,h=h,w=w,hw=h*w,cos=cos,sin=sin)

    def _blend(self, tiles):
        if self.world == 1:
            return super()._blend(tiles)
        count = max(hi-lo for lo,hi in self._parts)
        padded = tiles.new_zeros((count, *tiles.shape[1:]))
        padded[:tiles.shape[0]].copy_(tiles)
        gathered = [torch.empty_like(padded) for _ in self._parts] if self.rank == 0 else None
        dist.gather(padded, gather_list=gathered, dst=0)
        if self.rank:
            # Keeps inherited temporal bookkeeping; no dummy pixels are published.
            return tiles.new_empty((1,3,self.ratio,0,0))
        ordered = torch.cat([t[:hi-lo] for t,(lo,hi) in zip(gathered,self._parts)],0)
        return super()._blend(ordered)

    def _compensate(self, index, slot, out):
        return out if self.rank else super()._compensate(index,slot,out)
