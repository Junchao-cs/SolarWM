# SPDX-License-Identifier: Apache-2.0 AND MIT
# ruff: noqa
# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
from typing import Callable, List, Optional, Tuple

import torch

from ..camera import transform_relative_viewmats


def prope_qkv(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int = None,  # How many patches wide is each image?
    patches_y: int = None,  # How many patches tall is each image?
    image_width: int = None,  # Width of the image. Used to normalize intrinsics.
    image_height: int = None,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    mask: Optional[torch.Tensor] = None,
    kv_cache=None,
    is_cache: bool = False,
    camera_translation_transform: str = "linear",
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, _, _, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)

    transformed_viewmats = transform_relative_viewmats(
        viewmats,
        camera_translation_transform,
    )
    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns_all_dim(
        head_dim=head_dim,
        viewmats=transformed_viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
    )

    query = apply_fn_q(q)
    key = apply_fn_kv(k)
    value = apply_fn_kv(v)

    return query, key, value, apply_fn_o


def prope_qkv_separate(
    q: torch.Tensor,  # (batch, num_heads, query_len, head_dim)
    k: torch.Tensor,  # (batch, num_heads, kv_len, head_dim)
    v: torch.Tensor,  # (batch, num_heads, kv_len, head_dim)
    *,
    q_viewmats: torch.Tensor,  # (batch, query_len, 4, 4)
    q_Ks: Optional[torch.Tensor],  # (batch, query_len, 3, 3)
    kv_viewmats: torch.Tensor,  # (batch, kv_len, 4, 4)
    kv_Ks: Optional[torch.Tensor],  # (batch, kv_len, 3, 3)
    camera_translation_transform: str = "linear",
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Apply SolarWM PRoPE when query and visible KV lengths differ.

    This is the KV-cache counterpart of :func:`prope_qkv`.  It deliberately
    reuses the same camera normalization, intrinsic lifting, view-matrix
    convention, and tiled feature transform.  Only the camera tensors used to
    prepare the query/output transforms and KV transforms are separated.

    The caller is responsible for applying ordinary RoPE to ``q`` and ``k``
    first.  ``v`` must remain raw until this function applies ``P^-1``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("PRoPE expects q/k/v in [batch, heads, sequence, head_dim] layout")
    if k.shape != v.shape:
        raise ValueError(
            f"PRoPE visible k/v shapes must match, got {tuple(k.shape)} and {tuple(v.shape)}"
        )
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError(
            "PRoPE query and KV must share batch/head/head_dim: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}"
        )

    batch, _, query_len, head_dim = q.shape
    kv_len = k.shape[2]
    expected_q_viewmats = (batch, query_len, 4, 4)
    expected_kv_viewmats = (batch, kv_len, 4, 4)
    if tuple(q_viewmats.shape) != expected_q_viewmats:
        raise ValueError(
            f"q_viewmats must have shape {expected_q_viewmats}, got {tuple(q_viewmats.shape)}"
        )
    if tuple(kv_viewmats.shape) != expected_kv_viewmats:
        raise ValueError(
            f"kv_viewmats must have shape {expected_kv_viewmats}, got {tuple(kv_viewmats.shape)}"
        )
    if q_Ks is not None and tuple(q_Ks.shape) != (batch, query_len, 3, 3):
        raise ValueError(
            "q_Ks must align with the query sequence, got "
            f"{tuple(q_Ks.shape)} for q={tuple(q.shape)}"
        )
    if kv_Ks is not None and tuple(kv_Ks.shape) != (batch, kv_len, 3, 3):
        raise ValueError(
            "kv_Ks must align with the visible KV sequence, got "
            f"{tuple(kv_Ks.shape)} for k={tuple(k.shape)}"
        )

    transformed_q_viewmats = transform_relative_viewmats(
        q_viewmats,
        camera_translation_transform,
    )
    transformed_kv_viewmats = transform_relative_viewmats(
        kv_viewmats,
        camera_translation_transform,
    )
    apply_fn_q, _, apply_fn_o = _prepare_apply_fns_all_dim(
        head_dim=head_dim,
        viewmats=transformed_q_viewmats,
        Ks=q_Ks,
        patches_x=None,
        patches_y=None,
        image_width=None,
        image_height=None,
    )
    _, apply_fn_kv, _ = _prepare_apply_fns_all_dim(
        head_dim=head_dim,
        viewmats=transformed_kv_viewmats,
        Ks=kv_Ks,
        patches_x=None,
        patches_y=None,
        image_width=None,
        image_height=None,
    )
    return apply_fn_q(q), apply_fn_kv(k), apply_fn_kv(v), apply_fn_o


def _prepare_apply_fns_all_dim(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    (batch, cameras, _, _) = viewmats.shape

    # Normalize camera intrinsics.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 0, 2] = 0
        Ks_norm[..., 1, 2] = 0
        Ks_norm[..., 2, 2] = 1.0
        Ks_norm = Ks_norm.to(dtype=Ks.dtype)
        del Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        ).to(dtype=viewmats.dtype)

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D) or (batch, seqlen, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    D = matrix.shape[-1]
    assert feat_dim % D == 0, f"feat_dim={feat_dim} must be divisible by D={D}"
    # Keep camera construction/inversion in FP32, then cast only the derived
    # projection matrix used by this activation. Camera metadata stays FP32 and
    # retains its cache identity across chunk commits.
    matrix = matrix.to(dtype=feats.dtype)

    if matrix.shape[1] == seqlen:
        # Per-ray projection: matrix shape [B, seqlen, D, D]
        feats_ = feats.view(batch, num_heads, seqlen, feat_dim // D, D)
        out = torch.einsum("btij,bntpj->bntpi", matrix, feats_)
        return out.reshape(feats.shape)

    # Per-camera projection.
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    out = out.to(dtype=transforms.dtype)
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out
