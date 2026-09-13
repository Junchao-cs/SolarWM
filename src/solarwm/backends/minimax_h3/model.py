"""Diffusers MiniMax-H3 adapter with W6 masks and token-selective fused PRoPE.

The default FM adapter preserves upstream checkpoint/state-dict names. AnyFlow
target-time conditioning is an explicit extension enabled after strict loading.
The adapter relies on the released diffusers H3 block layout and fails loudly
if that dependency is unavailable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM,
        MiniMaxH3AttnProcessor,
        MiniMaxH3Transformer3DModel,
        MiniMaxH3TransformerBlock,
        MiniMaxH3TransformerOutput,
        _apply_rotary_emb,
    )
    from diffusers.utils import apply_lora_scale
    from torch.nn.attention.flex_attention import (
        flex_attention as _native_flex_attention,
    )
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - dependency environment
    raise ImportError(
        "The SolarWM H3 adapter requires a diffusers build containing "
        "MiniMaxH3Transformer3DModel (and its matching torch dependency)."
    ) from exc

from .anyflow_conditioning import H3AnyFlowConditioningMixin
from .camera import fixed_intrinsics_like
from .diffusers_compat import patch_minimax_h3_parameter_dtype
from .distributed import (
    get_sp_size,
    is_sequence_parallel_enabled,
    prepare_flex_attention_input,
    validate_packed_attention_mask,
)
from .distributed import (
    sequence_all_gather as sequence_model_parallel_all_gather,
)
from .distributed import (
    sequence_all_to_all as sequence_model_parallel_all_to_all_4D,
)
from .distributed import (
    shard_stage0p5_packed_sequence as shard_packed_sequence,
)
from .distributed import (
    state as get_sequence_parallel_state,
)

get_parameter_dtype = patch_minimax_h3_parameter_dtype()


FusedPrope = Callable[
    ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]
]


def _inductor_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Any,
) -> torch.Tensor:
    """A fail-closed graph boundary around the native FlexAttention HOP."""

    # If Dynamo hits a cache limit, a graph break, suppress_errors, or a
    # disabled compile context, it may try to run this Python function eagerly.
    # Raw torch 2.6 FlexAttention then uses a dense composite implementation;
    # for H3 that would allocate hundreds of GiB. Never permit that fallback.
    if not torch.compiler.is_dynamo_compiling():
        raise RuntimeError(
            "H3 Stage1 FlexAttention escaped its Inductor graph; refusing the dense eager fallback"
        )
    return _native_flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
    )


def _training_flex_attention(query, key, value, block_mask):
    return _inductor_flex_attention(query, key, value, block_mask)


def _validation_flex_attention(query, key, value, block_mask):
    return _inductor_flex_attention(query, key, value, block_mask)


def _compile_flex_attention(entry):
    return torch.compile(
        entry,
        backend="inductor",
        fullgraph=True,
        dynamic=True,
        # PyTorch 2.6 training needs the no-cudagraphs mode for Flex backward.
        mode="max-autotune-no-cudagraphs",
    )


# Dynamo budgets recompilations per Python code object. Validation introduces
# shorter rollout shapes and lazily registered decoder dependencies; these
# must not consume the training entry's bounded specialization budget.
_compiled_training_flex_attention = _compile_flex_attention(_training_flex_attention)
_compiled_validation_flex_attention = _compile_flex_attention(_validation_flex_attention)


def _compiled_h3_flex_attention(query, key, value, block_mask, *, training=True):
    # Detached AnyFlow target forwards are still training. Grad mode cannot
    # distinguish them from validation; use the attention module's mode.
    compiled = (
        _compiled_training_flex_attention if training else _compiled_validation_flex_attention
    )
    return compiled(query, key, value, block_mask)


@dataclass(frozen=True)
class H3AttentionControl:
    """Non-parameter payload carried through the upstream block forward."""

    attention_mask: Any
    fused_prope: bool | FusedPrope | None
    prope_token_indices: torch.Tensor | None
    prope_frame_ids: torch.Tensor | None
    cam_viewmats: torch.Tensor | None
    cam_K: torch.Tensor | None
    prope_kwargs: dict[str, Any] | None
    sgf_attention: Any = None
    sequence_lengths: tuple[int, ...] = ()


def _is_flex_block_mask(mask: Any) -> bool:
    """Recognise torch FlexAttention BlockMask without importing it eagerly."""

    return (
        mask is not None and type(mask).__name__ == "BlockMask" and hasattr(mask, "kv_num_blocks")
    )


def _default_prope() -> FusedPrope:
    """Resolve H3's shared head-sliced/logd4 PRoPE runtime."""

    try:
        from .torch_prope import prope_qkv
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "fused PRoPE was requested, but h3.torch_prope.prope_qkv is unavailable"
        ) from exc
    return prope_qkv


def _normalise_camera_rows(
    tensor: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    num_selected_rows: int,
    frame_ids: torch.Tensor | None,
    matrix_size: int,
) -> torch.Tensor:
    """Accept either token-aligned or frame-aligned camera tensors."""

    if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (matrix_size, matrix_size):
        raise ValueError(
            f"{name} must be [B,N,{matrix_size},{matrix_size}], got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] not in (1, batch_size):
        raise ValueError(
            f"{name} batch {tensor.shape[0]} cannot serve attention batch {batch_size}"
        )
    if tensor.shape[0] == 1 and batch_size != 1:
        tensor = tensor.expand(batch_size, -1, -1, -1)
    if tensor.shape[1] == num_selected_rows:
        return tensor
    if frame_ids is None:
        raise ValueError(
            f"{name} has {tensor.shape[1]} camera frames "
            f"for {num_selected_rows} selected token rows; "
            "pass prope_frame_ids to expand frame-aligned cameras"
        )
    frame_ids = frame_ids.to(device=tensor.device, dtype=torch.long)
    if frame_ids.ndim != 1 or frame_ids.numel() != num_selected_rows:
        raise ValueError(
            f"prope_frame_ids must have {num_selected_rows} entries, got {tuple(frame_ids.shape)}"
        )
    if frame_ids.numel() and (int(frame_ids.min()) < 0 or int(frame_ids.max()) >= tensor.shape[1]):
        raise ValueError(f"prope_frame_ids addresses outside {name}'s {tensor.shape[1]} frames")
    return tensor.index_select(1, frame_ids)


class SolarMiniMaxH3AttnProcessor(MiniMaxH3AttnProcessor):
    """Upstream-compatible processor that unwraps SolarWM attention controls."""

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: Any = None,
    ) -> torch.Tensor:
        if not isinstance(attention_mask, H3AttentionControl):
            return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
        if rotary_emb is None:
            raise ValueError("SolarWM H3 block attention requires native MM-RoPE")
        control = attention_mask
        if control.sequence_lengths:
            # Activation recomputation may follow a score-model forward with
            # a different packed length. Restore this document's SP layout.
            get_sequence_parallel_state().sequence_lengths = control.sequence_lengths
        return SolarMiniMaxH3Transformer3DModel._attention_forward(
            attn,
            hidden_states,
            rotary_emb,
            control.attention_mask,
            fused_prope=control.fused_prope,
            prope_token_indices=control.prope_token_indices,
            prope_frame_ids=control.prope_frame_ids,
            cam_viewmats=control.cam_viewmats,
            cam_K=control.cam_K,
            prope_kwargs=control.prope_kwargs,
            sgf_attention=control.sgf_attention,
        )


class SolarMiniMaxH3Transformer3DModel(H3AnyFlowConditioningMixin, MiniMaxH3Transformer3DModel):
    """Upstream H3 attention plumbing with optional AnyFlow time conditioning."""

    def _install_solar_processors(self) -> None:
        """Install non-module processors without changing model state keys."""

        # Processors are plain callables, not nn.Modules: replacing them adds no
        # state_dict keys. Installing before FSDP wrapping keeps every block's
        # ordinary forward boundary intact for sharding/checkpoint hooks.
        for layer_index, block in enumerate(self.transformer_blocks):
            block.attn.set_processor(SolarMiniMaxH3AttnProcessor())
            block.attn.h3_layer_index = layer_index

    @classmethod
    def from_config(
        cls,
        config: Any = None,
        return_unused_kwargs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Construct with the inherited upstream init signature, then adapt attention."""

        result = super().from_config(
            config,
            return_unused_kwargs=return_unused_kwargs,
            **kwargs,
        )
        if return_unused_kwargs:
            model, unused_kwargs = result
            model._install_solar_processors()
            return model, unused_kwargs
        result._install_solar_processors()
        return result

    @classmethod
    def strict_from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any):
        """Load a diffusers H3 checkpoint and reject every key/shape mismatch."""

        kwargs.pop("output_loading_info", None)
        model, loading_info = cls.from_pretrained(
            pretrained_model_name_or_path,
            output_loading_info=True,
            **kwargs,
        )
        problems = {
            name: loading_info.get(name)
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(name)
        }
        if problems:
            raise RuntimeError(f"MiniMax-H3 checkpoint did not strict-load: {problems}")
        return model

    @staticmethod
    def _prepare_prope(
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fused_prope: bool | FusedPrope | None,
        prope_token_indices: torch.Tensor | None,
        prope_frame_ids: torch.Tensor | None,
        cam_viewmats: torch.Tensor | None,
        cam_K: torch.Tensor | None,
        prope_kwargs: dict[str, Any] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Callable[[torch.Tensor], torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        cameras_present = cam_viewmats is not None or cam_K is not None
        if fused_prope is False:
            return query, key, value, None, None
        if not cameras_present and fused_prope is not True and not callable(fused_prope):
            return query, key, value, None, None
        if cam_viewmats is None or cam_K is None:
            raise ValueError("fused PRoPE requires both cam_viewmats and cam_K")
        if prope_token_indices is None:
            raise ValueError(
                "fused PRoPE requires explicit prope_token_indices; "
                "token_tags cannot distinguish Qwen vision rows"
            )

        indices = prope_token_indices.to(device=query.device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError(
                f"prope_token_indices must be one-dimensional, got {tuple(indices.shape)}"
            )
        if indices.numel() == 0:
            return query, key, value, None, indices
        if int(indices.min()) < 0 or int(indices.max()) >= query.shape[1]:
            raise ValueError("prope_token_indices contains an out-of-range packed row")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("prope_token_indices must not contain duplicates")

        viewmats = _normalise_camera_rows(
            cam_viewmats,
            name="cam_viewmats",
            batch_size=query.shape[0],
            num_selected_rows=indices.numel(),
            frame_ids=prope_frame_ids,
            matrix_size=4,
        ).to(device=query.device)
        intrinsics = _normalise_camera_rows(
            cam_K,
            name="cam_K",
            batch_size=query.shape[0],
            num_selected_rows=indices.numel(),
            frame_ids=prope_frame_ids,
            matrix_size=3,
        ).to(device=query.device)
        # Match the established Wan camera-training convention.  Artifact K is
        # still loaded and shape/alignment-validated above; only the PRoPE
        # runtime value is replaced.
        intrinsics = fixed_intrinsics_like(intrinsics)
        if callable(fused_prope):
            selected_q = query.index_select(1, indices).transpose(1, 2)
            selected_k = key.index_select(1, indices).transpose(1, 2)
            selected_v = value.index_select(1, indices).transpose(1, 2)
            selected_q, selected_k, selected_v, apply_output = fused_prope(
                selected_q,
                selected_k,
                selected_v,
                viewmats=viewmats,
                Ks=intrinsics,
                **(prope_kwargs or {}),
            )
            query = query.index_copy(1, indices, selected_q.transpose(1, 2))
            key = key.index_copy(1, indices, selected_k.transpose(1, 2))
            value = value.index_copy(1, indices, selected_v.transpose(1, 2))
            return query, key, value, apply_output, indices
        else:
            # Einsum requires camera matrices and q/k/v to share a dtype. Cast
            # only the small camera tensors. Identity matrices on non-camera
            # rows make the operation token-selective without materialising
            # selected q/k/v plus three full-sequence index_copy results.
            dtype = query.dtype
            batch, sequence = query.shape[:2]
            full_viewmats = (
                torch.eye(4, device=query.device, dtype=dtype)
                .view(1, 1, 4, 4)
                .expand(batch, sequence, 4, 4)
                .clone()
            )
            full_intrinsics = (
                torch.eye(3, device=query.device, dtype=dtype)
                .view(1, 1, 3, 3)
                .expand(batch, sequence, 3, 3)
                .clone()
            )
            full_viewmats = full_viewmats.index_copy(1, indices, viewmats.to(dtype))
            full_intrinsics = full_intrinsics.index_copy(1, indices, intrinsics.to(dtype))
            query, key, value, apply_output = _default_prope()(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                viewmats=full_viewmats,
                Ks=full_intrinsics,
                **(prope_kwargs or {}),
            )
            return (
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                apply_output,
                None,
            )

    @staticmethod
    def _sgf_attention_forward(attn, query, key, value, control):
        from .sgf_attention import sgf_attention as attend_sgf

        sp_enabled = is_sequence_parallel_enabled()
        if sp_enabled:
            if attn.heads % get_sp_size():
                raise ValueError("H3 SGF heads must be divisible by SP size")
            query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
            key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
            value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def segment_attention(q, k, v):
            # Each segment contains exactly its visible keys. Flash
            # attention needs no token-level mask or dense score tensor.
            from flash_attn import flash_attn_func

            return flash_attn_func(
                q.contiguous(), k.contiguous(), v.contiguous(), dropout_p=0.0, causal=False
            )

        attended = attend_sgf(
            query,
            key,
            value,
            control=control,
            layer=attn.h3_layer_index,
            apply_rotary=_apply_rotary_emb,
            attend=segment_attention,
        )
        if sp_enabled:
            attended = sequence_model_parallel_all_to_all_4D(attended, scatter_dim=1, gather_dim=2)
        attended = attended.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](attended))

    @classmethod
    def _attention_forward(
        cls,
        attn: Any,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any,
        *,
        fused_prope: bool | FusedPrope | None,
        prope_token_indices: torch.Tensor | None,
        prope_frame_ids: torch.Tensor | None,
        cam_viewmats: torch.Tensor | None,
        cam_K: torch.Tensor | None,
        prope_kwargs: dict[str, Any] | None,
        sgf_attention: Any = None,
    ) -> torch.Tensor:
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if sgf_attention is not None:
            return cls._sgf_attention_forward(attn, query, key, value, sgf_attention)

        # H3 native MM-RoPE first, then token-selective camera PRoPE.
        query = _apply_rotary_emb(query, *rotary_emb)
        key = _apply_rotary_emb(key, *rotary_emb)
        query, key, value, apply_output, prope_indices = cls._prepare_prope(
            query=query,
            key=key,
            value=value,
            fused_prope=fused_prope,
            prope_token_indices=prope_token_indices,
            prope_frame_ids=prope_frame_ids,
            cam_viewmats=cam_viewmats,
            cam_K=cam_K,
            prope_kwargs=prope_kwargs,
        )

        sp_enabled = is_sequence_parallel_enabled()
        if sp_enabled:
            # Shard the already-packed heterogeneous document in both stages.
            # Ulysses exchanges local sequence for local heads only after both
            # native MM-RoPE and token-selective camera PRoPE have been applied,
            # so the full-sequence attention is numerically the same operation
            # as SP1. Stage1 retains its full, head-broadcast W6 BlockMask:
            # after this exchange each rank owns every row and fewer heads.
            sp_size = get_sp_size()
            if attn.heads % sp_size:
                raise RuntimeError(
                    f"MiniMax-H3 attention heads={attn.heads} must be divisible "
                    f"by sp_size={sp_size}"
                )
            query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
            key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
            value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        if _is_flex_block_mask(attention_mask):
            expected_shape = (int(query.shape[1]), int(key.shape[1]))
            if tuple(attention_mask.seq_lengths) != expected_shape:
                raise ValueError(
                    "H3 FlexAttention BlockMask/QK shape mismatch: "
                    f"mask={tuple(attention_mask.seq_lengths)} qk={expected_shape}"
                )
            if (
                query.device.type != "cuda"
                or key.device != query.device
                or value.device != query.device
            ):
                raise RuntimeError("H3 Stage1 FlexAttention requires Q/K/V on one CUDA device")
            # Release the sequence-major references as each head-major input
            # is prepared; retain native Flex and its exact global mask.
            query = prepare_flex_attention_input(query)
            key = prepare_flex_attention_input(key)
            value = prepare_flex_attention_input(value)
            attended = _compiled_h3_flex_attention(
                query,
                key,
                value,
                attention_mask,
                training=attn.training,
            ).transpose(1, 2)
        else:
            attended = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=getattr(attn.processor, "_attention_backend", None),
                parallel_config=getattr(attn.processor, "_parallel_config", None),
            )

        if sp_enabled:
            attended = sequence_model_parallel_all_to_all_4D(attended, scatter_dim=1, gather_dim=2)

        if apply_output is not None:
            if prope_indices is None:
                attended = apply_output(attended.transpose(1, 2)).transpose(1, 2)
            elif prope_indices.numel():
                selected = attended.index_select(1, prope_indices).transpose(1, 2)
                selected = apply_output(selected).transpose(1, 2)
                attended = attended.index_copy(1, prope_indices, selected)
        attended = attended.flatten(2, 3).type_as(query)
        attended = attn.to_out[0](attended)
        return attn.to_out[1](attended)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        *,
        r_timestep: torch.Tensor | None = None,
        attention_mask: Any = None,
        stage1_block_mask: Any = None,
        fused_prope: bool | FusedPrope | None = None,
        prope_token_indices: torch.Tensor | None = None,
        prope_frame_ids: torch.Tensor | None = None,
        cam_viewmats: torch.Tensor | None = None,
        cam_K: torch.Tensor | None = None,
        prope_kwargs: dict[str, Any] | None = None,
        stage0p5_sequence_parallel: bool = False,
        packed_sequence_parallel: bool | None = None,
        sgf_attention: Any = None,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        """Run upstream H3 with an optional dense/Flex mask and fused PRoPE.

        ``stage1_block_mask`` is an explicit alias for a FlexAttention
        ``BlockMask``. Camera tensors may be token-aligned or frame-aligned; in
        the latter case pass the layout's ``camera_frame_ids``. ``cam_viewmats``
        follows SolarWM PRoPE's world-to-camera convention; callers holding
        camera-to-world matrices must invert them before this boundary.
        AnyFlow requires ``r_timestep`` with exactly ``timestep.shape``, in
        H3's native data-ward time coordinate (``1 - sigma``).
        ``packed_sequence_parallel`` enables the shared packed/Ulysses path;
        ``stage0p5_sequence_parallel`` remains a compatibility alias.
        """

        del attention_kwargs  # LoRA scaling is handled by the decorator.
        if attention_mask is not None and stage1_block_mask is not None:
            raise ValueError("pass only one of attention_mask and stage1_block_mask")
        if stage1_block_mask is not None:
            attention_mask = stage1_block_mask
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"position_ids must be [sequence,3], got {tuple(position_ids.shape)}")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "token_tags and timestep_indices must match the packed sequence length"
            )
        sp_enabled = is_sequence_parallel_enabled()
        if stage0p5_sequence_parallel and packed_sequence_parallel is False:
            raise ValueError("conflicting H3 sequence-parallel flags")
        requested_sp = (
            bool(stage0p5_sequence_parallel)
            if packed_sequence_parallel is None
            else bool(packed_sequence_parallel)
        )
        if sp_enabled != requested_sp:
            raise RuntimeError(
                "MiniMax-H3 SP runtime/forward contract mismatch: "
                f"runtime_sp={get_sp_size()} "
                f"packed_sequence_parallel={requested_sp}"
            )
        if sp_enabled:
            validate_packed_attention_mask(attention_mask, total_tokens=sequence_length)
        if sgf_attention is not None:
            if attention_mask is not None or fused_prope is not True:
                raise ValueError("H3 SGF requires its own segmented attention and fused PRoPE")
            from .sgf_attention import prepare_sgf_rotaries

            sgf_attention = prepare_sgf_rotaries(sgf_attention, self.rope)

        video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio_embeds = self.audio_proj_in(
            audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in))
        )
        text_embeds = self.context_embedder(
            encoder_hidden_states.to(get_parameter_dtype(self.context_embedder))
        )
        text_embeds = self.token_refiner(text_embeds)
        packed = text_embeds.new_zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
        )
        packed = packed.index_copy(1, text_indices, text_embeds)
        packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self._time_condition(timestep, r_timestep, parameter_dtype=get_parameter_dtype)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        if sp_enabled:
            shard = shard_packed_sequence(
                hidden_states=packed,
                position_ids=position_ids,
                token_tags=token_tags,
                timestep_indices=timestep_indices,
                prope_token_indices=prope_token_indices,
                prope_frame_ids=prope_frame_ids,
                camera_viewmats=cam_viewmats,
                camera_K=cam_K,
            )
            packed = shard.hidden_states
            position_ids = shard.position_ids
            token_tags = shard.token_tags
            timestep_indices = shard.timestep_indices
            adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
            prope_token_indices = shard.prope_token_indices
            prope_frame_ids = shard.prope_frame_ids
            cam_viewmats = shard.camera_viewmats
            cam_K = shard.camera_K
        rotary_emb = self.rope(position_ids)
        needs_control = (
            _is_flex_block_mask(attention_mask)
            or fused_prope is not None
            or cam_viewmats is not None
            or cam_K is not None
            or prope_token_indices is not None
            or prope_frame_ids is not None
            or prope_kwargs is not None
        )
        attention_control = (
            H3AttentionControl(
                attention_mask=attention_mask,
                fused_prope=fused_prope,
                prope_token_indices=prope_token_indices,
                prope_frame_ids=prope_frame_ids,
                cam_viewmats=cam_viewmats,
                cam_K=cam_K,
                prope_kwargs=prope_kwargs,
                sgf_attention=sgf_attention,
                sequence_lengths=shard.sequence_lengths if sp_enabled else (),
            )
            if needs_control
            else attention_mask
        )
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def custom_forward(
                    states: torch.Tensor, current_block: Any = block
                ) -> torch.Tensor:
                    return current_block(states, temb, adaln_indices, rotary_emb, attention_control)

                packed = self._gradient_checkpointing_func(custom_forward, packed)
            else:
                packed = block(packed, temb, adaln_indices, rotary_emb, attention_control)

        packed = self.norm_out(packed, temb, timestep_indices).to(
            get_parameter_dtype(self.proj_out)
        )
        video_packed_output = self.proj_out(packed)
        audio_packed_output = self.audio_proj_out(packed)
        if sp_enabled:
            # Restore only the small output-head widths, not the 5,376-wide
            # residual stream. Every SP rank then computes the identical full
            # logical loss; the trainer scales it by 1/sp_size before backward.
            video_packed_output = sequence_model_parallel_all_gather(video_packed_output, dim=1)
            audio_packed_output = sequence_model_parallel_all_gather(audio_packed_output, dim=1)
            if (
                int(video_packed_output.shape[1]) != sequence_length
                or int(audio_packed_output.shape[1]) != sequence_length
            ):
                raise RuntimeError(
                    "MiniMax-H3 SP output gather did not restore the full packed "
                    f"sequence={sequence_length}: video={video_packed_output.shape[1]} "
                    f"audio={audio_packed_output.shape[1]}"
                )
        video_output = video_packed_output.index_select(1, video_indices)
        audio_output = audio_packed_output.index_select(1, audio_indices)
        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


__all__ = [
    "FusedPrope",
    "H3AttentionControl",
    "MiniMaxH3TransformerBlock",
    "SolarMiniMaxH3AttnProcessor",
    "SolarMiniMaxH3Transformer3DModel",
]
