# MiniMax-H3 training speedups

Opt-in environment switches that make H3 training faster on the same GPUs and the
same released recipes. With every switch unset, training runs upstream's code paths.

| Stage | Reference step | With the switches | Speedup |
|---|---:|---:|---:|
| Stage0.5 FM (`stage0p5-158f-lora384-sp2`) | 35.64 s | 19.24 s | 1.85× |
| Stage1 TF-AnyFlow (`stage1-158f-lora384-w6-sp2`) | 95.81 s | 45.15 s | 2.12× |
| Stage2 SGF (`stage2-158f-lora384-w6-sp4`), per outer step | 79.69 s | 42.45 s | 1.88× |

Measured on one node with 8 GPUs, timing the reference, the switched run and the
reference again in one session. Peak allocated memory per GPU: Stage0.5 78.0 → 101.0 GiB,
Stage1 97.8 → 92.2 GiB, Stage2 116.5 → 112.8 GiB.

## Usage

Export these before `torchrun ... -m solar train` (the training commands in
[MiniMax-H3](minimax-h3.md) are unchanged):

```bash
export SOLARWM_H3_ATTENTION=cudnn SOLARWM_H3_CUDNN_STAGES=stage0p5,stage1
export SOLARWM_H3_SGF_CRITIC_CUDNN=1 SOLARWM_H3_SGF_ROLLOUT_CUDNN=1
export SOLARWM_H3_W6=segments SOLARWM_H3_STAGE1_PREFIX_REUSE=1
export SOLARWM_H3_SGF_PLAN=1 SOLARWM_H3_SGF_PREFIX_REUSE=1 SOLARWM_H3_SGF_KV_WINDOW=1
export SOLARWM_H3_SGF_PIECES=1 SOLARWM_H3_DEFER_FINITE=1
export SOLARWM_H3_FUSED=1 SOLARWM_H3_SWIGLU_FUSED=1 SOLARWM_H3_PROPE_PLAN=1
export SOLARWM_H3_PROPE_FUSED=1 SOLARWM_H3_PROPE_V2=1
export SOLARWM_H3_LORA_FAST=1 SOLARWM_H3_LORA_ADDMM=1
export SOLARWM_H3_A2A_FAST=1 SOLARWM_H3_A2A_COPY=1
export SOLARWM_H3_SAVED_ATTENTION_BLOCKS=stage0p5=50,critic=30
export SOLARWM_H3_NO_AC_BLOCKS=stage0p5=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`SAVED_ATTENTION_BLOCKS` and `NO_AC_BLOCKS` trade memory for speed. On GPUs with less
memory, lower them or leave them unset.

## Switches

All names carry the `SOLARWM_H3_` prefix. "Rounding" marks switches whose results
differ from upstream in the last bits; the others were bitwise equal to upstream
in unit tests.

| Switch | Effect | Stages | Rounding |
|---|---|---|---|
| `ATTENTION=cudnn` | cuDNN fused SDPA instead of FlashAttention-2 for unmasked attention | all | yes |
| `CUDNN_STAGES=<list>` | limit `ATTENTION=cudnn` to the listed stages | all | – |
| `SGF_CRITIC_CUDNN=1` | cuDNN attention for the Stage2 critic update (forward and backward) | 2 | yes |
| `SGF_ROLLOUT_CUDNN=1` | cuDNN attention for the detached Stage2 rollouts | 2 | yes |
| `W6=segments` | W6 mask as per-run dense cuDNN attention instead of FlexAttention | 1 | yes |
| `STAGE1_PREFIX_REUSE=1` | the two later detached AnyFlow targets reuse the W6 prefix keys/values | 1 | no |
| `SGF_PLAN=1` | camera and rotary tensors built once per forward instead of per layer | 2 | no |
| `SGF_PREFIX_REUSE=1` | rollout reuses the prefix keys/values across denoising steps | 2 | yes |
| `SGF_KV_WINDOW=1` | rollout K/V window built once per chunk and layer, one allocation | 2 | no |
| `SGF_PIECES=1` | replay with gradients takes segment rows from one split of q/k/v | 2 | yes |
| `DEFER_FINITE=1` | rollout finiteness check once per rollout instead of per chunk | 2 | no |
| `FUSED=1` | Triton RMSNorm+AdaLN modulation, gated residual, q/k RMSNorm+rotary | all | yes |
| `SWIGLU_FUSED=1` | Triton SwiGLU forward and backward | all | no |
| `PROPE_PLAN=1` | camera PRoPE matrices built once per forward | 0.5, 1 | no |
| `PROPE_FUSED=1` | camera PRoPE projection (and SGF MM-RoPE+PRoPE) as one Triton kernel | all | yes |
| `PROPE_V2=1` | coalesced variant of the `PROPE_FUSED` kernels | all | no |
| `LORA_FAST=1` | PEFT LoRA forward without the ×1.0 scaling and cast kernels | all | no |
| `LORA_ADDMM=1` | LoRA up-projection accumulated onto the base output in the GEMM epilogue | all | yes |
| `A2A_FAST=1` | Ulysses all-to-all with one packing copy per exchange | all | no |
| `A2A_COPY=1` | Triton block-transpose copy for that packing | all | no |
| `SAVED_ATTENTION_BLOCKS=stage0p5=<n>,critic=<n>` | activation checkpointing keeps the attention output of the first n blocks, so recomputation skips attention | 0.5, 2 | no |
| `NO_AC_BLOCKS=stage0p5=<n>` | the last n student blocks run without activation checkpointing | 0.5 | no |

Stage2 keeps FlashAttention-2 for the student's replay and score forwards
(`CUDNN_STAGES` without `stage2`). The student's DMD gradient is the difference of two
near-identical score evaluations (critic and teacher), so it follows the attention
kernel's rounding. With cuDNN there, the student's weight updates moved about 3.5 seed
standard deviations away from the reference in the trajectory check below.

## Verification

The switches above were checked against upstream 0193da0 in three ways:

- **Structure.** The released recipe configs and the launch path are unchanged, and
  the data reader yields bitwise the same first batches on every rank.
- **Short deterministic runs.** Two-step cases per stage in deterministic mode: RNG
  states, forward-call counts, invariants, learning rates, tensor names and shapes
  match exactly; tensor values differ only in rounding.
- **Training trajectories.** Loss, gradient norm and weight-update endpoints stay
  within 3 seed standard deviations of the reference in every stage. The seed spread
  comes from six reference seeds.

Tests: `tests/backends/minimax_h3/test_speedup_*.py` (CUDA, skipped on CPU).
