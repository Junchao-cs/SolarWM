---
name: solar-training-ops
description: Configure, launch, resume, monitor, validate, run inference, and diagnose SolarWM jobs for Wan 2.2 TI2V-5B, Wan 2.2 I2V-A14B, LTX-2.5, and MiniMax-H3. Use for supported-route selection, distributed topology, checkpoint and EMA behavior, periodic validation, standalone generation, or training-runtime failures. Excludes dataset construction, preencoding and publication, model or data acquisition, container construction, and provider-specific cluster provisioning.
---

# SolarWM Training and Inference Operations

Operate from a SolarWM source checkout. Treat the current checkout, resolved
configuration, active runtime, and generated manifests as authoritative. A
configuration filename or run name is descriptive, not proof of the route or
runtime state.

## Scope

Own these tasks:

- select a supported training or inference route;
- adapt and resolve a checked-in example configuration;
- validate distributed topology and batch arithmetic;
- launch, initialize, resume, monitor, and stop a task-owned run;
- verify checkpoints, periodic validation, standalone inference, and comparison
  outputs;
- diagnose non-finite values, OOM, distributed hangs, shape failures, checkpoint
  incompatibility, and missing validation artifacts.

Do not use this skill to build or publish datasets, preencode samples, acquire
model weights, build containers, or provision a particular cloud or cluster.
Do not turn a diagnosis-only request into a restart or code change.

## Read only the relevant documentation

Read the document for the selected backend before acting:

- [Wan 2.2 TI2V-5B](../../docs/backends/wan22-ti2v-5b.md)
- [Wan 2.2 I2V-A14B](../../docs/backends/wan22-i2v-a14b.md)
- [LTX-2.5](../../docs/backends/ltx25.md)
- [MiniMax-H3](../../docs/backends/minimax-h3.md)

Read [runtime environments](../../environments/README.md) for backend setup.
Read the [data contract](../../docs/data-contract.md) only when the request
involves indexes, sampling, storage transport, camera inputs, or shard access.

## What each training route learns

Stage names describe the role of a training phase; they do not imply that every
backend supports the same objective. Use the current route list as the capability
boundary.

| Backend | Stage and route | What is trained |
|---|---|---|
| Wan 2.2 TI2V-5B | Stage0.5, bidirectional FM, 81f/153f | Denoise the full latent clip in both temporal directions. The model predicts the straight-path velocity from clean video to noise with a masked, timestep-weighted MSE. |
| Wan 2.2 TI2V-5B | Stage1, teacher forcing with AnyFlow loss, 81f | Train causal generation with paired `[clean context | noisy target]` tokens while conditioning on a start time `t` and endpoint `r`. The objective mixes diffusion (`r=t`), consistency (`r=0`), and intermediate forward-map (`0<r<t`) samples so one stage learns both denoising and finite-step flow maps. |
| Wan 2.2 TI2V-5B | Stage2, DMD via SGF, 81f | Train a causal student on its own autoregressive rollout. A frozen teacher and a trainable critic estimate the direction used by the student's surrogate loss; the critic itself receives an FM loss on student-generated samples. |
| Wan 2.2 I2V-A14B | Stage0.5, bidirectional FM, 81f/153f | Train full-clip velocity prediction while conditioning on A14B's separate first-image tensor. Its latent layout and conditioning path are not interchangeable with TI2V-5B. |
| LTX-2.5 video-only | Stage0.5, native rectified flow, 153f | Predict LTX's native clean-to-noise velocity. The first latent remains clean as the image condition and is excluded from the velocity MSE. |
| MiniMax-H3 | Stage0.5, bidirectional FM, 158f | Predict the video velocity over the target latent sequence while using H3's anchor, prompt, camera, and packed sequence conditioning. The loss is video-velocity MSE. |
| MiniMax-H3 | Stage1, teacher forcing with AnyFlow, 158f | Train five-latent chunks over a W6 window, using the first 45 encoded latents and native absolute RoPE. |
| MiniMax-H3 | Stage2, SGF, 158f | Roll out 50 latents with cached raw K/V and sliding local RoPE; supervise the first 47 with a bidirectional teacher and critic. |

The training progression is **Stage0.5 FM → Stage1 TF-AnyFlow → Stage2 DMD via
SGF**.

- Stage0.5 establishes the bidirectional full-clip model.
- Stage1 combines causal teacher forcing and the AnyFlow loss in one stage,
  producing weights that can initialize Stage2/DMD without a separate ODE or
  consistency-distillation initialization stage.
- Stage2 DMD via SGF starts a new three-role run. Initialize the student from
  Stage1 AnyFlow EMA and the frozen teacher plus trainable critic from compatible
  Stage0.5 FM weights, exactly as declared in the Stage2 config.

The optional Stage1 TF-FM baseline remains implemented through
`train_stage1_tf_fm_81f.yaml`, but it is not the released Stage2 initializer.

Validate each transition with the next stage's actual load path and deterministic
test-index selection before a long run.

Loss terminology:

- **FM** uses a point on a straight clean/noise path and regresses its velocity.
  Wan and LTX use `x_sigma=(1-sigma)*x0+sigma*eps` with target `eps-x0`.
- **H3 FM** uses the equivalent reverse time coordinate,
  `x_t=t*x0+(1-t)*eps`, with target `x0-eps`.
- **AnyFlow** predicts a map from noise level `t` toward a requested endpoint
  `r`; it is not an alias for ordinary FM or merely a different loss weight.
- **DMD via SGF** trains the student through a teacher/critic-derived gradient
  on the student's own rollout, while retaining a separate FM objective for the
  critic.

Wan I2V-A14B and LTX-2.5 expose no Stage1 or Stage2 training route
in the current release. Reject those combinations before model allocation.

For H3 Stage1/2, use the [backend guide](../../docs/backends/minimax-h3.md#stage1--stage2-setup).
Keep the automatically frozen validation plan, encoded-silence condition and tail-camera
rules fixed. Stage2 EMA begins at student update 39 (outer step 196); validation
at steps 20 and 100 is LIVE only, and step 200 includes LIVE and EMA.

## Operating workflow

### 1. Establish the exact action and route

Identify the model family, stage, objective, frame profile, data encoding,
checkpoint mode, and whether the request is training, periodic validation, or
standalone inference.

List the routes implemented by the current checkout:

```bash
solarwm config routes
```

Select the closest file under `configs/examples/`. Do not transfer stage,
objective, latent layout, model assets, or inference settings between backends.
If the requested combination is not listed, report it as unsupported instead
of allocating a model.

### 2. Resolve the run before allocating GPUs

Inspect the active environment and the selected data controls:
Set the `SOLAR_*` shell variables below from the intended configuration before
running the examples.

```bash
solarwm environment probe
solarwm data inspect /path/to/index.jsonl.gz
solarwm data plan /path/to/index.jsonl.gz \
  --seed "${SOLAR_SEED}" \
  --pixel-frames "${SOLAR_PIXEL_FRAMES}" \
  --world-size "${SOLAR_RAW_WORLD_SIZE}" --rank 0
solarwm config resolve \
  --config configs/examples/<backend>/<config>.yaml \
  --output /path/to/resolved-preview.json
```

Review the resolved action and route, model assets, frame geometry, data index
and transport, camera transform, world and sequence-parallel sizes, effective
global batch, initialization or resume mode, EMA, validation passes, output
directory, and save/validation intervals.

Supply deployment paths with explicit `--set` values or a task-local config.
Do not commit credentials or deployment-specific absolute paths. Reject a run
when unresolved `/path/to/...` placeholders remain in fields it will use.

The stored camera arrays keep their published convention and scale. A configured
camera-translation transform is applied by the model and must agree across
training, checkpoint loading, periodic validation, and standalone inference.

### 3. Check distributed arithmetic

Compute the topology from the resolved values:

```text
logical_dp_world_size = world_size / sequence_parallel_size
global_batch_size = logical_dp_world_size * micro_batch_size * gradient_accumulation
```

Require divisibility and agreement with the configured global batch. Sequence-
parallel peers must consume the same logical occurrence, start frame, prompt,
camera, noise, and timestep state. Dataset ownership uses logical data-parallel
rank, not raw rank. Every collective must execute in the same order on every
member.

Do not assume a topology validated for one backend, resolution, frame count, or
attention layout applies to another.

### 4. Run the smallest representative preflight

Keep preflight proportional to the change. Before a long or multi-node run,
use the selected backend's documented launch path with the intended environment,
data path, and distributed settings. Start with the shortest representative run
that proves the route-specific gates:

- every rank joins and reads the intended logical samples;
- forward, backward, optimizer, scheduler, and EMA behavior are finite;
- a checkpoint transaction completes and can be loaded;
- the configured recipe test-index selection materializes successfully;
- every configured validation pass produces its videos, per-sample manifests,
  comparison outputs, and completion marker.

A short data or distributed bootstrap does not replace the model-step and full
validation-output checks. Stop at the first topology, compatibility, non-finite,
or artifact-count mismatch, fix the cause, and rerun only the affected gate.

### 5. Launch through the unified entrypoints

Use `--standalone` only for a single node. A multi-node deployment must supply
its own rendezvous coordinates while preserving the resolved world size and
rank map.

```bash
torchrun --standalone --nproc-per-node="${SOLAR_LOCAL_WORLD_SIZE}" \
  -m solarwm train \
  --config configs/examples/<backend>/<train-config>.yaml \
  --set distributed.world_size="${SOLAR_RAW_WORLD_SIZE}" \
  --set train.global_batch_size="${SOLAR_GLOBAL_BATCH_SIZE}" \
  --set runtime.output_dir=/path/to/new-output

torchrun --standalone --nproc-per-node="${SOLAR_LOCAL_WORLD_SIZE}" \
  -m solarwm infer \
  --config configs/examples/<backend>/<infer-config>.yaml \
  --set distributed.world_size="${SOLAR_RAW_WORLD_SIZE}" \
  --set runtime.output_dir=/path/to/new-output
```

Inference sample counts are backend-configured: Wan and H3 use
`validation.sample_count`, while LTX uses `inference.sample_count`. Override
the field declared by the selected inference YAML rather than adding a training
batch field to an inference run.

Use a new output directory for a new run. Rank zero writes
`resolved-config.json`, `launch-manifest.json`, and `run-result.json`; inspect
them instead of reconstructing the effective command from memory.

### 6. Preserve checkpoint semantics

Follow the selected config's `checkpoint` block and backend documentation.
Distinguish full resume from weight initialization:

- full resume restores the model, optimizer, scheduler, completed step, EMA,
  random state, and data-stream state required by that route;
- weight initialization starts a new optimization run and does not inherit the
  prior optimizer or step counter;
- role-based routes require every declared role and transaction member;
- an expected checkpoint that is missing, incomplete, or incompatible is an
  error, not permission to start fresh.

Validate model family, stage, objective, camera transform, tensor layout, and
the weight selection declared by the chosen route before model allocation.
The Wan5B Stage2 camera-length inference route accepts the released checkpoint
directory as one model and does not expose LIVE/EMA selection. Accept a saved
checkpoint only after its completion marker and declared members are durable.
Resolve the model role from `release-manifest.json` and retain that resolved
role in output provenance. Decode horizons longer than 60 latents as consecutive
cached VAE tiles, clearing the temporal cache only before and after the sequence.

### 7. Keep inference and validation aligned

Select an inference example matching the checkpoint's backend, stage,
objective, frame profile, and camera transform. The inference config must use
`action: infer` and an explicit checkpoint path. Follow the config when it
selects a weight source; `infer_stage2_sgf_camera_length.yaml` intentionally
loads the released Wan5B Stage2 model without a user-selected weight role.

Periodic validation and standalone inference use the same backend generation
path. Keep the recipe test index, selection seed, sample count, start frames, noise identities,
rollout length, solver, pass names, and weight sources fixed when comparing
checkpoints. If any of those change, compare by manifest identity rather than
rank or filename. Camera-length Stage2 inference is a generation route rather
than a validation-parity comparison: it resolves each sample to the longest
complete three-latent chunks supported by that sample's camera track.

A validation step is complete only when all intended logical slots and passes
have terminal manifests, the successful slots have videos, comparison outputs
are present, and the final completion marker exists. Preserve error manifests
and report partial output sets as partial.

### 8. Monitor and diagnose from evidence

Record the latest optimizer step, loss, gradient norm, step time, checkpoint
count, validation pass/count, and process state. Rank zero may be the only rank
printing global progress; inspect every rank for fatal errors before declaring
the run healthy.

Start diagnosis with the first failing boundary:

- non-finite loss or gradients: first bad rank/step, sample identity, camera,
  timestep, model prediction, and optimizer state;
- OOM: other CUDA clients, frame/resolution shape, activation checkpointing,
  sequence parallelism, accumulation, and optimizer/EMA memory;
- distributed hang: last collective entered by each rank, rank liveness,
  collective order, topology, and network backend logs;
- SP-only mismatch: peer sample, start, prompt, camera, noise, timestep, padding,
  and RNG identity;
- checkpoint load failure: visibility on every rank, completion state, selected
  weight source, and resolved route compatibility;
- missing validation output: logical data-parallel ownership, configured passes,
  per-slot error manifests, fixed-plan starts, decode/encoder availability, and
  output transaction state.

Do not classify a quiet worker, a transient launcher disconnect, or one missing
progress line as a failed collective without checking process and rank state.

## Completion report

Report:

- action and resolved route;
- backend environment and code revision;
- world size, sequence-parallel size, logical data-parallel size, and global
  batch;
- initialization or resume mode and LIVE/EMA selection;
- final observed step, loss, gradient norm, and step time;
- checkpoint completion and validation/inference pass counts;
- comparison-output status and any partial or failed slots;
- remaining failures or unverified conditions.

Only stop task-owned processes identified by exact command and process identity.
Do not delete or overwrite checkpoints, outputs, indexes, or model assets unless
the user explicitly requests that exact mutation.
