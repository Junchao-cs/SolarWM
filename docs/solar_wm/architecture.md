# Architecture

SolarWM is organized around four independent contracts: configuration, data,
model backends, and execution. A model family owns model-specific math; it does
not own dataset semantics, process topology, checkpoint policy, or command-line
entrypoints.

```text
solar CLI
  -> strict versioned run config
  -> deterministic data plan -> local or object-store transport
  -> model backend registry   -> Wan 2.2 / LTX-2.5 / MiniMax-H3
  -> train, infer, or preencode engine
  -> manifest + checkpoint + validation outputs
```

## Package boundaries

- `solar.config`: loading, path resolution, strict validation, and stable
  resolved-config output.
- `solar.data`: canonical indexes, virtual occurrences, rank/worker/SP
  ownership, frame starts, camera conventions, and transport-independent sample
  materialization.
- `solar.backends`: lazily imported model-family plugins. A backend declares
  its supported stages and validates its backend-specific configuration.
- `solar.training`: stage/objective orchestration, distributed topology, EMA,
  checkpoint transactions, and finite-value policy.
- `solar.inference`: the same backend and camera path used by validation.
- `solar.preencode`: versioned raw-to-latent production and schema checks.

## Dependency direction

Core modules never import a concrete backbone. Backends may import core
protocols, but backend-to-backend imports are prohibited. Dataset readers do
not infer camera normalization from a class or dataset name. Local and bucket
access are transports beneath one canonical index and sampler.

## Component contracts

Each component has a documented input/output contract, deterministic fixtures,
and a run manifest that binds the resolved configuration and input identities.
