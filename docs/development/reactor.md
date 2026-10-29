# Reactor development

For installation and operation, use the [deployment guide](../../deploy/solar_renderer/reactor/README.md).

## Application and frontend checks

Use a separate environment with Python 3.12 or later. Reactor Runtime 3.8.3
and its NumPy requirements differ from the model training environments;
Runtime 2.x is incompatible with this adapter. From `deploy/solar_renderer/reactor/`:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install reactor-runtime==3.8.3 Pillow pytest
PYTHONPATH=. python -m pytest tests/ -q
python -m reactor_runtime.schema --path . --out schema.json
reactor validate
cd demo
npm ci
npm run build
```

In a renderer CPU environment, run the sampler integration checks from the
repository root:

```bash
python -m pytest tests/renderer/test_reactor_integration.py -q
```

These checks cover application contracts, sampling and frontend compilation.
They do not establish GPU inference, Docker execution or visual equivalence.

## Module responsibilities

- `solarrenderer.py`: client commands and output.
- `solarrenderer_types.py`: public application state.
- `solarrenderer_model.py`: Runtime-independent generation steps.
- `stage_source.py`: package the shared `solar` source and renderer assets for
  the Docker build. The snapshot includes local changes and records its revision
  and file hashes in `provenance.json`; startup verifies those hashes.

The encoder and LoRA loader come from the shared renderer package. Sampling and
decoding retain per-video state between calls. Changes to input preprocessing
must preserve the native encoder's inputs; compare those inputs before comparing
sampled outputs. Upstream source attribution is retained in [NOTICE](../../deploy/solar_renderer/reactor/NOTICE).
