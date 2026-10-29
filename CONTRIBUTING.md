# Contributing

SolarWM treats reproducibility contracts as public API. A change to sample
order, camera convention, tensor shape/dtype, objective math, checkpoint
schema, or validation generation must include a focused contract test and a
clear user-facing change note.

Before opening a change:

```bash
python -m pip install -e '.[dev]'
ruff check solar deploy tests tools
ruff format --check solar deploy tests tools
pytest -q
python -m build
```

Keep model-family dependencies lazy and isolated. Core configuration and data
inspection commands must remain usable without torch. Do not add
credentials, private data payloads, mutable image tags, generated checkpoints,
or large model artifacts to the repository. User-facing examples and published
indexes must use relative paths or explicit `/path/to/...` placeholders.

New model stages require configuration, data, objective, checkpoint, inference,
validation, and runtime tests before they are registered as supported routes.

## Source layout

| Directory | Contents |
| --- | --- |
| `solar/` | Shared Python package and model backends. |
| `configs/solar_wm/`, `docs/solar_wm/` | Camera-controlled world-model recipes and guides. |
| `configs/solar_renderer/`, `docs/solar_renderer/` | Structure-conditioned rendering recipes and guides. |
| `deploy/` | Realtime, Reactor and ComfyUI applications. |
| `data-engine/` | Independently versioned data pipeline submodules. |
| `examples/` | Example inputs and prompts. |

See the [architecture](docs/solar_wm/architecture.md),
[data contract](docs/solar_wm/data-contract.md), and
[developer guides](docs/development/README.md) before changing shared behavior.

Keep usage guides focused on current inputs, commands, outputs and limitations.
Record one-time migrations and integration history in commit or release notes.
Source attribution belongs in `NOTICE`; versioned validation reports must identify
the measured runtime and distinguish CPU checks from GPU and quality evidence.
