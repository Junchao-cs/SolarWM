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

## Reactor checks

For changes to the Reactor application, use a separate Python 3.12 environment
with `reactor-runtime==3.8.3`, Pillow and pytest. From
`deploy/solar_renderer/reactor/`, run:

```bash
PYTHONPATH=. python -m pytest tests/ -q
python -m reactor_runtime.schema --path . --out schema.json
reactor validate
cd demo
npm ci
npm run build
```

In the renderer environment, run `python -m pytest tests/renderer/test_reactor_integration.py -q` from the repository root.
These checks do not establish GPU inference or Docker execution.

Keep usage guides focused on current inputs, commands, outputs and limitations.
Record one-time migrations and integration history in commit or release notes;
keep source attribution in `NOTICE`.
