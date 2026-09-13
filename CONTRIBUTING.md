# Contributing

SolarWM treats reproducibility contracts as public API. A change to sample
order, camera convention, tensor shape/dtype, objective math, checkpoint
schema, or validation generation must include a focused contract test and a
clear user-facing change note.

Before opening a change:

```bash
python -m pip install -e '.[dev]'
ruff check src tests tools
ruff format --check src tests tools
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
