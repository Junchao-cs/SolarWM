# Solar repository layout

SolarWM and SolarRenderer share the root [`solar/`](../solar/) Python package
and one [`pyproject.toml`](../pyproject.toml). Their configurations, documentation,
examples, and deployment entrypoints are grouped by project.

| Previous location | Current location |
| --- | --- |
| `src/solarwm/` | [`solar/`](../solar/) |
| `src/solarwm/realtime_interaction/` | [`deploy/solar_wm/`](../deploy/solar_wm/) |
| `reactor/solar_renderer/` | [`deploy/solar_renderer/reactor/`](../deploy/solar_renderer/reactor/) |
| `comfyui/solar_renderer/` | [`deploy/solar_renderer/comfyui/`](../deploy/solar_renderer/comfyui/) |
| `configs/examples/` | [`configs/solar_wm/`](../configs/solar_wm/) |
| `docs/solarwm/` | [`docs/solar_wm/`](solar_wm/) |
| `examples/solarwm/` | [`examples/solar_wm/`](../examples/solar_wm/) |

## Installation and commands

Reinstall from this checkout after moving to the new layout, using the same
backend environment as before:

```bash
python -m pip install --no-deps -e .
python -m solar --version
solar config resolve --config configs/solar_wm/minimax_h3/stage0p5-158f-lora384-sp2.yaml
solar-renderer --help
python -m solar.renderer.training --help
bash deploy/solar_wm/launch.sh configure --help
```

New Python imports use `solar.*`, and module commands use `python -m solar` or
`python -m solar.renderer.<module>`. The `solarwm` console command remains an
alias of the unified CLI; the old `python -m solarwm` module path is retired.
Update external scripts and any explicit module strings accordingly.

Use the repository root, rather than its former `src/` directory, when setting
`PYTHONPATH`. Update external launch configurations and ComfyUI symlinks to the
new deployment paths. Renderer entrypoints continue to use `solar-renderer`.

## Deployment resources

SolarWM deployment source and assets live in `deploy/solar_wm/` in both the
checkout and installed wheel. Canonical images and prompts live in
`examples/solar_wm/` in a checkout and are installed as `solar.examples` resources.
Configuration generation discovers the appropriate location automatically.

Reactor stages the root `solar/` package into its existing ignored
`.solarwm-source/` build context. Run `stage_source.py` again after updating the
checkout and rebuild the deployment. Its client resolves bundled examples from
the repository root. ComfyUI resolves the same shared renderer resources through
the new node-pack location.

The realtime source pins were updated only after verifying that the pinned
source changes are exactly the `solarwm.*` to `solar.*` import relocation.
Compiled kernel identities and calibration assets are retained. Source checks
and CPU tests do not substitute for a full GPU deployment smoke run after
changing environments or rebuilding compiled caches.

## Existing data and checkpoints

Versioned configuration schemas, checkpoint formats, model identifiers, asset
hashes, and environment variable names retain their existing values. A package
rename does not require converting training data or checkpoint contents. Legacy
built-in gallery manifests continue to be adapted in memory.

Externally stored configuration files are not rewritten automatically. Update
any paths they contain that point into the old checkout layout, including the
image paths in existing scene manifests, before launching from the new checkout.
