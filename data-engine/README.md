# Data engines

This directory groups the data preparation pipelines for SolarWM and SolarRenderer.
Each pipeline is an independent Git submodule pinned to a specific upstream commit.

| Directory | Upstream repository | Branch |
|---|---|---|
| `solarwm-data` | [Enderfga/solarwm-data](https://github.com/Enderfga/solarwm-data) | `main` |
| `solar_renderer-data` | [Enderfga/SolarRenderer](https://github.com/Enderfga/SolarRenderer/tree/data) | `data` |

From the root of a SolarWM Git checkout, initialize both pipelines:

```bash
git submodule update --init --recursive -- data-engine/solarwm-data data-engine/solar_renderer-data
```

This checks out the versions recorded by SolarWM. Follow each pipeline's own README
for its environment, data sources, and processing commands. The SolarRenderer
pipeline uses the upstream `data` branch.

If an existing checkout has initialized the former top-level `data-engine`
submodule, commit or stash any changes inside it, then run
`git submodule deinit -- data-engine` **before pulling this layout change**.
After updating SolarWM, run the initialization command above.
