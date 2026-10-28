# SolarWM guides

SolarWM provides shared data, training, and inference interfaces for
camera-controlled video world models. Run the commands in these guides from
the repository root.

- [Quickstart](quickstart.md): set up MiniMax-H3 and run training or inference.
- [Runtime environments](environments.md): install the environment for each backbone.
- [Architecture](architecture.md): framework structure and shared interfaces.
- [Data contract](data-contract.md): sample, camera, and storage conventions.
- [Dataset overview and statistics](datasets.md).
- [Download and access](data-access.md): choose raw data, latent data, or the test set.
- [Latent-WDS releases](latent-wds.md): available preencoded generations.

## Models and training

SolarWM supports four 5B–33B models. Each guide covers the available training
stages, model weights, and inference workflows.

| Model | Stage0.5 · Bid-Cam | Stage1 · TF-AnyFlow | Stage2 · SGF | Guide |
| --- | :---: | :---: | :---: | --- |
| **Wan2.2-5B** | ✓ | ✓ | ✓ | [Guide](backends/wan22-ti2v-5b.md) |
| **Wan2.2-14B** | ✓ | Coming soon | Coming soon | [Guide](backends/wan22-i2v-a14b.md) |
| **LTX-2.5** | ✓ | Coming soon | Coming soon | [Guide](backends/ltx25.md) |
| **MiniMax-H3** | ✓ | ✓ | ✓ | [Guide](backends/minimax-h3.md) |

## Examples and development

- [Training and inference configurations](../../configs/solar_wm/) · [Image and prompt examples](../../examples/solar_wm/).
- [H3 offline inference](backends/minimax-h3.md#full-length-stage2-inference) · [Realtime deployment](../../deploy/solar_wm/README.md).
- [Data pipeline](../../data-engine/solarwm-data) · [Pipeline setup](../../data-engine/README.md).

For structure-conditioned rendering, use the separate
[SolarRenderer guide](../solar_renderer/README.md). Return to the
[project overview](../../README.md) for release information and citations.
