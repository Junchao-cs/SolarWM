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

## Backbone guides

- [Wan 2.2 TI2V-5B](backends/wan22-ti2v-5b.md).
- [Wan 2.2 I2V-A14B](backends/wan22-i2v-a14b.md).
- [LTX-2.5](backends/ltx25.md).
- [MiniMax-H3](backends/minimax-h3.md).

For structure-conditioned rendering, use the separate
[SolarRenderer guide](../solar_renderer/README.md). Return to the
[project overview](../../README.md) for release information and citations.
