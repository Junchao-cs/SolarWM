<div align="center">

# SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models

<p>
Junchao Huang<sup>1,2</sup> &nbsp; Guian Fang<sup>3</sup> &nbsp;
Shengju Qian<sup>4</sup> &nbsp; Xianghao Kong<sup>5</sup> &nbsp;
Zhuoran Zhao<sup>5,6</sup><br>
Wei Huang<sup>7</sup> &nbsp; Yihua Du<sup>6</sup> &nbsp;
Zixin Zhang<sup>6</sup> &nbsp; Justin Cui<sup>8</sup> &nbsp;
Yuchao Gu<sup>7</sup> &nbsp; Yukang Chen<sup>7</sup> &nbsp; Xinting Hu<br>
Tianyu He<sup>9</sup> &nbsp; Shaoshuai Shi &nbsp;
Zhuotao Tian<sup>2</sup> &nbsp; Xin Wang &nbsp;
Mike Zheng Shou<sup>3</sup> &nbsp; Li Jiang<sup>1,2</sup>
<br>
<sub>
<sup>1</sup>CUHK-SZ &nbsp; <sup>2</sup>SLAI &nbsp; <sup>3</sup>NUS &nbsp;
<sup>4</sup>CUHK &nbsp; <sup>5</sup>HKUST &nbsp; <sup>6</sup>HKUST-GZ &nbsp;
<sup>7</sup>NVIDIA &nbsp; <sup>8</sup>UCLA &nbsp; <sup>9</sup>MSRA
</sub>
</p>

<p>
  <a href="https://junchao-cs.github.io/SolarWM-Web/"><img alt="Project page" src="https://img.shields.io/badge/-Project%20Page-0A66C2?logo=googlechrome&amp;logoColor=white&amp;labelColor=555"></a>
  <a href="https://arxiv.org/pdf/2609.02886"><img alt="Paper" src="https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&amp;logoColor=white&amp;labelColor=555"></a>
  <a href="https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data"><img alt="Hugging Face dataset" src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow"></a>
  <a href="https://modelscope.cn/datasets/junchao2003/SolarWM-Data"><img alt="ModelScope dataset" src="https://img.shields.io/badge/-Dataset-624AFF?logo=modelscope&amp;logoColor=white&amp;labelColor=555"></a>
  <a href="https://docs.google.com/forms/d/e/1FAIpQLSfS-SLOiSRVDWwZ2kPl9ywN27aB6QplN0jpdKaBu-gG8aNsvQ/viewform"><img alt="Dataset Access Form" src="https://img.shields.io/badge/-Dataset%20Access%20Form-7248B9?logo=googleforms&amp;logoColor=white&amp;labelColor=555"></a>
  <a href="https://huggingface.co/collections/junchaoh-cs/solarwm"><img alt="Model weights" src="https://img.shields.io/badge/%F0%9F%A4%97-Model%20Weights-yellow"></a>
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache%202.0-green"></a>
</p>

<img src="assets/solar_teaser.png" width="100%" alt="SolarWM teaser: one framework for diverse interactive worlds">

</div>

We present **SolarWM**, a fully open, end-to-end framework for building
interactive video world models, from data preparation to scalable training
and long-horizon inference.

- **Four models. One codebase.** A unified framework supports **four
  5B–33B models** across Wan2.2, LTX-2.5, and MiniMax-H3, with shared
  data, training, and inference interfaces while preserving each backbone's
  native representation and objective.
- **A simple three-stage training recipe.** Bidirectional adaptation,
  teacher-forced autoregressive initialization, and distribution matching
  distillation form a shared route across heterogeneous video backbones,
  without specialized ODE or consistency-distillation initialization.
  The resulting causal models support *few-step interaction over
  arbitrarily long horizons** through sliding-window inference.
- **1.43M clips · 25 TB. Fully open.** We release the **complete dataset
  and data-processing pipeline**, unifying 14 datasets into a frame-aligned
  format with metric camera geometry, captions, quality metadata, and
  provenance. Source processing is decoupled from training-mixture design,
  allowing flexible data selection and mixing without reprocessing the
  source datasets.

## News

- **September 13, 2026:** We release **SolarWM-H3**, the first MiniMax-H3-based world model for streaming, few-step interaction. Training and inference code, [weights](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B), and [pre-encoded training data](docs/latent-wds.md) are now available for all three stages: Stage0.5, Stage1, and Stage2. Get started with the [H3 guide](docs/backends/minimax-h3.md).
- **September 3, 2026** — We open-source the training and inference code, the
  complete dataset, the data pipeline, model weights for all SolarWM-5B training
  stages, and bidirectional weights for SolarWM-14B, SolarWM-LTX, and SolarWM-H3.

## Training progression

The staged workflow is **Stage0.5 FM → Stage1 TF-AnyFlow → Stage2 DMD via
SGF**, turning a bidirectional video model into a camera-controlled few-step
autoregressive model.

- **Stage0.5** learns full-clip bidirectional flow matching and establishes the
  base video, text, and camera-conditioned representation.
- **Stage1** combines teacher forcing with the AnyFlow loss in one training
  stage. Clean history conditions noisy target chunks while the model learns
  both denoising and finite-step flow maps. This removes the need for a separate
  ODE or consistency-distillation initialization before Stage2/DMD.
- **Stage2** performs DMD via self-gradient forcing (SGF), training the causal
  student on its own autoregressive rollout with a frozen teacher and a
  trainable critic.

| Backend | Stage0.5 (Bid-Cam) | Stage1 (TF-AnyFlow) | Stage2 (SGF) | Runtime interfaces |
|---|:---:|:---:|:---:|---|
| **Wan2.2-5B** | ✓ | ✓ | ✓ | train, infer, preencode |
| **Wan2.2-14B** | ✓ | Coming soon | Coming soon | train, infer, preencode |
| **LTX-2.5** | ✓ | Coming soon | Coming soon | train, infer, preencode |
| **MiniMax-H3** | ✓ | ✓ | ✓ | train, infer, preencode |

## Install

Wan, LTX, and MiniMax-H3 require
[separate runtime environments](environments/README.md). Activate the
environment for the selected backbone, then install the shared SolarWM source:

```bash
python -m pip install -e .
solarwm environment probe
```

Model weights are available from the
[`SolarWM model collection`](https://huggingface.co/collections/junchaoh-cs/solarwm). See the
[Wan2.2 TI2V-5B guide](docs/backends/wan22-ti2v-5b.md) for Stage0.5, Stage1,
and Stage2 commands. The [MiniMax-H3 guide](docs/backends/minimax-h3.md) provides
launch commands for Stage0.5, Stage1, and Stage2.

## Data

The public
[SolarWM-Data release](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data)
contains release controls, licenses, recipe and test indexes, small format
examples, and the `SolarWM-Data-Annotation/` package. It does **not** include
the full `releases-v1/raw-wds/` or `releases-v1/latent-wds/` payloads.

For a released recipe that uses preencoded data, download its matching latent
generation. That is sufficient for training and does not require raw-WDS.
Raw-WDS is needed only when you want the full processed video corpus, online
encoding, your own latent generation, or another workflow whose index points
to raw data.

1. **Use preencoded latents.** Each latent generation is published in a
   separate repository. See the [latent-WDS release list](docs/latent-wds.md)
   for available downloads and generations that are still being uploaded.
2. **Rebuild raw-WDS from annotations.** `SolarWM-Data-Annotation/` is an
   annotation-only release with no videos. It contains the released camera
   trajectories, captions, metadata, source identities, and reconstruction
   tools. Follow its README to download the original videos from their source
   publishers and process them into the expected `raw-wds/` layout.
3. **Request prepared raw-WDS.** Submit the
   [Dataset Access Form](https://docs.google.com/forms/d/e/1FAIpQLSfS-SLOiSRVDWwZ2kPl9ywN27aB6QplN0jpdKaBu-gG8aNsvQ/viewform).
   Approved applicants receive download instructions by email.
4. **Download the standalone test set.** The
   [SolarWM standalone test set v1](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data_test-set-v1)
   contains the current 1,300-example, 13-source evaluation set in 60
   WebDataset tar files (approximately 77.4 GB), without the training payload.

See the [dataset access guide](docs/data-access.md) for download commands and
the payload required by each training, validation, and inference example.

## Unified commands

```bash
# Validate and render the exact resolved configuration.
solarwm config resolve \
  --config configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml \
  --set distributed.world_size=8 \
  --set train.global_batch_size=4 \
  --set model.checkpoint_path=/path/to/SolarWM-models/SolarWM-h3-33B-base \
  --set data.index_root=/path/to/SolarWM-Data/releases-v1 \
  --set data.transport.root=/path/to/SolarWM-Data/releases-v1 \
  --set data.silence_latents_path=/path/to/SolarWM-Data/releases-v1/latent-wds/minimax-h3-158f-768p-nomind-v1/support/h3_silence_153_158_170.safetensors \
  --set data.encoder_contract_path=/path/to/SolarWM-Data/releases-v1/latent-wds/minimax-h3-158f-768p-nomind-v1/support/encoder_contract.json \
  --set runtime.output_dir=/path/to/output/h3-stage0p5-158f

# Train H3 Stage0.5 with preencoded latents.
torchrun --standalone --nproc-per-node=8 \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml \
  --set distributed.world_size=8 \
  --set train.global_batch_size=4 \
  --set model.checkpoint_path=/path/to/SolarWM-models/SolarWM-h3-33B-base \
  --set data.index_root=/path/to/SolarWM-Data/releases-v1 \
  --set data.transport.root=/path/to/SolarWM-Data/releases-v1 \
  --set data.silence_latents_path=/path/to/SolarWM-Data/releases-v1/latent-wds/minimax-h3-158f-768p-nomind-v1/support/h3_silence_153_158_170.safetensors \
  --set data.encoder_contract_path=/path/to/SolarWM-Data/releases-v1/latent-wds/minimax-h3-158f-768p-nomind-v1/support/encoder_contract.json \
  --set runtime.output_dir=/path/to/output/h3-stage0p5-158f

# Run H3 Stage2 inference for 158 frames.
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-stage2-158f-sp4.yaml \
  --set model.checkpoint_path=/path/to/SolarWM-models/SolarWM-h3-33B-base \
  --set checkpoint.resume_from=/path/to/SolarWM-models/SolarWM-h3-33B-sgf-stage2-158f \
  --set checkpoint.weight_source=ema \
  --set data.index_root=/path/to/SolarWM-Data/releases-v1 \
  --set data.transport.root=/path/to/SolarWM-Data/releases-v1 \
  --set data.silence_latents_path=/path/to/SolarWM-Data/releases-v1/latent-wds/minimax-h3-158f-768p-nomind-v1/support/h3_silence_153_158_170.safetensors \
  --set data.encoder_contract_path=/path/to/SolarWM-Data/releases-v1/latent-wds/minimax-h3-158f-768p-nomind-v1/support/encoder_contract.json \
  --set runtime.output_dir=/path/to/output/h3-stage2-infer

# Preencode raw-WDS for the H3 158f recipe.
torchrun --standalone --nproc-per-node=8 \
  -m solarwm preencode \
  --config configs/examples/minimax_h3/preencode-158f.yaml \
  --set model.checkpoint_path=/path/to/SolarWM-models/SolarWM-h3-33B-base \
  --set data.index_root=/path/to/SolarWM-Data/releases-v1 \
  --set data.transport.root=/path/to/SolarWM-Data/releases-v1 \
  --set preencode.output_root=/path/to/output/preencoded/minimax-h3-158f-768p-nomind-v1 \
  --set runtime.output_dir=/path/to/output/h3-preencode
```

The [quickstart](docs/quickstart.md) covers H3 setup, training, and inference.
For Stage1/Stage2 training and [full-length inference](docs/backends/minimax-h3.md#full-length-stage2-inference),
see the [H3 guide](docs/backends/minimax-h3.md). Other models have their own
backend guides.

Every launch writes `resolved-config.json` and `launch-manifest.json` before
model allocation. Config overrides are explicit and included in the resolved
configuration identity.

## Local and bucket data

Index rows always contain POSIX shard keys relative to the release directory:

```json
{"sample_id":"...","shard":"raw-wds/abot/shards/kept-high-000001.tar"}
```

Only the runtime root changes:

```yaml
# Locally mounted storage
data:
  index_root: /path/to/SolarWM-Data/releases-v1
  transport:
    kind: local
    root: /path/to/SolarWM-Data/releases-v1

# Object-store streaming uses the same release-relative rows. Supply the
# release root from the distribution channel or deployment environment.
data:
  index_root: /path/to/SolarWM-Data/releases-v1
  transport:
    kind: gcs
    root: ${SOLAR_RELEASE_ROOT}
    cache_dir: /path/to/solar-cache
    cache_max_gib: 256
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Runtime environments](environments/README.md)
- [Architecture](docs/architecture.md)
- [Data contract](docs/data-contract.md)
- [Dataset overview and statistics](docs/datasets.md)
- [Download and access](docs/data-access.md)
- [Wan 2.2 TI2V-5B backend](docs/backends/wan22-ti2v-5b.md)
- [Wan 2.2 I2V-A14B backend](docs/backends/wan22-i2v-a14b.md)
- [LTX-2.5 backend](docs/backends/ltx25.md)
- [MiniMax-H3 backend](docs/backends/minimax-h3.md)

## Acknowledgements

We gratefully thank the teams behind
[Wan2.2](https://github.com/Wan-Video/Wan2.2),
[LTX-2.5](https://github.com/Lightricks/LTX-2), and
[MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) for releasing the code
and pretrained models that make the SolarWM backbone family possible. We also
thank the authors of all datasets and open-source projects listed in
[NOTICE](NOTICE).

## Citation

**If you use SolarWM-Data, the data engine, or the released models in your research,
please cite our paper.**

Paper: https://arxiv.org/abs/2609.02886

```bibtex
@misc{huang2026solarwmopendatascalable,
      title={SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models}, 
      author={Junchao Huang and Guian Fang and Shengju Qian and Xianghao Kong and Zhuoran Zhao and Wei Huang and Yihua Du and Zixin Zhang and Justin Cui and Yuchao Gu and Yukang Chen and Xinting Hu and Tianyu He and Shaoshuai Shi and Zhuotao Tian and Xin Wang and Mike Zheng Shou and Li Jiang},
      year={2026},
      eprint={2609.02886},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.02886}, 
}
```

## License and attribution

SolarWM is licensed under Apache-2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). Model weights and backbone packages may carry their own
licenses; review the license and model card in the corresponding release
package before use or redistribution. In particular, LTX-2.5 derivatives are
subject to the LTX-2.x Community License, and the MiniMax H3 Community License
contains territory restrictions. Those packages are not relicensed under the
SolarWM code license.
