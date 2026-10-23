# Runtime environments

Wan, LTX, and H3 require separate environments. The following matrix is the
release-tested baseline; nearby compatible versions may work but are not part
of the reproduced result.

| Backend | Python | PyTorch / CUDA | Required model stack |
|---|---|---|---|
| Wan 2.2 | 3.10 | 2.5.1 / 12.4 | FlashAttention 2.8.3, Diffusers 0.38.0, Transformers 5.12.1 |
| LTX-2.5 | 3.12 | 2.9.1 / 12.8 | `ltx-core`, `ltx-pipelines`, and `ltx-trainer` 1.2.0; Transformers 5.14.1 |
| MiniMax-H3 | 3.10 | 2.6.0 / 12.4 | FlashAttention 2.8.3, PEFT 0.20.0, Transformers 5.12.1, Diffusers 0.40.0 |
| H3 Realtime deployment | 3.12 | 2.10.0 / 13.0 | Triton 3.6.0; use the deployment guide's pinned requirements |
| SolarRenderer | 3.10 | 2.8.0 / 12.8 | FlashAttention 2.7.4.post1, PEFT 0.20.0, Transformers 5.14.1, Diffusers 0.40.0 |

H3 Realtime deployment uses a separate environment and currently requires eight
Blackwell GPUs. Follow the
[realtime deployment guide](../../src/solarwm/realtime_interaction/README.md)
for its installation order, model downloads, and web demo commands.

SolarRenderer uses its own environment. Follow the
[SolarRenderer tutorial](../solar_renderer/README.md#install) for installation,
structure-conditioned inference, and LoRA training.

Do not install these stacks into one environment. Start from the matching
CUDA image or build an equivalent environment, then install SolarWM without
letting its optional helpers replace the already selected CUDA packages:

```bash
cd /path/to/SolarWM
python -m pip install -e .
solarwm environment probe
```

If the environment is managed by `uv` and intentionally has no pip module:

```bash
uv pip install --python /path/to/environment/bin/python -e /path/to/SolarWM
/path/to/environment/bin/python -m solarwm environment probe
```

## Building without a prepared image

For Wan, create a Python 3.10 CUDA 12.4 environment and install PyTorch before
FlashAttention. The following order avoids asking FlashAttention's build step
to import a PyTorch installation that does not yet exist:

```bash
cd /path/to/SolarWM
python3.10 -m venv .venv-wan
source .venv-wan/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[train]" \
  diffusers==0.38.0 transformers==5.12.1 peft==0.20.0 \
  "ftfy>=6.2" "omegaconf>=2.3" "regex>=2024.0"
python -m pip install --no-build-isolation flash-attn==2.8.3
solarwm environment probe
```

For LTX, create a Python 3.12 environment, install the tested CUDA 12.8
PyTorch wheels first, then install the three official LTX packages from
tag `v1.2.0`. Do not use the upstream `uv sync` command for this environment:
that release has no lock file, and its default PyTorch index is not the CUDA
12.8 baseline used here.

```bash
cd /path/to/SolarWM
SOLAR_REPO="$PWD"
uv venv --python 3.12 .venv-ltx
uv pip install --python .venv-ltx/bin/python \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/Lightricks/LTX-2.git ../LTX-2
cd ../LTX-2
git checkout v1.2.0
uv pip install --python "$SOLAR_REPO/.venv-ltx/bin/python" \
  -e packages/ltx-core \
  -e packages/ltx-pipelines \
  -e packages/ltx-trainer \
  transformers==5.14.1 peft==0.20.0 safetensors==0.8.0
uv pip install --python "$SOLAR_REPO/.venv-ltx/bin/python" \
  -e "${SOLAR_REPO}[train,ltx]"
"$SOLAR_REPO/.venv-ltx/bin/python" -m solarwm environment probe
```

For H3, install PyTorch 2.6.0 and torchvision 0.21.0 first, then Diffusers
0.40.0, Transformers 5.12.1, PEFT 0.20.0, and FlashAttention 2.8.3.
FlashAttention must be installed with build isolation disabled after PyTorch:

```bash
cd /path/to/SolarWM
python3.10 -m venv .venv-h3
source .venv-h3/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  diffusers==0.40.0 \
  transformers==5.12.1 peft==0.20.0 \
  imageio==2.37.4 imageio-ffmpeg==0.6.0
python -m pip install -e ".[train]"
python -m pip install --no-build-isolation flash_attn==2.8.3
solarwm environment probe
```
