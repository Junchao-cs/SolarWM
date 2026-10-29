# SolarWM-H3 Realtime Interaction

Interactive video generation with the original VAE or **StreamRender (Faster)**.

## 1. What you need before starting

Run this guide **on your GPU server**, in Bash, from the root of your cloned
SolarWM repository (the directory containing `pyproject.toml`).

- **Server:** Linux with eight NVIDIA Blackwell GPUs. The current fused kernels
  use Blackwell-specific features; this guide is not a verified setup for other
  GPU architectures.
- **System software:** Python 3.12 with `venv`, Git, a C/C++ compiler, CUDA Toolkit
  13.0 (`nvcc`), a compatible NVIDIA driver, and FFmpeg with `libx264`.
  Install these first or ask your server administrator to provide them. The Python
  installation below does not install the driver, CUDA Toolkit or FFmpeg.
- **Model access:** a Hugging Face account with access to the
  [SolarWM-H3 weights](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B).
  Open the link and accept the access terms before downloading.
- **Storage:** a writable directory on a local disk, outside this source checkout,
  for the Python environment, downloaded weights, generated scene features and caches.

**You do not need training data, pre-extracted scene features or an existing
compiler cache.** The repository includes two example images and their prompts
in [`examples/solar_wm`](../../examples/solar_wm/README.md): `snow` and
`fantasy_world`.
The commands below download the models, create the configuration and start the
webpage. You do not need to extract features or edit Python or JSON files.

## 2. Choose where to store files

Change **only the first two paths** below to match your server. `/data/solarwm-demo`
is an example directory you own; `CUDA_HOME` must point to an **already installed**
CUDA Toolkit 13.0, containing `bin/nvcc`. Setting it does not install CUDA.

```bash
export SOLARWM_WORKDIR="/data/solarwm-demo"
export CUDA_HOME="/usr/local/cuda-13.0"

export SOLARWM_ENV="$SOLARWM_WORKDIR/venv"
export SOLARWM_WEIGHTS="$SOLARWM_WORKDIR/models"
export SOLARWM_DATA="$SOLARWM_WORKDIR/web-data"
export SOLARWM_PYTHON="$SOLARWM_ENV/bin/python"
export HF_HOME="$SOLARWM_WORKDIR/huggingface"
export PATH="$CUDA_HOME/bin:$PATH"
umask 077
mkdir -p "$SOLARWM_WORKDIR"
```

| Directory | What goes here | Who creates it? |
|---|---|---|
| `venv/` | Isolated Python environment | Step 3 |
| `models/` | Model weights listed in step 4 | The download commands in step 4 |
| `web-data/` | Generated configuration, scene features, outputs and compiler caches | `configure` and the runtime in step 5 |
| `huggingface/` | Your Hugging Face login and download cache | The Hugging Face commands |

Leave `web-data/` **nonexistent for the first setup**: `configure` creates it and
will not overwrite an existing directory. Keep the same directory for later starts.
Keep `huggingface/` private; do not commit or share its credentials.

## 3. Install the Python environment (once)

The tested runtime is **Python 3.12, PyTorch 2.10.0 / CUDA 13.0 and Triton 3.6.0**.
Use the separate environment below, not an existing training or serving environment.

```bash
python3.12 -m venv "$SOLARWM_ENV"
"$SOLARWM_PYTHON" -m pip install --upgrade pip setuptools wheel packaging ninja psutil
"$SOLARWM_PYTHON" -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130
"$SOLARWM_PYTHON" -m pip install --no-build-isolation \
  -r "deploy/solar_wm/requirements.txt"
"$SOLARWM_PYTHON" -m pip install --no-deps -e .
"$SOLARWM_PYTHON" -m pip check
```

Keep this installation order and the pinned versions. Dependencies for both VAEs
are included.

## 4. Download the required model files

| File or directory | Required? | Purpose / source |
|---|---|---|
| `SolarWM-h3-33B-base/` | Yes | Base model, image/text encoders and original VAE, from [SolarWM-H3](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B). Download the whole folder. |
| `SolarWM-h3-33B-sgf-stage2-158f-fix/` | Yes | Camera-controlled inference checkpoint from the same repository. Use this `-fix` folder, not the older stage-2 folder. |
| `h3_silence_153_158_170.safetensors` | Yes | Fixed silence conditioning used by H3, from that repository's `support/` folder. Still required for this video-only demo. |
| `causal_vit24_step6000.safetensors` | Optional | Additional decoder weights from [StreamRender-H3](https://huggingface.co/Efficient-Large-Model/StreamRender-H3). Needed only for the **StreamRender (Faster)** option. |

Log in using the account that accepted the model access terms. Enter your token
at the terminal prompt, not in this README or a configuration file.

```bash
"$SOLARWM_PYTHON" -m huggingface_hub.cli.hf auth login --no-add-to-git-credential
```

Download the **three required items**:

```bash
"$SOLARWM_PYTHON" -m huggingface_hub.cli.hf download junchaoh-cs/SolarWM-H3-33B \
  --local-dir "$SOLARWM_WEIGHTS" --include 'SolarWM-h3-33B-base/**'
"$SOLARWM_PYTHON" -m huggingface_hub.cli.hf download junchaoh-cs/SolarWM-H3-33B \
  --local-dir "$SOLARWM_WEIGHTS" --include 'SolarWM-h3-33B-sgf-stage2-158f-fix/**'
"$SOLARWM_PYTHON" -m huggingface_hub.cli.hf download junchaoh-cs/SolarWM-H3-33B \
  support/h3_silence_153_158_170.safetensors --local-dir "$SOLARWM_WEIGHTS"
ln -s support/h3_silence_153_158_170.safetensors \
  "$SOLARWM_WEIGHTS/h3_silence_153_158_170.safetensors"
```

To enable **both VAEs in the webpage**, also download this single file **before
running `configure`**. Accept any access terms on its model page first.

```bash
"$SOLARWM_PYTHON" -m huggingface_hub.cli.hf download Efficient-Large-Model/StreamRender-H3 \
  causal_decoder/causal_vit24_step6000.safetensors --local-dir "$SOLARWM_WEIGHTS"
ln -s causal_decoder/causal_vit24_step6000.safetensors \
  "$SOLARWM_WEIGHTS/causal_vit24_step6000.safetensors"
```

The `ln -s` commands create shortcuts at the locations expected by the launcher;
they do not copy the weights. Run them once. If you already have these files, skip
their downloads and link or arrange them into the following layout instead:

```text
models/
├── SolarWM-h3-33B-base/
├── SolarWM-h3-33B-sgf-stage2-158f-fix/
├── h3_silence_153_158_170.safetensors
└── causal_vit24_step6000.safetensors       # optional: enables StreamRender
```

The `support/` and optional `causal_decoder/` download folders remain alongside
these entries. Keep the base-model directory intact, including its `text_encoder`,
`tokenizer`, `processor` and `vae` subfolders. No Stage0.5/Stage1 checkpoints or
training dataset are needed.

With the optional decoder file present, `configure` enables both VAEs and selects
StreamRender by default; without it, the original VAE is used.

## 5. Create the configuration and start

Run these commands in order. `configure` reads the image/prompt pairs from
`examples/solar_wm`; `serve` starts the eight-GPU web service. The webpage extracts
and caches each scene's features automatically on first use.

```bash
# Once: create web-data/ and its configuration files.
bash "deploy/solar_wm/launch.sh" configure \
  --weights "$SOLARWM_WEIGHTS" --output "$SOLARWM_DATA"

# Start the web service; leave this terminal running.
bash "deploy/solar_wm/launch.sh" serve \
  --config "$SOLARWM_DATA/deployment.json" --scene snow
```

`snow` is the default. Both
[included examples](../../examples/solar_wm/README.md) appear immediately.
Click either scene, or open **Your image + prompt** to upload your own.
**Start** is enabled when the scene is ready. You can cancel preparation.
Features and uploaded images stay in `web-data/runtime/` and
are reused on later selections and restarts. Do not run a separate encoder on the
same GPUs while the service is running.

**Later starts:** from the repository root, set the paths from step 2 again and
run only the `serve` command above. Do not rerun `configure` or reinstall/download
everything. Reuse the same data directory to retain prepared scenes and caches.
The first startup or a new scene may take
time to prepare; wait until the page finishes loading and enables **Start**.
Disk caches are retained, but a restarted process still loads the models and
prepares its in-memory CUDA Graphs.

The built-in gallery contains `snow` and `fantasy_world`. Reuse your data
directory to keep custom uploads, generated files and scene caches.

## 6. Open and use

On the server, open **http://127.0.0.1:8900/** in a current Chrome or Edge browser.
From your computer, run the following, replacing **`user@server`** with your SSH
login, then open the same address locally:

```bash
ssh -N -L 8900:127.0.0.1:8900 user@server
```

Select a scene and VAE, wait for preparation, then click **Start**. Use the keyboard or on-screen
buttons to move the camera; use **Pause / Resume** or **Stop** as needed. Stop takes
effect after the current chunk finishes. Full-resolution 1344×768 preview and
video reconnection are enabled by default.

For your own scene, choose a PNG, JPEG or WebP image (up to 8 MiB / 16 megapixels)
and enter a prompt, then click **Prepare scene**. **Use selected image** lets you
try another prompt with an existing image. Inputs are converted to the model's
1344×768 resolution; prompts are not padded. The current accelerated runtime
accepts 1000–1300 joint image/text tokens, not characters. An unsupported input
shows a scene error; edit the prompt and try again without restarting the model.
