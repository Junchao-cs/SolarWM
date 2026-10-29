# SolarRenderer on Reactor

Render a recorded structure video with [SolarRenderer](https://github.com/Enderfga/SolarRenderer),
served through [Reactor Runtime](https://github.com/reactor-team/reactor-runtime).
Select a structure video, optionally supply an aligned appearance image, describe the scene,
and receive the rendered video in a small browser demo.

![SolarRenderer rendering the official synthwave example through Reactor](assets/demo.png)

**Status:** experimental; this package's GPU inference and Docker image have
not been validated. The upstream four-B200 browser demo reports 1.8–2.1 seconds
per 17-frame block, with more than a minute possible for initial preparation.
These upstream measurements are not performance guarantees for this package.
FA4 and static-prefix caching change floating-point results and can change local
appearance relative to native inference.

## Features

- A Reactor service with explicit input selection, Start, Pause and Resume.
- Streaming output as completed video segments become available.
- A browser client using the official Reactor JavaScript SDK.
- Four-GPU sequence parallelism, FA4, modulation/text/static-prefix caching and parallel VAE decoding.
- Clickable example inputs and a side-by-side structure/render view.

The service uses the released causal four-step model. Each video contains 158 frames
and ends after ten generation blocks. Its native timeline is 24 FPS (about 6.58 seconds).
Runtime adapts playback to generation time. The upstream measured generation rate
is substantially below the native 24 FPS timeline.

## What you need

- A license and deployment location permitted by the model's weight terms.
- A Linux GPU machine with Docker and NVIDIA Container Toolkit. The upstream recipe was
  tested on four B200 GPUs on one host, with shared weights/scratch storage.
- The prepared MiniMax-H3 base and the two **causal** SolarRenderer LoRA files.
- Node.js 22.12+ and npm for the demo.

## 1. Install the Reactor CLI

Install the CLI using the [Reactor documentation](https://docs.reactor.inc/deploy).
Use a current CLI that supports `reactor run --shm-size`; older releases forward
unknown flags to Runtime. Check `reactor version` and `reactor run --help`.
From the repository root, enter the isolated adapter directory:

```sh
cd deploy/solar_renderer/reactor
```

## 2. Get the weights

Read the [weight license](https://huggingface.co/Enderfga/SolarRenderer/blob/main/LICENSE)
before downloading. Follow the integrated [setup and causal inference instructions](../../../docs/solar_renderer/inference.md).
Use `python -m solar.renderer.prepare_h3` from an installed checkout to prepare
the base and localize its component index. General framework documentation lives
under [docs/solar_wm](../../../docs/solar_wm/README.md).
Arrange the files under one directory:

```text
weights/
  MiniMax-H3/
    modular_model_index.json
    transformer_ref/
    text_encoder/
    processor/
    tokenizer/
    vae/
    ...
  solarrenderer_causal_control_lora.safetensors
  solarrenderer_causal_4step_lora.safetensors
```

Keep the prepared base at the same absolute path when mounting it into the container:
the localized component index refers to that path. The CLI's filesystem `--weights`
mount does this. The service does not automatically download or substitute checkpoints.

## 3. Start the model

```sh
python stage_source.py
reactor build
reactor run --gpus device=0,1,2,3 --shm-size 8g --port 8080 --weights ./weights
```

Select four available GPUs on a shared machine. In another terminal, check
`curl http://localhost:8080/health`. The service validates asset paths at startup;
the first **Start new video** performs encoding and loads the renderer. The original
text encoder and renderer use the GPU in sequence, so starting a new video has a
substantial preparation phase. Do not interpret HTTP readiness as first-frame readiness.

Run `stage_source.py` again and rebuild after changing the adapter or the
shared `solar` package. Keep its generated `.solarwm-source/` directory in the
Docker build context. `Ctrl+C` stops the service started by this command.

## 4. Open the client

```sh
cd demo
npm ci
npm run dev -- --host 127.0.0.1
```

Vite reads the example inputs directly from this repository's
`examples/solar_renderer/sem-highway_synthwave/` directory and includes them in production builds.
No sample path configuration is needed.

Open http://localhost:5173. The default Runtime address is http://localhost:8080;
set `VITE_RUNTIME_URL` before starting the demo to use another local endpoint.
Click **Connect**, choose **Synthwave Highway** to fill the official video, image
and prompt, then click **Start new video**. **Structure only** uses the same scene
without an appearance image and adjusts the prompt accordingly. You can also upload
your own inputs. File or example selection alone
does not run the model. No example or reference image is selected automatically.

This checkout includes the authors' matched
[structure video, image and prompt](../../../examples/solar_renderer/sem-highway_synthwave).
Use that complete set for the first comparison. Structure videos must cover at least
158 frames after native 24 FPS resampling; longer videos use the opening segment.
Landscape input produces 1344×768 output and portrait input 768×1344. The original
preprocessing center-fits the optional image to the selected canvas.

### Remote machines and SSH

Forward the HTTP and demo ports if running both on a remote machine:

```sh
ssh -L 8080:127.0.0.1:8080 -L 5173:127.0.0.1:5173 user@gpu-host
```

HTTP forwarding alone does not carry WebRTC media. Configure a supported TURN-over-TCP
relay and forward its TCP port as well, or use a network route that carries WebRTC.
Docker host networking alone does not make SSH forward UDP. Keep relay credentials
in deployment configuration; never put them in this repository. Check the
SSH tunnel and relay configuration before sharing the service.

## Controls

| Control | Effect |
| --- | --- |
| Structure video | Select the recorded motion and layout for the next video. |
| Appearance reference | Optionally constrain the opening appearance. |
| Rendering prompt / Seed | Captured when Start is pressed. |
| Start new video | Start a fresh finite video, replacing any current render. |
| Pause / Resume | Stop or continue at a generation boundary; buffered video may still play. |
| Disconnect | Close the demo's connection through the SDK. |

Commands take effect after the current generation block. The demo allows up to
180 seconds for their replies; a Pause during generation can take about 35 seconds,
and initial preparation can take longer. This changes the client wait limit, not
the model's generation speed.

Prompt edits apply to the **next video**; text conditioning stays fixed
throughout each render.
There are no driving keys or game-engine controls: motion is supplied by the uploaded video.
The decoder waits for the necessary temporal overlap before showing its first frames.

## Credits and licence

Thanks to [Rising0321](https://github.com/Rising0321) for the Reactor integration.
SolarRenderer and SolarWM remain the work of their original authors. The sampling and
decoder helpers derive from SolarRenderer's Apache-2.0 code; retain its attribution and
[NOTICE](https://github.com/Enderfga/SolarRenderer/blob/main/NOTICE).
Weights remain subject to the MiniMax H3 Community License, including territorial
restrictions. Please contact [ruixing@reactor.inc](mailto:ruixing@reactor.inc) with integration
feedback or collaboration ideas.
