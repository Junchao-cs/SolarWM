# Scoring the SolarRenderer benchmark

The benchmark has two splits, scored with the same metrics and the same judge. Every case gives a structure video, a
first frame and a prompt:

| Split | Cases | First frame | Ground truth |
|---|---|---|---|
| `hard` | 208 cases over 16 structure types, 11 never seen in training (74 cases, tag `unseen condition`), tagged for large motion, fast camera, stylized looks, night, multi-person and unseen condition | made by an image model from the structure's frame 0 | none |
| `heldout` | 53 held-out clips from CARLA, engine recordings with near-range depth, Magpie and UnrealZoo | the real one | the real clip |

Scoring has three parts:

1. **Automatic metrics** (`metrics.py`), in four groups:
   - quality: DOVER, LAION aesthetic and VideoScore2;
   - stability: low-pass flicker and DINOv2 adjacent-frame similarity;
   - structure: CoTracker3 motion error and edge F1 against the structure video;
   - first frame: DINOv2 similarity to the first frame over the last second.

   On the held-out split, `gt_metrics.py` adds DISTS, LPIPS, PSNR, SSIM and sharpness against the real clip.
   `summary.py` gives the mean of every reading and the **automatic score** (0–100, the four groups).
2. **Pairwise VLM judge** (`judge.py`): Gemini compares two methods on every case in both orders and answers four
   questions: visual quality, structure and motion, first frame, overall. The **net score** is wins minus losses over
   cases.
3. **Elo** (`elo.py`): a Bradley–Terry fit of every judged pair on a split gives one rating per method.

The released results (`results/` in the dataset) hold the SolarRenderer renders of both splits and every verdict
behind the paper's Elo table, so a new method only needs its own renders and its comparison with SolarRenderer.

## Setup

Run commands from the SolarWM repository root after installing the separate
[Renderer environment](../../docs/solar_renderer/README.md#install) and following
[model preparation](../../docs/solar_renderer/README.md#download-the-models).
The latter defines `H3_ROOT` and `RENDERER_WEIGHTS` for the render command below.
This directory
is adapted from [SolarRenderer revision e018b0f](https://github.com/Enderfga/SolarRenderer/tree/e018b0f673de865711fe99844d3976865722b8d0/eval).
It uses the integrated `solar.renderer` runtime. Model caches default to
`eval_models/solar_renderer/`; set `SOLAR_EVAL_MODELS` to override this location.
Install `ffmpeg` and `ffprobe` before scoring.

```bash
hf download Enderfga/SolarRenderer-Data --repo-type dataset --include "benchmark/*" --local-dir ./SolarRenderer-Data
bash eval/solar_renderer/setup.sh            # DOVER, the LAION aesthetic head, CoTracker3 (pinned versions, checksummed)
```

VideoScore2 needs its own environment, because it requires a different transformers version:

```bash
python -m venv vs2 && vs2/bin/pip install "torch==2.8.*" "torchvision==0.23.*" "transformers==4.53.2" "qwen-vl-utils==0.0.14" accelerate av decord
```

## 1. Render

Put a method's renders in one directory:

```
<renders>/hard/<case_id>.mp4
<renders>/heldout/<clip_id>.mp4      e.g. heldout/carla/cc000014066.mp4
```

Render each case from its structure video, first frame and prompt. Each clip should be 24 fps; the published renders are 1344×768 (768×1344 portrait),
158 frames. For SolarRenderer itself:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/solar_renderer/render_benchmark.py --bench ./SolarRenderer-Data --out ./renders/solarrenderer \
    --h3 "$H3_ROOT" --control "$RENDERER_WEIGHTS/solarrenderer_control_lora.safetensors" \
    --fewstep "$RENDERER_WEIGHTS/solarrenderer_4step_lora.safetensors" --shard 0 --shards 8
```

Run one process per GPU with `--shard k --shards n`.

## 2. Automatic metrics

```bash
CUDA_VISIBLE_DEVICES=0 python eval/solar_renderer/metrics.py --bench ./SolarRenderer-Data --renders ./renders/solarrenderer --method solarrenderer \
    --out ./scores --shard 0 --shards 8
CUDA_VISIBLE_DEVICES=0 vs2/bin/python eval/solar_renderer/metrics.py --bench ./SolarRenderer-Data --renders ./renders/solarrenderer \
    --method solarrenderer --out ./scores --metrics videoscore2 --shard 0 --shards 8
python eval/solar_renderer/gt_metrics.py --bench ./SolarRenderer-Data --renders ./renders/solarrenderer --method solarrenderer --out ./scores

python eval/solar_renderer/summary.py --scores ./scores --methods solarrenderer,other --split hard
python eval/solar_renderer/summary.py --scores ./scores --methods solarrenderer,other --split hard --tag "unseen condition"
python eval/solar_renderer/summary.py --scores ./scores --methods solarrenderer,other --split heldout
```

On the released renders (`results/solarrenderer`, and `results/solarrenderer_causal` for the causal variant) these
commands give the paper's means; DOVER and VideoScore2 depend
slightly on library versions (VideoScore2 generates its answer, so expect differences of about 0.02).

Results are one JSON per case and metric, under `<out>/<metric>/<method>/hard/<case_id>.json` and
`<out>/<metric>/<method>/heldout/<clip_id>.json`. A rerun only computes what is missing.

The automatic score maps each reading linearly from a fixed range, so it does not depend on the other methods:

| Reading | Range | Group |
|---|---|---|
| DOVER overall | [0, 1] | quality |
| LAION aesthetic | [1, 10] | quality |
| VideoScore2 visual quality | [1, 5] | quality |
| low-pass flicker | [0, 10], reversed, clipped at 0 | stability |
| DINOv2 adjacent-frame similarity | [0, 1] | stability |
| motion error | [0, 0.5], reversed, clipped at 0 | structure |
| edge F1 | [0, 1] | structure |
| DINOv2 similarity to the first frame | [0, 1] | first frame |

The readings of each group are averaged, and the score is the mean of the four groups, on the cases every listed
method has. Ground-truth readings are listed but not scored.

## 3. Pairwise judge

```bash
export GEMINI_API_KEY=...
python eval/solar_renderer/judge.py run --bench ./SolarRenderer-Data --a ./renders/solarrenderer --b ./renders/other \
    --name-a solarrenderer --name-b other --out ./judge/solarrenderer__vs__other.jsonl
python eval/solar_renderer/judge.py report ./judge/solarrenderer__vs__other.jsonl
```

How it judges:

- **Model**: `gemini-3.8-flash` at temperature 0. The judge sees the structure video, the first frame, video A and
  video B, compressed as described at the top of `judge.py`.
- **Two orders**: every case is judged twice, with A and B swapped. A case is a win for X when both orders pick X, or
  one picks X and the other is a tie. When each order picks its own position, the case counts as a tie.
- **Net score**: wins minus losses, divided by the number of cases (from −1 to 1). `*` marks a 95% bootstrap interval
  that excludes zero. On the hard split the interval resamples whole structure videos (2,000 draws), because each
  structure video is rendered in several looks; held-out clips are resampled one by one (10,000 draws).
- **Counts**: the report says how many calls were answered and failed, and how many cases have only one order and
  are left out.
- **Report**: given for each split and for each hard-split tag.

A rerun retries calls that were rate-limited. Even at temperature 0, a few answers can change from run to run, so
compare methods within one run.

## 4. Elo

```bash
hf download Enderfga/SolarRenderer-Data --repo-type dataset --include "results/*" --local-dir ./SolarRenderer-Data
python eval/solar_renderer/judge.py run --bench ./SolarRenderer-Data --a ./SolarRenderer-Data/results/solarrenderer --b ./renders/mine \
    --name-a solarrenderer --name-b mine --out ./judge/solarrenderer__vs__mine.jsonl
python eval/solar_renderer/elo.py ./SolarRenderer-Data/results/judge/hard/*.jsonl ./judge/solarrenderer__vs__mine.jsonl --split hard
python eval/solar_renderer/elo.py ./SolarRenderer-Data/results/judge/hard/*.jsonl ./judge/solarrenderer__vs__mine.jsonl --split hard \
    --tag "unseen condition"
python eval/solar_renderer/elo.py ./SolarRenderer-Data/results/judge/heldout/*.jsonl ./judge/solarrenderer__vs__mine.jsonl --split heldout
```

`elo.py` fits a Bradley–Terry model to the overall verdicts (`--question` picks another) of every pair in the given
files, with a tie counting half a win to each side, and reports each method on the Elo scale: 400 points per factor of
ten in odds, mean of the rated methods 1000. Every method also plays one virtual tie against a fixed reference, so a
method that loses every comparison still gets a finite rating. The interval resamples the same units as the net
score, 400 draws. `--methods` rates only the listed methods, from the comparisons among them; on the released
verdicts, the nine methods of the paper's main table give its Elo columns:

```bash
python eval/solar_renderer/elo.py ./SolarRenderer-Data/results/judge/hard/*.jsonl --split hard --methods solarrenderer,lynnreal_omni,\
control_model_50step,official_backbone_50step,pai_controlnet_union,streaming_cwm,streamrender_h3,cosmos_transfer2_5,\
wan2_2_vace_fun
```

The causal variant (`results/solarrenderer_causal`) is judged only against SolarRenderer. Adding its verdicts and
`solarrenderer_causal` as a tenth method rates it and moves the other nine by a few points at most; shifting all ten
ratings so that the other nine average 1000 gives its row in the paper:

```bash
python eval/solar_renderer/elo.py ./SolarRenderer-Data/results/judge/hard/*.jsonl --split hard --methods solarrenderer,lynnreal_omni,\
control_model_50step,official_backbone_50step,pai_controlnet_union,streaming_cwm,streamrender_h3,cosmos_transfer2_5,\
wan2_2_vace_fun,solarrenderer_causal
```
