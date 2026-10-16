---
name: solarrenderer-caption
description: Write and validate SolarRenderer prompts, preserve structure and first-frame conditioning, and prepare matching first-frame guidance. Use for SolarRenderer rendering inputs, training captions, or prompt corrections.
---

# Writing SolarRenderer prompts

SolarRenderer uses a fixed six-part prompt format. Preserve the training-time wording for input definitions and
retention rules, then describe the target video concretely in the free fields.

All helper paths below are relative to the directory containing this `SKILL.md`. Keep the `scripts/` directory
beside it when copying the skill into an agent's skill directory. The prompt helpers need only Python; frame
extraction and fitting also require `ffmpeg` on the path. Use absolute input and output paths when your media
and prompt files are outside the skill directory.

## The six parts, in this order

```
subject_definitions:
<Video 1> is <what the structure clip is>.
<Video 2> is the first frame of [Shot 1], held still.

summary:
[keyframe completion + reference generation] The target video begins from <Video 2> and follows the frame-by-frame structure and motion of <Video 1>. <one short paragraph: who is where doing what, how the camera moves>

retention_analysis:
<Video 1> <what of it is followed, what it does not give, what of its look is not shown>.
<Video 2> ([Shot 1] first frame): fully_preserved - the composition, subjects, materials, colours and lighting of <Video 2> are the target video's first frame.

detailed_description:
<1-2 sentences on visual style and light> [Shot 1] The shot begins from <Video 2>. <first-frame state, then what happens, in playback order>

overall_soundscape:
<1-4 sentences of ambience and physical sounds>

non_diegetic_music:
N/A
```

Everything except the summary paragraph, the detailed description and the two sound parts is fixed text. Do not
paraphrase it: [`scripts/conditions.py`](scripts/conditions.py) holds the exact sentences and `assemble()` builds the prompt from the free parts.

## Which inputs, which labels

| Inputs given | Task type in the summary | Labels |
|---|---|---|
| structure + first frame | `[keyframe completion + reference generation]` | structure `<Video 1>`, first frame `<Video 2>` |
| structure only | `[reference generation]` | structure `<Video 1>` |
| first frame only | `[keyframe completion]` | first frame `<Video 1>` |
| text only | none: write `integrated_multimodal_description: [Shot 1] ...`, `overall_soundscape:`, `non_diegetic_music:` | none |

Define only the inputs actually given, and never use a label that is not defined.

The SolarWM rendering command requires a structure video: use `full` for structure plus a first frame, or
`structure` for structure alone. The `first` and `text` modes remain available for upstream training-caption
workflows; they are not input modes of `python -m solarwm.renderer`.

## The structure sentence

For the six trained condition types use the sentence in `conditions.DEFINE` word for word: `scene_graybox`
(untextured grey block-out, Magpie / engine white-box), `semantic` (CARLA-style class colours), `mask_skeleton`
(flat object masks with skeletons), `near_depth` (grey shaded engine depth, black beyond ~30 m), `smpl_mesh`
(grey body mesh on white), `scene_whitebox` (untextured grey rendering of the scene and the main character).

For any other structure, say plainly what the clip looks like and what of its look must not appear, with
`conditions.custom_condition(looks_like, not_shown)`:

```python
custom_condition(
    "a sparse 3D point cloud scan of the scene and its characters, white points on black",
    "the dotted point look",
)
custom_condition(
    "a black pencil line sketch of the scene on white paper, outlines only, like an animated storyboard",
    "the pencil lines and white paper",
)
```

## Writing the free parts

- **summary**: one short paragraph after the fixed lead: who is where doing what, and how the camera moves.
- **detailed_description**: 350-500 words (aim for about 420).
  - Open with one or two sentences on the visual style and light ("The target video is a realistic 3D CG game render
    in bright midday sun with crisp shadows."), then `[Shot 1]`.
  - With structure and a first frame, `[Shot 1]` opens with `The shot begins from <Video 2>.`; a first-frame-only
    caption uses `<Video 1>` instead. `assemble()` adds the matching sentence.
  - Describe the first frame's composition, subjects and scene, then what happens in playback order.
  - Spend the words on what the structure does not carry: materials and surfaces with colours and wear; light
    (direction, colour, hardness, time of day, shadows, reflections, emissive lights); weather and atmosphere;
    effects (sparks, dust, smoke, particles, glows); characters' full appearance (each outfit piece with colour and
    material, gear, hair, skin); small objects; events and state changes; camera motion (type, amplitude, speed).
  - Keep what the structure fixes consistent with it: same layout, same number of characters, the same facing
    (if we see someone's back, say so), the same camera path.
  - Time events in words ("midway", "near the end"). A timestamp appears only at a hard cut:
    `[Shot 2] At 00:03.500, the shot cuts to ...`. Most clips are one shot.
- **overall_soundscape**: ambience and physical sounds; **non_diegetic_music**: `N/A` unless it is a scored trailer.
- English only; on-screen text is quoted exactly in its original language.
- No labels such as `<Video 1>` inside the free parts you write (assemble() places them), no guesses about story or
  names, no mood words standing in for visible detail.

To restyle a clip (same structure, another world), write the description of the world you want, keeping what the
structure fixes. The first frame, if given, must already be in that world.

## The first frame

A first frame is optional, but it is the strongest control over the look: the render starts from it and keeps its
materials, colours and light. When a first frame is requested but not provided, create one with an image model
that accepts a reference image, after writing the prompt so that the two agree:

1. `python scripts/first_frame.py frame0 structure.mp4 frame0.png` - frame 0 of the structure clip, the reference image.
2. `python scripts/first_frame.py prompt prompt.txt` (add `--stylised` for pixel art, toon, clay, painterly and other
   non-photoreal looks) - the image prompt: keep the camera, layout, characters, poses and facing of the reference
   exactly; replace the grey or flat surfaces with the look of the detailed description's opening sentences.
3. Generate a 16:9 image from `frame0.png` with that prompt (9:16 for a portrait clip).
4. `python scripts/first_frame.py fit generated.png first.png` - crop and scale to 1344x768 (`768x1344` for portrait).
5. Check it against `frame0.png`: same number of characters, same facing, same horizon and layout. An image that
   moved things will fight the structure; regenerate it. If you have a frame of the real scene with the same layout
   (a photo, a game capture), use it as the reference instead of the structure frame - image models restyle a real
   frame more faithfully than a grey one.

The prompt must then describe that image: its `[Shot 1]` opens with what the first frame shows.

## Steps

1. Look at the structure clip (and the first frame, if any): count the characters, note their facing and the camera
   path, the layout and what the structure leaves open.
2. Write the free parts as JSON `{summary, detailed_description, overall_soundscape, non_diegetic_music}`.
3. `assemble(free, inputs, condition)` from [`scripts/conditions.py`](scripts/conditions.py).
4. `python scripts/check.py prompt.txt` - sections in order, labels defined, task type present, first-frame opening, length,
   English, shot markers. Fix and re-check until it prints OK.

Save the free fields as `free-parts.json`. From this skill directory, assemble a structure-plus-first-frame
prompt with the semantic-map condition, then validate it:

```bash
python - <<'PYTHON'
import json
import runpy
from pathlib import Path

assemble = runpy.run_path("scripts/conditions.py")["assemble"]
free = json.loads(Path("free-parts.json").read_text(encoding="utf-8"))
prompt = assemble(free, inputs="full", condition="semantic")
Path("prompt.txt").write_text(prompt + "\n", encoding="utf-8")
PYTHON
python scripts/check.py prompt.txt
```

Choose `inputs` and `condition` to match the supplied media. Use `custom_condition()` from the same module for
an unlisted structure type. The checker validates prompt structure and wording constraints; inspect the clip
and first frame to verify that the visual description is accurate.

For batch training-caption production, see the
[upstream SolarRenderer data branch](https://github.com/Enderfga/SolarRenderer/tree/data).
Its dataset-processing and captioning pipeline is maintained separately from this self-contained prompt skill.
