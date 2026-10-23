# SolarRenderer prompts and first frames

Adapted from the upstream [prompt guide at revision
`e018b0f`](https://github.com/Enderfga/SolarRenderer/blob/e018b0f673de865711fe99844d3976865722b8d0/docs/prompts.md).
See the [integrated tutorial](README.md) for installation and a complete render.
Run all helper commands from the repository root.

## Use the six-part training format

SolarRenderer was trained with a six-part prompt. Preserve that format rather
than replacing it with a short free-form description:

```text
subject_definitions:   what each input is (fixed sentences from conditions.py)
summary:               task type and a paragraph describing action and camera motion
retention_analysis:    what is retained from each input (fixed sentences)
detailed_description:  350–500 words on materials, light, weather, subjects, and events
overall_soundscape:    ambience
non_diegetic_music:    N/A
```

The soundscape sections are part of the learned caption format. The integrated
renderer exports video only.

Give the local [caption skill](../../skills/solarrenderer-caption/SKILL.md) to a
coding agent together with the structure video and optional first frame, and
ask it to write `prompt.txt` in the desired appearance. Keep the skill's
`scripts/` directory when copying it into an agent's skill directory.

The workflow is:

1. Inspect the structure video's subjects, facing, layout, and camera path.
2. Choose one of the six trained structure definitions in
   [`conditions.DEFINE`](../../skills/solarrenderer-caption/scripts/conditions.py)
   verbatim. For another representation, use
   `custom_condition(looks_like, not_shown)` to describe what it looks like and
   which aspects of its appearance should not be reproduced.
3. Write the free sections, describing the appearance and details that the
   structure leaves open, while preserving visible actions and scene geometry.
4. Assemble with `conditions.assemble(free, inputs, condition)` and validate:

```bash
python skills/solarrenderer-caption/scripts/check.py prompt.txt
```

For structure plus a first frame, use `full` conditioning. Without a first
frame, use `structure` conditioning and remove descriptions of a supplied
first image. The skill also supports first-frame-only and text-only captions
for training data; the integrated rendering CLI always requires `--structure`.

The [included prompt](../../examples/solar_renderer/sem-highway_synthwave/prompt.txt)
is a complete six-part example.

## Prepare a matching first frame

A first frame gives the renderer a concrete appearance reference. If you do
not have one, extract frame zero from the structure video and use it as the
reference for an image model. Preserve the camera, layout, subjects, poses,
and facing, while changing the appearance to match the video prompt:

```bash
python skills/solarrenderer-caption/scripts/first_frame.py frame0 \
  structure.mp4 frame0.png
python skills/solarrenderer-caption/scripts/first_frame.py prompt prompt.txt
```

For toon, pixel-art, or another stylized appearance, append `--stylised` to
the `prompt` command. Use its printed text and `frame0.png` with an image model,
then save the result as `generated.png`. Fit it to the rendering canvas:

```bash
python skills/solarrenderer-caption/scripts/first_frame.py fit \
  generated.png first.png
```

The default canvas is 1344×768. For portrait inputs, append `768x1344`. Check
that cropping has preserved the structure's subjects and composition before
passing the result as `--first first.png` to the [renderer](inference.md).

To restyle a clip, keep its structure video, write a prompt for the new
appearance, and make a matching first frame using the same layout.

For the integrated upstream commit, date, and adaptation scope, see
[Upstream synchronization](README.md#upstream-synchronization).
