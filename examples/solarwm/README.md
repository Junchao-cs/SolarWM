# SolarWM examples

Each example contains a first-frame image and its prompt. These files are shared
with the [realtime interaction demo](../../src/solarwm/realtime_interaction/README.md).

| Example | Image | Prompt |
|---|---|---|
| `snow` (default) | [first.png](snow/first.png) | [prompt.txt](snow/prompt.txt) |
| `fantasy_world` | [first.png](fantasy_world/first.png) | [prompt.txt](fantasy_world/prompt.txt) |

The demo reads `first.png` and `prompt.txt` from each example directory when
creating a deployment. To add an example, create another directory containing
both files; use letters, numbers, underscores or hyphens for the directory name.
