"""Repository assets for the isolated ComfyUI node pack.

Resolve symlinks because ComfyUI loads this directory through custom_nodes/solar_renderer.
Keep this module independent of SolarWM's training dependencies and ComfyUI imports.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SILENCE = str(ROOT / "src/solarwm/renderer/assets/h3_silence_153_158_170.safetensors")
CAPTION_HELPERS = ROOT / "skills/solarrenderer-caption/scripts/conditions.py"
EXAMPLES = str(ROOT / "examples/solar_renderer")
WEB = str(HERE / "web")
