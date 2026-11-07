"""Load scene images and prompts from the canonical SolarWM examples."""
from pathlib import Path
import re


def bundled_cases(directory=None):
    if directory is None:
        root = Path(__file__).resolve().parents[3]
        checkout_examples = root / "examples" / "solar_wm"
        directory = (checkout_examples if (root / "pyproject.toml").is_file()
                     else root / "solar" / "examples")
    directory = Path(directory).resolve(strict=True)
    cases = {}
    for folder in sorted(directory.iterdir(), key=lambda path: (path.name != "snow", path.name)):
        if not folder.is_dir() or folder.name == "__pycache__":
            continue
        identity = folder.name
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identity):
            raise ValueError("invalid example scene id: " + identity)
        image = (folder / "first.png").resolve(strict=True)
        prompt_file = (folder / "prompt.txt").resolve(strict=True)
        if not all(path.is_relative_to(directory) and path.is_file()
                   for path in (image, prompt_file)):
            raise ValueError("example files must stay within the examples directory: " + identity)
        prompt = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError("nonempty example prompt required: " + identity)
        cases[identity] = dict(scene_id=identity, title=identity, first=str(image), prompt=prompt)
    if not cases:
        raise ValueError("at least one image/prompt example is required")
    return cases
