"""The SolarRenderer benchmark as a list of cases, read from a local copy of the Hugging Face dataset.

    hf download Enderfga/SolarRenderer-Data --repo-type dataset --include "benchmark/*" --local-dir ./SolarRenderer-Data

Two splits, one case format; every case gives a structure video, a first frame and a prompt:
  hard     benchmark/hard/cases.jsonl: 208 cases, the first frame made by an image model, no ground truth
  heldout  benchmark/heldout/cases.jsonl: 53 held-out clips with their real first frame and the real video

A method's renders live in one directory: <renders>/hard/<case_id>.mp4 and <renders>/heldout/<clip_id>.mp4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Case:
    split: str  # hard | heldout
    id: str  # case_id (hard) or clip_id (heldout)
    prompt: str
    condition: str  # the structure video
    first_frame: str
    ground_truth: str | None  # held-out clips only
    tags: list[str] = field(default_factory=list)
    kind: str | None = None  # the structure type

    @property
    def key(self) -> str:
        return self.id

    def render(self, renders: str | Path) -> Path:
        return Path(renders) / self.split / f"{self.id}.mp4"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(root: str | Path, split: str = "all") -> list[Case]:
    """Every case of the split (hard, heldout or all) under a local copy of the dataset."""

    root = Path(root)
    cases: list[Case] = []
    if split in ("hard", "all"):
        for r in _rows(root / "benchmark" / "hard" / "cases.jsonl"):
            files = r["files"]
            prompt = (root / files["prompt.txt"]).read_text().strip()
            cases.append(
                Case(
                    "hard",
                    r["case_id"],
                    prompt,
                    str(root / files["condition.mp4"]),
                    str(root / files["first_frame.png"]),
                    None,
                    list(r.get("tags") or []),
                    r.get("condition"),
                )
            )
    if split in ("heldout", "all"):
        for r in _rows(root / "benchmark" / "heldout" / "cases.jsonl"):
            files = r["files"]
            cases.append(
                Case(
                    "heldout",
                    r["clip_id"],
                    r["prompt"],
                    str(root / files["condition.mp4"]),
                    str(root / files["first_frame.png"]),
                    str(root / r["ground_truth"]),
                    [],
                    r.get("condition"),
                )
            )
    return cases
