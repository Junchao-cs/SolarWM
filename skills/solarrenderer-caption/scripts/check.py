"""Check SolarRenderer prompts and normalize the free caption fields.

Adapted from SolarRenderer, Copyright 2026 SolarRenderer authors, Apache-2.0.
Modified to validate required fields and provide a command-line interface.
"""

import argparse
import math
import re
from pathlib import Path

FIELDS = ["summary", "detailed_description", "overall_soundscape", "non_diegetic_music"]
SECTIONS = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
]
CJK = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef\u3000-\u303f]"
)
QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|「[^」]*」")
LABEL = re.compile(r"<\s*(Subject|Picture|Video|Audio)\s*\d+\s*>")


def normalize_shots(d):
    """Remove the first shot's timestamp and number unlabelled cuts in playback order."""
    d = re.sub(
        r"\[Shot 1\]\s*At 00:00(?:\.0+)?,?\s*(\w)", lambda m: "[Shot 1] " + m.group(1).upper(), d
    )
    d = re.sub(
        "(?<!\\] )(?<!\\]\\s)\\bAt (\\d{2}:\\d{2}\\.\\d{3}), the (shot|camera|scene) "
        "(cuts|switches|changes|transitions)",
        r"[Shot N] At \1, the \2 \3",
        d,
    )
    n = iter(range(1, 1000))
    return re.sub(r"\[Shot (?:\d+|N)\]", lambda m: f"[Shot {next(n)}]", d)


def shots_ok(d, duration):
    """Require ordered shots with timestamps only at later cuts, inside the clip."""
    shots = [(m.start(), int(m.group(1))) for m in re.finditer(r"\[Shot (\d+)\]", d)]
    ok, prev = True, 0.0
    if not shots or shots[0][1] != 1 or re.match(r"\[Shot 1\]\s*At \d", d[shots[0][0] :]):
        ok = False
    for i, (pos, n) in enumerate(shots[1:], start=2):
        m = re.match(r"\[Shot \d+\]\s*At (\d{2}):(\d{2})\.(\d{3})", d[pos:])
        if n != i or not m:
            ok = False
            continue
        t = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000
        ok &= prev < t < duration
        prev = t
    stray = len(re.findall(r"\d{1,2}:\d{2}(?:\.\d+)?", d)) - len(shots[1:])
    return ok and stray == 0, len(shots), stray


def check_free(ans, duration):
    f = {k: str(ans.get(k) or "").strip() for k in FIELDS}
    f["summary"] = re.sub(r"^(\[[^\]]*\]\s*)+[.:;,\s]*", "", f["summary"])
    f["detailed_description"] = normalize_shots(f["detailed_description"])
    s = ans.get("subject")
    f["subject"] = (
        s.strip().rstrip(".")
        if isinstance(s, str) and s.strip() and s.strip().lower() not in ("null", "none", "n/a")
        else None
    )
    c = {"fields": all(f[k] for k in FIELDS)}
    c["words"] = len(f["detailed_description"].split())
    c["length"] = 350 <= c["words"] <= 500
    text = " ".join(f[k] for k in FIELDS) + " " + (f["subject"] or "")
    c["english"] = not CJK.search(QUOTED.sub("", text))
    c["no_labels"] = not LABEL.search(text)
    c["cuts"], c["shots"], c["stray_timestamps"] = shots_ok(f["detailed_description"], duration)
    c["pass"] = all(c[k] for k in ("fields", "length", "english", "no_labels", "cuts"))
    return f, c


def problems(c):
    p = []
    if not c.get("fields"):
        p.append("every field must be filled")
    if not c.get("length"):
        p.append(
            f"detailed_description has {c.get('words')} words; "
            "it must have 350-500 (aim for about 420)"
            + ("; add concrete visible detail" if c.get("words", 0) < 350 else "; tighten it")
        )
    if not c.get("english"):
        p.append("only English is allowed outside double-quoted on-screen text")
    if not c.get("no_labels"):
        p.append("remove every reference label such as <Picture 1> or <Video 1>")
    if not c.get("cuts"):
        p.append(
            "[Shot 1] has no timestamp; a later shot starts '[Shot N] At MM:SS.mmm, the "
            "shot cuts to', times "
            "strictly increasing inside the clip; no other timestamps anywhere (describe "
            "timing inside a shot in "
            "words: 'midway', 'near the end')"
        )
    return p


def check_prompt(text, duration=6.583):
    """Problems of a finished prompt (an empty list when it is fine)."""
    p = []
    if text.startswith("integrated_multimodal_description:"):  # text only
        names = ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"]
        headings = list(re.finditer(r"^([a-z_]+):[ \t]*", text, flags=re.M))
        if [match.group(1) for match in headings] != names:
            return [f"text-only sections must be {names} in this order"]
        values = [
            text[match.end() : headings[index + 1].start() if index + 1 < len(headings) else None]
            for index, match in enumerate(headings)
        ]
        if any(not value.strip() for value in values):
            p.append("every text-only field must be filled; use N/A when appropriate")
        body = values[0]
        if LABEL.search(text):
            p.append("a text-only prompt names no inputs")
        words = len(body.split())
    else:
        heads = re.findall(r"^([a-z_]+):\s*$", text, flags=re.M)
        if heads != SECTIONS:
            p.append(f"sections must be {SECTIONS} in this order, found {heads}")
            return p
        part = dict(zip(SECTIONS, re.split(r"^[a-z_]+:\s*$", text, flags=re.M)[1:], strict=True))
        if any(not value.strip() for value in part.values()):
            p.append("every section must be filled; use N/A when appropriate")
        defined = set(re.findall(r"^(<Video \d+>) is ", part["subject_definitions"], flags=re.M))
        used = set(m.group(0) for m in LABEL.finditer(text))
        if used - defined:
            p.append(f"labels used but not defined: {sorted(used - defined)}")
        tasks = re.match(r"\s*\[([a-z +]+)\]", part["summary"])
        if not tasks:
            p.append(
                "summary must open with the task type, e.g. [keyframe completion + "
                "reference generation]"
            )
        first = re.search(
            r"^(<Video \d+>) is the first frame of \[Shot 1\]",
            part["subject_definitions"],
            flags=re.M,
        )
        if (
            first
            and f"[Shot 1] The shot begins from {first.group(1)}."
            not in part["detailed_description"]
        ):
            p.append(
                "with a first frame, [Shot 1] must open with "
                f"'The shot begins from {first.group(1)}.'"
            )
        body = part["detailed_description"]
        words = len(body.split())
    if not 350 <= words <= 520:  # the fixed opening sentence adds a few words to the free 350-500
        p.append(f"detailed_description has {words} words; it should have about 350-500")
    if CJK.search(QUOTED.sub("", text)):
        p.append("only English outside double-quoted on-screen text")
    ok, _, _stray = shots_ok(body, duration)
    if not ok:
        p.append(
            "shot markers: [Shot 1] without a timestamp, later shots open with 'At "
            "MM:SS.mmm', no other timestamps"
        )
    return p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check SolarRenderer prompt files.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--duration", type=float, default=158 / 24, help="clip duration in seconds")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    bad = 0
    for path in args.paths:
        try:
            issues = check_prompt(path.read_text(encoding="utf-8"), duration=args.duration)
        except (OSError, UnicodeError) as error:
            issues = [str(error)]
        print(path, "OK" if not issues else "; ".join(issues))
        bad += bool(issues)
    return int(bool(bad))


if __name__ == "__main__":
    raise SystemExit(main())
