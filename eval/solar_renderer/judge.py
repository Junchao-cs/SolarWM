#!/usr/bin/env python
"""Pairwise VLM judge of the SolarRenderer benchmark: two methods, every case both have, both orders.

    export GEMINI_API_KEY=...
    python eval/solar_renderer/judge.py run --bench ./SolarRenderer-Data --a ./renders/mine --b ./renders/other \
        --name-a mine --name-b other --out ./judge/mine_vs_other.jsonl [--split hard|heldout|all] [--threads 8]
    python eval/solar_renderer/judge.py report ./judge/mine_vs_other.jsonl [more.jsonl ...]

The judge (gemini-3.8-flash, temperature 0, through the Gemini API) sees, in order, the structure video, the first
frame, video A and video B, and answers four questions with A, B or tie: q1 visual quality (the two videos on their
own), q2 structure and motion faithfulness, q3 first-frame appearance kept, q4 overall. Every case is judged twice
with A and B swapped; a case is a win for X when both orders pick X, or one picks X and the other ties; opposite picks
(each order choosing its own position) count as a tie. The net score is win rate minus loss rate, with a bootstrap
95% interval. eval/solar_renderer/elo.py fits Elo ratings from the verdicts of several methods.

Videos are sent as they were for the published numbers: 672x384 (384x672 portrait), 12 fps, H.264 from the original
at CRF 30, raised in steps of 3 (up to 45) while over 0.7 MB, no audio; the first frame as a 672x384 JPEG. They are cached under <out>.media/.
Results are appended to <out> as they arrive; a rerun skips what is answered (rate-limited calls are retried then).
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
Q = ["q1_visual_quality", "q2_structure_motion", "q3_first_frame", "q4_overall"]
PROMPT = """You are an expert judge of AI-generated video for film and game rendering.
Inputs, in order: (1) REFERENCE STRUCTURE - a rough proxy render (grey/flat/schematic) that fixes layout, object and character motion and camera for every frame; (2) FIRST FRAME - the image the generated video must start from and keep the look of; (3) VIDEO A; (4) VIDEO B. Both A and B were generated from the same reference structure, first frame and text prompt.
Compare A and B. Look closely at fine detail, textures, faces/hands/limbs, edges, and frame-to-frame stability (flicker, warping, smearing, melting, popping). Ignore that the reference structure itself looks crude - judge how well each video follows its layout, motion timing and camera.
The order of A and B is random; do not prefer either position. Answer each question with "A", "B" or "tie" (use tie only if you genuinely cannot tell):
q1_visual_quality: judging the two videos ON THEIR OWN (ignore the first frame and the reference here - a video that departs from the first frame's look can still win this question), which has higher visual quality: sharper fine detail and texture, fewer artifacts (blur, smearing, melting, distorted bodies/faces), better temporal stability (no flicker or warping)?
q2_structure_motion: which follows the reference structure's layout, motion and camera more faithfully over the whole clip?
q3_first_frame: which better keeps the first frame's appearance throughout - colours, lighting, contrast, materials, identities and style - without drifting?
q4_overall: which would you rather ship as the final rendered shot?
Return only JSON: {"q1_visual_quality":"A|B|tie","q2_structure_motion":"A|B|tie","q3_first_frame":"A|B|tie","q4_overall":"A|B|tie","reason":"<=40 words on the decisive differences"}"""


# ---------------------------------------------------------------------------------------------- media


def _size(path: str) -> tuple[int, int]:
    out = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split(",")
    )
    return int(out[0]), int(out[1])


def small_video(src: str, dst: Path) -> Path:
    """672x384 (384x672 portrait), 12 fps, H.264 from the original at CRF 30, raised in steps of 3 (up to 45) while
    the file is over 0.7 MB."""
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        w, h = _size(src)
        scale = "384:672" if h > w else "672:384"
        tmp = dst.with_suffix(".tmp.mp4")
        crf = 30
        while True:
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    src,
                    "-vf",
                    f"fps=12,scale={scale}:flags=lanczos,format=yuv420p",
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(crf),
                    "-an",
                    "-movflags",
                    "+faststart",
                    str(tmp),
                ],
                check=True,
            )
            if tmp.stat().st_size <= 700_000 or crf >= 45:
                break
            crf += 3
        tmp.rename(dst)
    return dst


def small_image(src: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        w, h = Image.open(src).size
        scale = "384:672" if h > w else "672:384"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                src,
                "-vf",
                f"scale={scale}:flags=lanczos",
                "-q:v",
                "3",
                "-frames:v",
                "1",
                str(dst),
            ],
            check=True,
        )
    return dst


# ---------------------------------------------------------------------------------------------- judge


def part(path: str) -> dict:
    mime = "video/mp4" if path.endswith(".mp4") else "image/jpeg"
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(open(path, "rb").read()).decode(),
        }
    }


def ask(row: dict, key: str) -> dict:
    parts = [
        {"text": PROMPT},
        {"text": "REFERENCE STRUCTURE:"},
        part(row["cond"]),
        {"text": "FIRST FRAME:"},
        part(row["first"]),
    ]
    parts += [{"text": "VIDEO A:"}, part(row["a"]), {"text": "VIDEO B:"}, part(row["b"])]
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4000},
    }
    err = ""
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                URL.format(model=MODEL),
                data=json.dumps(body).encode(),
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            )
            d = json.load(urllib.request.urlopen(req, timeout=300))
            txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
            ans = json.loads(txt[txt.find("{") : txt.rfind("}") + 1])
            return {**row, "answer": ans, "usage": d.get("usageMetadata")}
        except Exception as e:  # rate limits and transient errors: back off, then record the error
            err = str(e)[:200]
            time.sleep(5 * (attempt + 1))
    return {**row, "error": err}


def pairs(a: argparse.Namespace, media: Path) -> list[dict]:
    rows = []
    for c in bench.load(a.bench, a.split):
        ra, rb = c.render(a.a), c.render(a.b)
        if not (ra.exists() and rb.exists()):
            continue
        tag = c.key.replace("/", "__")
        sa = str(small_video(str(ra), media / a.name_a / f"{tag}.mp4"))
        sb = str(small_video(str(rb), media / a.name_b / f"{tag}.mp4"))
        cond = str(small_video(c.condition, media / "condition" / f"{tag}.mp4"))
        first = str(small_image(c.first_frame, media / "first" / f"{tag}.jpg"))
        for swap in (0, 1):
            (x, px), (y, py) = (
                ((a.name_a, sa), (a.name_b, sb)) if not swap else ((a.name_b, sb), (a.name_a, sa))
            )
            rows.append(
                {
                    "key": f"{c.split}/{c.key}|{a.name_a}|{a.name_b}|{swap}",
                    "case": f"{c.split}/{c.key}",
                    "split": c.split,
                    "tags": c.tags,
                    "a": px,
                    "b": py,
                    "cond": cond,
                    "first": first,
                    "pair": [a.name_a, a.name_b],
                    "swap": swap,
                    "a_m": x,
                    "b_m": y,
                }
            )
    return rows


def run(a: argparse.Namespace) -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    media = Path(str(out) + ".media")
    rows = pairs(a, media)
    done = set()
    if out.exists():
        done = {r["key"] for r in map(json.loads, out.read_text().splitlines()) if "answer" in r}
    todo = [r for r in rows if r["key"] not in done]
    print(f"{len(rows) // 2} cases, {len(todo)} calls to make", flush=True)
    lock, n = threading.Lock(), [0]

    def work(r):
        res = ask(r, key)
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(res) + "\n")
            n[0] += 1
            if n[0] % 20 == 0:
                print(f"{n[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(work, todo))
    report(argparse.Namespace(results=[str(out)]))


# ---------------------------------------------------------------------------------------------- report


def verdict(rs: dict, q: str):
    v = [{"A": r["a_m"], "B": r["b_m"]}.get(r["answer"].get(q, "tie"), "tie") for r in rs.values()]
    if v[0] == v[1]:
        return v[0]
    if "tie" in v:
        return next(t for t in v if t != "tie")  # one order decided, the other tied
    return "tie"  # each order picked its own position: position bias


def ci(xs, groups=None, n=2000, seed=0):
    """Mean and 95% bootstrap interval; with ``groups`` (one label per x) whole groups are resampled - on the hard
    split the cases of one structure video share its layout and motion, so they are not independent."""
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    if groups is None:
        b = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    else:
        by = collections.defaultdict(list)
        for x, g in zip(xs, groups, strict=False):
            by[g].append(x)
        units = list(by.values())
        b = []
        for _ in range(n):
            draw = [x for _ in units for x in rng.choice(units)]
            b.append(sum(draw) / len(draw))
        b.sort()
    return m, b[int(0.025 * n)], b[int(0.975 * n)]


def segment(case: str) -> str:
    """hard/<segment>__<look> -> the structure video it was rendered from; a held-out clip is its own unit."""
    return case.split("__")[0] if case.startswith("hard/") else case


def net_ci(cases: list[str], xs: list[int]):
    """The intervals of the paper: on the hard split whole structure videos are resampled (2,000 draws), since the
    cases of one structure video share its layout and motion; held-out clips are resampled one by one (10,000)."""
    if cases and all(c.startswith("hard/") for c in cases):
        return ci(xs, [segment(c) for c in cases], n=2000)
    return ci(xs, n=10000)


def report(a: argparse.Namespace) -> None:
    by = collections.defaultdict(dict)
    meta, calls = {}, collections.Counter()
    for f in a.results:
        with open(f, encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                calls[(tuple(r["pair"]), "answered" if "answer" in r else "failed")] += 1
                if "answer" in r:
                    by[(r["case"], tuple(r["pair"]))][r["swap"]] = r
                    meta[r["case"]] = r
    for p in sorted({p for _, p in by}):
        x, y = p
        cases = [c for (c, pp) in by if pp == p and len(by[(c, pp)]) == 2]
        single = sum(1 for (c, pp) in by if pp == p and len(by[(c, pp)]) == 1)
        print(
            f"\n=== {x} vs {y}: {len(cases)} cases judged in both orders (net = ({x} wins - {y} wins) / cases)"
        )
        print(
            f"    calls answered {calls[(p, 'answered')]}, failed {calls[(p, 'failed')]} (a rerun retries them); "
            f"{single} cases with one order only are left out; hard intervals resample whole structure videos"
        )
        groups = [(s, [c for c in cases if meta[c]["split"] == s]) for s in ("hard", "heldout")]
        tags = sorted({t for c in cases for t in meta[c].get("tags") or []})
        groups += [
            (f"hard, {t}", [c for c in cases if t in (meta[c].get("tags") or [])]) for t in tags
        ]
        for label, sel in groups:
            if not sel:
                continue
            cells = []
            for q in Q:
                v = [verdict(by[(c, p)], q) for c in sel]
                w, l = v.count(x), v.count(y)
                m, lo, hi = net_ci(sel, [1 if t == x else -1 if t == y else 0 for t in v])
                star = (
                    "*" if len(v) >= 10 and (lo > 0 or hi < 0) else " "
                )  # no interval from a handful of cases
                cells.append(
                    f"{q.split('_', 1)[1][:9]:9s} {w:3d}/{len(v) - w - l:3d}/{l:3d} {m:+.2f}{star}"
                )
            print(f"  {label:28s} n={len(sel):3d}  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bench", required=True)
    r.add_argument("--a", required=True, help="renders of method A")
    r.add_argument("--b", required=True, help="renders of method B")
    r.add_argument("--name-a", required=True)
    r.add_argument("--name-b", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--split", default="all", choices=("hard", "heldout", "all"))
    r.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("results", nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
