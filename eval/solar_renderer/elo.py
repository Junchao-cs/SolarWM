#!/usr/bin/env python
"""Elo ratings from the pairwise judge: one rating per method from every verdict between any two methods.

    python eval/solar_renderer/elo.py ./judge/*.jsonl [--split hard|heldout] [--tag "unseen condition"] [--question q4_overall]

Reads the outputs of eval/solar_renderer/judge.py run (any set of method pairs, each judged in both orders) and fits a Bradley-Terry
model to the per-case verdicts of the chosen question (both-order rule of eval/solar_renderer/judge.py; a tie counts half a win to
each side). Strengths are reported on the Elo scale: 400 points per factor of ten in odds, mean of the rated methods
1000. Every method also plays one virtual tie against a fixed reference player, so a method that loses every
comparison still gets a finite rating (methods with the same record share one rating). The 95% interval resamples
whole structure videos on the hard split and clips on the held-out split, 400 draws.

To add a method to the published table, judge its renders against the released SolarRenderer renders (or any rated
method) on the same split and fit again with the released verdicts.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import Q, segment, verdict


def games(
    files: list[str], split: str, tag: str | None, question: str, methods: set[str] | None = None
):
    """(unit, a, b, winner or 'tie') for every case of the split judged in both orders, between two of ``methods``
    (all methods if None)."""
    by = collections.defaultdict(dict)
    for f in files:
        with open(f, encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                if methods and not set(r["pair"]) <= methods:
                    continue
                if (
                    "answer" in r
                    and r["split"] == split
                    and (not tag or tag in (r.get("tags") or []))
                ):
                    by[(r["case"], tuple(r["pair"]))][r["swap"]] = r
    return [
        (segment(c), p[0], p[1], verdict(rs, question)) for (c, p), rs in by.items() if len(rs) == 2
    ]


def fit(gs) -> dict[str, float]:
    """Bradley-Terry strengths by minorization-maximization, with one virtual tie per method against a reference of
    strength 1; returned on the Elo scale with the rated methods' mean at 1000."""
    methods = sorted({g[1] for g in gs} | {g[2] for g in gs})
    wins, n = collections.Counter(), collections.Counter()
    for _, a, b, w in gs:
        n[a, b] += 1
        if w == a:
            wins[a, b] += 1
        elif w == b:
            wins[b, a] += 1
        else:
            wins[a, b] += 0.5
            wins[b, a] += 0.5
    p = {m: 1.0 for m in methods}
    for _ in range(2000):
        new = {}
        for i in methods:
            w = sum(wins[i, j] for j in methods if j != i) + 0.5
            d = sum((n[i, j] + n[j, i]) / (p[i] + p[j]) for j in methods if j != i) + 1 / (p[i] + 1)
            new[i] = w / d
        step = max(abs(math.log(new[m] / p[m])) for m in methods)
        p = new
        if step < 1e-7:
            break
    g = math.exp(sum(math.log(v) for v in p.values()) / len(p))
    return {m: 1000 + 400 * math.log10(p[m] / g) for m in methods}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results", nargs="+", help="outputs of eval/solar_renderer/judge.py run")
    ap.add_argument("--split", default="hard", choices=("hard", "heldout"))
    ap.add_argument(
        "--tag", help='hard split only: rate on the cases with this tag, e.g. "unseen condition"'
    )
    ap.add_argument("--question", default="q4_overall", choices=Q)
    ap.add_argument(
        "--methods", help="comma list: rate only these methods, from the comparisons among them"
    )
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--json", help="also write the ratings here")
    a = ap.parse_args()
    gs = games(
        a.results, a.split, a.tag, a.question, set(a.methods.split(",")) if a.methods else None
    )
    if not gs:
        sys.exit("no case judged in both orders on this split")
    point = fit(gs)
    by = collections.defaultdict(list)
    for g in gs:
        by[g[0]].append(g)
    units = list(by.values())
    rng, boot = random.Random(0), collections.defaultdict(list)
    for _ in range(a.draws):
        for m, v in fit([g for _ in units for g in rng.choice(units)]).items():
            boot[m].append(v)

    def interval(
        xs,
    ):  # a method absent from a draw (judged on few structure videos) has fewer values
        xs = sorted(xs)
        return xs[int(0.025 * len(xs))], xs[max(0, int(0.975 * len(xs)) - 1)]

    rows = {m: (point[m], *interval(boot[m])) for m in point}
    label = a.split + (f", {a.tag}" if a.tag else "")
    print(f"{label}: {a.question}, {len(gs)} judged cases, {len(units)} resampling units")
    for m, (e, l, h) in sorted(rows.items(), key=lambda x: -x[1][0]):
        print(f"  {m:24s} {e:6.0f}  [{l:.0f}, {h:.0f}]")
    if a.json:
        Path(a.json).write_text(
            json.dumps({m: {"elo": e, "lo": l, "hi": h} for m, (e, l, h) in rows.items()}, indent=1)
        )


if __name__ == "__main__":
    main()
