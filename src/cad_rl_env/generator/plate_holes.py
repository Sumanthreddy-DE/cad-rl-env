"""Flat plate with 2 holes in a row or 4 on a rectangle, sized by ISO 273."""

from __future__ import annotations

import math
import random

from cad_rl_env.generator import FamilySample
from cad_rl_env.generator._checks import (
    EXPLICIT_TOL,
    PATTERN_TOL_ENGINEER,
    bbox_sorted,
    cut_chain,
    hole_set,
    tol,
)
from cad_rl_env.generator.standards import ISO273

EXPLICIT = [
    "Make a {w} x {d} x {t} mm plate with {holes} {dia} mm through holes {layout}.",
    "A flat plate, {w} by {d} mm and {t} mm thick, with {holes} through holes of {dia} mm diameter {layout}.",
    "Model a {t} mm thick plate measuring {w} x {d} mm. Drill {holes} {dia} mm holes right through it, {layout}.",
    "I need a {w} x {d} mm plate, {t} mm thick, with {holes} {dia} mm diameter through holes {layout}.",
]
ENGINEER = [
    "Make a {w} x {d} x {t} mm plate with {holes} {thread} clearance holes (ISO 273) {layout}.",
    "A {t} mm thick mounting plate, {w} x {d} mm, with {holes} through holes that clear {thread} screws per ISO 273, {layout}.",
    "Mounting plate {w} by {d} mm, {t} mm thick: {holes} ISO 273 clearance holes for {thread} bolts, {layout}.",
    "I need a {w} x {d} x {t} mm plate drilled for {holes} {thread} bolts (ISO 273 clearance), {layout}.",
]
WORDS = {2: "two", 4: "four"}


def generate(rng: random.Random) -> FamilySample:
    thread = rng.choice(sorted(ISO273))
    fine, medium, coarse = ISO273[thread]
    t = rng.choice([3, 4, 5, 6, 8])
    n = rng.choice([2, 4])
    margin = max(rng.choice([5, 6, 8, 10]), math.ceil(coarse / 2 + 2))
    gap = math.ceil(coarse) + 4  # minimum centre distance between holes
    if n == 4:
        w = rng.randrange(max(30, 2 * margin + gap), 125, 5)
        d = rng.randrange(max(30, 2 * margin + gap), 125, 5)
        a, b = w - 2 * margin, d - 2 * margin
        points: list[tuple[float, float]] = [
            (float(margin), float(margin)),
            (float(w - margin), float(margin)),
            (float(margin), float(d - margin)),
            (float(w - margin), float(d - margin)),
        ]
        layout = f"on a {a} x {b} mm rectangular pattern centred on the plate"
    else:
        w = rng.randrange(max(60, 2 * margin + gap), 125, 5)
        d = rng.randrange(max(15, math.ceil(coarse) + 6), 45, 5)
        a = w - 2 * margin
        points = [(float(margin), d / 2), (float(w - margin), d / 2)]
        layout = f"{a} mm apart on the centre line along the {w} mm side, centred"

    base = {"op": "box", "dx": w, "dy": d, "dz": t}
    tools = [{"op": "cyl", "r": medium / 2, "h": t, "at": [x, y, 0]} for x, y in points]
    fields = {"w": w, "d": d, "t": t, "holes": WORDS[n], "layout": layout}
    return FamilySample(
        params={"thread": thread, "w": w, "d": d, "t": t, "n": n, "margin": margin},
        reference_spec=cut_chain(base, tools),
        prompts={
            "explicit": rng.choice(EXPLICIT).format(dia=medium, **fields),
            "engineer": rng.choice(ENGINEER).format(thread=thread, **fields),
        },
        checks={
            "explicit": [
                bbox_sorted([w, d, t]),
                hole_set(tol(medium, EXPLICIT_TOL), points, EXPLICIT_TOL),
            ],
            "engineer": [
                bbox_sorted([w, d, t]),
                hole_set([fine, coarse], points, PATTERN_TOL_ENGINEER),
            ],
        },
    )
