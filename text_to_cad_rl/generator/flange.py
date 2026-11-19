"""Disc + hub flange with a through bore and 3 or 4 bolt holes on a pitch circle."""

from __future__ import annotations

import math
import random
from typing import Any

from text_to_cad_rl.generator import FamilySample
from text_to_cad_rl.generator._checks import (
    EXPLICIT_TOL,
    PATTERN_TOL_ENGINEER,
    bbox_sorted,
    hole_set,
    tol,
)
from text_to_cad_rl.generator.standards import ISO273

BOLT_THREADS = ["M4", "M5", "M6"]
EXPLICIT = [
    "Make a flange: a {d1} mm diameter disc {t1} mm thick with a {d2} mm diameter hub {t2} mm tall on top, a {s} mm bore through both, and {n} {dia} mm through holes in the disc equally spaced on a {pcd} mm pitch circle.",
    "A round flange, {d1} mm across and {t1} mm thick, carrying a {d2} mm hub that stands {t2} mm above it. Bore {s} mm right through. {n} holes of {dia} mm, equally spaced on a {pcd} mm pitch circle, go through the disc.",
    "Model a hub flange: disc {d1} x {t1} mm, hub {d2} x {t2} mm, centre bore {s} mm through, {n} x {dia} mm holes through the disc on a {pcd} mm pitch circle, equally spaced.",
    "I need a {d1} mm flange plate, {t1} mm thick, with a {d2} mm hub {t2} mm high, a {s} mm through bore and {n} {dia} mm bolt holes equally spaced on a {pcd} mm pitch circle.",
]
ENGINEER = [
    "Make a flange: a {d1} mm diameter disc {t1} mm thick with a {d2} mm diameter hub {t2} mm tall on top, a {s} mm bore through both, and {n} {thread} clearance bolt holes (ISO 273) equally spaced on a {pcd} mm pitch circle.",
    "A round flange, {d1} mm across and {t1} mm thick, with a {d2} mm hub {t2} mm tall and a {s} mm through bore, bolted down by {n} {thread} bolts (ISO 273 clearance holes) equally spaced on a {pcd} mm pitch circle.",
    "Hub flange, disc {d1} x {t1} mm, hub {d2} x {t2} mm, {s} mm through bore, {n} ISO 273 clearance holes for {thread} on a {pcd} mm pitch circle, equally spaced.",
    "I need a {d1} mm flange, {t1} mm thick, with a {d2} mm hub {t2} mm high, {s} mm bore, drilled for {n} {thread} bolts (ISO 273) equally spaced on a {pcd} mm pitch circle.",
]


def _draw(rng: random.Random) -> dict[str, Any] | None:
    d1 = rng.randrange(40, 105, 5)
    t1 = rng.choice([4, 5, 6, 8])
    d2 = 2 * round(d1 * rng.uniform(0.35, 0.5) / 2)
    t2 = rng.choice([5, 8, 10, 12])
    shafts = [x for x in (6, 8, 10, 12, 16) if x <= d2 - 6]
    thread = rng.choice(BOLT_THREADS)
    coarse = ISO273[thread][2]
    lo, hi = math.ceil(d2 + coarse + 4), math.floor(d1 - coarse - 4)
    if not shafts or lo > hi:
        return None
    pcd = rng.randrange(lo + lo % 2, hi + 1, 2) if lo + lo % 2 <= hi else None
    if pcd is None:
        return None
    return {
        "d1": d1,
        "t1": t1,
        "d2": d2,
        "t2": t2,
        "s": rng.choice(shafts),
        "thread": thread,
        "n": rng.choice([3, 4]),
        "pcd": pcd,
    }


def generate(rng: random.Random) -> FamilySample:
    for _ in range(100):
        p = _draw(rng)
        if p is not None:
            break
    else:
        raise RuntimeError("flange: no feasible draw in 100 tries")
    fine, medium, coarse = ISO273[p["thread"]]
    radius = p["pcd"] / 2
    points = [
        (
            round(radius * math.cos(2 * math.pi * k / p["n"]), 6),
            round(radius * math.sin(2 * math.pi * k / p["n"]), 6),
        )
        for k in range(p["n"])
    ]
    steps: list[dict[str, Any]] = [
        {"op": "cyl", "r": p["d1"] / 2, "h": p["t1"]},
        {"op": "cyl", "r": p["d2"] / 2, "h": p["t2"], "at": [0, 0, p["t1"]]},
        {"op": "fuse", "a": 0, "b": 1},
        {"op": "cyl", "r": p["s"] / 2, "h": p["t1"] + p["t2"]},
        {"op": "cut", "a": 2, "b": 3},
    ]
    for x, y in points:
        steps.append({"op": "cyl", "r": medium / 2, "h": p["t1"], "at": [x, y, 0]})
        steps.append({"op": "cut", "a": len(steps) - 2, "b": len(steps) - 1})
    common: list[dict[str, Any]] = [
        bbox_sorted([p["d1"], p["d1"], p["t1"] + p["t2"]]),
        {"type": "boss", "diameter": tol(p["d1"], EXPLICIT_TOL), "height_min": p["t1"] - EXPLICIT_TOL},
        {"type": "boss", "diameter": tol(p["d2"], EXPLICIT_TOL), "height_min": p["t2"] - EXPLICIT_TOL},
        {"type": "bore", "diameter": tol(p["s"], EXPLICIT_TOL), "through": True},
    ]
    fields = {k: p[k] for k in ("d1", "t1", "d2", "t2", "s", "n", "pcd")}
    return FamilySample(
        params=p,
        reference_spec={"steps": steps},
        prompts={
            "explicit": rng.choice(EXPLICIT).format(dia=medium, **fields),
            "engineer": rng.choice(ENGINEER).format(thread=p["thread"], **fields),
        },
        checks={
            "explicit": common + [hole_set(tol(medium, EXPLICIT_TOL), points, EXPLICIT_TOL)],
            "engineer": common + [hole_set([fine, coarse], points, PATTERN_TOL_ENGINEER)],
        },
    )
