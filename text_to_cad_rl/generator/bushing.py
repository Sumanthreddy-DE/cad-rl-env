"""Round spacer tube: bore + OD + length. Engineer phrasing sizes the bore by ISO 273."""

from __future__ import annotations

import math
import random

from text_to_cad_rl.generator import FamilySample
from text_to_cad_rl.generator._checks import EXPLICIT_TOL, bbox_sorted, cut_chain, tol
from text_to_cad_rl.generator.standards import ISO273

EXPLICIT = [
    "Model a round tube {od} mm outside diameter, {bore} mm inside diameter and {length} mm long.",
    "A cylindrical spacer: {od} mm OD, {bore} mm through bore, {length} mm long.",
    "Make a {length} mm long sleeve with a {od} mm outer diameter and a {bore} mm hole through its axis.",
    "I need a spacer tube, {length} mm long, outside diameter {od} mm, bore {bore} mm.",
]
ENGINEER = [
    "A {length} mm long spacer for an {thread} bolt, {od} mm outside diameter, bore per ISO 273 clearance.",
    "Spacer tube for {thread} screws (ISO 273 clearance bore), {od} mm OD, {length} mm long.",
    "Make a {od} mm diameter standoff spacer, {length} mm long, that an {thread} bolt passes through (ISO 273 clearance).",
    "I need an {thread} clearance spacer (ISO 273), {od} mm across and {length} mm long.",
]


def generate(rng: random.Random) -> FamilySample:
    thread = rng.choice(sorted(ISO273))
    fine, medium, coarse = ISO273[thread]
    od = math.ceil(coarse) + rng.choice([3, 4, 6, 8, 10])
    length = rng.choice([3, 5, 8, 10, 12, 15, 20, 25, 30])
    spec = cut_chain(
        {"op": "cyl", "r": od / 2, "h": length},
        [{"op": "cyl", "r": medium / 2, "h": length}],
    )
    boss = {"type": "boss", "diameter": tol(od, EXPLICIT_TOL), "height_min": length - EXPLICIT_TOL}
    fields = {"od": od, "length": length}
    return FamilySample(
        params={"thread": thread, "od": od, "length": length},
        reference_spec=spec,
        prompts={
            "explicit": rng.choice(EXPLICIT).format(bore=medium, **fields),
            "engineer": rng.choice(ENGINEER).format(thread=thread, **fields),
        },
        checks={
            "explicit": [
                bbox_sorted([od, od, length]),
                {"type": "bore", "diameter": tol(medium, EXPLICIT_TOL), "through": True},
                boss,
            ],
            "engineer": [
                bbox_sorted([od, od, length]),
                {"type": "bore", "diameter": [fine, coarse], "through": True},
                boss,
            ],
        },
    )
