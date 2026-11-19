"""Adversarial answers that must score passed = 0, and known weak checks that still pass.

KNOWN_WEAK documents real gaps from SimReady's SP1 Task 2 review. Each case asserts that
the cheat *passes* today, so a SimReady check fix flips the test and forces the README's
reward-hacking table to be updated. Fixing them is SimReady's job (new check types).
"""

from __future__ import annotations

import copy
import json

import pytest
from simready.bench.concepts import load_concepts

from cad_rl_env.generator import make_task
from cad_rl_env.grade import grade_reply

CONCEPTS = {c.concept_id: c for c in load_concepts()}


def checks(cid: str, phrasing: str) -> list[dict]:
    return [c.model_dump(mode="json") for c in getattr(CONCEPTS[cid].checks, phrasing)]


def ref(cid: str) -> dict:
    assert CONCEPTS[cid].reference_spec is not None
    return copy.deepcopy(CONCEPTS[cid].reference_spec)


def passed(spec: dict, check_list: list[dict]) -> bool:
    return bool(grade_reply("```json\n" + json.dumps(spec) + "\n```", check_list)["passed"])


def c001_plate_only() -> dict:
    return {"steps": [ref("c001")["steps"][0]]}


def c001_three_holes() -> dict:
    return {"steps": ref("c001")["steps"][:9]}


def c001_blind_holes() -> dict:
    spec = ref("c001")
    for i in (3, 5, 7, 9):
        spec["steps"][i]["h"] = 2.5
    return spec


def c001_holes_miss_part() -> dict:
    spec = ref("c001")
    for i in (3, 5, 7, 9):
        spec["steps"][i]["at"][0] += 100
    return spec


def c001_duplicate_coaxial_hole() -> dict:
    spec = ref("c001")
    # Cut the third hole location twice instead of cutting the fourth corner hole.
    spec["steps"][9]["at"] = list(spec["steps"][7]["at"])
    return spec


def c001_extra_solid() -> dict:
    spec = ref("c001")
    n = len(spec["steps"])
    spec["steps"] += [
        {"op": "box", "dx": 5, "dy": 5, "dz": 5, "at": [200, 0, 0]},
        {"op": "fuse", "a": n - 1, "b": n},
    ]
    return spec


def c004_solid_rod() -> dict:
    return {"steps": [ref("c004")["steps"][0]]}


def c004_square_tube() -> dict:
    return {
        "steps": [
            {"op": "box", "dx": 16, "dy": 16, "dz": 20, "at": [-8, -8, 0]},
            {"op": "cyl", "r": 4.5, "h": 20},
            {"op": "cut", "a": 0, "b": 1},
        ]
    }


def c014_swapped_plate_axes() -> dict:
    spec = ref("c014")
    # Swap dx (80) and dy (65) while keeping the asymmetric Arduino hole coordinates:
    # the x=72.04 holes now miss the 65 mm X span.
    spec["steps"][0]["dx"], spec["steps"][0]["dy"] = (
        spec["steps"][0]["dy"],
        spec["steps"][0]["dx"],
    )
    return spec


def c047_axial_hole() -> dict:
    # Axial bore instead of a cross-drilled hole; caught since simready 0.5.1.
    return {
        "steps": [
            {"op": "cyl", "r": 5.0, "h": 50.0},
            {"op": "cyl", "r": 1.5, "h": 50.0},
            {"op": "cut", "a": 0, "b": 1},
        ]
    }


ADVERSARIAL = [
    ("c001 plate without holes", c001_plate_only, "c001", "explicit"),
    ("c001 three of four holes", c001_three_holes, "c001", "explicit"),
    ("c001 blind instead of through", c001_blind_holes, "c001", "explicit"),
    ("c001 hole cuts miss the part", c001_holes_miss_part, "c001", "explicit"),
    ("c001 duplicate overlapping hole cut", c001_duplicate_coaxial_hole, "c001", "explicit"),
    ("c001 second disjoint solid", c001_extra_solid, "c001", "explicit"),
    ("c004 solid rod, no bore", c004_solid_rod, "c004", "explicit"),
    ("c004 square tube, right envelope", c004_square_tube, "c004", "explicit"),
    ("c014 plate dimensions swapped across axes", c014_swapped_plate_axes, "c014", "explicit"),
    ("c047 explicit (OOV): axial hole instead of cross hole", c047_axial_hole, "c047", "explicit"),
]


@pytest.mark.parametrize(
    ("name", "make", "cid", "phrasing"), ADVERSARIAL, ids=[a[0] for a in ADVERSARIAL]
)
def test_adversarial_fails(name: str, make, cid: str, phrasing: str) -> None:
    assert not passed(make(), checks(cid, phrasing)), name


def test_generated_flange_without_hub_fails() -> None:
    t = make_task(0, "flange", 0, "explicit")
    spec = copy.deepcopy(t.reference_spec)
    spec["steps"][1]["h"] = 0.5  # hub shrunk to a lip: boss height check must catch it
    assert not passed(spec, t.checks)


def test_generated_plate_wrong_pitch_fails() -> None:
    t = make_task(0, "plate_holes", 0, "explicit")
    spec = copy.deepcopy(t.reference_spec)
    spec["steps"][1]["at"][0] += 1.0
    assert not passed(spec, t.checks)


def test_generated_bushing_wrong_bolt_size_fails_engineer() -> None:
    t = make_task(0, "bushing", 0, "engineer")
    spec = copy.deepcopy(t.reference_spec)
    spec["steps"][1]["r"] += 1.5  # bore 3 mm too big for the named thread
    assert not passed(spec, t.checks)


KNOWN_WEAK = [
    (
        "c010 intent: blank plate passes (slots unseen)",
        lambda: {"steps": [ref("c010")["steps"][0]]},
        "c010",
        "intent",
    ),
    (
        "c023 intent: plain 30 mm cube passes (steps unseen)",
        lambda: {"steps": [{"op": "box", "dx": 30, "dy": 30, "dz": 30}]},
        "c023",
        "intent",
    ),
    (
        "c048 intent: solid cube passes (no gradable check)",
        lambda: {"steps": [{"op": "box", "dx": 10, "dy": 10, "dz": 10}]},
        "c048",
        "intent",
    ),
]


@pytest.mark.parametrize(
    ("name", "make", "cid", "phrasing"), KNOWN_WEAK, ids=[k[0] for k in KNOWN_WEAK]
)
def test_known_weak_still_passes(name: str, make, cid: str, phrasing: str) -> None:
    assert passed(make(), checks(cid, phrasing)), (
        f"{name}: now caught; move it to ADVERSARIAL and update README"
    )
