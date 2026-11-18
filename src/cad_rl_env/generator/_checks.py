"""Builders for SimReady check dicts. Shapes only; the check semantics are SimReady's."""

from __future__ import annotations

import itertools
import math
from typing import Any

EXPLICIT_TOL = 0.1  # SP1 design §1 rule 3: ±0.1 mm on stated numbers
PATTERN_TOL_ENGINEER = 0.2  # design choice, as in concepts_v1 (not a standard value)


def tol(x: float, t: float) -> list[float]:
    return [round(x - t, 4), round(x + t, 4)]


def bbox_sorted(dims: list[float], t: float = EXPLICIT_TOL) -> dict[str, Any]:
    return {"type": "bbox_sorted", "ranges": [tol(d, t) for d in sorted(dims)]}


def pairwise(points: list[tuple[float, float]]) -> list[float]:
    """Sorted pairwise centre distances (SimReady hole_set ``pattern``)."""
    return sorted(math.dist(a, b) for a, b in itertools.combinations(points, 2))


def hole_set(
    diameter: list[float], points: list[tuple[float, float]], pattern_tol: float
) -> dict[str, Any]:
    n = len(points)
    return {
        "type": "hole_set",
        "diameter": diameter,
        "count": [n, n],
        "through": True,
        "pattern": [tol(d, pattern_tol) for d in pairwise(points)],
    }


def cut_chain(base: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    """``base`` minus each tool in turn: [base, t1, cut(0,1), t2, cut(2,3), ...]."""
    steps: list[dict[str, Any]] = [base]
    last = 0
    for tool in tools:
        steps.append(tool)
        steps.append({"op": "cut", "a": last, "b": len(steps) - 1})
        last = len(steps) - 1
    return {"steps": steps}
