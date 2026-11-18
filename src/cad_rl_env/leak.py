"""Keep train parts away from eval parts: compare check-list signatures.

A signature is the sorted bbox midpoints plus the (kind, diameter midpoint, count)
of every hole/bore/boss check. Two signatures collide when every bbox side is within
BBOX_TOL and the features match one-to-one within DIA_TOL. A missing bbox on either
side compares features only (stricter, never looser).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

BBOX_TOL = 1.0
DIA_TOL = 0.5

Signature = tuple[tuple[float, ...], tuple[tuple[str, float, int], ...]]


def _mid(rng: list[float | None] | tuple[float | None, ...]) -> float:
    lo, hi = rng
    if lo is None and hi is None:
        return 0.0
    if lo is None:
        return float(hi)  # type: ignore[arg-type]
    if hi is None:
        return float(lo)
    return round((lo + hi) / 2, 4)


def signature(checks: list[dict[str, Any]]) -> Signature:
    bbox: tuple[float, ...] = ()
    feats: list[tuple[str, float, int]] = []
    for c in checks:
        kind = c["type"]
        if kind == "bbox_sorted":
            bbox = tuple(_mid(r) for r in c["ranges"])
        elif kind == "hole_set":
            feats.append(("hole", _mid(c["diameter"]), int(_mid(c["count"]))))
        elif kind in ("bore", "boss"):
            feats.append((kind, _mid(c["diameter"]), 1))
    return bbox, tuple(sorted(feats))


def collides(a: Signature, b: Signature) -> bool:
    (bbox_a, feats_a), (bbox_b, feats_b) = a, b
    if bbox_a and bbox_b and any(abs(x - y) > BBOX_TOL for x, y in zip(bbox_a, bbox_b)):
        return False
    if len(feats_a) != len(feats_b):
        return False
    return all(
        ka == kb and na == nb and abs(da - db) <= DIA_TOL
        for (ka, da, na), (kb, db, nb) in zip(feats_a, feats_b)
    )


@lru_cache(maxsize=1)
def eval_signatures() -> tuple[Signature, ...]:
    from simready.bench.concepts import load_concepts

    return tuple(
        signature([c.model_dump(mode="json") for c in concept.checks.explicit])
        for concept in load_concepts()
        if concept.in_vocab
    )
