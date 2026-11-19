"""Keep train parts away from eval parts: compare check-list signatures.

A signature is the sorted bbox midpoints plus the (kind, diameter midpoint, count)
of every hole/bore/boss check. Two signatures collide when every constrained bbox side
is within BBOX_TOL and the canonical features match one-to-one within DIA_TOL. A missing
bbox on either side compares features only (stricter, never looser).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

BBOX_TOL = 1.0
DIA_TOL = 0.5

Feature = tuple[str, float, int]
Signature = tuple[tuple[float | None, ...], tuple[Feature, ...]]


def _mid(rng: list[float | None] | tuple[float | None, ...]) -> float | None:
    lo, hi = rng
    if lo is None and hi is None:
        return None
    if lo is None:
        return float(hi)  # type: ignore[arg-type]
    if hi is None:
        return float(lo)
    return round((lo + hi) / 2, 4)


def signature(checks: list[dict[str, Any]]) -> Signature:
    bbox: tuple[float | None, ...] = ()
    feats: list[Feature] = []
    for c in checks:
        kind = c["type"]
        if kind == "bbox_sorted":
            bbox = tuple(_mid(r) for r in c["ranges"])
        elif kind == "hole_set":
            dia = _mid(c["diameter"])
            cnt = _mid(c["count"])
            feats.append(("hole", dia if dia is not None else 0.0, int(cnt) if cnt is not None else 1))
        elif kind == "bore":
            dia = _mid(c["diameter"])
            feats.append(("bore", dia if dia is not None else 0.0, 1))
        elif kind == "boss":
            dia = _mid(c["diameter"])
            cnt_rng = c.get("count")
            cnt = (
                int(_mid(cnt_rng))  # type: ignore[arg-type]
                if cnt_rng and cnt_rng[0] is not None and cnt_rng[1] is not None
                else 1
            )
            feats.append(("boss", dia if dia is not None else 0.0, cnt))
    return bbox, tuple(sorted(feats))


def _circular_envelope_dia(bbox: tuple[float | None, ...]) -> float | None:
    concrete = [x for x in bbox if x is not None]
    for i in range(len(concrete)):
        for j in range(i + 1, len(concrete)):
            if abs(concrete[i] - concrete[j]) <= DIA_TOL:
                return round((concrete[i] + concrete[j]) / 2, 4)
    return None


def _strip_redundant_envelope_boss(
    bbox: tuple[float | None, ...], feats: tuple[Feature, ...]
) -> tuple[Feature, ...]:
    """Drop a single outer-OD boss redundant with a circular bbox when inner features exist.

    Some concepts_v1 parts (e.g. c007 flange) check bbox_sorted + hub boss + bore + holes
    without repeating a boss check on the outer disc OD, while the flange generator emits
    both boss checks. Canonicalizing that outer envelope boss prevents false non-collisions.
    """
    env_dia = _circular_envelope_dia(bbox)
    if env_dia is None or len(feats) <= 1:
        return feats
    out: list[Feature] = []
    removed = False
    for kind, dia, cnt in feats:
        if not removed and kind == "boss" and cnt == 1 and abs(dia - env_dia) <= DIA_TOL:
            removed = True
            continue
        out.append((kind, dia, cnt))
    return tuple(out)


def collides(a: Signature, b: Signature) -> bool:
    (bbox_a, raw_a), (bbox_b, raw_b) = a, b
    if bbox_a and bbox_b:
        if len(bbox_a) != len(bbox_b):
            return False
        if any(
            x is not None and y is not None and abs(x - y) > BBOX_TOL
            for x, y in zip(bbox_a, bbox_b)
        ):
            return False
    feats_a = _strip_redundant_envelope_boss(bbox_a, raw_a)
    feats_b = _strip_redundant_envelope_boss(bbox_b, raw_b)
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
