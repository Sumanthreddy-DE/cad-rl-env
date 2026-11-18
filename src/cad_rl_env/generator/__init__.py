"""Seeded train-task generator. A task is fully determined by (seed, family, index)."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PHRASINGS = ("explicit", "engineer")
MAX_DRAWS = 50  # redraws allowed when a sample collides with an eval part


@dataclass(frozen=True)
class FamilySample:
    params: dict[str, Any]
    reference_spec: dict[str, Any]
    prompts: dict[str, str]
    checks: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class GeneratedTask:
    family: str
    index: int
    phrasing: str
    prompt: str
    checks: list[dict[str, Any]]
    reference_spec: dict[str, Any]
    params: dict[str, Any]


def task_rng(seed: int, family: str, index: int) -> random.Random:
    # A str seed is hashed deterministically by random.Random (unlike hash(), which
    # Python salts per process).
    return random.Random(f"{seed}:{family}:{index}")


def _families() -> dict[str, Callable[[random.Random], FamilySample]]:
    from cad_rl_env.generator import plate_holes

    return {"plate_holes": plate_holes.generate}


FAMILIES = _families()


def make_task(seed: int, family: str, index: int, phrasing: str) -> GeneratedTask:
    from cad_rl_env.leak import collides, eval_signatures, signature

    if phrasing not in PHRASINGS:
        raise ValueError(f"train phrasing must be one of {PHRASINGS}, got {phrasing!r}")
    rng = task_rng(seed, family, index)
    sigs = eval_signatures()
    for _ in range(MAX_DRAWS):
        sample = FAMILIES[family](rng)
        if not any(collides(signature(sample.checks["explicit"]), e) for e in sigs):
            return GeneratedTask(
                family,
                index,
                phrasing,
                sample.prompts[phrasing],
                sample.checks[phrasing],
                sample.reference_spec,
                sample.params,
            )
    raise RuntimeError(f"{family} seed={seed} index={index}: {MAX_DRAWS} draws all hit eval parts")
