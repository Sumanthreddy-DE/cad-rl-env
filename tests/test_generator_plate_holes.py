from __future__ import annotations

import json

import pytest
from simready.bench.concepts import load_concepts

from text_to_cad_rl.generator import make_task
from text_to_cad_rl.grade import grade_reply
from text_to_cad_rl.leak import collides, eval_signatures, signature


def test_same_seed_same_task() -> None:
    assert make_task(0, "plate_holes", 3, "explicit") == make_task(0, "plate_holes", 3, "explicit")
    assert make_task(0, "plate_holes", 3, "explicit") != make_task(0, "plate_holes", 4, "explicit")


def test_engineer_prompt_names_standard_not_diameter() -> None:
    t = make_task(0, "plate_holes", 0, "engineer")
    assert "ISO 273" in t.prompt and "clearance" in t.prompt


@pytest.mark.parametrize("index", range(20))
@pytest.mark.parametrize("phrasing", ["explicit", "engineer"])
def test_reference_passes_own_checks(index: int, phrasing: str) -> None:
    t = make_task(0, "plate_holes", index, phrasing)
    out = grade_reply("```json\n" + json.dumps(t.reference_spec) + "\n```", t.checks)
    assert out["passed"], (t.params, out["failure_class"], out["check_report"])


def test_no_collision_with_eval_in_200_tasks() -> None:
    sigs = eval_signatures()
    for i in range(200):
        t = make_task(0, "plate_holes", i, "explicit")
        assert not any(collides(signature(t.checks), e) for e in sigs)


def test_guard_positive_control() -> None:
    c001 = [
        c.model_dump(mode="json")
        for c in next(c for c in load_concepts() if c.concept_id == "c001").checks.explicit
    ]
    assert any(collides(signature(c001), e) for e in eval_signatures())
