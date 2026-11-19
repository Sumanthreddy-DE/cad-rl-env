from __future__ import annotations

import json

import pytest

from text_to_cad_rl.generator import FAMILIES, make_task
from text_to_cad_rl.grade import grade_reply


def test_three_families_registered() -> None:
    assert set(FAMILIES) == {"plate_holes", "bushing", "flange"}


@pytest.mark.parametrize("family", ["bushing", "flange"])
@pytest.mark.parametrize("index", range(15))
@pytest.mark.parametrize("phrasing", ["explicit", "engineer"])
def test_reference_passes_own_checks(family: str, index: int, phrasing: str) -> None:
    t = make_task(0, family, index, phrasing)
    assert len(t.reference_spec["steps"]) <= 16
    out = grade_reply("```json\n" + json.dumps(t.reference_spec) + "\n```", t.checks)
    assert out["passed"], (family, t.params, out["failure_class"], out["check_report"])


def test_flange_engineer_names_pitch_circle() -> None:
    t = make_task(0, "flange", 0, "engineer")
    assert "pitch circle" in t.prompt and "ISO 273" in t.prompt
