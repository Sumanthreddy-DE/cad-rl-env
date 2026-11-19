from __future__ import annotations

import json
import multiprocessing as mp

import pytest
from simready.bench import attempt
from simready.bench.concepts import load_concepts

from cad_rl_env.grade import GraderError, grade_reply
from tests._worker_stubs import broken_checker_worker, post_ready_hang

C001 = next(c for c in load_concepts() if c.concept_id == "c001")
CHECKS = [c.model_dump(mode="json") for c in C001.checks.explicit]
REF = C001.reference_spec


def reply(spec: dict) -> str:
    return "Here is the part:\n```json\n" + json.dumps(spec) + "\n```"


def test_reference_spec_passes() -> None:
    out = grade_reply(reply(REF), CHECKS)
    assert out["passed"] is True
    assert out["failure_class"] is None
    assert out["partial"] == 1.0
    assert out["checker_version"]


def test_empty_reply_is_no_parse() -> None:
    assert grade_reply(None, CHECKS)["failure_class"] == "F1_no_parse"
    assert grade_reply("I cannot do that.", CHECKS)["failure_class"] == "F1_no_parse"


def test_negative_dimension_is_schema() -> None:
    bad = {"steps": [{"op": "box", "dx": -1, "dy": 1, "dz": 1}]}
    assert grade_reply(reply(bad), CHECKS)["failure_class"] == "F2_schema"


def test_orphan_step_is_reference() -> None:
    bad = {"steps": [{"op": "box", "dx": 1, "dy": 1, "dz": 1}, {"op": "box", "dx": 2, "dy": 2, "dz": 2}]}
    assert grade_reply(reply(bad), CHECKS)["failure_class"] == "F3_reference"


def test_unknown_op_is_out_of_vocab() -> None:
    bad = {"steps": [{"op": "box", "dx": 1, "dy": 1, "dz": 1}, {"op": "fillet", "a": 0, "r": 1}]}
    assert grade_reply(reply(bad), CHECKS)["failure_class"] == "F9_out_of_vocab"


def test_disjoint_solids_fail_gate() -> None:
    two = {
        "steps": [
            {"op": "box", "dx": 42, "dy": 42, "dz": 5},
            {"op": "box", "dx": 5, "dy": 5, "dz": 5, "at": [100, 0, 0]},
            {"op": "fuse", "a": 0, "b": 1},
        ]
    }
    assert grade_reply(reply(two), CHECKS)["failure_class"] == "F5_invalid_solid"


def test_wrong_plate_size_is_dimension_miss() -> None:
    wrong = json.loads(json.dumps(REF))
    wrong["steps"][0]["dx"] = 50
    assert grade_reply(reply(wrong), CHECKS)["failure_class"] in {"F6_dimension_miss", "F7_feature_miss"}


def test_startup_timeout_raises_grader_error() -> None:
    with pytest.raises(GraderError, match="infra_error"):
        grade_reply(reply(REF), CHECKS, timeout_s=0.001)


def test_running_timeout_is_f8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attempt, "_grade_worker", post_ready_hang)
    before = {p.pid for p in mp.active_children()}
    out = grade_reply(reply(REF), CHECKS, timeout_s=1.0)
    assert out["failure_class"] == "F8_timeout"
    assert out["passed"] is False and out["partial"] == 0.0
    assert {p.pid for p in mp.active_children()} == before


def test_checker_error_and_invalid_checks_raise_grader_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(GraderError):
        grade_reply(reply(REF), [{"type": "not_a_real_check"}])

    monkeypatch.setattr(attempt, "_grade_worker", broken_checker_worker)
    with pytest.raises(GraderError, match="checker_error"):
        grade_reply(reply(REF), CHECKS, timeout_s=5.0)


def test_same_reply_same_grade() -> None:
    a, b = grade_reply(reply(REF), CHECKS), grade_reply(reply(REF), CHECKS)
    assert a == b


def test_missing_binding_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import cad_rl_env.grade as g

    monkeypatch.setattr(g.occ_compat, "BACKEND", None)
    with pytest.raises(GraderError):
        grade_reply(reply(REF), CHECKS)


def test_unhashable_op_is_schema_failure_not_type_error() -> None:
    for bad_op in (["box"], {"type": "box"}):
        out = grade_reply(reply({"steps": [{"op": bad_op, "dx": 1, "dy": 1, "dz": 1}]}), CHECKS)
        assert out["failure_class"] == "F2_schema"
        assert out["passed"] is False and out["partial"] == 0.0


def test_deeply_nested_json_is_no_parse_not_recursion_error() -> None:
    nested = '{"spec":' * 1200 + '{"steps":[]}' + "}" * 1200
    out = grade_reply("```json\n" + nested + "\n```", CHECKS)
    # simready >= 0.5.1 skips the too-deep prefixes and recovers the inner empty spec.
    assert out["failure_class"] == "F2_schema"
    assert out["passed"] is False and out["partial"] == 0.0


def test_non_string_reply_raises_type_error() -> None:
    for bad_reply in (123, [], {}):
        with pytest.raises(TypeError):
            grade_reply(bad_reply, CHECKS)  # type: ignore[arg-type]

