from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from typing import Any

import pytest
from simready.bench.attempt import (
    CHECKER_ERROR,
    INFRA,
    evaluate_reply,
    parse_reply,
    validate_grade_result,
)

from text_to_cad_rl.generator import FAMILIES, make_task
from text_to_cad_rl.grade import GraderError, grade_reply
from text_to_cad_rl.leak import collides, eval_signatures, signature

N = 1000
SUBPROCESS_SPOT_N = 15
WORKERS = min(6, os.cpu_count() or 4)


def _in_worker_grade(spec_dict: dict[str, Any], checks: list[Any], expected_solids: int) -> dict[str, Any]:
    from simready.bench.checker import grade_shape
    from simready.bench.concepts import CHECK_LIST
    from simready.bench.geometry import fingerprint
    from simready.gen.build import build_shape
    from simready.gen.spec import PartSpec

    spec = PartSpec.model_validate(spec_dict)
    validated_checks = CHECK_LIST.validate_python(checks)
    shape = build_shape(spec)
    if shape is None or shape.IsNull():
        return {"stage": "build_error", "error": "build produced a null shape"}
    report = grade_shape(shape, validated_checks, expected_solids)
    return validate_grade_result({"stage": "graded", "fingerprint": fingerprint(shape), "report": report})


def _grade_in_pool(args: tuple[str, int, str, str, list[dict[str, Any]]]) -> tuple[int, str, bool, str | None]:
    _, i, phrasing, reply_text, checks = args
    report = evaluate_reply(parse_reply([], reply_text), checks, 1, grade=_in_worker_grade)
    if report["failure_class"] in (INFRA, CHECKER_ERROR) or report["passed"] is None:
        raise GraderError(f"{report['failure_class']}: {report.get('error')}")
    return (i, phrasing, bool(report["passed"]), report["failure_class"])


@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_bulk_self_check_and_no_leak(family: str) -> None:
    sigs = eval_signatures()
    jobs: list[tuple[str, int, str, str, list[dict[str, Any]]]] = []
    spot_tasks = []
    for i in range(N):
        explicit = make_task(0, family, i, "explicit")
        assert not any(collides(signature(explicit.checks), e) for e in sigs), (family, i)
        for phrasing in ("explicit", "engineer"):
            t = explicit if phrasing == "explicit" else make_task(0, family, i, phrasing)
            reply_text = "```json\n" + json.dumps(t.reference_spec) + "\n```"
            jobs.append((family, i, phrasing, reply_text, t.checks))
            if i < SUBPROCESS_SPOT_N and phrasing == "explicit":
                spot_tasks.append((i, reply_text, t.checks))

    failures: list[tuple[int, str, str | None]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        futures = [pool.submit(_grade_in_pool, job) for job in jobs]
        for fut in futures:
            i, phrasing, ok, fc = fut.result(timeout=60.0)
            if not ok:
                failures.append((i, phrasing, fc))

    for i, reply_text, checks in spot_tasks:
        out = grade_reply(reply_text, checks)
        if not out["passed"]:
            failures.append((i, "explicit_subprocess", out["failure_class"]))

    assert failures == [], failures[:10]
