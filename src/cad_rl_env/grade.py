"""Grade one model reply against a check list, through SimReady's SP1 attempt pipeline.

No geometry or check logic lives here: parsing, schema validation, the build
subprocess, the gate and the checks are all ``simready.bench``. This module only
adapts the call and turns grader/infrastructure faults into ``GraderError``, so
a broken grader errors the rollout instead of scoring a wrong model.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from pydantic import ValidationError
from simready import occ_compat
from simready.bench import CHECKER_VERSION
from simready.bench.attempt import CHECKER_ERROR, INFRA, evaluate_reply, grade_spec, parse_reply
from simready.bench.concepts import CHECK_LIST


class GraderError(RuntimeError):
    """The grader failed; the rollout must error, not score 0.0."""


def grade_reply(
    reply: str | None,
    checks: list[dict[str, Any]],
    expected_solids: int = 1,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """SP1 grade report for ``reply`` (failure classes F1-F9, ``partial``, ``passed``)."""
    if occ_compat.BACKEND is None or not occ_compat.OCC_AVAILABLE:
        raise GraderError("no OCC binding on the evaluator: install cadquery-ocp")
    try:
        validated_checks = CHECK_LIST.validate_python(checks)
    except ValidationError as exc:
        raise GraderError(f"invalid task checks: {exc}") from exc
    report = evaluate_reply(
        parse_reply([], reply or ""),
        validated_checks,
        expected_solids,
        grade=partial(grade_spec, timeout_s=timeout_s),
    )
    if (
        report["failure_class"] in (INFRA, CHECKER_ERROR)
        or report["passed"] is None
        or report["partial"] is None
    ):
        detail = report.get("error") or report.get("build_error") or "grader failed"
        kind = report.get("error_type") or "unknown"
        raise GraderError(f"{report['failure_class']} ({kind}): {detail}")
    return {**report, "checker_version": CHECKER_VERSION}
