"""Grade one model reply against a check list, through SimReady's SP1 attempt pipeline.

No geometry or check logic lives here: parsing, schema validation, the build
subprocess, the gate and the checks are all ``simready.bench``. This module only
adapts the call and turns grader/infrastructure faults into ``GraderError``, so
a broken grader errors the rollout instead of scoring a wrong model.
"""

from __future__ import annotations

from collections.abc import Hashable
from functools import partial
from typing import Any

from pydantic import ValidationError
from simready import occ_compat
from simready.bench import CHECKER_VERSION
from simready.bench.attempt import (
    CHECKER_ERROR,
    F2,
    INFRA,
    ParsedReply,
    evaluate_reply,
    grade_spec,
    parse_reply,
)
from simready.bench.concepts import CHECK_LIST
from simready.gen.spec import PartSpec


class GraderError(RuntimeError):
    """The grader failed; the rollout must error, not score 0.0."""


def _has_unhashable_op(spec: dict[str, Any]) -> bool:
    steps = spec.get("steps")
    if not isinstance(steps, list):
        return False
    return any(
        isinstance(s, dict) and "op" in s and not isinstance(s["op"], Hashable)
        for s in steps
    )


def grade_reply(
    reply: str | None,
    checks: list[dict[str, Any]],
    expected_solids: int = 1,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """SP1 grade report for ``reply`` (failure classes F1-F9, ``partial``, ``passed``)."""
    if reply is not None and not isinstance(reply, str):
        raise TypeError(f"reply must be str or None, got {type(reply).__name__}")
    if occ_compat.BACKEND is None or not occ_compat.OCC_AVAILABLE:
        raise GraderError("no OCC binding on the evaluator: install cadquery-ocp")
    try:
        validated_checks = CHECK_LIST.validate_python(checks)
    except ValidationError as exc:
        raise GraderError(f"invalid task checks: {exc}") from exc

    try:
        parsed = parse_reply([], reply if reply is not None else "")
    except RecursionError:
        parsed = ParsedReply(None, None, 0, "JSON nesting exceeded recursion limit")

    if parsed.spec is not None and _has_unhashable_op(parsed.spec):
        try:
            PartSpec.model_validate(parsed.spec)
        except ValidationError as exc:
            messages = [
                f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}"
                for e in exc.errors()
            ]
        else:  # pragma: no cover
            messages = ["unhashable step op"]
        return {
            "via": parsed.via,
            "n_tool_calls": parsed.n_tool_calls,
            "spec": parsed.spec,
            "schema_errors": messages,
            "build_error": None,
            "fingerprint": None,
            "check_report": None,
            "partial": 0.0,
            "passed": False,
            "failure_class": F2,
            "error": None,
            "error_type": None,
            "checker_version": CHECKER_VERSION,
        }

    report = evaluate_reply(
        parsed,
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
