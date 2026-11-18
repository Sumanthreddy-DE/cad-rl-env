"""One text-to-CAD task: single turn, graded once, cached on the trace."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from cad_rl_env.grade import grade_reply

_PARSE_FAILS = {"F1_no_parse"}
_SPEC_FAILS = _PARSE_FAILS | {"F2_schema", "F3_reference", "F9_out_of_vocab"}
_BUILD_FAILS = _SPEC_FAILS | {"F4_build_error", "F8_timeout"}


class CadData(vf.TaskData):
    checks: list[dict[str, Any]]
    """SimReady check dicts for this phrasing. Hidden from the model."""
    expected_solids: int = 1
    reference_spec: dict[str, Any] | None = None
    """Known-good PartSpec dict; graded by ``validate``."""
    split: str
    source_id: str
    """Concept id (eval) or ``<family>-<seed>-<index>`` (train)."""
    phrasing: str
    params: dict[str, Any] = Field(default_factory=dict)


class CadTaskConfig(vf.TaskConfig):
    build_timeout_s: float = 60.0


class CadTask(vf.Task[CadData, vf.State, CadTaskConfig]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    async def _grade(self, trace: vf.Trace) -> dict[str, Any]:
        from simready.bench import CHECKER_VERSION

        cached = trace.info.get("grade")
        # A checker change invalidates the cache, so rescoring after a fix regrades.
        if cached is None or cached.get("checker_version") != CHECKER_VERSION:
            cached = await asyncio.to_thread(
                grade_reply,
                trace.last_reply,
                self.data.checks,
                self.data.expected_solids,
                self.config.build_timeout_s,
            )
            trace.info["grade"] = cached
        return cached

    @vf.reward(weight=1.0)
    async def passed(self, trace: vf.Trace) -> float:
        return float((await self._grade(trace))["passed"])

    @vf.reward(weight=0.0)
    async def partial(self, trace: vf.Trace) -> float:
        return float((await self._grade(trace))["partial"])

    @vf.metric
    async def funnel(self, trace: vf.Trace) -> dict[str, float]:
        grade = await self._grade(trace)
        fc = grade["failure_class"]
        report = grade["check_report"] or {}
        return {
            "parse_ok": float(fc not in _PARSE_FAILS),
            "schema_ok": float(fc not in _SPEC_FAILS),
            "build_ok": float(fc not in _BUILD_FAILS),
            "gate_ok": float(bool(report) and report["gate"]["passed"]),
            "checks_passed": float(report.get("n_pass", 0)),
            "checks_failed": float(report.get("n_fail", 0)),
        }

    async def validate(self, runtime: vf.Runtime | None) -> bool | None:
        if self.data.reference_spec is None:
            return None
        reply = "```json\n" + json.dumps(self.data.reference_spec) + "\n```"
        grade = await asyncio.to_thread(
            grade_reply,
            reply,
            self.data.checks,
            self.data.expected_solids,
            self.config.build_timeout_s,
        )
        return bool(grade["passed"])
