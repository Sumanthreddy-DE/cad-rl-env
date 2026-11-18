from __future__ import annotations

import asyncio
import copy
import json

import pytest
import verifiers.v1 as vf
from verifiers.v1.trace import TraceTask

from cad_rl_env.taskset import CadConfig, CadTaskset


def load(**kw):
    return CadTaskset(CadConfig(split="eval", **kw)).load()


def _trace_with_reply(task, text: str | None) -> vf.Trace:
    trace = vf.Trace(
        task=TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
        state=vf.State(),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="test", trainable=False),
        root_reply=text,
    )
    return trace


def test_eval_is_111_in_vocab_prompts() -> None:
    tasks = load()
    assert len(tasks) == 111
    assert len({t.data.id for t in tasks}) == 111


def test_eval_ids_and_hidden_fields() -> None:
    t = next(t for t in load() if t.data.id == "c001-explicit")
    assert t.data.split == "eval" and t.data.phrasing == "explicit"
    assert t.data.reference_spec["steps"][0]["op"] == "box"
    assert "reference_spec" not in t.data.prompt and "31" in t.data.prompt
    assert "500" in (t.data.system_prompt or "") and "1000" in (t.data.system_prompt or "")


def test_eval_phrasing_filter() -> None:
    assert len(load(phrasings=["engineer"])) == 37


def test_eval_rejects_unknown_phrasing() -> None:
    with pytest.raises(ValueError):
        load(phrasings=["bogus"])


def test_task_rewards_funnel_and_validate() -> None:
    t = next(t for t in load() if t.data.id == "c001-explicit")
    assert asyncio.run(t.validate(None)) is True

    gold_reply = "```json\n" + json.dumps(t.data.reference_spec) + "\n```"
    tr_pass = _trace_with_reply(t, gold_reply)
    asyncio.run(t.score(tr_pass))
    assert tr_pass.rewards["passed"].score == 1.0
    assert tr_pass.rewards["passed"].weight == 1.0
    assert tr_pass.rewards["partial"].score == 1.0
    assert tr_pass.rewards["partial"].weight == 0.0
    assert tr_pass.reward == 1.0
    assert tr_pass.metrics == {
        "parse_ok": 1.0,
        "schema_ok": 1.0,
        "build_ok": 1.0,
        "gate_ok": 1.0,
        "checks_passed": 3.0,
        "checks_failed": 0.0,
    }

    # Partial credit: plate without the 4 mounting holes passes bbox + centre bore (2/3)
    partial_spec = copy.deepcopy(t.data.reference_spec)
    partial_spec["steps"] = partial_spec["steps"][:3]
    tr_partial = _trace_with_reply(t, "```json\n" + json.dumps(partial_spec) + "\n```")
    asyncio.run(t.score(tr_partial))
    assert tr_partial.rewards["passed"].score == 0.0
    assert tr_partial.rewards["partial"].score == pytest.approx(2.0 / 3.0)
    assert tr_partial.metrics["gate_ok"] == 1.0
    assert tr_partial.metrics["checks_passed"] == 2.0
    assert tr_partial.metrics["checks_failed"] == 1.0

    # Untouched / empty reply scores 0.0
    tr_empty = _trace_with_reply(t, None)
    asyncio.run(t.score(tr_empty))
    assert tr_empty.reward == 0.0
    assert tr_empty.metrics["parse_ok"] == 0.0


def test_single_turn_stop_hook() -> None:
    t = next(t for t in load() if t.data.id == "c001-explicit")
    tr = _trace_with_reply(t, None)
    assert asyncio.run(t.single_turn(tr)) is False
