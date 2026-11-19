from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "report", Path(__file__).resolve().parents[1] / "scripts" / "report.py"
)
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def episode(model: str, task_id: str, phrasing: str, passed: float, fc: str | None,
            out_tok: int = 100, ok: bool = True) -> dict:
    trace = {
        "run": {"name": f"cad-rl-env--{model}--null--abcd"},
        "task": {"data": {"id": task_id, "phrasing": phrasing}},
        "rewards": {"passed": {"score": passed, "weight": 1.0}},
        "info": {"grade": {"failure_class": fc}},
        "num_output_tokens": out_tok,
        "ok": ok,
    }
    return {"ok": ok, "traces": [trace]}


@pytest.fixture
def traces_file(tmp_path: Path) -> Path:
    rows = [
        episode("m", "c001-explicit", "explicit", 1.0, None),
        episode("m", "c001-explicit", "explicit", 0.0, "F6_dimension_miss"),
        episode("m", "c001-engineer", "engineer", 0.0, "F1_no_parse", out_tok=4096),
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_summarise_three_trace_fixture(traces_file: Path) -> None:
    s = report.summarise(report.load([traces_file]), max_tokens=4096)["m"]
    assert s["episodes"] == 3 and s["excluded"] == 0
    assert s["pass@1"] == pytest.approx(1 / 3)
    assert s["pass@k"] == pytest.approx(1 / 2)  # c001-explicit passed once, engineer never
    assert s["by_phrasing"] == {"explicit": 0.5, "engineer": 0.0}
    assert s["gap"] == pytest.approx(0.5)
    assert s["failures"] == {"F6_dimension_miss": 1, "F1_no_parse": 1}
    assert s["hit_token_cap"] == 1


def test_errored_episode_is_excluded_not_scored(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [episode("m", "c001-explicit", "explicit", 1.0, None),
            episode("m", "c002-explicit", "explicit", 0.0, None, ok=False)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    s = report.summarise(report.load([path]), max_tokens=4096)["m"]
    assert s["excluded"] == 1 and s["pass@1"] == 1.0


def test_markdown_table_has_one_row_per_model(traces_file: Path) -> None:
    md = report.markdown(report.summarise(report.load([traces_file]), max_tokens=4096))
    assert "| `m` | 3 | 33.3% | 50.0% |" in md
