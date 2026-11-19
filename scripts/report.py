"""Summarise `vf-run` eval traces into the README § Results table.

Usage: .venv/bin/python scripts/report.py outputs/c1/*/traces.jsonl [--max-tokens 16384]

Errored episodes (`ok = false`, e.g. GraderError or provider failure) are excluded from
scoring and counted, matching SimReady bench_v1's infra/checker exclusion.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(paths: list[Path]) -> list[dict[str, Any]]:
    traces = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    traces.extend(json.loads(line)["traces"])
    return traces


def _model(trace: dict[str, Any]) -> str:
    return trace["run"]["name"].split("--")[1]


def summarise(traces: list[dict[str, Any]], max_tokens: int) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        by_model[_model(t)].append(t)

    out = {}
    for model, rows in by_model.items():
        scored = [t for t in rows if t["ok"]]
        passed = [(t["task"]["data"], t["rewards"]["passed"]["score"]) for t in scored]
        per_task: dict[str, float] = defaultdict(float)
        per_phrasing: dict[str, list[float]] = defaultdict(list)
        for data, p in passed:
            per_task[data["id"]] = max(per_task[data["id"]], p)
            per_phrasing[data["phrasing"]].append(p)
        by_phrasing = {ph: sum(v) / len(v) for ph, v in per_phrasing.items()}
        failures = Counter(
            t["info"]["grade"]["failure_class"]
            for t in scored
            if t["info"]["grade"]["failure_class"] is not None
        )
        out[model] = {
            "episodes": len(rows),
            "excluded": len(rows) - len(scored),
            "pass@1": sum(p for _, p in passed) / len(passed) if passed else 0.0,
            "pass@k": sum(per_task.values()) / len(per_task) if per_task else 0.0,
            "by_phrasing": by_phrasing,
            "gap": by_phrasing.get("explicit", 0.0) - by_phrasing.get("engineer", 0.0),
            "failures": dict(failures),
            "hit_token_cap": sum(t["num_output_tokens"] >= max_tokens for t in scored),
            "mean_output_tokens": (
                sum(t["num_output_tokens"] for t in scored) / len(scored) if scored else 0.0
            ),
        }
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def markdown(summary: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| model | episodes | pass@1 | pass@k | explicit | engineer | intent "
        "| gap (explicit - engineer) | hit token cap | excluded |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, s in sorted(summary.items(), key=lambda kv: -kv[1]["pass@1"]):
        ph = s["by_phrasing"]
        lines.append(
            f"| `{model}` | {s['episodes']} | {_pct(s['pass@1'])} | {_pct(s['pass@k'])} "
            f"| {_pct(ph.get('explicit', 0))} | {_pct(ph.get('engineer', 0))} "
            f"| {_pct(ph.get('intent', 0))} | {_pct(s['gap'])} "
            f"| {s['hit_token_cap']} | {s['excluded']} |"
        )
    classes = sorted({fc for s in summary.values() for fc in s["failures"]})
    lines += ["", "| model | " + " | ".join(f"`{c}`" for c in classes) + " |",
              "|---|" + "---|" * len(classes)]
    for model, s in sorted(summary.items(), key=lambda kv: -kv[1]["pass@1"]):
        lines.append(f"| `{model}` | " + " | ".join(
            str(s["failures"].get(c, 0)) for c in classes) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--max-tokens", type=int, default=16384)
    args = ap.parse_args()
    print(markdown(summarise(load(args.traces), args.max_tokens)))


if __name__ == "__main__":
    main()
