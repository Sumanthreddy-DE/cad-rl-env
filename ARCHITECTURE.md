# Architecture: cad-rl-env

## Domain

[What this system does in 2-3 sentences. Business context, not tech.]

## Layer model

Dependency direction flows top → bottom. Upper layers import lower; never reverse.

```
[Domain]
├── types/     — pure types, no logic, no imports from sibling layers
├── config/    — constants, env. Imports: types/
├── repo/      — data access. Imports: types/, config/
├── service/   — business logic. Imports: types/, config/, repo/
├── api/       — HTTP handlers. Imports: service/
└── ui/        — frontend. Imports: types/, api/ (via generated client)
```

Cross-cutting (auth, logging, telemetry) → inject via provider/DI, not direct import.

## Package map

```
cad_rl_env/          — the verifiers environment package (flat layout, see decision log)
├── grade.py         — adapter over simready.bench.attempt
├── prompt.py, task.py, taskset.py — prompt, CadTask, CadTaskset, CadEnv
├── leak.py          — train/eval leak checks
└── generator/       — seeded train-task generator over ISO 273 tables
```

## Enforced invariants

- No circular imports (enforced by: _add linter_)
- Layer boundaries respected (enforced by: _add linter_)
- Max file size: 800 lines
- No magic strings for domain values (enums/const maps only)

## Decision log

<!-- YYYY-MM-DD — decision — why (one line each) -->
2026-11-19 — package moved from `src/cad_rl_env/` to `cad_rl_env/` — `prime env push` 0.22.1 rejects a repo with no root-level `.py` file or package.
