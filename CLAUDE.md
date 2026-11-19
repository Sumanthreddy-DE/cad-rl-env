# text-to-cad-rl

A verifiers v1 environment for single-shot text-to-CAD, where the model writes a part as a JSON `PartSpec` (SimReady's 4-op DSL) and SimReady's SP1 checker grades the part.

## Navigation

| Resource | Purpose |
|---|---|
| `ARCHITECTURE.md` | Domain model, layers, dependency rules |
| `docs/design-docs/core-beliefs.md` | Non-negotiable operating principles |
| `docs/design-docs/index.md` | All design docs + verification status |
| `docs/exec-plans/active/` | In-progress execution plans |
| `docs/exec-plans/completed/` | Finished plans (historical record) |
| `docs/references/` | External API docs, vendor llms.txt |

## Stack

- **Language**: Python 3.12
- **Framework**: verifiers v1
- **Database**: none
- **Package manager**: uv

## Dev commands

```bash
# Test:     .venv/bin/python -m pytest -q -m "not slow"
# Lint:     bash scripts/lint-arch.sh
# Validate: .venv/bin/vf-validate text-to-cad-rl --runtime.type subprocess -n 5
```

## Rules (project overrides)

- **exec-plans**: Plans save to `docs/exec-plans/active/YYYY-MM-DD-<slug>.md`. This overrides the write-plan skill default (`docs/superpowers/plans/`). Move to `docs/exec-plans/completed/` when task is done.
- **lint**: Before committing, run `bash scripts/lint-arch.sh`. Do not suppress violations without an inline comment explaining why.
- **design-docs/index.md**: Add a row whenever a new design doc is created. Keep Status column current (✓ current / ⚠ needs review / ✗ outdated).
