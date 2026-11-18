# cad-rl-env

`cad-rl-env` is a [verifiers v1](https://github.com/PrimeIntellect-ai/verifiers) reinforcement-learning and evaluation environment for single-shot text-to-CAD. Given a natural-language mechanical-part specification, the model replies in one turn with a JSON `PartSpec` in [SimReady](https://github.com/Sumanthreddy-DE/simready)'s 4-operation constructive solid geometry DSL; SimReady's OpenCASCADE (OCCT) B-rep geometry checker builds the solid in an isolated subprocess and grades dimensional and feature conformance directly on the 3D shape. No model-written Python code is ever executed.

## Task format

Each rollout is single-turn (`single_turn` stop hook). The model receives the DSL specification in the system prompt and a mechanical-part request in the user prompt, and replies with one fenced ```` ```json ```` block containing `{"steps": [...]}` (1 to 16 steps):

- `{"op": "box", "dx": W, "dy": D, "dz": H, "at": [x, y, z]}` — axis-aligned box (`at` = minimum corner, default `[0, 0, 0]`)
- `{"op": "cyl", "r": R, "h": H, "at": [x, y, z]}` — cylinder along `+Z` (`at` = base centre, default `[0, 0, 0]`)
- `{"op": "fuse", "a": i, "b": j}` — boolean union of earlier steps `i` and `j`
- `{"op": "cut", "a": i, "b": j}` — boolean subtraction of step `j` from step `i`

Schema bounds (millimetres): `dx`, `dy`, `dz`, `h` in `(0, 1000]`; `r` in `(0, 500]`; `at` coordinates in `[-10000, 10000]`. Indices `a` and `b` must be distinct 0-based indices strictly earlier than the boolean step, every non-final step must be consumed by a later `fuse` or `cut` (no orphan primitives), and the final step must be a single watertight solid.

## Splits

| Split | Size | Phrasings | Source |
|---|---|---|---|
| `eval` | 111 tasks (fixed) | `explicit`, `engineer`, `intent` | SimReady `concepts_v1.jsonl` (37 in-vocab mechanical parts × 3 phrasings), loaded from the installed `simready` package. **Held-out test split — never train on `eval`.** |
| `train` | Configurable (`n_train`, default `1000`; `5000` in `configs/train.toml`) | `explicit`, `engineer` | Deterministic seeded generator (`seed:family:index`) across `plate_holes`, `bushing`, and `flange` using ISO 273 clearance tables, filtered by a geometric signature leak guard (`leak.py`) so no generated part collides with any `eval` concept. |

 Verified via `vf-validate` and the bulk self-check suite:
- `eval`: **111/111** gold checks pass (`passed = 1.0`) and **111/111** untouched noop checks score `0.0`.
- `train`: **6,000/6,000** seeded tasks (`1,000` indices × `3` families × `2` phrasings) pass their own OCCT checks with **0** `eval` signature collisions.

## Reward

| Signal | Decorator | Default weight | Train config weight | Definition |
|---|---|---|---|---|
| `passed` | `@vf.reward` | `1.0` | `1.0` | `1.0` iff the B-rep validity gate passes (valid B-rep, 0 free edges, positive volume, expected solid count) **and** every gradable check passes; else `0.0`. |
| `partial` | `@vf.reward` | `0.0` | `0.5` | `0.0` if pre-build or validity gate fails; else `n_pass / (n_pass + n_fail)` across the task's geometric checks (`not_checked` entries excluded). |
| `funnel` | `@vf.metric` | observational | observational | Per-stage funnel indicators: `parse_ok`, `schema_ok`, `build_ok`, `gate_ok`, `checks_passed`, `checks_failed`. |

Infrastructure and checker faults (`infra_error`, `checker_error`, missing OCCT bindings, child crashes, or startup timeouts before worker readiness) raise `GraderError` so `verifiers` records an execution error rather than assigning a false `0.0` reward to the policy.

## Failure classes

Every failed model attempt receives one primary failure class from SimReady's SP1 attempt pipeline (first failing stage wins):

| Class | Stage | Meaning |
|---|---|---|
| `F1_no_parse` | Parse | Reply contains no parseable JSON `PartSpec` (`{"steps": [...]}`). |
| `F2_schema` | Schema | JSON violates `PartSpec` bounds, types, or the 16-step cap. |
| `F3_reference` | Schema | Forward/self boolean index (`a >= i`, `b >= i`, `a == b`) or unreferenced orphan step. |
| `F9_out_of_vocab` | Schema | Step requests an operation outside `{"box", "cyl", "fuse", "cut"}` (e.g. `fillet`). |
| `F4_build_error` | OCCT build | Native OpenCASCADE construction/boolean kernel error or null shape. |
| `F8_timeout` | OCCT build | Worker exceeds `build_timeout_s` (default `60.0` s) after startup readiness. |
| `F5_invalid_solid` | Validity gate | Invalid B-rep, open shell (`free_edges > 0`), non-positive volume, or wrong solid count (e.g. disjoint fuse). |
| `F6_dimension_miss` | Checker | First failing check is `bbox_sorted`, `bbox_min_dim`, `volume`, `bore`, `boss`, or `any_of`. |
| `F7_feature_miss` | Checker | First failing check is `hole_set` (hole count, diameter, through/blind state, or pairwise pitch pattern). |

## Reward-hacking table

Measured directly by `tests/test_redteam.py` against the installed `simready[bench]` checker (`CHECKER_VERSION = "1.0.0-dev"`):

| Category | Case (`tests/test_redteam.py`) | Target | `passed` | `partial` | `failure_class` |
|---|---|---|---|---|---|
| Adversarial | `c001 plate without holes` | `c001-explicit` | `False` | `0.333` | `F6_dimension_miss` |
| Adversarial | `c001 three of four holes` | `c001-explicit` | `False` | `0.667` | `F7_feature_miss` |
| Adversarial | `c001 blind instead of through` | `c001-explicit` | `False` | `0.667` | `F7_feature_miss` |
| Adversarial | `c001 hole cuts miss the part` | `c001-explicit` | `False` | `0.667` | `F7_feature_miss` |
| Adversarial | `c001 duplicate overlapping hole cut` | `c001-explicit` | `False` | `0.667` | `F7_feature_miss` |
| Adversarial | `c001 second disjoint solid` | `c001-explicit` | `False` | `0.000` | `F5_invalid_solid` |
| Adversarial | `c004 solid rod, no bore` | `c004-explicit` | `False` | `0.667` | `F6_dimension_miss` |
| Adversarial | `c004 square tube, right envelope` | `c004-explicit` | `False` | `0.667` | `F6_dimension_miss` |
| Adversarial | `c014 plate dimensions swapped across axes` | `c014-explicit` | `False` | `0.500` | `F7_feature_miss` |
| Generated adversarial | `generated flange without hub (h=0.5)` | `flange-0-explicit` | `False` | `0.600` | `F6_dimension_miss` |
| Generated adversarial | `generated plate wrong hole pitch (+1.0 mm)` | `plate_holes-0-explicit` | `False` | `0.500` | `F7_feature_miss` |
| Generated adversarial | `generated bushing wrong bolt size (+1.5 mm r)` | `bushing-0-engineer` | `False` | `0.667` | `F6_dimension_miss` |
| Known weak check | `c010 intent: blank plate passes (slots unseen)` | `c010-intent` | `True` | `1.000` | `None` |
| Known weak check | `c023 intent: plain 30 mm cube passes (steps unseen)` | `c023-intent` | `True` | `1.000` | `None` |
| Known weak check | `c047 explicit (OOV): axial hole passes cross-drilled shaft checks` | `c047-explicit` *(excluded from `eval`)* | `True` | `1.000` | `None` |
| Known weak check | `c048 intent: solid cube passes (no gradable check)` | `c048-intent` | `True` | `1.000` | `None` |

The four `KNOWN_WEAK` rows document real v1 checker blind spots identified during SimReady's SP1 Task 2 review (partial-cylinder stadium slots, ungraded step treads on open-ended `intent` prompts, hole axis orientation on out-of-vocab `c047`, and `c048-intent` having only `not_checked`). They are asserted in `tests/test_redteam.py` so any upstream check addition in SimReady immediately surfaces here; note also why `eval` keeps `partial` weight at `0.0` (several failed cheats achieve `partial` between `0.333` and `0.667` by matching bounding-box envelopes alone).

## Results

Live model evaluation runs (`pass@1`, `pass@5`, `explicit - engineer` gap, and failure-class distribution) are scheduled in Phase C (Task C1) once API run budgets are approved. SimReady's own SP1 benchmark runner (`bench_v1.md`, tool-calling channel) remains the canonical benchmark record; `cad-rl-env` evaluates the single-turn text-JSON channel used for RL rollouts.

## Install and run

### Local install (pip / uv only — no conda required)

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

### Prime Intellect Environments Hub install (once published)

```bash
prime env install <owner>/cad-rl-env
```

### Validate gold and noop controls (no model calls)

```bash
# All 111 held-out eval tasks
.venv/bin/vf-validate cad-rl-env --runtime.type subprocess --taskset.split eval -c 6

# Sample of 30 procedurally generated train tasks
.venv/bin/vf-validate cad-rl-env --runtime.type subprocess --taskset.split train --taskset.n-train 30 -c 6
```

### Run evaluations or dry-run configs

```bash
# Verify resolved config without spending API tokens
.venv/bin/vf-run @ configs/eval.toml --dry-run
.venv/bin/vf-run @ configs/train.toml --dry-run

# Evaluate an OpenAI-compatible model endpoint on the eval split
.venv/bin/vf-run @ configs/eval.toml -m <model-id> --client.base-url <https://api.example.com/v1> --client.api-key-var <API_KEY_ENV_VAR> -r 5 -o outputs
```

### Run tests and architecture lint

```bash
bash scripts/lint-arch.sh
.venv/bin/python -m pytest -q -m "not slow"
.venv/bin/python -m pytest -q -m slow tests/test_generator_bulk.py
```

## Provenance

- **Upstream checker & benchmark:** [Sumanthreddy-DE/simready](https://github.com/Sumanthreddy-DE/simready), installed via `simready[bench] @ git+https://github.com/Sumanthreddy-DE/simready@efe3f15cf7ce92fd2ad8bc85ef402c28091e8b50` (`CHECKER_VERSION = "1.0.0-dev"`, OpenCASCADE via pip `cadquery-ocp`). `cad-rl-env` imports `simready.bench` and `simready.gen` directly and copies neither check logic nor `concepts_v1.jsonl`.
- **Trusted DSL execution:** Follows SimReady ADR 0001 — models emit schema-validated JSON (`PartSpec`) executed by `simready.gen.build.build_shape` inside a killable `spawn` worker with atomic mailbox supervision.
- **Engineering standards in `concepts_v1` and `train`:** ISO 273:1979 fastener clearance holes (`M2.5`–`M8` in `train`), plus ISO 286-2 limits and fits, ISO 7089 / DIN 125-1 washers, DIN 705 shaft collars, DIN 988 shim rings, ISO 4379 sleeve bushes, EN 60423 cable glands, NEMA 17 / NEMA 23 motor mounts, 608 bearings, HIWIN MGN12H carriages, Arduino Uno R3, and Raspberry Pi 4 mechanical drawings in `eval`.
