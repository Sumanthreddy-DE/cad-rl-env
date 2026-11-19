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

Measured directly by `tests/test_redteam.py` against the installed `simready[bench]` checker (`CHECKER_VERSION = "1.0.0"`):

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
| Adversarial | `c047 explicit (OOV): axial hole instead of cross hole` | `c047-explicit` *(excluded from `eval`)* | `False` | `0.750` | `F6_dimension_miss` |
| Generated adversarial | `generated flange without hub (h=0.5)` | `flange-0-explicit` | `False` | `0.600` | `F6_dimension_miss` |
| Generated adversarial | `generated plate wrong hole pitch (+1.0 mm)` | `plate_holes-0-explicit` | `False` | `0.500` | `F7_feature_miss` |
| Generated adversarial | `generated bushing wrong bolt size (+1.5 mm r)` | `bushing-0-engineer` | `False` | `0.667` | `F6_dimension_miss` |
| Known weak check | `c010 intent: blank plate passes (slots unseen)` | `c010-intent` | `True` | `1.000` | `None` |
| Known weak check | `c023 intent: plain 30 mm cube passes (steps unseen)` | `c023-intent` | `True` | `1.000` | `None` |
| Known weak check | `c048 intent: solid cube passes (no gradable check)` | `c048-intent` | `True` | `1.000` | `None` |

The three `KNOWN_WEAK` rows document real v1 checker blind spots identified during SimReady's SP1 Task 2 review (partial-cylinder stadium slots, ungraded step treads on open-ended `intent` prompts, and `c048-intent` having only `not_checked`). A fourth, the axial-hole cheat on out-of-vocab `c047`, is caught since SimReady `v0.5.1` and moved to the adversarial rows. The `KNOWN_WEAK` rows are asserted in `tests/test_redteam.py` so any upstream check addition in SimReady immediately surfaces here; note also why `eval` keeps `partial` weight at `0.0` (several failed cheats achieve `partial` between `0.333` and `0.750` by matching bounding-box envelopes alone).

## Results

Run on 2026-11-19 with `configs/eval.toml`: all 111 `eval` tasks, `k = 5` rollouts each (`pass@k` is `pass@5`), `temperature = 0.7`, `max_tokens = 16384`, Vultr Inference (`https://api.vultrinference.com/v1`), `simready` `v0.5.1`, `CHECKER_VERSION = "1.0.0"`. Gap CIs are paired 95% bootstrap intervals over the 37 concepts. Reproduce the tables with `.venv/bin/python scripts/report.py outputs/<run>/traces.jsonl`.

| model | episodes | pass@1 | pass@k | explicit | engineer | intent | gap (explicit - engineer) | gap 95% CI | hit token cap | excluded |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-v4.1-flash-1017` | 555 | 91.5% | 98.2% | 100.0% | 85.9% | 88.6% | 14.1% | [5.9%, 23.8%] | 3 | 0 |
| `glm-5.3-flash` | 555 | 86.8% | 96.4% | 98.9% | 76.2% | 85.4% | 22.7% | [11.9%, 34.6%] | 16 | 0 |
| `qwen3.8-27b` | 555 | 74.4% | 79.3% | 96.8% | 57.3% | 69.2% | 39.5% | [22.7%, 55.7%] | 44 | 0 |
| `nemotron-3-nano-omni-30b-a3b-reasoning` | 555 | 43.6% | 66.7% | 71.7% | 23.2% | 35.9% | 48.5% | [35.1%, 61.1%] | 7 | 2 |

| model | `F1_no_parse` | `F2_schema` | `F3_reference` | `F5_invalid_solid` | `F6_dimension_miss` | `F7_feature_miss` | `F9_out_of_vocab` |
|---|---|---|---|---|---|---|---|
| `deepseek-v4.1-flash-1017` | 3 | 1 | 0 | 0 | 28 | 15 | 0 |
| `glm-5.3-flash` | 16 | 0 | 1 | 0 | 34 | 22 | 0 |
| `qwen3.8-27b` | 44 | 0 | 0 | 0 | 64 | 34 | 0 |
| `nemotron-3-nano-omni-30b-a3b-reasoning` | 61 | 2 | 19 | 12 | 139 | 78 | 1 |

**Reading the results**

- **The environment has headroom below the frontier.** SimReady's [`bench_v1`](https://github.com/Sumanthreddy-DE/simready/blob/v0.5.1/docs/validation/bench_v1.md) measured the same 111 prompts and checker on two frontier models: `gpt-6.1-sol` 96.2% and `gemini-4-argon` 95.5% `pass@1`. The small open models here range from 43.6% to 91.5%.
- **The explicit - engineer gap opens on smaller models.** At the frontier the gap is 3.2% and 4.9%, and both CIs include zero. Here every model's gap CI excludes zero, and the gap grows as `pass@1` falls (14.1% to 48.5%). Explicit prompts state every dimension; engineer prompts name standards and fits (for example the `D12` bore of a DIN 988 shim) that the model must resolve to numbers. That resolution step is the skill this environment rewards.
- **Most failures are geometry, not format.** `F6_dimension_miss` and `F7_feature_miss` dominate for every model. Every `F1_no_parse` for `deepseek-v4.1-flash-1017`, `glm-5.3-flash` and `qwen3.8-27b` is a reply that hit the 16384-token cap during reasoning (the "hit token cap" column). For `nemotron-3-nano-omni-30b-a3b-reasoning`, 7 of its 61 `F1_no_parse` replies hit the cap; the rest returned no parsable `PartSpec`.
- **Excluded episodes:** 2 of 2,220. One was a provider `502`. The other exposed a SimReady checker bug: a spec that cuts a shape from itself builds an empty solid, and the checker raises `checker_error` instead of returning `F5_invalid_solid`. It is filed upstream as `empty-solid-checker-error`.
- **Answer-channel difference:** `bench_v1` collects answers through a tool call; `cad-rl-env` reads the `PartSpec` from the single-turn text reply used for RL rollouts. The prompts, checks and checker are the same, but the numbers are not a like-for-like comparison.

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

- **Upstream checker & benchmark:** [Sumanthreddy-DE/simready](https://github.com/Sumanthreddy-DE/simready), installed via `simready[bench] @ git+https://github.com/Sumanthreddy-DE/simready@v0.5.1` (`CHECKER_VERSION = "1.0.0"`, OpenCASCADE via pip `cadquery-ocp`). `cad-rl-env` imports `simready.bench` and `simready.gen` directly and copies neither check logic nor `concepts_v1.jsonl`.
- **Trusted DSL execution:** Follows SimReady ADR 0001 — models emit schema-validated JSON (`PartSpec`) executed by `simready.gen.build.build_shape` inside a killable `spawn` worker with atomic mailbox supervision.
- **Engineering standards in `concepts_v1` and `train`:** ISO 273:1979 fastener clearance holes (`M2.5`–`M8` in `train`), plus ISO 286-2 limits and fits, ISO 7089 / DIN 125-1 washers, DIN 705 shaft collars, DIN 988 shim rings, ISO 4379 sleeve bushes, EN 60423 cable glands, NEMA 17 / NEMA 23 motor mounts, 608 bearings, HIWIN MGN12H carriages, Arduino Uno R3, and Raspberry Pi 4 mechanical drawings in `eval`.
