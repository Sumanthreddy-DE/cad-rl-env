# cad-rl-env: design

**Status:** done
**Last verified:** 2026-11-19

A verifiers v1 environment for single-shot text-to-CAD. The model writes a part as a JSON
`PartSpec` (SimReady's 4-op DSL). SimReady's SP1 checker grades the part. The environment is
published on Prime Intellect's Environments Hub as proof of work for a Mercor domain-expert
application (computational engineering).

Brainstormed and approved section by section with the user on 2026-11-14. The kickoff brief
is an internal working note and is not published. Facts below were checked against disk and live docs on
2026-11-14; corrections to the brief are listed in § Brief corrections.

## Brief corrections (verified 2026-11-14)

1. **verifiers v0 is deprecated.** `load_environment` / `vf-eval` is "v0", listed on the Hub as
   Legacy. Build on v1: `import verifiers.v1 as vf` (PyPI `verifiers` 0.4.1, Python >=3.11),
   `Taskset` / `Task` / `TaskData`, `@vf.reward`, `@vf.metric`, `vf-validate`, `vf-run`.
   Publishing is unchanged: `prime env push`.
2. **The SP1 checker does not exist yet.** On `sp1-bench` (`58a6906`, worktree
   `/tmp/simready-sp1-review`), `simready/bench/` holds only `concepts.py` and
   `prompts/concepts_v1.jsonl`. Plan Tasks 3 (geometry) and 4 (checker) are code inside
   `docs/exec-plans/active/2026-10-04-sp1-bench-implementation.md`, not modules. The checker on
   `main` (`checks.py`, `analyze_file_safe`) checks simulation readiness, not spec conformance.
3. **"111 reference specs" is 37 specs × 3 check lists.** `concepts_v1` has 50 concepts:
   37 in-vocab with a `reference_spec`, 13 out-of-vocab. The 111/111 result came from a scratch
   evaluator that copied the plan's Task 3/4 code (record:
   `2026-11-12-sp1-task2-review.md` on `sp1-bench`). It must be re-confirmed on shipped code.
4. **SimReady is not pip-installable.** No `pyproject.toml`; `requirements.txt` pulls torch,
   streamlit and sentence-transformers.
5. **pythonocc-core is not on PyPI in usable form** (PyPI has 0.16; real releases are
   conda-forge only). Hub environment dependencies install with pip/uv on the evaluator. The
   pip route to OCCT is `cadquery-ocp` (8.0.1.1.0, `OCP.*` namespace).

## Decisions

### D1. Reward runs on the evaluator through an OCP backend (user, 2026-11-14)

The reward is a `@vf.reward` Python hook on the evaluator. SimReady's new geometry module
(SP1 Task 3) is written against a small OCC/OCP shim so it runs under both pythonocc (SimReady's
`sr` env) and `cadquery-ocp` (this environment). `simready/gen/build.py` moves its three OCC
imports to the same shim.

**Why:** the model emits validated JSON, never code (SimReady ADR 0001), so evaluator-side
grading executes nothing untrusted. Pip-only install means `vf-validate` works locally without
Docker and Hub users need no image. The same answer settles SimReady's `sp3-base-model-choice`
question (can the reward run in Colab).

**Rejected:** a grading container image with micromamba + pythonocc. It keeps SimReady unchanged
but adds a public image to maintain, needs `docker`/`prime` runtimes for every test, and slows RL
rollouts, all to isolate code the model never sends.

### D2. Two splits: generated train, `concepts_v1` eval (user, 2026-11-14)

- `eval`: the 37 in-vocab `concepts_v1` concepts × 3 phrasings (explicit, engineer, intent) =
  111 prompts, read from the installed `simready` package. Never trained on.
- `train`: a seeded procedural generator over three families the v1 check types already cover:
  `plate_holes` (standard hole patterns: ISO 273 clearances, NEMA, Raspberry Pi), `bushing`
  (bore + OD + length), `flange` (stacked cylinders + cuts).

**Why:** SP1's concepts are SimReady's frozen test set, and SP3's metric (single-shot pass@1 on
held-out prompts, SP1 design D6) needs them unseen. A generator also acts as the hidden-variant
guard against hard-coded answers.

**Rejected:** eval only (SP3 has nothing to train on; weaker proof of RL-environment skill);
training on perturbed `concepts_v1` specs (leaks the benchmark).

### D3. Single turn, JSON `PartSpec` in the final reply (user, 2026-11-14)

**Why:** matches SP1 D6 and SP3's fixed metric, gives clean failure classes, and pass@k comes
free from rollout groups.

**Deferred to v0.2:** a tool-calling `build_part` repair loop as a config flag, with feedback
limited to the validity gate (check-level feedback would leak the hidden ranges). Tracks
SimReady's `sp1-repair-mode`.

**Rejected:** free-form build123d/CadQuery code. The reward would run model-written code, which
breaks ADR 0001 and forces container grading (D1).

### D4. Reward = binary `passed` + configurable `partial` (user, 2026-11-14)

- `passed` at weight 1.0: gate passes and every check passes (SP1 design §1 rule 4).
- `partial` = gate × fraction of checks passed, weight from `TaskConfig.partial_weight`
  (default 0.0; training configs set it above 0).
- Funnel metrics at weight 0: `parse_ok`, `schema_ok`, `build_ok`, `gate_ok`, failure class,
  checks passed / total.

**Why:** eval numbers stay pure pass@1, comparable with SP1. Dense signal is available when a
small model's pass@k is near 0, where GRPO gets no gradient.

**Rejected:** binary only (risks a dead signal for a 3B model); partial only (the Task 2 review
found weak checks such as c010's blank plate, so partial credit rewards dropping features).
**Rejected:** IoU against the reference. Checks are placement-invariant and leave free shape
choices `not_checked`, so many correct parts exist; IoU would punish valid alternatives.

### D5. Sequencing (user, 2026-11-14)

1. SimReady: `gencad-strong-baseline`, then submit the MecAgent test
   (`apply-submit-mecagent-test`). A pending test in a sent application outranks this repo.
2. SimReady on `sp1-bench`: SP1 Task 3 (with the OCP shim and the bbox fix), Task 4, a minimal
   `pyproject.toml`, a tag.
3. This repo.


**Rejected:** building this repo now against a stubbed checker interface. Task 3/4 signatures
are plan code until written, so the interface would drift.

### D6. Ownership

- **SimReady owns:** check types, the checker, the geometry module, the bbox fix
  (`sp1-task3-bbox-gap`), `concepts_v1`, the DSL.
- **cad-rl-env owns:** the verifiers wrapper, the train generator and its standards tables, the
  red-team suite, Hub packaging, model runs through verifiers.
- **SP3 trains against cad-rl-env.**
- SP1's own runner (`bench_v1.md`, SP1 Task 11) stays the canonical benchmark number.

Recording this in SimReady's `BACKLOG.md` (on `main`, per SimReady's branch rule) is pending the
user's approval.

## Architecture

The repo root is the package (`pyproject.toml`, `src/cad_rl_env/`); `prime env push` runs from
the root.

| Unit | Job | Depends on |
|---|---|---|
| `taskset.py` | `CadTaskset(TasksetConfig: split="train"\|"eval", seed, n_train, families, phrasings)`; `load()` returns `CadTask`s | `generator`, `simready.bench.concepts` |
| `task.py` | `CadTask`: `@vf.reward passed`, `@vf.reward partial`, funnel `@vf.metric`s; `validate()` grades the task's own `reference_spec` | `grade` |
| `grade.py` | pure `grade(reply: str, checks) -> GradeResult`: extract JSON, `PartSpec` check, build in a subprocess with timeout, gate, checks. No geometry logic: adapts `simready.bench` | `simready.gen`, `simready.bench` |
| `generator/` | one module per family; each takes `rng`, returns `(prompt, checks, reference_spec)`; train-only standards tables | `simready.bench` check schema |
| `prompt.py` | system prompt stating the DSL grammar; import SP1 Task 6's prompt module instead if it ships one | — |

Dependencies: `verifiers`, `simready @ git+https://github.com/Sumanthreddy-DE/simready@<tag>`
(core + a `bench` extra, PEP 508 form in `dependencies`, not `[tool.uv.sources]`),
`cadquery-ocp`, `pydantic`.

Boundary rules:
- No range, tolerance or check logic in this repo except the generator's train tables. A new
  check type is a SimReady change and a new tag.
- `eval` reads `concepts_v1.jsonl` from the installed package (SimReady's `pyproject` must ship
  it as package data), never from a copy.

To verify before planning:
- `simready.bench` and `simready.gen` run on Python >=3.11 (`sr` env is 3.10).
- `cadquery-ocp` and pythonocc are not installed into one env (both bundle OCCT); SimReady's
  test matrix gets a separate OCP env.

## Data flow

1. `load()` builds `TaskData`. Visible: `prompt` (system grammar + part request). Hidden:
   `checks`, `reference_spec`, `split`, `family` or `concept_id`, `phrasing`, `params`. Extra
   `TaskData` fields stay on the evaluator and are never put into the prompt (verifiers docs).
2. The model replies once.
3. `grade(reply, checks)` returns a `GradeResult`: stage reached, failure class, per-check
   results, `passed`, `partial`.
4. Rewards and metrics read the `GradeResult`, computed once and cached on `trace.state`; a
   summary is copied to `trace.info` for saved traces.

### Generator

- **Reproducible:** task *i* uses `rng = Random(hash((seed, family, i)))`; a run is replayable
  from `(seed, n_train)`.
- **Phrasings:** train generates explicit and engineer only. Engineer prompts name the standard
  ("M4 clearance holes on a 40 mm square pattern"); the generator's table turns it into the
  range. Intent needs human-written text and stays eval-only.
- **Templates:** 4–6 hand-written sentence templates per family, chosen at random.
- **Leak guard:** reject a task whose family and key parameters match an eval reference part
  (sorted extents + hole pattern within tolerance). A test runs the guard over a large seed range
  and asserts zero collisions.
- **Self-check:** every generated task builds its own `reference_spec` and passes its own checks
  (in `validate()` and in a bulk test), as in SP1 design §1 rule 7.

**Rejected:** LLM paraphrasing of train prompts (model dependency, nondeterminism, cost per task).

## Error handling

A failure the model caused scores 0.0 with a class:

| Class | Trigger |
|---|---|
| `no_json` | no parseable JSON. Rule: last fenced ```` ```json ```` block, else the whole reply |
| `schema` | `PartSpec` validation fails (bounds, >16 steps, orphan steps, bad indices) |
| `build_error` | OCC raises during the build |
| `build_timeout` | build subprocess exceeds `TaskConfig.build_timeout_s` (default 30 s) |
| `gate_fail` | invalid BRep, open shell, volume <=0, wrong solid count |
| `check_fail` | gate passes, at least one check fails; `partial` still gets the fraction |

Infrastructure failures raise instead of scoring 0.0 (verifiers docs): OCP import fails, the
subprocess dies for reasons outside the spec, `simready` cannot load the concepts.

`build_timeout` on a `reference_spec` is a grader bug and fails the bulk test. On a model reply it
counts as a model failure and its rate is reported; a nonzero rate gets investigated.

Determinism: the same reply always gives the same `GradeResult` (needed for re-scoring saved runs).

Existing bounds: DSL caps (dimensions <=1000 mm, <=16 steps); the gate (blocks empty-shell
cheats); `sampling.max_tokens` in the default eval config so runaway output fails cheaply (all
GenCAD failures were runaway generations).

## Testing

Every test has a known pass and a known fail.

- `grade()` units: one fixed reply per failure class, plus a correct reply.
- Eval bulk: all 37 in-vocab reference specs pass their explicit and engineer checks
  (re-confirms the 111 on shipped code).
- Train bulk: 1,000 seeded tasks per family pass their own self-check; leak guard 0 collisions.
- Determinism: a fixed reply set graded twice, results equal.
- `vf-validate` on both splits with the `subprocess` runtime.

### Red-team suite

These adversarial specs must each score `passed = 0`: holes dropped, wrong hole count, envelope-only
box, a cut that misses the solid, duplicate overlapping cylinders, two disjoint solids, dimensions
swapped across axes. Results go in a README table.

Known weak checks from the Task 2 review are listed as known passes, not hidden: c010 (blank
plate), c023 (any 30 mm block), c047 (axial-hole fake), c048 (no gradable check). Fixing them is
SimReady's job through new check types.

## Model runs

- GLM 5.3 and DeepSeek V4.1 Flash on Vultr (ids verified in SimReady's backlog,
  `sp1-resume` item (a2)). Eval split, k = 5 rollouts, about 1,110 calls.
- Report pass@1, pass@5, the explicit − engineer gap (SP1 D3), and the failure funnel.
- Check the Vultr balance first (credit was $5 on 2026-10-05; SimReady `vultr-budget`).
- Cross-check: these numbers must agree with SP1's `bench_v1.md` on the same models within
  sampling noise. A mismatch means one harness has a bug.

## Publishing

1. `prime env push --visibility=PRIVATE`; `vf-run` smoke from the Hub install.
2. Credential and PII history audit (global Working Rule 3).
3. GitHub repo and Hub environment go public together.
4. Add the link to the Mercor application.

## Done when

1. Every test in § Testing passes.
2. The red-team table is in the README.
3. Runs exist for both models, and they agree with `bench_v1.md`.
4. The environment is public on the Hub and installs with pip alone.

## Out of scope for v0.1

- Tool-calling repair loop (v0.2 flag, D3).
- Out-of-vocab concepts (SP2's DSL v2).
- New check types (SimReady).
- Train families beyond the three in D2, until real runs show where models fail.

## Sources (fetched 2026-11-14)

- <https://docs.primeintellect.ai/tutorials-environments/create.md>
- <https://docs.primeintellect.ai/verifiers/overview.md>
- <https://docs.primeintellect.ai/verifiers/building-environments.md>
- <https://docs.primeintellect.ai/verifiers/tasksets.md>
- <https://docs.primeintellect.ai/verifiers/runtimes.md>
- <https://docs.primeintellect.ai/verifiers/architecture.md>
- PyPI JSON API: `pythonocc-core` 0.16, `cadquery-ocp` 8.0.1.1.0, `verifiers` 0.4.1
