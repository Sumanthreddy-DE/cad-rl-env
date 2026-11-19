# cad-rl-env Implementation Plan

**Status:** active
**Last verified:** 2026-11-18

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a verifiers v1 text-to-CAD environment on the Prime Intellect Environments Hub. The environment is graded by SimReady's SP1 checker, which it imports and never copies.

**Architecture:** Phase A happens in SimReady on branch `sp1-bench`. It adds an OCC/OCP shim and builds SP1 Tasks 3–5 on top of it (geometry, checker, attempt pipeline), with the bbox fix. It also adds a `pyproject.toml` and a tag. Phase B happens in this repo: a thin verifiers taskset that wraps `simready.bench.attempt`, a seeded train generator over 3 part families, a leak guard and a red-team suite. Phase C covers model runs and publishing, and the user runs most of it.

**Tech Stack:** Python 3.12 (evaluator), verifiers 0.4.1 (`verifiers.v1`), cadquery-ocp (OCP), pythonocc-core (SimReady `sr` env, conda-forge), pydantic 2, pytest, uv, micromamba, Prime CLI.

**Spec:** `docs/exec-plans/active/2026-11-14-cad-rl-env-design.md` (approved 2026-11-14). **SimReady plan this builds on:** `docs/exec-plans/active/2026-10-04-sp1-bench-implementation.md` on SimReady `sp1-bench`, called "the SP1 plan" below.

## Global Constraints

- Reward runs as `@vf.reward` on the evaluator, through an OCP backend in SimReady (spec D1). No grading container image.
- `eval` split = 37 in-vocab `concepts_v1` concepts × 3 phrasings = 111 prompts, read from the installed `simready` package. Never trained on (spec D2).
- `train` split = seeded generator over `plate_holes`, `bushing`, `flange`; explicit and engineer phrasings only (spec D2).
- Single turn; the model answers with a JSON `PartSpec` in its reply (spec D3).
- `passed` reward weight 1.0; `partial` reward weight 0.0 by default (spec D4).
- No range, tolerance or check logic in this repo, except the generator's train tables (spec § Architecture).
- `eval` reads `concepts_v1.jsonl` from the installed package, never from a copy.
- Infrastructure failures raise. They never score 0.0 (spec § Error handling).
- No LLM-written code is ever executed (SimReady ADR 0001). The executor is `simready.gen.build.build_shape`.
- DSL unchanged: `box`, `cyl` (+Z only), `fuse`, `cut`, at most 16 steps.
- Git dependencies use PEP 508 `pkg @ git+https://…` inside `[project].dependencies`, never `[tool.uv.sources]` (Hub requirement).
- No AI attribution in commits. Never `git push`: the user pushes. Stage and commit in the same Bash call.
- Installing software, creating envs, and anything outside the working repo is a separate ask (global Working Rule 5). Steps marked **USER GATE** stop until the user says yes.
- Never `Read` `.env`. API keys appear only as env-var names.
- SimReady rule: STATE.md and BACKLOG.md are edited on `main` only. Side branches merge `main` before their next session.

## Deviations from the spec (Rule 2: fixed here, reported)

1. **`grade.py` wraps SP1 Task 5 (`simready.bench.attempt`), not Task 3/4 directly.** SP1 Task 5 already provides text-JSON parsing, the spawn subprocess with timeout, and the failure classes. The spec's own rule ("no geometry or check logic here") means reusing it. So SimReady owes **Tasks 3, 4 and 5**, not only 3 and 4.
2. **Failure classes are SP1's, not the spec's names.** `F1_no_parse`, `F2_schema`, `F3_reference`, `F4_build_error`, `F5_invalid_solid`, `F6_dimension_miss`, `F7_feature_miss`, `F8_timeout`, `F9_out_of_vocab`, plus `checker_error` (raised as `GraderError` here). The spec's `no_json`/`schema`/`build_error`/`build_timeout`/`gate_fail`/`check_fail` map onto F1/F2+F3+F9/F4/F8/F5/F6+F7. A second set of names would drift.
3. **JSON extraction rule is SP1's:** the first fenced block that is a valid spec, else the first bare `{…}` that is one. The spec said "last fenced block". One rule in one place.
4. **Timeout default 60 s**, from SP1's `GRADE_TIMEOUT_S`. The spec said 30 s.
5. **`partial` weight uses verifiers' native config override**, not a custom `TaskConfig.partial_weight`. `[env.taskset.task.rewards] partial = { weight = 0.5 }` in `configs/train.toml` (verifiers docs § Plugging hooks via config).
6. **Seeded RNG uses `random.Random(f"{seed}:{family}:{index}")`, not `Random(hash((seed, family, i)))`.** Python salts `hash()` of strings per process (`PYTHONHASHSEED`), so the spec's form is not reproducible across runs.
7. **`simready/gen/build.py` has 4 OCC import lines, not 3** (BRepAlgoAPI, BRepPrimAPI, gp, STEPControl). In addition, `simready/occ_utils.py` must move to the shim, because SP1 Task 3 consumes `count_shapes`, `iter_faces` and `shape_bounding_box` from it.
8. **The cross-check against `bench_v1.md` is not a done criterion.** Per spec D5, this repo ships before SP1 Tasks 6–11 (runner, official runs). `bench_v1.md` will likely not exist yet, and SP1 answers through a `build_part` tool call while this environment answers in text. Task C1 compares the two only if `bench_v1.md` exists, and reports the answer channel as a known difference.
9. **SimReady dependency pin and extra (reconciled 2026-11-18).** Upstream Phase A shipped extra `bench` (not `ocp`) in `c255c60` and repaired subprocess/backend fault classification in `efe3f15cf7ce92fd2ad8bc85ef402c28091e8b50` (`sp1-bench`; `v0.5.0` predates that fix and `bench-api-0.1` was never tagged). Phase B pins `simready[bench] @ git+https://github.com/Sumanthreddy-DE/simready@efe3f15cf7ce92fd2ad8bc85ef402c28091e8b50` until immutable `v0.5.1` / `bench-checker-1.0.0` is published.
10. **Grader infrastructure classification (reconciled 2026-11-18).** SimReady `efe3f15` emits both `INFRA = "infra_error"` and `CHECKER_ERROR = "checker_error"` with `passed=None, partial=None` and details in `report["error"]` (`build_error` is `None`); pre-`ready` startup expiry (`timeout_s=0.001`) is `infra_error`, whereas post-`ready` expiry is `F8_timeout`. `grade_reply` raises `GraderError` on both unscored classes (and invalid task `checks`), and `tests/test_grade.py` tests both pre-ready `GraderError` and post-ready `F8_timeout`.
11. **`SYSTEM_PROMPT` exact `PartSpec` bounds (reconciled 2026-11-18).** `simready/gen/spec.py` bounds `dx, dy, dz, h` to `(0, 1000]`, `r` to `(0, 500]`, and `at` coordinates to `[-10000, 10000]` (matching SimReady `6d5fb47`). `prompt.py` states those exact bounds instead of "at most 1000" across all dimensions.

## Prerequisite gate (before Task A0)

- [x] SimReady `apply-submit-mecagent-test` is done (spec D5; closed 2026-11-15 in SimReady `a105642`).

## File map

**SimReady (`sp1-bench`):**

| File | Responsibility |
|---|---|
| `simready/occ_compat.py` (new) | one OCCT import surface: pythonocc or OCP |
| `simready/occ_utils.py` (modify) | imports via the shim |
| `simready/gen/build.py` (modify) | imports via the shim |
| `simready/bench/geometry.py` (new, SP1 Task 3) | gate, extents (bbox fix), cylindrical features, fingerprint |
| `simready/bench/checker.py` (new, SP1 Task 4) | `grade_shape` |
| `simready/bench/attempt.py` (new, SP1 Task 5) | `parse_reply`, `classify_spec`, `grade_spec`, `evaluate_reply` |
| `pyproject.toml` (new) | installable package, light core, extras |
| `tests/test_occ_compat.py` (new) | symbol parity under both backends |

**cad-rl-env:**

| File | Responsibility |
|---|---|
| `pyproject.toml`, `.gitignore`, `README.md` | package + Hub page |
| `src/cad_rl_env/__init__.py` | exports `CadTaskset` |
| `src/cad_rl_env/grade.py` | `grade_reply` adapter over `simready.bench.attempt` |
| `src/cad_rl_env/prompt.py` | `SYSTEM_PROMPT` |
| `src/cad_rl_env/task.py` | `CadData`, `CadTaskConfig`, `CadTask` (rewards, metrics, validate) |
| `src/cad_rl_env/taskset.py` | `CadConfig`, `CadTaskset.load()` for both splits |
| `src/cad_rl_env/leak.py` | check-list signatures + collision test |
| `src/cad_rl_env/generator/__init__.py` | `FamilySample`, `GeneratedTask`, `task_rng`, `FAMILIES`, `make_task` |
| `src/cad_rl_env/generator/_checks.py` | check-dict builders (`tol`, `bbox_sorted`, `pairwise`) |
| `src/cad_rl_env/generator/standards.py` | ISO 273 table |
| `src/cad_rl_env/generator/plate_holes.py`, `bushing.py`, `flange.py` | one family each |
| `configs/eval.toml`, `configs/train.toml` | run configs |
| `tests/` | one file per module + `test_redteam.py` |

## Execution notes (Orca)

Each task below is one brief: about an hour of code or more, with tests and a fixed file list. A0, B0, C1–C3 involve user gates, so run them inline in Claude Code, never through a worker. Per global Working Rule 4, re-run every command in a brief against the live repo the day it is handed off.

---

## Phase A: SimReady (`sp1-bench`) — COMPLETED UPSTREAM (2026-11-15..17, `efe3f15`)

> Completed upstream in `Mech/SimReady` on `sp1-bench`: OCC/OCP shim + Task 3 (`ca9f861`, `28afebe`), Task 4 (`5f830db`), Task 5 (`b36c625`), `pyproject.toml` with `bench` extra (`c255c60`), and attempt infrastructure repair (`efe3f15cf7ce92fd2ad8bc85ef402c28091e8b50`, 175 passed / 1 deselected under OCP). Do not re-run Phase A or merge `gpt/simready-pyproject`.

### Task A0: Workspace and environments (USER GATE)

**Files:** none in git.

- [x] **Step 1: Pick the worktree.** `/tmp/simready-sp1-review` is on `sp1-bench` at `58a6906`, but `/tmp` does not survive a reboot (SimReady S3 `sp1-review-worktree`). Ask the user: keep it, or `git -C ~/projects/Mech/SimReady worktree add ../SimReady-sp1 sp1-bench`. The second option creates a new top-level folder, so it needs its own yes. Record the choice as `$WT` in the task log.
- [x] **Step 2: Sync.** Run
  ```bash
  git -C "$WT" fetch origin && git -C "$WT" status -sb && git -C "$WT" rev-list --left-right --count sp1-bench...origin/sp1-bench
  ```
  Expected: clean, `0 0` (SimReady STATE says `58a6906` may be unpushed; if `1 0`, ask the user to push first). Then merge `main` (SimReady branch rule): `git -C "$WT" merge --no-edit origin/main`. If it conflicts, stop and ask.
- [x] **Step 3: USER GATE: create the sr env** (conda-forge pythonocc; no dotfile edits). It follows the claude-lab fresh-clone procedure `claude-lab/docs/exec-plans/completed/2026-11-03-simready-fresh-clone-test.md`:
  ```bash
  export MAMBA_ROOT_PREFIX=~/simready-fresh/mamba/root PATH=~/simready-fresh/mamba/bin:$PATH
  micromamba create -y -n simready -f "$WT/environment.yml"
  micromamba run -n simready python -c "import OCC.Core.gp, pydantic; print('ok')"
  ```
  Expected: `ok`. If `pydantic` is missing, `micromamba run -n simready pip install "pydantic>=2"` (it is not in `environment.yml`; it came in through `openai`).
- [x] **Step 4: USER GATE: create the OCP venv:**
  ```bash
  uv venv ~/simready-fresh/ocp --python 3.12
  uv pip install --python ~/simready-fresh/ocp/bin/python cadquery-ocp "pydantic>=2" pytest
  ~/simready-fresh/ocp/bin/python -c "import OCP.gp; print('ok')"
  ```
  Expected: `ok`. Never install `cadquery-ocp` into the sr env: both bundle OCCT.
- [x] **Step 5: Baseline.** Record both counts in the task log:
  ```bash
  cd "$WT" && micromamba run -n simready python -m pytest -q 2>&1 | tail -3
  ```
  The suite count is the regression baseline for A1.

Below, `SR` means `micromamba run -n simready python` and `OCPPY` means `~/simready-fresh/ocp/bin/python`, both run from `$WT`.

### Task A1: OCC/OCP shim; `occ_utils` and `gen/build` on it

**Files:**
- Create: `simready/occ_compat.py`
- Create: `tests/test_occ_compat.py`
- Modify: `simready/occ_utils.py` (import block lines 1–25, the two `breptools` fallbacks at lines 81–92, `topods.Face` at line 118)
- Modify: `simready/gen/build.py:31-45` (import block)

**Interfaces:**
- Produces: `BACKEND: str | None` (`"pythonocc"`, `"ocp"` or `None`); `occ(module: str) -> ModuleType`; `static(module: str, pythonocc_name: str, ocp_class: str | None = None) -> Any`. Env var `SIMREADY_OCC_BACKEND` forces a backend.

- [x] **Step 1: Write the failing parity test**

```python
# tests/test_occ_compat.py
"""Every OCCT symbol SimReady's bench + gen path uses resolves under the active backend."""
from __future__ import annotations

import pytest

from simready import occ_compat

pytestmark = pytest.mark.skipif(occ_compat.BACKEND is None, reason="no OCC binding")

CLASSES = [
    ("Bnd", "Bnd_Box"),
    ("BRep", "BRep_Builder"),
    ("BRepAdaptor", "BRepAdaptor_Curve"),
    ("BRepAdaptor", "BRepAdaptor_Surface"),
    ("BRepAlgoAPI", "BRepAlgoAPI_Cut"),
    ("BRepAlgoAPI", "BRepAlgoAPI_Fuse"),
    ("BRepCheck", "BRepCheck_Analyzer"),
    ("BRepClass3d", "BRepClass3d_SolidClassifier"),
    ("BRepGProp", "BRepGProp_Face"),
    ("BRepPrimAPI", "BRepPrimAPI_MakeBox"),
    ("BRepPrimAPI", "BRepPrimAPI_MakeCylinder"),
    ("GCPnts", "GCPnts_AbscissaPoint"),
    ("GeomAbs", "GeomAbs_Cylinder"),
    ("GProp", "GProp_GProps"),
    ("ShapeAnalysis", "ShapeAnalysis_FreeBounds"),
    ("STEPControl", "STEPControl_AsIs"),
    ("STEPControl", "STEPControl_Writer"),
    ("TopAbs", "TopAbs_EDGE"),
    ("TopAbs", "TopAbs_FACE"),
    ("TopAbs", "TopAbs_OUT"),
    ("TopAbs", "TopAbs_SOLID"),
    ("TopExp", "TopExp_Explorer"),
    ("TopTools", "TopTools_IndexedDataMapOfShapeListOfShape"),
    ("gp", "gp_Ax2"),
    ("gp", "gp_Dir"),
    ("gp", "gp_Pnt"),
    ("gp", "gp_Vec"),
]

# (module, pythonocc helper, OCP class, methods used)
STATICS = [
    ("BRepBndLib", "brepbndlib", "BRepBndLib", ["Add", "AddOptimal"]),
    ("BRepGProp", "brepgprop", "BRepGProp", ["VolumeProperties", "SurfaceProperties"]),
    ("BRepTools", "breptools", "BRepTools", ["UVBounds"]),
    ("TopExp", "topexp", "TopExp", ["MapShapesAndAncestors"]),
    ("TopoDS", "topods", "TopoDS", ["Face", "Solid", "Edge", "Shell"]),
]


@pytest.mark.parametrize(("module", "name"), CLASSES)
def test_class_resolves(module, name):
    assert getattr(occ_compat.occ(module), name) is not None


@pytest.mark.parametrize(("module", "py_name", "ocp_class", "methods"), STATICS)
def test_static_helper_methods_resolve(module, py_name, ocp_class, methods):
    helper = occ_compat.static(module, py_name, ocp_class)
    for method in methods:
        assert callable(getattr(helper, method)), f"{module}.{method}"


def test_box_volume_roundtrip():
    box = occ_compat.occ("BRepPrimAPI").BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    props = occ_compat.occ("GProp").GProp_GProps()
    occ_compat.static("BRepGProp", "brepgprop").VolumeProperties(box, props)
    assert props.Mass() == pytest.approx(6000.0)
```

- [x] **Step 2: Run under both backends to verify it fails**

Run: `SR -m pytest tests/test_occ_compat.py -q` and `OCPPY -m pytest tests/test_occ_compat.py -q`
Expected: both FAIL with `ImportError: cannot import name 'occ_compat' from 'simready'`.

- [x] **Step 3: Write the shim**

```python
# simready/occ_compat.py
"""One import surface for OpenCASCADE: pythonocc (``OCC.Core``) or OCP (``cadquery-ocp``).

pythonocc ships on conda-forge only; OCP ships as a pip wheel. The DSL executor and
the bench geometry import OCCT through this module, so the same code runs in the sr
env (pythonocc) and in a pip-only env such as the cad-rl-env evaluator (OCP).

    from simready.occ_compat import occ, static
    BRepPrimAPI_MakeBox = occ("BRepPrimAPI").BRepPrimAPI_MakeBox
    brepbndlib = static("BRepBndLib", "brepbndlib")
    brepbndlib.AddOptimal(shape, box, False, False)

``static`` returns pythonocc's lowercase helper object (``brepbndlib``) or, under OCP,
a proxy that maps ``Name`` to the class's ``Name_s`` static method.
Set ``SIMREADY_OCC_BACKEND=pythonocc|ocp`` to force one.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any

_PROBE = {"pythonocc": "OCC.Core.gp", "ocp": "OCP.gp"}
_PREFIX = {"pythonocc": "OCC.Core.", "ocp": "OCP."}


def _detect() -> str | None:
    forced = os.environ.get("SIMREADY_OCC_BACKEND")
    for name in [forced] if forced else ["pythonocc", "ocp"]:
        try:
            importlib.import_module(_PROBE[name])
            return name
        except ImportError:
            continue
    return None


BACKEND: str | None = _detect()


def occ(module: str) -> ModuleType:
    """The OCCT module ``module`` (e.g. ``"BRepPrimAPI"``) from the active backend."""
    if BACKEND is None:
        raise ImportError(
            "no OpenCASCADE binding: install pythonocc-core (conda-forge) or cadquery-ocp (pip)"
        )
    return importlib.import_module(_PREFIX[BACKEND] + module)


class _StaticProxy:
    """OCP exposes static methods as ``Class.Name_s``; present them as ``Name``."""

    def __init__(self, cls: Any) -> None:
        self._cls = cls

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._cls, name + "_s")
        except AttributeError:
            return getattr(self._cls, name)


def static(module: str, pythonocc_name: str, ocp_class: str | None = None) -> Any:
    """pythonocc helper ``pythonocc_name``, or an OCP proxy over class ``ocp_class``
    (default: the module name, which is how OCP names e.g. ``BRepBndLib``)."""
    mod = occ(module)
    if BACKEND == "pythonocc":
        return getattr(mod, pythonocc_name)
    return _StaticProxy(getattr(mod, ocp_class or module))
```

- [x] **Step 4: Run the parity test under both backends**

Run: `SR -m pytest tests/test_occ_compat.py -q` then `OCPPY -m pytest tests/test_occ_compat.py -q`
Expected: both PASS. If an OCP name differs (for example an enum not exported at module level), fix it in the shim with a named alias and add a comment naming the OCP version. Never special-case it in a caller.

- [x] **Step 5: Move `occ_utils.py` onto the shim.** Replace the `try: from OCC.Core… except ImportError: … = None` block with:

```python
from simready import occ_compat

if occ_compat.BACKEND is not None:  # pragma: no cover — needs an OCC binding
    Bnd_Box = occ_compat.occ("Bnd").Bnd_Box
    BRepAdaptor_Curve = occ_compat.occ("BRepAdaptor").BRepAdaptor_Curve
    brepbndlib = occ_compat.static("BRepBndLib", "brepbndlib")
    GCPnts_AbscissaPoint = occ_compat.occ("GCPnts").GCPnts_AbscissaPoint
    _topabs = occ_compat.occ("TopAbs")
    TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID = _topabs.TopAbs_EDGE, _topabs.TopAbs_FACE, _topabs.TopAbs_SOLID
    TopExp_Explorer = occ_compat.occ("TopExp").TopExp_Explorer
    topexp = occ_compat.static("TopExp", "topexp")
    topods = occ_compat.static("TopoDS", "topods")
    TopTools_IndexedDataMapOfShapeListOfShape = occ_compat.occ("TopTools").TopTools_IndexedDataMapOfShapeListOfShape
else:  # pragma: no cover — base env without OCC
    Bnd_Box = BRepAdaptor_Curve = brepbndlib = GCPnts_AbscissaPoint = None
    TopAbs_EDGE = TopAbs_FACE = TopAbs_SOLID = TopExp_Explorer = topexp = topods = None
    TopTools_IndexedDataMapOfShapeListOfShape = None
```

Keep every name the old block bound. Check with `git diff` that none was dropped. In `uv_bounds`, replace both `breptools` branches with `return tuple(float(v) for v in occ_compat.static("BRepTools", "breptools").UVBounds(face))`. Leave `shape_bounding_box` on `brepbndlib.Add`. The main pipeline's numbers depend on it, and the bbox fix belongs in `geometry.py` (Task A2).

- [x] **Step 6: Move `gen/build.py` onto the shim.** Replace lines 31–45 with the same pattern: `occ_compat.occ("BRepAlgoAPI").BRepAlgoAPI_Cut`, `…Fuse`, `occ_compat.occ("BRepPrimAPI").BRepPrimAPI_MakeBox`, `…MakeCylinder`, `occ_compat.occ("gp").gp_Ax2 / gp_Dir / gp_Pnt`, `occ_compat.occ("STEPControl").STEPControl_AsIs / STEPControl_Writer`. The `else` branch sets all nine names to `None`, as the old `except ImportError` branch did.

- [x] **Step 7: Regression check (sr env).** Run: `SR -m pytest -q 2>&1 | tail -3`. Expected: the same pass count as the A0 Step 5 baseline, plus the new parity tests, and 0 failed.

- [x] **Step 8: Build smoke under OCP**

Run:
```bash
OCPPY -c "from simready.gen.spec import PartSpec; from simready.gen.build import build_shape; s=build_shape(PartSpec.model_validate({'steps':[{'op':'box','dx':10,'dy':20,'dz':30}]})); print(s.IsNull())"
```
Expected: `False`. Run it from `$WT` so `simready` imports from the checkout.

- [x] **Step 9: Commit**

```bash
git -C "$WT" add simready/occ_compat.py tests/test_occ_compat.py simready/occ_utils.py simready/gen/build.py && git -C "$WT" commit -m "feat: occ_compat shim so gen and occ_utils run on pythonocc or OCP"
```

### Task A2: SP1 Task 3 (geometry module) on the shim, with the bbox fix

**Files:** as in the SP1 plan Task 3 (lines 578–1036): create `simready/bench/geometry.py` and `tests/test_bench_geometry.py`.

**Interfaces:**
- Consumes: Task A1 shim; `simready.occ_utils.count_shapes, count_topology, iter_faces`.
- Produces: exactly the SP1 plan Task 3 "Produces" line: `GateResult`, `run_gate`, `sorted_extents`, `volume`, `surface_area`, `CylFeature`, `cylindrical_features`, `axis_distance`, `fingerprint`.

Follow the SP1 plan Task 3 steps and code as written, with these four deviations. Record them in the SP1 design spec under "Deviations", per its Rule 2.

- [x] **Step 1: Tests import through the shim.** In `tests/test_bench_geometry.py`, replace `pytest.importorskip("OCC.Core.BRepPrimAPI")` and the three `from OCC.Core…` lines with:

```python
from simready import occ_compat

pytestmark = pytest.mark.skipif(occ_compat.BACKEND is None, reason="no OCC binding")

BRep_Builder = occ_compat.occ("BRep").BRep_Builder if occ_compat.BACKEND else None
BRepPrimAPI_MakeBox = occ_compat.occ("BRepPrimAPI").BRepPrimAPI_MakeBox if occ_compat.BACKEND else None
TopoDS_Shell = occ_compat.occ("TopoDS").TopoDS_Shell if occ_compat.BACKEND else None
topods = occ_compat.static("TopoDS", "topods") if occ_compat.BACKEND else None
```

- [x] **Step 2: Add the bbox-gap tests** (from SimReady `sp1-task3-bbox-gap`) to `tests/test_bench_geometry.py`:

```python
def test_sorted_extents_have_no_bounding_box_gap():
    # brepbndlib.Add pads 1e-7 per side: a 2 mm plate measured 2.0000002.
    assert geo.sorted_extents(mk([{"op": "box", "dx": 42, "dy": 42, "dz": 2}])) == pytest.approx(
        [2.0, 42.0, 42.0], abs=1e-9
    )


def test_exact_nominal_cylinder_passes_upper_bound_range():
    from simready.bench.concepts import in_range

    ext = geo.sorted_extents(mk([{"op": "cyl", "r": 10, "h": 30}]))
    assert in_range(ext[0], (19.987, 20.0)) and in_range(ext[1], (19.987, 20.0))
```

- [x] **Step 3: Module imports through the shim.** In `simready/bench/geometry.py`, replace the SP1 plan's `from OCC.Core…` block (plan lines 189–198) with:

```python
from simready import occ_compat

BRepAdaptor_Surface = occ_compat.occ("BRepAdaptor").BRepAdaptor_Surface
BRepCheck_Analyzer = occ_compat.occ("BRepCheck").BRepCheck_Analyzer
BRepClass3d_SolidClassifier = occ_compat.occ("BRepClass3d").BRepClass3d_SolidClassifier
BRepGProp_Face = occ_compat.occ("BRepGProp").BRepGProp_Face
brepgprop = occ_compat.static("BRepGProp", "brepgprop")
brepbndlib = occ_compat.static("BRepBndLib", "brepbndlib")
breptools = occ_compat.static("BRepTools", "breptools")
Bnd_Box = occ_compat.occ("Bnd").Bnd_Box
GeomAbs_Cylinder = occ_compat.occ("GeomAbs").GeomAbs_Cylinder
GProp_GProps = occ_compat.occ("GProp").GProp_GProps
ShapeAnalysis_FreeBounds = occ_compat.occ("ShapeAnalysis").ShapeAnalysis_FreeBounds
TopAbs_EDGE = occ_compat.occ("TopAbs").TopAbs_EDGE
TopAbs_OUT = occ_compat.occ("TopAbs").TopAbs_OUT
gp_Pnt = occ_compat.occ("gp").gp_Pnt
gp_Vec = occ_compat.occ("gp").gp_Vec
```

The module docstring keeps the SP1 rule: "import it only inside the build/check subprocess or in tests that skip without OCC". Then grep the plan's Task 3 code for any other static call, such as `BRep_Tool.` or `topods.`, and route each one through `occ_compat.static(...)`:
```bash
grep -n "BRep_Tool\.\|topods\.\|topexp\.\|breptools\.\|brepgprop\.\|brepbndlib\." simready/bench/geometry.py
```
Each hit must use a name bound above.

- [x] **Step 4: Replace `sorted_extents`** (SP1 plan lines 833–836) with the gap-free box:

```python
def sorted_extents(shape: Any) -> list[float]:
    """Bounding-box side lengths, ascending. Placement- and axis-order-invariant.

    Uses AddOptimal without triangulation or shape tolerance: brepbndlib.Add pads
    1e-7 mm per side, which failed exact-nominal answers against ranges whose upper
    bound is the nominal (SimReady sp1-task3-bbox-gap).
    """
    box = Bnd_Box()
    brepbndlib.AddOptimal(shape, box, False, False)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return sorted([xmax - xmin, ymax - ymin, zmax - zmin])
```

- [x] **Step 5: Run under both backends**

Run: `SR -m pytest tests/test_bench_geometry.py -q` then `OCPPY -m pytest tests/test_bench_geometry.py -q`
Expected: both PASS, including the two new tests. If the gap test fails with a residual close to 1e-7, the box still carries a gap: subtract `box.GetGap()` from each side and note that in the deviation record. Do not widen the test tolerance.

- [x] **Step 6: Commit** (geometry + tests + the SP1 spec deviation note):

```bash
git -C "$WT" add simready/bench/geometry.py tests/test_bench_geometry.py docs/exec-plans/active/2026-09-25-sp1-cad-benchmark-design.md && git -C "$WT" commit -m "feat(bench): geometry module on occ_compat; gap-free sorted extents"
```

### Task A3: SP1 Task 4 (checker + reference-spec positive controls)

**Files:** as in the SP1 plan Task 4 (lines 1037–1348): `simready/bench/checker.py`, `tests/test_bench_checker.py`, `tests/test_bench_reference_specs.py`.

**Interfaces:**
- Consumes: A2 `geometry`; `simready.bench.concepts` (`CHECK_LIST`, `in_range`, check models); `CHECKER_VERSION`.
- Produces: `grade_shape(shape, checks, expected_solids=1) -> dict` with the keys listed in the SP1 plan Task 4 "Produces" line; constants `F5`, `F6`, `F7`, `PASS`, `FAIL`, `NOT_CHECKED`.

- [x] **Step 1:** Follow the SP1 plan Task 4 steps and code as written. The only deviation: any `importorskip("OCC…")` in the tests becomes `pytestmark = pytest.mark.skipif(occ_compat.BACKEND is None, reason="no OCC binding")`, the same as A2 Step 1.
- [x] **Step 2: Re-confirm the 111.** Run: `SR -m pytest tests/test_bench_reference_specs.py -q` and the same with `OCPPY`. Expected: under both backends, the 37 in-vocab references pass all three check lists (111 phrasing checks), and the Task 2 review's three negative controls fail (c001 with one 3.0 mm hole, c003 with a 6 mm seat, c039 with square corners). Record both counts in the task log. If anything differs from 111/111 now that the code is shipped, stop and report. The spec's eval split depends on it.
- [x] **Step 3: Commit**

```bash
git -C "$WT" add simready/bench/checker.py tests/test_bench_checker.py tests/test_bench_reference_specs.py && git -C "$WT" commit -m "feat(bench): checker and reference-spec positive controls"
```

### Task A4: SP1 Task 5 (attempt pipeline)

**Files:** as in the SP1 plan Task 5 (lines 1349–1691): `simready/bench/attempt.py`, `tests/test_bench_attempt.py`.

**Interfaces:**
- Consumes: `PartSpec`, `build_shape`, `grade_shape`, `fingerprint`, `CHECK_LIST`.
- Produces: exactly the SP1 plan Task 5 "Produces" line. cad-rl-env uses `parse_reply`, `evaluate_reply`, `grade_spec`, `F1`, `F2`, `F3`, `F4`, `F8`, `F9`, `CHECKER_ERROR`, `GRADE_TIMEOUT_S`.

- [x] **Step 1:** Follow the SP1 plan Task 5 steps and code as written. One deviation: OCC test skips use the shim, as in A2 Step 1.
- [x] **Step 2: Add a missing-binding test.** `_grade_worker` reports a missing OCC binding as `build_error`, which is F4, a model failure. That is the infra-as-model-failure case the cad-rl-env spec forbids. Add to `tests/test_bench_attempt.py`:

```python
def test_missing_occ_binding_is_checker_error_not_build_error(monkeypatch):
    from simready import occ_compat
    from simready.bench import attempt

    monkeypatch.setattr(occ_compat, "BACKEND", None)
    out = attempt._grade_worker_inline({"steps": [{"op": "box", "dx": 1, "dy": 1, "dz": 1}]}, [], 1)
    assert out["stage"] == "checker_error"
```

and, in `attempt.py`, split the worker body into a plain function `_grade_worker_inline(spec_dict, checks_json, expected_solids) -> dict`, which `_grade_worker` calls and puts on the queue. It begins:

```python
    from simready import occ_compat

    if occ_compat.BACKEND is None:
        return {"stage": "checker_error", "error": "no OCC binding in the grading process"}
```

Record this as a deviation in the SP1 design spec. It is a correctness fix: "a crash inside our own checker must not be blamed on the model" (SP1 deviation 3).

- [x] **Step 3: Run under both backends.** Run: `SR -m pytest tests/test_bench_attempt.py -q` and the same with `OCPPY`. Expected: PASS.
- [x] **Step 4: Commit**

```bash
git -C "$WT" add simready/bench/attempt.py tests/test_bench_attempt.py docs/exec-plans/active/2026-09-25-sp1-cad-benchmark-design.md && git -C "$WT" commit -m "feat(bench): attempt pipeline; missing OCC binding is checker_error"
```

### Task A5: `pyproject.toml`, package data, tag

**Files:**
- Create: `pyproject.toml`
- Modify: `README.md` (install section: one line for `pip install "simready[ocp] @ git+…"`)

**Interfaces:**
- Produces: `pip install "simready[ocp] @ git+https://github.com/Sumanthreddy-DE/simready@bench-api-0.1"` gives an importable `simready.bench` with `concepts_v1.jsonl` inside the wheel.

- [x] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "simready"
version = "0.1.0"
description = "Simulation-readiness checks and a checker-scored text-to-CAD benchmark."
requires-python = ">=3.10"
dependencies = ["pydantic>=2"]

[project.optional-dependencies]
# pythonocc-core is conda-forge only (environment.yml); OCP is the pip route.
ocp = ["cadquery-ocp"]
app = [
    "click>=8.1", "rich>=13.0", "jinja2>=3.1", "torch>=2.0", "streamlit>=1.30",
    "openai>=1.40,<3", "python-dotenv>=1.0", "pypdf>=4.0", "requests>=2.31",
    "sentence-transformers>=3.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["simready"]

[tool.hatch.build.targets.wheel.force-include]
"simready/bench/prompts/concepts_v1.jsonl" = "simready/bench/prompts/concepts_v1.jsonl"
```

Before writing `app`, re-check every pin against `requirements.txt` on the merged branch. The `openai` cap comes from SimReady merge `57a105e`. Copy, never retype.

- [x] **Step 2: Wheel check (positive control)**

Run:
```bash
cd "$WT" && uv build --wheel -o /tmp/claude-sr-wheel && python3 -m zipfile -l /tmp/claude-sr-wheel/simready-0.1.0-py3-none-any.whl | grep -c "concepts_v1.jsonl\|bench/attempt.py\|occ_compat.py"
```
Expected: `3`.

- [x] **Step 3: Clean install under OCP.** Run: `uv venv /tmp/claude-sr-clean --python 3.12 && uv pip install --python /tmp/claude-sr-clean/bin/python "/tmp/claude-sr-wheel/simready-0.1.0-py3-none-any.whl[ocp]" && cd /tmp && /tmp/claude-sr-clean/bin/python -c "from simready.bench.concepts import load_concepts; print(sum(c.in_vocab for c in load_concepts()))"`. Expected: `37`. Running from `/tmp` proves the import comes from the wheel, not the checkout. This venv lives in `/tmp` and is throwaway, so it needs no gate. Delete it after.
- [x] **Step 4: Full suites.** Run: `SR -m pytest -q 2>&1 | tail -3` (0 failed) and `OCPPY -m pytest tests/test_occ_compat.py tests/test_bench_*.py -q` (0 failed).
- [x] **Step 5: Commit and tag**

```bash
git -C "$WT" add pyproject.toml README.md && git -C "$WT" commit -m "build: pyproject with light core and ocp extra" && git -C "$WT" tag bench-api-0.1
```

- [x] **Step 6: Hand the push to the user.** The tag must be on `origin` before Phase B can install from git:
  ```
  ! git -C "<WT>" push origin sp1-bench bench-api-0.1
  ```
  Substitute the real `$WT`. Then update SimReady `BACKLOG.md` on `main`: close `sp1-task3-bbox-gap`, update `sp1-resume` (Tasks 3–5 done) and `cad-rl-env-ownership` (items 1–4 done, tag `bench-api-0.1`).

---

## Phase B: cad-rl-env

Below, `PY` means `.venv/bin/python` in `/home/sumanth/projects/cad-rl-env`.

### Task B0: Package skeleton and venv (USER GATE for the venv)

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/cad_rl_env/__init__.py` (temporary: empty)
- Modify: `CLAUDE.md` (Stack + Dev commands)

- [x] **Step 1: `pyproject.toml`**

```toml
[project]
name = "cad-rl-env"
version = "0.1.0"
description = "Text-to-CAD RL environment: a JSON CAD-DSL part graded by SimReady's geometry checker."
requires-python = ">=3.11"
dependencies = [
    "verifiers>=0.4.1",
    "pydantic>=2",
    "simready[bench] @ git+https://github.com/Sumanthreddy-DE/simready@efe3f15cf7ce92fd2ad8bc85ef402c28091e8b50",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["src/cad_rl_env"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: bulk generator checks (minutes)"]
```

- [x] **Step 2: `.gitignore`:** `.venv/`, `__pycache__/`, `outputs/`, `dist/`, `*.egg-info/`.
- [x] **Step 3: USER GATE: venv**

```bash
cd /home/sumanth/projects/cad-rl-env && uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -c "import verifiers.v1, simready.bench.attempt, OCP.gp; from simready import occ_compat; print(occ_compat.BACKEND)"
```
Expected: `ocp`.

- [x] **Step 4: `CLAUDE.md`.** Fill in **Language** Python 3.12, **Framework** verifiers v1, **Database** none, **Package manager** uv. Dev commands: `Test: .venv/bin/python -m pytest -q -m "not slow"`, `Lint: bash scripts/lint-arch.sh`, `Validate: .venv/bin/vf-validate cad-rl-env -n 5`. Also replace the `[One sentence…]` line with the spec's first sentence.
- [x] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/cad_rl_env/__init__.py CLAUDE.md && git commit -m "build: package skeleton"
```

### Task B1: `grade.py`

**Files:**
- Create: `src/cad_rl_env/grade.py`
- Test: `tests/test_grade.py`

**Interfaces:**
- Consumes: `simready.bench.attempt.parse_reply, evaluate_reply, grade_spec, CHECKER_ERROR`; `simready.bench.concepts.CHECK_LIST`; `simready.occ_compat.BACKEND`.
- Produces: `class GraderError(RuntimeError)`; `grade_reply(reply: str | None, checks: list[dict], expected_solids: int = 1, timeout_s: float = 60.0) -> dict`. The dict is `evaluate_reply`'s output plus `"checker_version": str`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_grade.py
import json

import pytest

from cad_rl_env.grade import GraderError, grade_reply
from simready.bench.concepts import load_concepts

C001 = next(c for c in load_concepts() if c.concept_id == "c001")
CHECKS = [c.model_dump(mode="json") for c in C001.checks.explicit]
REF = C001.reference_spec


def reply(spec) -> str:
    return "Here is the part:\n```json\n" + json.dumps(spec) + "\n```"


def test_reference_spec_passes():
    out = grade_reply(reply(REF), CHECKS)
    assert out["passed"] is True and out["failure_class"] is None and out["partial"] == 1.0


def test_empty_reply_is_no_parse():
    assert grade_reply(None, CHECKS)["failure_class"] == "F1_no_parse"
    assert grade_reply("I cannot do that.", CHECKS)["failure_class"] == "F1_no_parse"


def test_negative_dimension_is_schema():
    bad = {"steps": [{"op": "box", "dx": -1, "dy": 1, "dz": 1}]}
    assert grade_reply(reply(bad), CHECKS)["failure_class"] == "F2_schema"


def test_orphan_step_is_reference():
    bad = {"steps": [{"op": "box", "dx": 1, "dy": 1, "dz": 1}, {"op": "box", "dx": 2, "dy": 2, "dz": 2}]}
    assert grade_reply(reply(bad), CHECKS)["failure_class"] == "F3_reference"


def test_unknown_op_is_out_of_vocab():
    bad = {"steps": [{"op": "box", "dx": 1, "dy": 1, "dz": 1}, {"op": "fillet", "a": 0, "r": 1}]}
    assert grade_reply(reply(bad), CHECKS)["failure_class"] == "F9_out_of_vocab"


def test_disjoint_solids_fail_gate():
    two = {"steps": [
        {"op": "box", "dx": 42, "dy": 42, "dz": 5},
        {"op": "box", "dx": 5, "dy": 5, "dz": 5, "at": [100, 0, 0]},
        {"op": "fuse", "a": 0, "b": 1},
    ]}
    assert grade_reply(reply(two), CHECKS)["failure_class"] == "F5_invalid_solid"


def test_wrong_plate_size_is_dimension_miss():
    wrong = json.loads(json.dumps(REF))
    wrong["steps"][0]["dx"] = 50
    assert grade_reply(reply(wrong), CHECKS)["failure_class"] in {"F6_dimension_miss", "F7_feature_miss"}


def test_timeout_is_f8():
    assert grade_reply(reply(REF), CHECKS, timeout_s=0.001)["failure_class"] == "F8_timeout"


def test_same_reply_same_grade():
    a, b = grade_reply(reply(REF), CHECKS), grade_reply(reply(REF), CHECKS)
    assert a == b


def test_missing_binding_raises(monkeypatch):
    import cad_rl_env.grade as g

    monkeypatch.setattr(g.occ_compat, "BACKEND", None)
    with pytest.raises(GraderError):
        grade_reply(reply(REF), CHECKS)
```

`test_wrong_plate_size_is_dimension_miss` accepts F6 or F7. Moving the plate edge leaves the holes in place, so whether the envelope or the pattern check fires first is SimReady's ordering, not this repo's concern.

- [x] **Step 2: Run to verify it fails.** Run: `PY -m pytest tests/test_grade.py -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'cad_rl_env.grade'`.

- [x] **Step 3: Implement**

```python
# src/cad_rl_env/grade.py
"""Grade one model reply against a check list, through SimReady's SP1 attempt pipeline.

No geometry or check logic lives here: parsing, schema validation, the build
subprocess, the gate and the checks are all ``simready.bench``. This module only
adapts the call and turns grader faults into ``GraderError``, so a broken grader
errors the rollout instead of scoring a wrong model.
"""

from __future__ import annotations

from functools import partial

from simready import occ_compat
from simready.bench import CHECKER_VERSION
from simready.bench.attempt import CHECKER_ERROR, evaluate_reply, grade_spec, parse_reply
from simready.bench.concepts import CHECK_LIST


class GraderError(RuntimeError):
    """The grader failed; the rollout must error, not score 0.0."""


def grade_reply(
    reply: str | None,
    checks: list[dict],
    expected_solids: int = 1,
    timeout_s: float = 60.0,
) -> dict:
    """SP1 grade report for ``reply`` (failure classes F1-F9, ``partial``, ``passed``)."""
    if occ_compat.BACKEND is None:
        raise GraderError("no OCC binding on the evaluator: install cadquery-ocp")
    report = evaluate_reply(
        parse_reply([], reply or ""),
        CHECK_LIST.validate_python(checks),
        expected_solids,
        grade=partial(grade_spec, timeout_s=timeout_s),
    )
    if report["failure_class"] == CHECKER_ERROR:
        raise GraderError(report["build_error"] or "checker_error")
    return {**report, "checker_version": CHECKER_VERSION}
```

- [x] **Step 4: Run tests.** Run: `PY -m pytest tests/test_grade.py -q`. Expected: PASS (10 tests).
- [x] **Step 5: Commit**

```bash
git add src/cad_rl_env/grade.py tests/test_grade.py && git commit -m "feat: grade_reply adapter over simready.bench.attempt"
```

### Task B2: Prompt, task, eval split, `vf-validate`

**Files:**
- Create: `src/cad_rl_env/prompt.py`, `src/cad_rl_env/task.py`, `src/cad_rl_env/taskset.py`, `configs/eval.toml`
- Modify: `src/cad_rl_env/__init__.py`
- Test: `tests/test_taskset_eval.py`

**Interfaces:**
- Consumes: B1 `grade_reply`, `GraderError`; `simready.bench.concepts.load_concepts, PHRASINGS`.
- Produces: `SYSTEM_PROMPT: str`; `CadData(vf.TaskData)` with fields `checks: list[dict]`, `expected_solids: int`, `reference_spec: dict | None`, `split: str`, `source_id: str`, `phrasing: str`, `params: dict`; `CadTaskConfig(vf.TaskConfig)` with `build_timeout_s: float = 60.0`; `CadTask`; `CadConfig(vf.TasksetConfig)` with `split`, `seed`, `n_train`, `families`, `phrasings`, `task`; `CadTaskset`; `eval_tasks(config) -> list[CadTask]`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_taskset_eval.py
from cad_rl_env.taskset import CadConfig, CadTaskset


def load(**kw):
    return CadTaskset(CadConfig(split="eval", **kw)).load()


def test_eval_is_111_in_vocab_prompts():
    tasks = load()
    assert len(tasks) == 111
    assert len({t.data.id for t in tasks}) == 111


def test_eval_ids_and_hidden_fields():
    t = next(t for t in load() if t.data.id == "c001-explicit")
    assert t.data.split == "eval" and t.data.phrasing == "explicit"
    assert t.data.reference_spec["steps"][0]["op"] == "box"
    assert "reference_spec" not in t.data.prompt and "31" in t.data.prompt


def test_eval_phrasing_filter():
    assert len(load(phrasings=["engineer"])) == 37
```

- [x] **Step 2: Run to verify it fails.** Run: `PY -m pytest tests/test_taskset_eval.py -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'cad_rl_env.taskset'`.

- [x] **Step 3: `prompt.py`.** The grammar is copied from `simready/gen/spec.py`'s docstring and bounds. If SimReady's SP1 Task 6 has shipped `simready.bench.prompting.SYSTEM_PROMPT` by now, still use this text-answer prompt: SP1's asks for a `build_part` tool call, and this environment is text-only (deviation 8).

```python
# src/cad_rl_env/prompt.py
"""System prompt: the SimReady CAD DSL, answered as one JSON block (single turn)."""

SYSTEM_PROMPT = """You design mechanical parts in a small CAD language. Units are millimetres.

A part is a JSON object {"steps": [...]}, with 1 to 16 steps. Each step is one of:
- {"op": "box", "dx": W, "dy": D, "dz": H, "at": [x, y, z]}  axis-aligned box; "at" is its min corner
- {"op": "cyl", "r": R, "h": H, "at": [x, y, z]}  cylinder along +Z; "at" is the centre of its base
- {"op": "fuse", "a": i, "b": j}  union of earlier steps i and j
- {"op": "cut", "a": i, "b": j}  step i minus step j

Rules: "a" and "b" are 0-based indices of earlier steps. Every step except the last must be used
by a later fuse or cut. The last step is the finished part, and it must be one solid. Dimensions
are positive and at most 1000; "at" defaults to [0, 0, 0]. There are no other operations (no
fillet, chamfer, rotate or thread). To make a hole, cut a cylinder that passes through the part.

Reply with exactly one ```json fenced block containing the part. If the part cannot be built with
these operations, say so instead of approximating it."""
```

- [x] **Step 4: `task.py`**

```python
# src/cad_rl_env/task.py
"""One text-to-CAD task: single turn, graded once, cached on the trace."""

from __future__ import annotations

import asyncio
import json

import verifiers.v1 as vf

from cad_rl_env.grade import grade_reply

_PARSE_FAILS = {"F1_no_parse"}
_SPEC_FAILS = _PARSE_FAILS | {"F2_schema", "F3_reference", "F9_out_of_vocab"}
_BUILD_FAILS = _SPEC_FAILS | {"F4_build_error", "F8_timeout"}


class CadData(vf.TaskData):
    checks: list[dict]
    """SimReady check dicts for this phrasing. Hidden from the model."""
    expected_solids: int = 1
    reference_spec: dict | None = None
    """Known-good PartSpec dict; graded by ``validate``."""
    split: str
    source_id: str
    """Concept id (eval) or ``<family>-<seed>-<index>`` (train)."""
    phrasing: str
    params: dict = {}


class CadTaskConfig(vf.TaskConfig):
    build_timeout_s: float = 60.0


class CadTask(vf.Task[CadData, vf.State, CadTaskConfig]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    async def _grade(self, trace: vf.Trace) -> dict:
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

    async def validate(self, runtime: vf.Runtime) -> bool | None:
        if self.data.reference_spec is None:
            return None
        reply = "```json\n" + json.dumps(self.data.reference_spec) + "\n```"
        grade = await asyncio.to_thread(
            grade_reply, reply, self.data.checks, self.data.expected_solids, self.config.build_timeout_s
        )
        return bool(grade["passed"])
```

- [x] **Step 5: `taskset.py` (eval part; B5 adds train)**

```python
# src/cad_rl_env/taskset.py
"""cad-rl-env taskset: ``eval`` = SimReady concepts_v1 in-vocab, ``train`` = generator."""

from __future__ import annotations

from typing import Literal

import verifiers.v1 as vf

from cad_rl_env.prompt import SYSTEM_PROMPT
from cad_rl_env.task import CadData, CadTask, CadTaskConfig

EVAL_PHRASINGS = ("explicit", "engineer", "intent")
TRAIN_PHRASINGS = ("explicit", "engineer")


class CadConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "eval"
    seed: int = 0
    n_train: int = 1000
    families: list[str] = ["plate_holes", "bushing", "flange"]
    phrasings: list[str] | None = None
    """None = all phrasings the split supports."""
    task: CadTaskConfig = CadTaskConfig()


def eval_tasks(config: CadConfig) -> list[CadTask]:
    from simready.bench.concepts import load_concepts

    phrasings = config.phrasings or list(EVAL_PHRASINGS)
    tasks = []
    for concept in load_concepts():
        if not concept.in_vocab:
            continue
        for phrasing in phrasings:
            tasks.append(
                CadTask(
                    CadData(
                        id=f"{concept.concept_id}-{phrasing}",
                        prompt=getattr(concept.prompts, phrasing),
                        system_prompt=SYSTEM_PROMPT,
                        checks=[c.model_dump(mode="json") for c in getattr(concept.checks, phrasing)],
                        expected_solids=concept.solids,
                        reference_spec=concept.reference_spec,
                        split="eval",
                        source_id=concept.concept_id,
                        phrasing=phrasing,
                    ),
                    config.task,
                )
            )
    return tasks


class CadTaskset(vf.Taskset[CadTask, CadConfig]):
    def load(self) -> list[CadTask]:
        if self.config.split == "eval":
            return eval_tasks(self.config)
        raise NotImplementedError("train split lands in Task B5")
```

`__init__.py`:
```python
from cad_rl_env.taskset import CadTaskset

__all__ = ["CadTaskset"]
```

- [x] **Step 6: Run the tests.** Run: `PY -m pytest tests/test_taskset_eval.py -q`. Expected: PASS. If `CadData(id=…)` is rejected, read `verifiers/v1/task.py` (`TaskData`) at the installed version for the id field's real name, and fix the plan text as well (Rule 2).
- [x] **Step 7: `vf-validate` with both controls.** First check the flags: `.venv/bin/vf-validate cad-rl-env --help` (read the whole output; never `head` it). Then:
  ```bash
  .venv/bin/vf-validate cad-rl-env --env.taskset.split eval
  ```
  Expected: 111 gold checks pass (the reference scores `passed = 1.0`), and 111 noop checks are valid (the untouched reply scores 0). Any gold failure stops the task. It means the eval split disagrees with A3's 111.
- [x] **Step 8: `configs/eval.toml`**

```toml
# Eval: pure pass@1 (partial stays at weight 0). Model and client are set per run.
[env.taskset]
id = "cad-rl-env"
split = "eval"

[sampling]
max_tokens = 4096
```

Confirm the key names against `vf-run cad-rl-env --help`.

- [x] **Step 9: Commit**

```bash
git add src/cad_rl_env/__init__.py src/cad_rl_env/prompt.py src/cad_rl_env/task.py src/cad_rl_env/taskset.py configs/eval.toml tests/test_taskset_eval.py && git commit -m "feat: eval split over concepts_v1 with passed/partial rewards"
```

### Task B3: Generator core, leak guard, `plate_holes`

**Files:**
- Create: `src/cad_rl_env/generator/__init__.py`, `generator/_checks.py`, `generator/standards.py`, `generator/plate_holes.py`, `src/cad_rl_env/leak.py`
- Test: `tests/test_leak.py`, `tests/test_generator_plate_holes.py`

**Interfaces:**
- Produces:
  - `FamilySample(params: dict, reference_spec: dict, prompts: dict[str, str], checks: dict[str, list[dict]])`, where `prompts` and `checks` are keyed by `"explicit"` and `"engineer"`;
  - `GeneratedTask(family, index, phrasing, prompt, checks, reference_spec, params)`;
  - `task_rng(seed: int, family: str, index: int) -> random.Random`;
  - `FAMILIES: dict[str, Callable[[random.Random], FamilySample]]`;
  - `make_task(seed: int, family: str, index: int, phrasing: str) -> GeneratedTask`;
  - `leak.signature(checks: list[dict]) -> Signature`, `leak.collides(a, b) -> bool`, `leak.eval_signatures() -> list[Signature]`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_leak.py
from cad_rl_env.leak import collides, eval_signatures, signature

PLATE = [
    {"type": "bbox_sorted", "ranges": [[4.9, 5.1], [41.9, 42.1], [41.9, 42.1]]},
    {"type": "hole_set", "diameter": [3.3, 3.5], "count": [4, 4], "through": True},
]


def test_signature_uses_range_midpoints():
    bbox, feats = signature(PLATE)
    assert bbox == (5.0, 42.0, 42.0)
    assert feats == (("hole", 3.4, 4),)


def test_same_part_collides_and_different_part_does_not():
    near = [dict(PLATE[0], ranges=[[4.9, 5.1], [42.4, 42.6], [41.9, 42.1]]), PLATE[1]]
    far = [dict(PLATE[0], ranges=[[4.9, 5.1], [59.9, 60.1], [41.9, 42.1]]), PLATE[1]]
    assert collides(signature(PLATE), signature(near))
    assert not collides(signature(PLATE), signature(far))


def test_feature_count_mismatch_does_not_collide():
    three = [PLATE[0], dict(PLATE[1], count=[3, 3])]
    assert not collides(signature(PLATE), signature(three))


def test_eval_signatures_cover_37_concepts():
    assert len(eval_signatures()) == 37
```

```python
# tests/test_generator_plate_holes.py
import pytest

from cad_rl_env.generator import FAMILIES, make_task, task_rng
from cad_rl_env.grade import grade_reply
from cad_rl_env.leak import collides, eval_signatures, signature
import json


def test_same_seed_same_task():
    assert make_task(0, "plate_holes", 3, "explicit") == make_task(0, "plate_holes", 3, "explicit")
    assert make_task(0, "plate_holes", 3, "explicit") != make_task(0, "plate_holes", 4, "explicit")


def test_engineer_prompt_names_standard_not_diameter():
    t = make_task(0, "plate_holes", 0, "engineer")
    assert "ISO 273" in t.prompt and "clearance" in t.prompt


@pytest.mark.parametrize("index", range(20))
@pytest.mark.parametrize("phrasing", ["explicit", "engineer"])
def test_reference_passes_own_checks(index, phrasing):
    t = make_task(0, "plate_holes", index, phrasing)
    out = grade_reply("```json\n" + json.dumps(t.reference_spec) + "\n```", t.checks)
    assert out["passed"], (t.params, out["failure_class"], out["check_report"])


def test_no_collision_with_eval_in_200_tasks():
    # The guard lives in make_task, so test the tasks it actually returns.
    sigs = eval_signatures()
    for i in range(200):
        t = make_task(0, "plate_holes", i, "explicit")
        assert not any(collides(signature(t.checks), e) for e in sigs)


def test_guard_positive_control():
    # A part identical to an eval concept must be detected (Working Rule 9).
    c001 = [c.model_dump(mode="json") for c in next(
        c for c in __import__("simready.bench.concepts", fromlist=["x"]).load_concepts() if c.concept_id == "c001"
    ).checks.explicit]
    assert any(collides(signature(c001), e) for e in eval_signatures())
```

Remove the unused `FAMILIES` and `task_rng` imports from this file.

- [x] **Step 2: Run to verify they fail.** Run: `PY -m pytest tests/test_leak.py tests/test_generator_plate_holes.py -q`. Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: `generator/standards.py` and `generator/_checks.py`**

```python
# src/cad_rl_env/generator/standards.py
"""Train-only standards tables.

ISO 273 clearance holes (fine, medium, coarse), mm. Values as verified in SimReady's
SP1 Task 2 review (IS 1821:1987 = identical adoption of ISO 273:1979, p.6 table).
"""

ISO273: dict[str, tuple[float, float, float]] = {
    "M2.5": (2.7, 2.9, 3.1),
    "M3": (3.2, 3.4, 3.6),
    "M4": (4.3, 4.5, 4.8),
    "M5": (5.3, 5.5, 5.8),
    "M6": (6.4, 6.6, 7.0),
    "M8": (8.4, 9.0, 10.0),
}
```

```python
# src/cad_rl_env/generator/_checks.py
"""Builders for SimReady check dicts. Shapes only; the check semantics are SimReady's."""

from __future__ import annotations

import itertools
import math

EXPLICIT_TOL = 0.1  # SP1 design §1 rule 3: ±0.1 mm on stated numbers
PATTERN_TOL_ENGINEER = 0.2  # design choice, as in concepts_v1 (not a standard value)


def tol(x: float, t: float) -> list[float]:
    return [round(x - t, 4), round(x + t, 4)]


def bbox_sorted(dims: list[float], t: float = EXPLICIT_TOL) -> dict:
    return {"type": "bbox_sorted", "ranges": [tol(d, t) for d in sorted(dims)]}


def pairwise(points: list[tuple[float, float]]) -> list[float]:
    """Sorted pairwise centre distances (SimReady hole_set ``pattern``)."""
    return sorted(math.dist(a, b) for a, b in itertools.combinations(points, 2))


def hole_set(diameter: list[float], points: list[tuple[float, float]], pattern_tol: float) -> dict:
    n = len(points)
    return {
        "type": "hole_set",
        "diameter": diameter,
        "count": [n, n],
        "through": True,
        "pattern": [tol(d, pattern_tol) for d in pairwise(points)],
    }


def cut_chain(base: dict, tools: list[dict]) -> dict:
    """``base`` minus each tool in turn: [base, t1, cut(0,1), t2, cut(2,3), ...]."""
    steps, last = [base], 0
    for tool in tools:
        steps.append(tool)
        steps.append({"op": "cut", "a": last, "b": len(steps) - 1})
        last = len(steps) - 1
    return {"steps": steps}
```

- [x] **Step 4: `generator/plate_holes.py`**

```python
# src/cad_rl_env/generator/plate_holes.py
"""Flat plate with 2 holes in a row or 4 on a rectangle, sized by ISO 273."""

from __future__ import annotations

import math
import random

from cad_rl_env.generator import FamilySample
from cad_rl_env.generator._checks import (
    EXPLICIT_TOL,
    PATTERN_TOL_ENGINEER,
    bbox_sorted,
    cut_chain,
    hole_set,
    tol,
)
from cad_rl_env.generator.standards import ISO273

EXPLICIT = [
    "Make a {w} x {d} x {t} mm plate with {holes} {dia} mm through holes {layout}.",
    "A flat plate, {w} by {d} mm and {t} mm thick, with {holes} through holes of {dia} mm diameter {layout}.",
    "Model a {t} mm thick plate measuring {w} x {d} mm. Drill {holes} {dia} mm holes right through it, {layout}.",
    "I need a {w} x {d} mm plate, {t} mm thick, with {holes} {dia} mm diameter through holes {layout}.",
]
ENGINEER = [
    "Make a {w} x {d} x {t} mm plate with {holes} {thread} clearance holes (ISO 273) {layout}.",
    "A {t} mm thick mounting plate, {w} x {d} mm, with {holes} through holes that clear {thread} screws per ISO 273, {layout}.",
    "Mounting plate {w} by {d} mm, {t} mm thick: {holes} ISO 273 clearance holes for {thread} bolts, {layout}.",
    "I need a {w} x {d} x {t} mm plate drilled for {holes} {thread} bolts (ISO 273 clearance), {layout}.",
]
WORDS = {2: "two", 4: "four"}


def generate(rng: random.Random) -> FamilySample:
    thread = rng.choice(sorted(ISO273))
    fine, medium, coarse = ISO273[thread]
    t = rng.choice([3, 4, 5, 6, 8])
    n = rng.choice([2, 4])
    margin = max(rng.choice([5, 6, 8, 10]), math.ceil(coarse / 2 + 2))
    gap = math.ceil(coarse) + 4  # minimum centre distance between holes
    if n == 4:
        w = rng.randrange(max(30, 2 * margin + gap), 125, 5)
        d = rng.randrange(max(30, 2 * margin + gap), 125, 5)
        a, b = w - 2 * margin, d - 2 * margin
        points = [(margin, margin), (w - margin, margin), (margin, d - margin), (w - margin, d - margin)]
        layout = f"on a {a} x {b} mm rectangular pattern centred on the plate"
    else:
        w = rng.randrange(max(60, 2 * margin + gap), 125, 5)
        d = rng.randrange(max(15, math.ceil(coarse) + 6), 45, 5)
        a = w - 2 * margin
        points = [(margin, d / 2), (w - margin, d / 2)]
        layout = f"{a} mm apart on the centre line along the {w} mm side, centred"

    base = {"op": "box", "dx": w, "dy": d, "dz": t}
    tools = [{"op": "cyl", "r": medium / 2, "h": t, "at": [x, y, 0]} for x, y in points]
    fields = {"w": w, "d": d, "t": t, "holes": WORDS[n], "layout": layout}
    return FamilySample(
        params={"thread": thread, "w": w, "d": d, "t": t, "n": n, "margin": margin},
        reference_spec=cut_chain(base, tools),
        prompts={
            "explicit": rng.choice(EXPLICIT).format(dia=medium, **fields),
            "engineer": rng.choice(ENGINEER).format(thread=thread, **fields),
        },
        checks={
            "explicit": [
                bbox_sorted([w, d, t]),
                hole_set(tol(medium, EXPLICIT_TOL), points, EXPLICIT_TOL),
            ],
            "engineer": [
                bbox_sorted([w, d, t]),
                hole_set([fine, coarse], points, PATTERN_TOL_ENGINEER),
            ],
        },
    )
```

- [x] **Step 5: `generator/__init__.py`**

```python
# src/cad_rl_env/generator/__init__.py
"""Seeded train-task generator. A task is fully determined by (seed, family, index)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

PHRASINGS = ("explicit", "engineer")
MAX_DRAWS = 50  # redraws allowed when a sample collides with an eval part


@dataclass(frozen=True)
class FamilySample:
    params: dict
    reference_spec: dict
    prompts: dict[str, str]
    checks: dict[str, list[dict]]


@dataclass(frozen=True)
class GeneratedTask:
    family: str
    index: int
    phrasing: str
    prompt: str
    checks: list[dict]
    reference_spec: dict
    params: dict


def task_rng(seed: int, family: str, index: int) -> random.Random:
    # A str seed is hashed deterministically by random.Random (unlike hash(), which
    # Python salts per process).
    return random.Random(f"{seed}:{family}:{index}")


def _families() -> dict[str, Callable[[random.Random], FamilySample]]:
    from cad_rl_env.generator import plate_holes

    return {"plate_holes": plate_holes.generate}


FAMILIES = _families()


def make_task(seed: int, family: str, index: int, phrasing: str) -> GeneratedTask:
    from cad_rl_env.leak import collides, eval_signatures, signature

    if phrasing not in PHRASINGS:
        raise ValueError(f"train phrasing must be one of {PHRASINGS}, got {phrasing!r}")
    rng = task_rng(seed, family, index)
    sigs = eval_signatures()
    for _ in range(MAX_DRAWS):
        sample = FAMILIES[family](rng)
        if not any(collides(signature(sample.checks["explicit"]), e) for e in sigs):
            return GeneratedTask(
                family, index, phrasing, sample.prompts[phrasing],
                sample.checks[phrasing], sample.reference_spec, sample.params,
            )
    raise RuntimeError(f"{family} seed={seed} index={index}: {MAX_DRAWS} draws all hit eval parts")
```

The guard compares on the *explicit* checks for both phrasings. Explicit midpoints are the nominal part, so a train engineer prompt can never describe an eval part either.

- [x] **Step 6: `leak.py`**

```python
# src/cad_rl_env/leak.py
"""Keep train parts away from eval parts: compare check-list signatures.

A signature is the sorted bbox midpoints plus the (kind, diameter midpoint, count)
of every hole/bore/boss check. Two signatures collide when every bbox side is within
BBOX_TOL and the features match one-to-one within DIA_TOL. A missing bbox on either
side compares features only (stricter, never looser).
"""

from __future__ import annotations

from functools import lru_cache

BBOX_TOL = 1.0
DIA_TOL = 0.5

Signature = tuple[tuple[float, ...], tuple[tuple[str, float, int], ...]]


def _mid(rng: list | tuple) -> float:
    lo, hi = rng
    if lo is None:
        return float(hi)
    if hi is None:
        return float(lo)
    return round((lo + hi) / 2, 4)


def signature(checks: list[dict]) -> Signature:
    bbox: tuple[float, ...] = ()
    feats = []
    for c in checks:
        kind = c["type"]
        if kind == "bbox_sorted":
            bbox = tuple(_mid(r) for r in c["ranges"])
        elif kind == "hole_set":
            feats.append(("hole", _mid(c["diameter"]), int(_mid(c["count"]))))
        elif kind in ("bore", "boss"):
            feats.append((kind, _mid(c["diameter"]), 1))
    return bbox, tuple(sorted(feats))


def collides(a: Signature, b: Signature) -> bool:
    (bbox_a, feats_a), (bbox_b, feats_b) = a, b
    if bbox_a and bbox_b and any(abs(x - y) > BBOX_TOL for x, y in zip(bbox_a, bbox_b)):
        return False
    if len(feats_a) != len(feats_b):
        return False
    return all(
        ka == kb and na == nb and abs(da - db) <= DIA_TOL
        for (ka, da, na), (kb, db, nb) in zip(feats_a, feats_b)
    )


@lru_cache(maxsize=1)
def eval_signatures() -> tuple[Signature, ...]:
    from simready.bench.concepts import load_concepts

    return tuple(
        signature([c.model_dump(mode="json") for c in concept.checks.explicit])
        for concept in load_concepts()
        if concept.in_vocab
    )
```

`eval_signatures` returns a tuple. `len()` works, so `test_eval_signatures_cover_37_concepts` is unchanged.

- [x] **Step 7: Run the tests.** Run: `PY -m pytest tests/test_leak.py tests/test_generator_plate_holes.py -q`. Expected: PASS. A reference that fails its own checks is a generator bug: fix the generator, never the checks.
- [x] **Step 8: Commit**

```bash
git add src/cad_rl_env/generator src/cad_rl_env/leak.py tests/test_leak.py tests/test_generator_plate_holes.py && git commit -m "feat: seeded generator core, leak guard, plate_holes family"
```

### Task B4: `bushing` and `flange` families

**Files:**
- Create: `src/cad_rl_env/generator/bushing.py`, `src/cad_rl_env/generator/flange.py`
- Modify: `src/cad_rl_env/generator/__init__.py` (`_families`)
- Test: `tests/test_generator_families.py`

**Interfaces:**
- Consumes: B3 `FamilySample`, `_checks`, `ISO273`.
- Produces: `bushing.generate(rng) -> FamilySample`, `flange.generate(rng) -> FamilySample`; `FAMILIES` keys `plate_holes`, `bushing`, `flange`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_generator_families.py
import json

import pytest

from cad_rl_env.generator import FAMILIES, make_task
from cad_rl_env.grade import grade_reply


def test_three_families_registered():
    assert set(FAMILIES) == {"plate_holes", "bushing", "flange"}


@pytest.mark.parametrize("family", ["bushing", "flange"])
@pytest.mark.parametrize("index", range(15))
@pytest.mark.parametrize("phrasing", ["explicit", "engineer"])
def test_reference_passes_own_checks(family, index, phrasing):
    t = make_task(0, family, index, phrasing)
    assert len(t.reference_spec["steps"]) <= 16
    out = grade_reply("```json\n" + json.dumps(t.reference_spec) + "\n```", t.checks)
    assert out["passed"], (family, t.params, out["failure_class"], out["check_report"])


def test_flange_engineer_names_pitch_circle():
    t = make_task(0, "flange", 0, "engineer")
    assert "pitch circle" in t.prompt and "ISO 273" in t.prompt
```

- [x] **Step 2: Run to verify it fails.** Run: `PY -m pytest tests/test_generator_families.py -q`. Expected: FAIL with `KeyError: 'bushing'`.

- [x] **Step 3: `bushing.py`**

```python
# src/cad_rl_env/generator/bushing.py
"""Round spacer tube: bore + OD + length. Engineer phrasing sizes the bore by ISO 273."""

from __future__ import annotations

import math
import random

from cad_rl_env.generator import FamilySample
from cad_rl_env.generator._checks import EXPLICIT_TOL, bbox_sorted, cut_chain, tol
from cad_rl_env.generator.standards import ISO273

EXPLICIT = [
    "Model a round tube {od} mm outside diameter, {bore} mm inside diameter and {length} mm long.",
    "A cylindrical spacer: {od} mm OD, {bore} mm through bore, {length} mm long.",
    "Make a {length} mm long sleeve with a {od} mm outer diameter and a {bore} mm hole through its axis.",
    "I need a spacer tube, {length} mm long, outside diameter {od} mm, bore {bore} mm.",
]
ENGINEER = [
    "A {length} mm long spacer for an {thread} bolt, {od} mm outside diameter, bore per ISO 273 clearance.",
    "Spacer tube for {thread} screws (ISO 273 clearance bore), {od} mm OD, {length} mm long.",
    "Make a {od} mm diameter standoff spacer, {length} mm long, that an {thread} bolt passes through (ISO 273 clearance).",
    "I need an {thread} clearance spacer (ISO 273), {od} mm across and {length} mm long.",
]


def generate(rng: random.Random) -> FamilySample:
    thread = rng.choice(sorted(ISO273))
    fine, medium, coarse = ISO273[thread]
    od = math.ceil(coarse) + rng.choice([3, 4, 6, 8, 10])
    length = rng.choice([3, 5, 8, 10, 12, 15, 20, 25, 30])
    spec = cut_chain(
        {"op": "cyl", "r": od / 2, "h": length},
        [{"op": "cyl", "r": medium / 2, "h": length}],
    )
    boss = {"type": "boss", "diameter": tol(od, EXPLICIT_TOL), "height_min": length - EXPLICIT_TOL}
    fields = {"od": od, "length": length}
    return FamilySample(
        params={"thread": thread, "od": od, "length": length},
        reference_spec=spec,
        prompts={
            "explicit": rng.choice(EXPLICIT).format(bore=medium, **fields),
            "engineer": rng.choice(ENGINEER).format(thread=thread, **fields),
        },
        checks={
            "explicit": [
                bbox_sorted([od, od, length]),
                {"type": "bore", "diameter": tol(medium, EXPLICIT_TOL), "through": True},
                boss,
            ],
            "engineer": [
                bbox_sorted([od, od, length]),
                {"type": "bore", "diameter": [fine, coarse], "through": True},
                boss,
            ],
        },
    )
```

- [x] **Step 4: `flange.py`**

```python
# src/cad_rl_env/generator/flange.py
"""Disc + hub flange with a through bore and 3 or 4 bolt holes on a pitch circle."""

from __future__ import annotations

import math
import random

from cad_rl_env.generator import FamilySample
from cad_rl_env.generator._checks import (
    EXPLICIT_TOL,
    PATTERN_TOL_ENGINEER,
    bbox_sorted,
    hole_set,
    tol,
)
from cad_rl_env.generator.standards import ISO273

BOLT_THREADS = ["M4", "M5", "M6"]
EXPLICIT = [
    "Make a flange: a {d1} mm diameter disc {t1} mm thick with a {d2} mm diameter hub {t2} mm tall on top, a {s} mm bore through both, and {n} {dia} mm through holes in the disc equally spaced on a {pcd} mm pitch circle.",
    "A round flange, {d1} mm across and {t1} mm thick, carrying a {d2} mm hub that stands {t2} mm above it. Bore {s} mm right through. {n} holes of {dia} mm, equally spaced on a {pcd} mm pitch circle, go through the disc.",
    "Model a hub flange: disc {d1} x {t1} mm, hub {d2} x {t2} mm, centre bore {s} mm through, {n} x {dia} mm holes through the disc on a {pcd} mm pitch circle, equally spaced.",
    "I need a {d1} mm flange plate, {t1} mm thick, with a {d2} mm hub {t2} mm high, a {s} mm through bore and {n} {dia} mm bolt holes equally spaced on a {pcd} mm pitch circle.",
]
ENGINEER = [
    "Make a flange: a {d1} mm diameter disc {t1} mm thick with a {d2} mm diameter hub {t2} mm tall on top, a {s} mm bore through both, and {n} {thread} clearance bolt holes (ISO 273) equally spaced on a {pcd} mm pitch circle.",
    "A round flange, {d1} mm across and {t1} mm thick, with a {d2} mm hub {t2} mm tall and a {s} mm through bore, bolted down by {n} {thread} bolts (ISO 273 clearance holes) equally spaced on a {pcd} mm pitch circle.",
    "Hub flange, disc {d1} x {t1} mm, hub {d2} x {t2} mm, {s} mm through bore, {n} ISO 273 clearance holes for {thread} on a {pcd} mm pitch circle, equally spaced.",
    "I need a {d1} mm flange, {t1} mm thick, with a {d2} mm hub {t2} mm high, {s} mm bore, drilled for {n} {thread} bolts (ISO 273) equally spaced on a {pcd} mm pitch circle.",
]


def _draw(rng: random.Random) -> dict | None:
    d1 = rng.randrange(40, 105, 5)
    t1 = rng.choice([4, 5, 6, 8])
    d2 = 2 * round(d1 * rng.uniform(0.35, 0.5) / 2)
    t2 = rng.choice([5, 8, 10, 12])
    shafts = [x for x in (6, 8, 10, 12, 16) if x <= d2 - 6]
    thread = rng.choice(BOLT_THREADS)
    coarse = ISO273[thread][2]
    lo, hi = math.ceil(d2 + coarse + 4), math.floor(d1 - coarse - 4)
    if not shafts or lo > hi:
        return None
    pcd = rng.randrange(lo + lo % 2, hi + 1, 2) if lo + lo % 2 <= hi else None
    if pcd is None:
        return None
    return {"d1": d1, "t1": t1, "d2": d2, "t2": t2, "s": rng.choice(shafts),
            "thread": thread, "n": rng.choice([3, 4]), "pcd": pcd}


def generate(rng: random.Random) -> FamilySample:
    for _ in range(100):
        p = _draw(rng)
        if p is not None:
            break
    else:
        raise RuntimeError("flange: no feasible draw in 100 tries")
    fine, medium, coarse = ISO273[p["thread"]]
    radius = p["pcd"] / 2
    points = [
        (round(radius * math.cos(2 * math.pi * k / p["n"]), 6), round(radius * math.sin(2 * math.pi * k / p["n"]), 6))
        for k in range(p["n"])
    ]
    steps = [
        {"op": "cyl", "r": p["d1"] / 2, "h": p["t1"]},
        {"op": "cyl", "r": p["d2"] / 2, "h": p["t2"], "at": [0, 0, p["t1"]]},
        {"op": "fuse", "a": 0, "b": 1},
        {"op": "cyl", "r": p["s"] / 2, "h": p["t1"] + p["t2"]},
        {"op": "cut", "a": 2, "b": 3},
    ]
    for x, y in points:
        steps.append({"op": "cyl", "r": medium / 2, "h": p["t1"], "at": [x, y, 0]})
        steps.append({"op": "cut", "a": len(steps) - 2, "b": len(steps) - 1})
    common = [
        bbox_sorted([p["d1"], p["d1"], p["t1"] + p["t2"]]),
        {"type": "boss", "diameter": tol(p["d1"], EXPLICIT_TOL), "height_min": p["t1"] - EXPLICIT_TOL},
        {"type": "boss", "diameter": tol(p["d2"], EXPLICIT_TOL), "height_min": p["t2"] - EXPLICIT_TOL},
        {"type": "bore", "diameter": tol(p["s"], EXPLICIT_TOL), "through": True},
    ]
    fields = {k: p[k] for k in ("d1", "t1", "d2", "t2", "s", "n", "pcd")}
    return FamilySample(
        params=p,
        reference_spec={"steps": steps},
        prompts={
            "explicit": rng.choice(EXPLICIT).format(dia=medium, **fields),
            "engineer": rng.choice(ENGINEER).format(thread=p["thread"], **fields),
        },
        checks={
            "explicit": common + [hole_set(tol(medium, EXPLICIT_TOL), points, EXPLICIT_TOL)],
            "engineer": common + [hole_set([fine, coarse], points, PATTERN_TOL_ENGINEER)],
        },
    )
```

Step bookkeeping: the loop appends the tool cylinder first, so `len(steps) - 2` is the previous result and `len(steps) - 1` is the new tool. Steps = 5 + 2n ≤ 13.

- [x] **Step 5: Register both** in `_families()`:

```python
def _families() -> dict[str, Callable[[random.Random], FamilySample]]:
    from cad_rl_env.generator import bushing, flange, plate_holes

    return {"plate_holes": plate_holes.generate, "bushing": bushing.generate, "flange": flange.generate}
```

- [x] **Step 6: Run the tests.** Run: `PY -m pytest tests/test_generator_families.py -q`. Expected: PASS. If a flange boss check fails, check SimReady's own `test_stacked_flange_bosses_and_bore` (A2) for how fused disc and hub faces are reported. Fix the *generator's* checks to the documented SimReady semantics, and never edit SimReady from this task.
- [x] **Step 7: Commit**

```bash
git add src/cad_rl_env/generator tests/test_generator_families.py && git commit -m "feat: bushing and flange families"
```

### Task B5: Train split, bulk self-check, `configs/train.toml`

**Files:**
- Modify: `src/cad_rl_env/taskset.py` (`load`, new `train_tasks`)
- Create: `configs/train.toml`
- Test: `tests/test_taskset_train.py`, `tests/test_generator_bulk.py`

**Interfaces:**
- Consumes: B3/B4 `make_task`, `FAMILIES`.
- Produces: `train_tasks(config: CadConfig) -> list[CadTask]`. Task i gets family `families[i % F]` and phrasing `phrasings[(i // F) % P]`, with id `f"{family}-{seed}-{i}-{phrasing}"`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_taskset_train.py
import pytest

from cad_rl_env.taskset import CadConfig, CadTaskset


def load(**kw):
    return CadTaskset(CadConfig(split="train", **kw)).load()


def test_train_size_and_unique_ids():
    tasks = load(n_train=60)
    assert len(tasks) == 60 and len({t.data.id for t in tasks}) == 60


def test_train_round_robins_families_and_phrasings():
    tasks = load(n_train=6)
    assert [t.data.params["family"] for t in tasks[:3]] == ["plate_holes", "bushing", "flange"]
    assert {t.data.phrasing for t in tasks} == {"explicit", "engineer"}


def test_train_is_reproducible():
    assert [t.data.prompt for t in load(n_train=9)] == [t.data.prompt for t in load(n_train=9)]


def test_train_rejects_intent():
    with pytest.raises(ValueError):
        load(n_train=3, phrasings=["intent"])
```

```python
# tests/test_generator_bulk.py
import json

import pytest

from cad_rl_env.generator import FAMILIES, make_task
from cad_rl_env.grade import grade_reply
from cad_rl_env.leak import collides, eval_signatures, signature

N = 1000


@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_bulk_self_check_and_no_leak(family):
    sigs = eval_signatures()
    failures = []
    for i in range(N):
        explicit = make_task(0, family, i, "explicit")
        assert not any(collides(signature(explicit.checks), e) for e in sigs), (family, i)
        for phrasing in ("explicit", "engineer"):
            t = explicit if phrasing == "explicit" else make_task(0, family, i, phrasing)
            out = grade_reply("```json\n" + json.dumps(t.reference_spec) + "\n```", t.checks)
            if not out["passed"]:
                failures.append((i, phrasing, out["failure_class"]))
    assert failures == [], failures[:10]
```

- [x] **Step 2: Run to verify it fails.** Run: `PY -m pytest tests/test_taskset_train.py -q`. Expected: FAIL with `NotImplementedError: train split lands in Task B5`.

- [x] **Step 3: Implement `train_tasks` and update `load`**

```python
def train_tasks(config: CadConfig) -> list[CadTask]:
    from cad_rl_env.generator import FAMILIES, make_task

    phrasings = config.phrasings or list(TRAIN_PHRASINGS)
    bad = [p for p in phrasings if p not in TRAIN_PHRASINGS]
    if bad:
        raise ValueError(f"train split supports {TRAIN_PHRASINGS}, got {bad}")
    unknown = [f for f in config.families if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families {unknown}; known: {sorted(FAMILIES)}")
    tasks = []
    for i in range(config.n_train):
        family = config.families[i % len(config.families)]
        phrasing = phrasings[(i // len(config.families)) % len(phrasings)]
        g = make_task(config.seed, family, i, phrasing)
        tasks.append(
            CadTask(
                CadData(
                    id=f"{family}-{config.seed}-{i}-{phrasing}",
                    prompt=g.prompt,
                    system_prompt=SYSTEM_PROMPT,
                    checks=g.checks,
                    reference_spec=g.reference_spec,
                    split="train",
                    source_id=f"{family}-{config.seed}-{i}",
                    phrasing=phrasing,
                    params={"family": family, **g.params},
                ),
                config.task,
            )
        )
    return tasks
```

`load` becomes `return eval_tasks(self.config) if self.config.split == "eval" else train_tasks(self.config)`.

- [x] **Step 4: Run the fast tests.** Run: `PY -m pytest -q -m "not slow"`. Expected: PASS.
- [x] **Step 5: Run the bulk test once.** Run it detached with a log, because it builds 6,000 shapes in subprocesses:
  ```bash
  nohup .venv/bin/python -m pytest -q -m slow tests/test_generator_bulk.py > outputs/bulk.log 2>&1 &
  ```
  Expected in `outputs/bulk.log`: `3 passed`. Record the wall time. If a family has failures, fix that family and re-run only that parametrisation.
- [x] **Step 6: `vf-validate` on train.** Run: `.venv/bin/vf-validate cad-rl-env --env.taskset.split train --env.taskset.n-train 30`. Expected: 30 gold pass, 30 noop valid.
- [x] **Step 7: `configs/train.toml`**

```toml
# Train: dense signal on. partial = gate × fraction of checks passed (SP1 §1 rule 4).
[env.taskset]
id = "cad-rl-env"
split = "train"
n_train = 5000
seed = 0

[env.taskset.task.rewards]
partial = { weight = 0.5 }
```

- [x] **Step 8: Commit**

```bash
git add src/cad_rl_env/taskset.py configs/train.toml tests/test_taskset_train.py tests/test_generator_bulk.py && git commit -m "feat: seeded train split with bulk self-check"
```

### Task B6: Red-team suite

**Files:**
- Create: `tests/test_redteam.py`

**Interfaces:**
- Consumes: B1 `grade_reply`; `load_concepts`; B3/B4 `make_task`.

- [x] **Step 1: Write the suite**

```python
# tests/test_redteam.py
"""Adversarial answers that must score passed = 0, and known weak checks that still pass.

KNOWN_WEAK documents real gaps from SimReady's SP1 Task 2 review. Each case asserts that
the cheat *passes* today, so a SimReady check fix flips the test and forces the README's
reward-hacking table to be updated. Fixing them is SimReady's job (new check types).
"""

import copy
import json

import pytest

from cad_rl_env.generator import make_task
from cad_rl_env.grade import grade_reply
from simready.bench.concepts import load_concepts

CONCEPTS = {c.concept_id: c for c in load_concepts()}


def checks(cid: str, phrasing: str) -> list[dict]:
    return [c.model_dump(mode="json") for c in getattr(CONCEPTS[cid].checks, phrasing)]


def ref(cid: str) -> dict:
    return copy.deepcopy(CONCEPTS[cid].reference_spec)


def passed(spec: dict, check_list: list[dict]) -> bool:
    return grade_reply("```json\n" + json.dumps(spec) + "\n```", check_list)["passed"]


def c001_plate_only():
    return {"steps": [ref("c001")["steps"][0]]}


def c001_three_holes():
    return {"steps": ref("c001")["steps"][:9]}


def c001_blind_holes():
    spec = ref("c001")
    for i in (3, 5, 7, 9):
        spec["steps"][i]["h"] = 2.5
    return spec


def c001_holes_miss_part():
    spec = ref("c001")
    for i in (3, 5, 7, 9):
        spec["steps"][i]["at"][0] += 100
    return spec


def c001_extra_solid():
    spec = ref("c001")
    n = len(spec["steps"])
    spec["steps"] += [{"op": "box", "dx": 5, "dy": 5, "dz": 5, "at": [200, 0, 0]}, {"op": "fuse", "a": n - 1, "b": n}]
    return spec


def c004_solid_rod():
    return {"steps": [ref("c004")["steps"][0]]}


def c004_square_tube():
    return {"steps": [
        {"op": "box", "dx": 16, "dy": 16, "dz": 20, "at": [-8, -8, 0]},
        {"op": "cyl", "r": 4.5, "h": 20},
        {"op": "cut", "a": 0, "b": 1},
    ]}


ADVERSARIAL = [
    ("c001 plate without holes", c001_plate_only, "c001", "explicit"),
    ("c001 three of four holes", c001_three_holes, "c001", "explicit"),
    ("c001 blind instead of through", c001_blind_holes, "c001", "explicit"),
    ("c001 hole cuts miss the part", c001_holes_miss_part, "c001", "explicit"),
    ("c001 second disjoint solid", c001_extra_solid, "c001", "explicit"),
    ("c004 solid rod, no bore", c004_solid_rod, "c004", "explicit"),
    ("c004 square tube, right envelope", c004_square_tube, "c004", "explicit"),
]


@pytest.mark.parametrize(("name", "make", "cid", "phrasing"), ADVERSARIAL, ids=[a[0] for a in ADVERSARIAL])
def test_adversarial_fails(name, make, cid, phrasing):
    assert not passed(make(), checks(cid, phrasing)), name


def test_generated_flange_without_hub_fails():
    t = make_task(0, "flange", 0, "explicit")
    spec = copy.deepcopy(t.reference_spec)
    spec["steps"][1]["h"] = 0.5  # hub shrunk to a lip: boss height check must catch it
    assert not passed(spec, t.checks)


def test_generated_plate_wrong_pitch_fails():
    t = make_task(0, "plate_holes", 0, "explicit")
    spec = copy.deepcopy(t.reference_spec)
    spec["steps"][1]["at"][0] += 1.0
    assert not passed(spec, t.checks)


def test_generated_bushing_wrong_bolt_size_fails_engineer():
    t = make_task(0, "bushing", 0, "engineer")
    spec = copy.deepcopy(t.reference_spec)
    spec["steps"][1]["r"] += 1.5  # bore 3 mm too big for the named thread
    assert not passed(spec, t.checks)


KNOWN_WEAK = [
    ("c010 intent: blank plate passes (slots unseen)", lambda: {"steps": [ref("c010")["steps"][0]]}, "c010", "intent"),
]


@pytest.mark.parametrize(("name", "make", "cid", "phrasing"), KNOWN_WEAK, ids=[k[0] for k in KNOWN_WEAK])
def test_known_weak_still_passes(name, make, cid, phrasing):
    assert passed(make(), checks(cid, phrasing)), f"{name}: now caught; move it to ADVERSARIAL and update README"
```

- [x] **Step 2: Run.** Run: `PY -m pytest tests/test_redteam.py -q -v`. Expected: every ADVERSARIAL case and every generated case PASS, which means the cheat fails. The KNOWN_WEAK case PASS means the cheat still passes.
  - An adversarial cheat that scores `passed`: do not weaken the test. Report it, and decide with the user whether it is a SimReady check gap (add to KNOWN_WEAK, and file a SimReady BACKLOG item on `main`) or a generator check bug (fix here).
  - Before adding the other Task 2 review weak cases (c023 any 30 mm block, c047 axial-hole fake, c048 no gradable check), read each concept's intent checks and write the concrete cheat spec against them. Add each one only once it really passes. The review's wording is a lead, not proof (Working Rule 7).
- [x] **Step 3: Commit**

```bash
git add tests/test_redteam.py && git commit -m "test: red-team suite and known weak checks"
```

### Task B7: README (Hub page) and lint

**Files:**
- Create: `README.md`
- Modify: `scripts/lint-arch.sh` only if `src` is missing from `SRC_DIRS` (it is present as of 2026-11-14)

- [x] **Step 1: Write `README.md`.** The Hub displays this file. Sections, in order:
  1. One-paragraph summary.
  2. **Task format:** the DSL and the single-turn JSON answer.
  3. **Splits:** `eval` (111, never train on it) and `train` (seeded generator).
  4. **Reward:** `passed` / `partial`, and the funnel metrics.
  5. **Failure classes:** F1–F9.
  6. **Reward-hacking table:** every ADVERSARIAL and KNOWN_WEAK row from `tests/test_redteam.py`, verbatim names with the result. Generate it by running `PY -m pytest tests/test_redteam.py -q -v` and copying the outcomes. Never type results by hand.
  7. **Results:** filled in by Task C1.
  8. **Install and run:** `prime env install …`, `vf-validate`, `vf-run` with `configs/*.toml`.
  9. **Provenance:** SimReady, the checker tag `bench-api-0.1`, and the concept sources.

  Every command in the README must have been run (global Working Rule 4).
- [x] **Step 2: Lint and tests.** Run: `bash scripts/lint-arch.sh && PY -m pytest -q -m "not slow"`. Expected: `lint-arch: PASS` and 0 failed.
- [x] **Step 3: Commit**

```bash
git add README.md && git commit -m "docs: README for the Hub page"
```

---

## Phase C: Runs and publishing

### Task C1: Model runs (USER GATE: spends API credit)

> **Done 2026-11-19 with deviations (Rule 2):** (1) Task C2's pin bump to `v0.5.1` ran first (`bc97f71`) so results carry the final checker. (2) Models changed by user decision: four small open models on Vultr (`qwen3.8-27b`, `nemotron-3-nano-omni-30b-a3b-reasoning`, `glm-5.3-flash`, `deepseek-v4.1-flash-1017`) instead of `glm-5.3` + `deepseek-v4.1-flash`; the frontier ceiling is cited from SimReady `bench_v1`. (3) Eval `max_tokens` raised 4096 → 16384 after the smoke showed truncation (`f180aa3`). Results: README § Results (`a49f370`).

- [x] **Step 1:** Ask the user to check the Vultr balance (SimReady `vultr-budget`; it was $5 on 2026-10-05). The estimate is about 1,110 calls. The user confirms before the run.
- [x] **Step 2: Check the run flags.** Run `.venv/bin/vf-run cad-rl-env --help`. Expected: confirm the spelling of the model, client base URL, API-key variable, rollout count and output directory flags. `RunnerConfig` at v0.4.1 has `model` (`-m`), `client.base_url`, `client.api_key_var`, `num_rollouts`, `output_dir` (`-o`).
- [x] **Step 3: Smoke run.** Load the key into the environment without printing it. The user exports `VULTR_API_KEY` in their shell, or uses SimReady's redacted loader. Then:
  ```bash
  .venv/bin/vf-run cad-rl-env @ configs/eval.toml -m glm-5.3 --client.base-url https://api.vultrinference.com/v1 --client.api-key-var VULTR_API_KEY -n 3 -r 1 -o outputs
  ```
  Adjust the config-file syntax to what `--help` shows. Expected: 3 traces with `passed`/`partial` and funnel metrics, and no `GraderError`.
- [x] **Step 4: Full runs.** Run detached with a log: all 111 eval prompts, 5 rollouts each, for `glm-5.3` and then `deepseek-v4.1-flash`. Both ids were verified in SimReady `sp1-resume` (a2). Re-check them with a list-models call the same day.
- [x] **Step 5: Report.** From `outputs/*/traces.jsonl`, compute per model and phrasing:
  - pass@1 (mean `passed`);
  - pass@5 (share of prompts with any pass);
  - the explicit − engineer gap;
  - the failure-class distribution.

  Write a small `scripts/report.py` with a test on a 3-trace fixture before using it on the real file. Put the table in README § Results, labelled with the model ids, dates, `CHECKER_VERSION` and the run's `simready` tag. If SimReady's `docs/validation/bench_v1.md` exists, compare against it and name the answer-channel difference (deviation 8). If it doesn't exist, say so.
- [x] **Step 6: Commit** the README update and `scripts/report.py` with its test.

### Task C2: Private Hub push (USER GATE: outward-facing)

- [ ] **Step 1:** The user runs `prime login`. Check that the username is set on the profile.
- [ ] **Step 2:** The user runs `prime env push --visibility=PRIVATE` from the repo root. Claude supplies the exact command after checking `prime env push --help`.
- [ ] **Step 3:** Hub smoke, in a throwaway venv in `/tmp`: `prime env install <owner>/cad-rl-env`, then `vf-validate cad-rl-env -n 5`. Expected: gold pass and noop valid, which proves the pip-only install pulls `simready[ocp]` from the git tag.

### Task C3: Audit and go public (USER GATE)

- [ ] **Step 1:** Credential and PII history audit per global Working Rule 3, in both directions:
  - every `.env` value checked against `git log --all -S`;
  - every commit message and file in history grepped for names, phones and emails.
- [ ] **Step 2:** The user creates the GitHub repo and makes it public, and makes the Hub environment public. This is one decision covering both, never one without the other.
- [ ] **Step 3:** The user adds the repo link to the Mercor application (Engineering Expert talent network).
- [ ] **Step 4:** Move this plan and the design spec to `docs/exec-plans/completed/` with `Status: done`. Close `cad-rl-env-ownership` in SimReady `BACKLOG.md` on `main`.

---

## Self-review (2026-11-14)

- **Spec coverage:**
  - D1 → A1–A5 and B1.
  - D2 → B2 (eval) and B3–B5 (train, leak guard).
  - D3 → B2 (`single_turn` stop, prompt).
  - D4 → B2 rewards and B5 `train.toml`.
  - D5 → prerequisite gate.
  - D6 → file map and boundary.
  - Error handling → B1 (F-classes, `GraderError`, timeout, determinism) and A4 (missing binding).
  - Testing → B1–B6.
  - Red-team → B6.
  - Model runs → C1.
  - Publishing → C2–C3.
  - Done-when 1–4 → B5/B6, B7, C1, C2/C3.
- **Placeholders:** SimReady Tasks A2–A4 point to code in the SP1 plan by line range instead of repeating ~1,100 lines. That code exists verbatim there and was the basis of the 111/111 result. Every deviation from it is written out here in full.
- **Type consistency:**
  - `grade_reply(reply, checks, expected_solids, timeout_s)` is the same in B1, B2, B3, B4, B5 and B6.
  - `FamilySample` and `GeneratedTask` fields are the same in B3, B4 and B5.
  - `signature` and `collides` are the same in B3 and B5.
  - `CadData` fields are the same in B2 and B5.
- **Open risks (checked during execution, not assumed):**
  - OCP symbol names (A1 Step 4).
  - Whether `AddOptimal` leaves a gap (A2 Step 5).
  - The `CadData(id=…)` field name (B2 Step 6).
  - `vf-run` flag spelling (C1 Step 2).
  - Flange boss semantics on fused faces (B4 Step 6).
