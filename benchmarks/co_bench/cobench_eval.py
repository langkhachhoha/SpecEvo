"""Shared CO-Bench evaluation engine for SpecEvo Baselines.

This module adapts the CO-Bench benchmark (Sun et al., 2025 —
https://github.com/sunnweiwei/CO-Bench) so a single ``solve`` function can be
scored the same way for BOTH integration paths in this repo:

  * the *baseline* runners (openevolve/gepa/adaevolve/evox) via
    ``benchmarks/co_bench/<slug>/evaluator.py``; and
  * the *SpecEvo / SpecEvo* runner via ``tasks/co_bench/<slug>/problem.py``.

Each CO-Bench task ships a ``config.py`` (vendored under ``data/<task>/``)
exporting ``load_data``, ``eval_func``, ``DESCRIPTION`` and a ``solve``
template, plus optionally ``norm_score`` and ``get_dev``.  We evaluate a
candidate ``solve`` on the task's test-case files exactly as CO-Bench does:
each *instance* runs in isolation under a hard per-instance time limit (10s in
the paper), raw scores are normalised against the best-known objective, and the
per-problem score is the mean over instances (a program error, constraint
violation or timeout scores 0 for that instance).

Two per-instance execution strategies keep the semantics identical across both
paths while respecting their process models:

  * **fork subprocess** (default, non-daemon caller e.g. the baseline
    evaluator): hard-kills a runaway/looping ``solve`` — matches CO-Bench.
  * **in-process SIGALRM** (daemon caller e.g. a SpecEvo worker, which forbids
    child processes): interrupts pure-Python loops in the worker's main thread.

Every task is split into a **dev** set (what the search optimises) and a
disjoint **test** set (held out, reported but never optimised).  The split is
deterministic: flatten all evaluated instances in case order (files sorted by
name) then instance order, take the first ``COBENCH_DEV_FRAC`` (7/10 by
default) as dev and the remaining tail as test.  With the default cap of 10
files x 3 instances (upper bound 30) that is roughly a 21/9 dev/test split.
The vendored ``get_dev`` is ignored so all tasks share this scheme.

Runtime knobs (env vars, all optional):
  COBENCH_TIMEOUT        per-instance time limit in seconds (default 10)
  COBENCH_MAX_CASES      cap on number of test-case files evaluated (default 10)
  COBENCH_MAX_INSTANCES  cap on instances per test-case file (default 3)
  COBENCH_DEV_FRAC       fraction of instances used for the dev split (default 0.7)
"""

from __future__ import annotations

import importlib.util
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

# Directory holding the vendored CO-Bench tasks: data/<task>/config.py + files.
DATA_ROOT = Path(__file__).resolve().parent / "data"

# Canonical slug -> CO-Bench task name (one representative per CO-Bench category,
# plus the full Packing category: all 9 CO-Bench packing problems).
TASKS: dict[str, str] = {
    "bin_packing_1d": "Bin packing - one-dimensional",          # Packing
    "mdmkp": "Multi-Demand Multidimensional Knapsack problem",  # Packing
    "mkp": "Multidimensional knapsack problem",                 # Packing
    "container_loading": "Container loading",                   # Packing
    "container_loading_weight": "Container loading with weight restrictions",  # Packing
    "packing_circles": "Packing unequal circles",              # Packing
    "packing_circles_area": "Packing unequal circles area",    # Packing
    "packing_rectangles": "Packing unequal rectangles and squares",       # Packing
    "packing_rectangles_area": "Packing unequal rectangles and squares area",  # Packing
    "non_guillotine_cutting": "Constrained non-guillotine cutting",  # Cutting
    "assortment": "Assortment problem",                         # Cutting
    "constrained_guillotine": "Constrained guillotine cutting",  # Cutting
    "unconstrained_guillotine": "Unconstrained guillotine cutting",  # Cutting
    "warehouse_location_uncap": "Uncapacitated warehouse location",  # Facility location
    "warehouse_location_cap": "Capacitated warehouse location",  # Facility location
    "pmedian_cap": "p-median - capacitated",                   # Facility location
    "pmedian_uncap": "p-median - uncapacitated",               # Facility location
    "flow_shop": "Flow shop scheduling",                        # Scheduling
    "aircraft_landing": "Aircraft landing",                    # Scheduling
    "crew_scheduling": "Crew scheduling",                      # Scheduling
    "common_due_date": "Common due date scheduling",           # Scheduling
    "hybrid_reentrant": "Hybrid Reentrant Shop Scheduling",    # Scheduling
    "job_shop": "Job shop scheduling",                         # Scheduling
    "open_shop": "Open shop scheduling",                       # Scheduling
    "tsp": "Travelling salesman problem",                       # Routing
    "period_vrp": "Vehicle routing: period routing",           # Routing
    "rcsp": "Resource constrained shortest path",              # Routing
    "gap": "Generalised assignment problem",                    # Assignment
    "assignment": "Assignment problem",                        # Assignment
    "steiner": "Euclidean Steiner problem",                     # Tree
    "corporate_structuring": "Corporate structuring",          # Tree
    "graph_coloring": "Graph colouring",                        # Graph & set
    "mis": "Maximal independent set",                          # Graph & set
    "equitable_partitioning": "Equitable partitioning problem",  # Graph & set
    "set_covering": "Set covering",                            # Graph & set
    "set_partitioning": "Set partitioning",                    # Graph & set
}

CATEGORY: dict[str, str] = {
    "bin_packing_1d": "Packing",
    "mdmkp": "Packing",
    "mkp": "Packing",
    "container_loading": "Packing",
    "container_loading_weight": "Packing",
    "packing_circles": "Packing",
    "packing_circles_area": "Packing",
    "packing_rectangles": "Packing",
    "packing_rectangles_area": "Packing",
    "non_guillotine_cutting": "Cutting",
    "assortment": "Cutting",
    "constrained_guillotine": "Cutting",
    "unconstrained_guillotine": "Cutting",
    "warehouse_location_uncap": "Facility location",
    "warehouse_location_cap": "Facility location",
    "pmedian_cap": "Facility location",
    "pmedian_uncap": "Facility location",
    "flow_shop": "Scheduling",
    "aircraft_landing": "Scheduling",
    "crew_scheduling": "Scheduling",
    "common_due_date": "Scheduling",
    "hybrid_reentrant": "Scheduling",
    "job_shop": "Scheduling",
    "open_shop": "Scheduling",
    "tsp": "Routing",
    "period_vrp": "Routing",
    "rcsp": "Routing",
    "gap": "Assignment",
    "assignment": "Assignment",
    "steiner": "Tree",
    "corporate_structuring": "Tree",
    "graph_coloring": "Graph & set",
    "mis": "Graph & set",
    "equitable_partitioning": "Graph & set",
    "set_covering": "Graph & set",
    "set_partitioning": "Graph & set",
}


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def default_timeout() -> float:
    raw = os.environ.get("COBENCH_TIMEOUT", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 10.0


# --------------------------------------------------------------------------- #
# CO-Bench scoring helpers (vendored verbatim from evaluation/utils.py so we do
# not depend on the CO-Bench package, which pulls litellm/openai/etc.).
# --------------------------------------------------------------------------- #
def _average_score(results: dict) -> float:
    if not results:
        return 0.0
    return sum(
        (sum(x if not isinstance(x, str) else 0 for x in scores) / len(scores)
         if not error_message and scores else 0)
        for scores, error_message in
        (results.get(case, (None, "No result")) for case in results.keys())
    ) / len(results)


def default_dev_frac() -> float:
    """Fraction of the ordered instance sequence assigned to the dev (search) split."""
    raw = os.environ.get("COBENCH_DEV_FRAC", "").strip()
    if raw:
        try:
            v = float(raw)
            if 0.0 < v < 1.0:
                return v
        except ValueError:
            pass
    return 0.7  # 7/3 dev/test split


def _split_indices(loaded: dict, dev_frac: float) -> tuple[dict, dict]:
    """Deterministic ordered dev/test split over the flattened instance list.

    Every task is split the SAME way (the vendored ``get_dev`` is ignored) so
    the search set and the held-out set are disjoint and reproducible.  We
    flatten all evaluated instances in case order (files sorted by name) then
    instance order, take the first ``dev_frac`` (7/10 by default) as **dev**
    (what the search optimises) and the remaining tail as **test** (held out,
    reported but never optimised).  Returns ``(dev, test)``, each a dict
    ``case -> [instance indices]``.
    """
    flat = [(case, idx) for case, insts in loaded.items() for idx in range(len(insts))]
    n = len(flat)
    if n == 0:
        return {}, {}
    n_dev = int(round(n * dev_frac))
    if n >= 2:
        n_dev = max(1, min(n - 1, n_dev))  # keep both splits non-empty
    dev: dict = {}
    test: dict = {}
    for case, idx in flat[:n_dev]:
        dev.setdefault(case, []).append(idx)
    for case, idx in flat[n_dev:]:
        test.setdefault(case, []).append(idx)
    return dev, test


def _select(results: dict, sel: dict) -> dict:
    """Keep only the instance indices named in ``sel`` (dict case -> [idx])."""
    out = {}
    for case, (scores, error_message) in results.items():
        if case not in sel:
            continue
        if scores is None:
            out[case] = (scores, error_message)
            continue
        idxs = sel[case]
        chosen = [s for i, s in enumerate(scores) if i in idxs]
        if chosen:
            out[case] = (chosen, error_message)
    return out


def _valid_rate(results: dict) -> float:
    total = numeric = 0
    for scores, err in results.values():
        if err or scores is None:
            total += 1
            continue
        for s in scores:
            total += 1
            if not isinstance(s, str):
                numeric += 1
    return numeric / total if total else 0.0


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_config(task: str, data_root: Path = DATA_ROOT):
    """Import a task's vendored config.py and return the module."""
    config_path = data_root / task / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"CO-Bench config not found: {config_path}")
    spec = importlib.util.spec_from_file_location(
        f"cobench_cfg_{abs(hash(str(config_path)))}", str(config_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod, str(config_path)


def problem_description(task: str, data_root: Path = DATA_ROOT) -> str:
    """The CO-Bench problem description + solve template (agent-facing prompt)."""
    import ast

    config_path = data_root / task / "config.py"
    src = config_path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    solve_src = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            solve_src = "\n".join(lines[node.lineno - 1: node.end_lineno])
            break
    mod, _ = load_config(task, data_root)
    return f"{mod.DESCRIPTION}\n\n# Implement in Solve Function\n\n{solve_src}"


def solve_template(task: str, data_root: Path = DATA_ROOT) -> str:
    import ast

    config_path = data_root / task / "config.py"
    src = config_path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    raise ValueError(f"No solve() found in {config_path}")


def list_test_cases(task: str, data_root: Path = DATA_ROOT) -> list[str]:
    """Names of the task's test-case entries (passed one at a time to load_data).

    Usually plain data files. A few CO-Bench tasks (e.g. Maximal independent
    set) instead ship each *case* as a sub-directory of instance files whose
    name matches a ``norm_score`` key, so directories are listed too. Hidden
    files (``.DS_Store``), ``__pycache__`` and the ``.py`` config are skipped.
    """
    d = data_root / task
    out = []
    for f in d.iterdir():
        if f.name.startswith(".") or f.name == "__pycache__":
            continue
        if f.is_file() and f.name.endswith(".py"):
            continue
        out.append(f.name)
    return sorted(out)


# --------------------------------------------------------------------------- #
# Per-instance execution
# --------------------------------------------------------------------------- #
def _subproc_target(solve_source, config_path, instance, q):
    # CRITICAL: this child is fork()'d from the search-runner process, so it
    # inherits the runner's SIGTERM/SIGINT handlers AND its shared mp.Event
    # shutdown flag. If we let a timeout-kill (SIGTERM) run the inherited handler,
    # the child would call request_shutdown() and set the *shared* event, aborting
    # the whole search after the first timed-out instance. Reset signal handlers
    # to default and detach the process group so killing this child never touches
    # the parent's shutdown state.
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, signal.SIG_DFL)
        except Exception:  # noqa: BLE001
            pass
    try:
        if hasattr(os, "setpgrp"):
            os.setpgrp()
    except Exception:  # noqa: BLE001
        pass
    try:
        ns: dict[str, Any] = {}
        exec(solve_source, ns)
        solve_fn = ns.get("solve")
        if not callable(solve_fn):
            q.put("Exception: candidate does not define solve()")
            return
        spec = importlib.util.spec_from_file_location("cobench_child_cfg", config_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        solution = solve_fn(**instance)
        solution = {str(k): v for k, v in solution.items()}
        score = cfg.eval_func(**instance, **solution)
        q.put(score)
    except Exception as e:  # noqa: BLE001
        q.put(f"Exception: {e}")


def _run_instance_subprocess(solve_source, config_path, instance, timeout) -> Any:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_subproc_target, args=(solve_source, config_path, instance, q))
    p.start()
    p.join(timeout + 1)
    if p.is_alive():
        # Use SIGKILL (uncatchable) rather than SIGTERM so no inherited handler
        # can run in the child — see _subproc_target for why that matters.
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            p.terminate()
        p.join(1)
        return f"Timeout ({timeout}s)"
    try:
        return q.get_nowait()
    except Exception:  # noqa: BLE001
        return "No result"


class _AlarmTimeout(Exception):
    pass


def _run_instance_sigalrm(solve_fn, eval_func, instance, timeout) -> Any:
    def _handler(signum, frame):
        raise _AlarmTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        solution = solve_fn(**instance)
        solution = {str(k): v for k, v in solution.items()}
        return eval_func(**instance, **solution)
    except _AlarmTimeout:
        return f"Timeout ({timeout}s)"
    except Exception as e:  # noqa: BLE001
        return f"Exception: {e}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _run_instance_thread(solve_fn, eval_func, instance, timeout) -> Any:
    """Soft-timeout fallback (daemon + non-main-thread): cannot hard-kill."""
    box: dict[str, Any] = {}

    def _work():
        try:
            solution = solve_fn(**instance)
            solution = {str(k): v for k, v in solution.items()}
            box["r"] = eval_func(**instance, **solution)
        except Exception as e:  # noqa: BLE001
            box["r"] = f"Exception: {e}"

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout + 1)
    if t.is_alive():
        return f"Timeout ({timeout}s)"
    return box.get("r", "No result")


# --------------------------------------------------------------------------- #
# Top-level evaluation
# --------------------------------------------------------------------------- #
def evaluate_source(
    task: str,
    solve_source: str,
    *,
    data_root: Path = DATA_ROOT,
    timeout: Optional[float] = None,
    max_cases: Optional[int] = None,
    max_instances: Optional[int] = None,
) -> dict[str, Any]:
    """Score a candidate ``solve`` (given as source text) on a CO-Bench task.

    Returns a metric dict:
      score / dev_score / test_score / overall_score  (higher is better, 1.0 = best-known)
      valid_rate, num_cases, num_instances, num_dev, num_test, feedback
    ``score`` == ``dev_score`` — the search signal, i.e. the mean over the DEV
    split (the first ``COBENCH_DEV_FRAC`` of the ordered instances).  The search
    only ever optimises this number.  ``test_score`` is the disjoint held-out
    tail (reported for generalisation, never optimised).  ``overall_score`` is
    the mean over EVERY evaluated instance (dev + test), kept for reference.
    """
    if timeout is None:
        timeout = default_timeout()
    if max_cases is None:
        max_cases = _env_int("COBENCH_MAX_CASES", None)
    if max_instances is None:
        max_instances = _env_int("COBENCH_MAX_INSTANCES", None)
    # Convention: 0 (or negative) means "no cap" — use every file / instance.
    # This matches the baseline evaluators' and SpecEvo problems' _cap() helper
    # (which maps 0 -> None) and the workflow help text ("set 0 for all
    # instances = full test set"). Without this, an env value of "0" reaches
    # the slice below as the literal int 0, so instances[:0] == [] yields an
    # empty score list and _average_score() divides by zero.
    if max_cases is not None and max_cases <= 0:
        max_cases = None
    if max_instances is not None and max_instances <= 0:
        max_instances = None

    cfg, config_path = load_config(task, data_root)
    cases = list_test_cases(task, data_root)
    if max_cases is not None:
        cases = cases[:max_cases]

    norm_score = getattr(cfg, "norm_score", None)

    # Decide per-instance execution strategy once.
    daemon = mp.current_process().daemon
    is_main_thread = threading.current_thread() is threading.main_thread()
    use_subprocess = not daemon

    # In-process paths need a compiled solve callable.
    solve_fn = None
    if not use_subprocess:
        ns: dict[str, Any] = {}
        try:
            exec(solve_source, ns)
            solve_fn = ns.get("solve")
        except Exception as e:  # noqa: BLE001
            return _error_result(f"Candidate failed to compile: {e}")
        if not callable(solve_fn):
            return _error_result("Candidate does not define solve()")

    # Load every case's instances up front (preserving order for norm_score).
    results: dict[str, tuple] = {}
    loaded: dict[str, list] = {}
    for case in cases:
        file_path = str(data_root / task / case)
        try:
            instances = cfg.load_data(file_path)
        except Exception as e:  # noqa: BLE001
            results[case] = (None, f"load_data error: {e}")
            continue
        if max_instances is not None:
            instances = instances[:max_instances]
        loaded[case] = list(instances)

    total_instances = sum(len(v) for v in loaded.values())
    # Evaluate every instance SEQUENTIALLY (one at a time), preserving order.
    # Non-daemon callers run each instance in a forked subprocess (hard-kill on
    # timeout); daemon callers (SpecEvo workers) use in-process SIGALRM / a soft
    # thread timeout.
    for case, instances in loaded.items():
        scores: list[Any] = []
        for inst in instances:
            if use_subprocess:
                s = _run_instance_subprocess(solve_source, config_path, inst, timeout)
            elif is_main_thread:
                s = _run_instance_sigalrm(solve_fn, cfg.eval_func, inst, timeout)
            else:
                s = _run_instance_thread(solve_fn, cfg.eval_func, inst, timeout)
            scores.append(s)
        results[case] = (scores, None)

    # Normalise raw scores against best-known objective.
    if callable(norm_score):
        try:
            results = norm_score(results)
        except Exception:  # noqa: BLE001 — fall back to raw on a broken norm
            pass

    # Deterministic 7/3 dev/test split over the flattened instance sequence.
    # The search optimises the DEV split; TEST is held out (reported only).
    dev_sel, test_sel = _split_indices(loaded, default_dev_frac())
    overall = _average_score(results)
    dev_score = _average_score(_select(results, dev_sel))
    test_score = _average_score(_select(results, test_sel))
    valid = _valid_rate(results)

    return {
        "score": dev_score,          # search signal — optimise the dev split
        "dev_score": dev_score,
        "test_score": test_score,
        "overall_score": overall,
        "valid_rate": valid,
        "num_cases": len(results),
        "num_instances": total_instances,
        "num_dev": sum(len(v) for v in dev_sel.values()),
        "num_test": sum(len(v) for v in test_sel.values()),
        "feedback": _feedback(results, overall, dev_score, test_score),
    }


def _error_result(msg: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "dev_score": 0.0,
        "test_score": 0.0,
        "overall_score": 0.0,
        "valid_rate": 0.0,
        "num_cases": 0,
        "num_instances": 0,
        "num_dev": 0,
        "num_test": 0,
        "feedback": msg,
        "error": msg,
    }


def _feedback(results: dict, avg: float, dev: Optional[float] = None,
              test: Optional[float] = None, feedback_length: int = 16) -> str:
    lines = []
    for case, (scores, err) in list(results.items())[:feedback_length]:
        if err:
            lines.append(f"{case} -> Caught Error: {err}")
        elif scores is None:
            lines.append(f"{case} -> No result")
        else:
            shown = [s if isinstance(s, str) else f"{float(s):.3f}" for s in scores][:feedback_length]
            lines.append(f"{case} -> Scores: {shown}")
    if dev is not None and test is not None:
        lines.append(f"Dev Score {dev} | Test Score {test} | Overall {avg}")
    else:
        lines.append(f"Avg Score {avg}")
    return "\n".join(lines)


def read_solve_source(program_path: str) -> str:
    """Return candidate source for the baseline path (whole file defines solve)."""
    return Path(program_path).read_text()
