"""SpecEvo example for CO-Bench :: Aircraft landing  (category: Scheduling).

Exports PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM and score_fn so
`scripts/run_specevo.py` can evolve a CO-Bench `solve` function. Scoring reuses
the shared CO-Bench engine (benchmarks/co_bench/cobench_eval.py): higher is
better, 1.0 == best-known; errors / infeasibility / timeouts score 0.
"""
import os
import sys
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(_REPO / "benchmarks" / "co_bench"))
import cobench_eval as ce  # noqa: E402

TASK = "Aircraft landing"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    planes = kwargs["planes"]\n    sep = kwargs["separation"]\n    R = kwargs["num_runways"]\n    n = kwargs["num_planes"]\n    # process planes by target time; greedily place on the runway giving least penalty\n    order = sorted(range(n), key=lambda i: planes[i]["target"])\n    runways = [[] for _ in range(R)]  # each: list of (landing_time, plane_idx)\n    schedule = {}\n    for idx in order:\n        p = planes[idx]\n        e, t, l = p["earliest"], p["target"], p["latest"]\n        best = None  # (penalty, landing, runway)\n        for r in range(R):\n            needed = e\n            for (lt, pi) in runways[r]:\n                needed = max(needed, lt + sep[pi][idx])\n            if needed > l + 1e-9:\n                continue\n            landing = max(needed, min(t, l))  # closest feasible time to target\n            if landing < t:\n                pen = (t - landing) * p["penalty_early"]\n            elif landing > t:\n                pen = (landing - t) * p["penalty_late"]\n            else:\n                pen = 0.0\n            if best is None or pen < best[0]:\n                best = (pen, landing, r)\n        if best is None:  # infeasible everywhere -> place at latest on runway 0\n            schedule[idx + 1] = {"landing_time": float(l), "runway": 1}\n            runways[0].append((l, idx))\n        else:\n            _, landing, r = best\n            schedule[idx + 1] = {"landing_time": float(landing), "runway": r + 1}\n            runways[r].append((landing, idx))\n    return {"schedule": schedule}\n'

INPUTS = None


def _cap(name, default):
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    v = int(raw)
    return None if v == 0 else v


def _source_of(fn) -> str:
    # SpecEvo stashes the candidate's full source in the function's globals.
    src = getattr(fn, "__globals__", {}).get("__source_code__")
    if isinstance(src, str) and src.strip():
        return src
    import inspect
    return inspect.getsource(fn)


def score_fn(solve_fn, _inputs=None) -> dict:
    try:
        source = _source_of(solve_fn)
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not recover candidate source: {e}"}
    r = ce.evaluate_source(
        TASK,
        source,
        max_cases=_cap("COBENCH_MAX_CASES", 10),
        max_instances=_cap("COBENCH_MAX_INSTANCES", 3),
    )
    out = {
        "score": r["score"],
        "valid": r["valid_rate"],
        "dev_score": r["dev_score"],
        "test_score": r["test_score"],
        "overall_score": r["overall_score"],
    }
    if "error" in r:
        out["error"] = r["error"]
    return out
