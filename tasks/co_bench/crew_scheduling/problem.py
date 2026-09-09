"""SpecEvo example for CO-Bench :: Crew scheduling  (category: Scheduling).

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

TASK = "Crew scheduling"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    N = kwargs["N"]\n    K = kwargs["K"]\n    time_limit = kwargs["time_limit"]\n    tasks = kwargs["tasks"]\n    arcs = kwargs["arcs"]\n    order = sorted(range(1, N + 1), key=lambda t: (tasks[t][0], tasks[t][1]))\n    crews = []  # each: {"seq":[ids], "last":id, "first_start":s}\n    for t in order:\n        s, f = tasks[t]\n        best, best_cost = None, None\n        for cr in crews:\n            last = cr["last"]\n            if tasks[last][1] > s:               # would overlap\n                continue\n            if (last, t) not in arcs:            # no valid transition\n                continue\n            if f - cr["first_start"] > time_limit:  # duty time exceeded\n                continue\n            c = arcs[(last, t)]\n            if best_cost is None or c < best_cost:\n                best_cost, best = c, cr\n        if best is not None:\n            best["seq"].append(t)\n            best["last"] = t\n        elif len(crews) < K:\n            crews.append({"seq": [t], "last": t, "first_start": s})\n        else:\n            # no room: append to the crew with the earliest last-finish (may be infeasible)\n            cr = min(crews, key=lambda c: tasks[c["last"]][1])\n            cr["seq"].append(t)\n            cr["last"] = t\n    return {"crews": [cr["seq"] for cr in crews]}\n'

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
