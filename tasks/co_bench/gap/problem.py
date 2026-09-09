"""SpecEvo example for CO-Bench :: Generalised assignment problem  (category: Assignment).

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

TASK = "Generalised assignment problem"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    m = kwargs["m"]\n    n = kwargs["n"]\n    cost = kwargs["cost_matrix"]\n    cons = kwargs["consumption_matrix"]\n    cap = list(kwargs["capacities"])\n    ptype = kwargs.get("problem_type", "max")\n    # Start: each job on the agent it consumes least on, then repair overflows.\n    assign = [min(range(m), key=lambda i: cons[i][j]) for j in range(n)]\n    load = [0.0] * m\n    for j in range(n):\n        load[assign[j]] += cons[assign[j]][j]\n    for _ in range(200 * n):\n        over = [i for i in range(m) if load[i] > cap[i] + 1e-9]\n        if not over:\n            break\n        i = max(over, key=lambda i: load[i] - cap[i])\n        best = None\n        for j in (j for j in range(n) if assign[j] == i):\n            for t in range(m):\n                if t == i or load[t] + cons[t][j] > cap[t] + 1e-9:\n                    continue\n                pen = (cost[i][j] - cost[t][j]) if ptype == "max" else (cost[t][j] - cost[i][j])\n                if best is None or pen < best[0]:\n                    best = (pen, j, t)\n        if best is None:\n            break\n        _, j, t = best\n        load[i] -= cons[i][j]\n        load[t] += cons[t][j]\n        assign[j] = t\n    return {"assignments": [a + 1 for a in assign]}\n'

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
