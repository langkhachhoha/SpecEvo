"""SpecEvo example for CO-Bench :: Multi-Demand Multidimensional Knapsack problem  (category: Packing).

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

TASK = "Multi-Demand Multidimensional Knapsack problem"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    n = kwargs["n"]\n    m = kwargs["m"]\n    q = kwargs["q"]\n    A_leq = kwargs["A_leq"]\n    b_leq = kwargs["b_leq"]\n    A_geq = kwargs["A_geq"]\n    b_geq = kwargs["b_geq"]\n    c = kwargs["cost_vector"]\n\n    x = [0] * n\n    leq = [0] * m\n    geq = [0] * q\n\n    def can_add(j):\n        for i in range(m):\n            if leq[i] + A_leq[i][j] > b_leq[i]:\n                return False\n        return True\n\n    # Phase 1: greedily satisfy the >= (demand) constraints while respecting <=.\n    for _ in range(n):\n        deficits = [max(0, b_geq[i] - geq[i]) for i in range(q)]\n        if sum(deficits) <= 0:\n            break\n        best, best_val = None, 0\n        for j in range(n):\n            if x[j] == 1 or not can_add(j):\n                continue\n            contrib = 0\n            for i in range(q):\n                if deficits[i] > 0:\n                    contrib += min(A_geq[i][j], deficits[i])\n            if contrib > best_val:\n                best_val, best = contrib, j\n        if best is None:\n            break\n        x[best] = 1\n        for i in range(m):\n            leq[i] += A_leq[i][best]\n        for i in range(q):\n            geq[i] += A_geq[i][best]\n\n    feasible = all(geq[i] >= b_geq[i] for i in range(q))\n    if feasible:\n        # Phase 2: add profitable items that keep <= feasible.\n        for j in sorted(range(n), key=lambda k: -c[k]):\n            if x[j] == 1 or c[j] <= 0:\n                continue\n            if can_add(j):\n                x[j] = 1\n                for i in range(m):\n                    leq[i] += A_leq[i][j]\n    return {"x": x}\n'

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
