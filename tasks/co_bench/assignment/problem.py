"""SpecEvo example for CO-Bench :: Assignment problem  (category: Assignment).

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

TASK = "Assignment problem"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    import numpy as np\n    n = kwargs["n"]\n    C = np.asarray(kwargs["cost_matrix"], dtype=float)\n    if np.isinf(C).any():  # replace inf so the solver stays finite\n        finite = C[np.isfinite(C)]\n        big = (finite.max() * 1e6) if finite.size else 1e18\n        C = np.where(np.isinf(C), big, C)\n    try:\n        from scipy.optimize import linear_sum_assignment\n        r, c = linear_sum_assignment(C)\n        assignment = [(int(i) + 1, int(j) + 1) for i, j in zip(r, c)]\n        total = float(C[r, c].sum())\n        return {"total_cost": total, "assignment": assignment}\n    except Exception:\n        # greedy fallback: cheapest available agent per item (row order)\n        used = np.zeros(n, dtype=bool)\n        assignment = []\n        total = 0.0\n        for i in range(n):\n            row = C[i].copy()\n            row[used] = np.inf\n            j = int(np.argmin(row))\n            used[j] = True\n            assignment.append((i + 1, j + 1))\n            total += float(C[i, j])\n        return {"total_cost": total, "assignment": assignment}\n'

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
