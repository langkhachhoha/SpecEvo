"""SpecEvo example for CO-Bench :: p-median - uncapacitated  (category: Facility location).

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

TASK = "p-median - uncapacitated"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    import numpy as np\n    n = kwargs["n"]\n    p = kwargs["p"]\n    D = np.asarray(kwargs["dist"], dtype=float)\n    best = np.full(n, np.inf)\n    chosen = []\n    chosen_set = set()\n    for _ in range(min(p, n)):\n        # cost if each candidate is added = sum_i min(best[i], D[i, c])\n        cand = np.minimum(best[:, None], D).sum(axis=0)\n        for c in chosen_set:\n            cand[c] = np.inf\n        c = int(np.argmin(cand))\n        chosen.append(c + 1)\n        chosen_set.add(c)\n        best = np.minimum(best, D[:, c])\n    # pad if p > n (degenerate) by adding any remaining vertices\n    i = 1\n    while len(chosen) < p and i <= n:\n        if i not in chosen:\n            chosen.append(i)\n        i += 1\n    return {"medians": chosen}\n'

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
