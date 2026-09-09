"""SpecEvo example for CO-Bench :: p-median - capacitated  (category: Facility location).

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

TASK = "p-median - capacitated"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    import math\n    n = kwargs["n"]\n    p = kwargs["p"]\n    Q = kwargs["Q"]\n    customers = kwargs["customers"]\n    ids = [c[0] for c in customers]\n    xs = [c[1] for c in customers]\n    ys = [c[2] for c in customers]\n    dem = [c[3] for c in customers]\n\n    def dist(i, j):\n        return math.hypot(xs[i] - xs[j], ys[i] - ys[j])\n\n    # farthest-first: spread p medians, seeded at the highest-demand customer\n    start = max(range(n), key=lambda i: dem[i])\n    med = [start]\n    mind = [dist(i, start) for i in range(n)]\n    while len(med) < min(p, n):\n        nxt = max((i for i in range(n) if i not in set(med)), key=lambda i: mind[i])\n        med.append(nxt)\n        for i in range(n):\n            d = dist(i, nxt)\n            if d < mind[i]:\n                mind[i] = d\n\n    cap = {mi: Q for mi in med}\n    assign = [None] * n\n    # assign heaviest customers first to the nearest median with spare capacity\n    for i in sorted(range(n), key=lambda i: -dem[i]):\n        best, bestd = None, None\n        for mi in med:\n            if cap[mi] + 1e-9 >= dem[i]:\n                d = dist(i, mi)\n                if bestd is None or d < bestd:\n                    bestd, best = d, mi\n        if best is None:  # no spare capacity anywhere -> nearest (may be infeasible)\n            best = min(med, key=lambda mi: dist(i, mi))\n        assign[i] = best\n        cap[best] -= dem[i]\n\n    obj = sum(math.floor(dist(i, assign[i])) for i in range(n))\n    return {\n        "objective": obj,\n        "medians": [ids[mi] for mi in med],\n        "assignments": [ids[assign[i]] for i in range(n)],\n    }\n'

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
