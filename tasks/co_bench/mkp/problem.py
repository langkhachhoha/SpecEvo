"""SpecEvo example for CO-Bench :: Multidimensional knapsack problem  (category: Packing).

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

TASK = "Multidimensional knapsack problem"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    n = kwargs["n"]\n    m = kwargs["m"]\n    p = kwargs["p"]\n    r = kwargs["r"]\n    b = kwargs["b"]\n    x = [0] * n\n    used = [0.0] * m\n\n    def ratio(j):\n        denom = 0.0\n        for i in range(m):\n            if b[i] > 0:\n                denom += r[i][j] / b[i]\n        return (p[j] / denom) if denom > 1e-12 else p[j]\n\n    order = sorted(range(n), key=lambda j: -ratio(j))\n    for j in order:\n        ok = True\n        for i in range(m):\n            if used[i] + r[i][j] > b[i] + 1e-9:\n                ok = False\n                break\n        if ok:\n            x[j] = 1\n            for i in range(m):\n                used[i] += r[i][j]\n    return {"x": x}\n'

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
