"""SpecEvo example for CO-Bench :: Set covering  (category: Graph & set).

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

TASK = "Set covering"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    m = kwargs["m"]\n    n = kwargs["n"]\n    costs = kwargs["costs"]\n    row_cover = kwargs["row_cover"]\n    col_rows = [set() for _ in range(n + 1)]\n    for i in range(m):\n        for c in row_cover[i]:\n            col_rows[c].add(i)\n    uncovered = set(range(m))\n    selected = []\n    # greedy: repeatedly take the column with the best cost / newly-covered ratio\n    while uncovered:\n        best, best_key = None, None\n        for c in range(1, n + 1):\n            new = len(col_rows[c] & uncovered)\n            if new <= 0:\n                continue\n            key = costs[c - 1] / new\n            if best_key is None or key < best_key:\n                best_key, best = key, c\n        if best is None:\n            break\n        selected.append(best)\n        uncovered -= col_rows[best]\n    return {"selected_columns": selected}\n'

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
