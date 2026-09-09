"""SpecEvo example for CO-Bench :: Set partitioning  (category: Graph & set).

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

TASK = "Set partitioning"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    R = kwargs["num_rows"]\n    cols = kwargs["columns_info"]\n    covered = set()\n    remaining = set(range(1, R + 1))\n    selected = []\n    # greedy exact cover: pick non-overlapping columns by best cost / new-rows ratio\n    while remaining:\n        best, best_key, best_rows = None, None, None\n        for c, (cost, rows) in cols.items():\n            if rows & covered:\n                continue                    # would cover a row twice\n            new = len(rows & remaining)\n            if new <= 0:\n                continue\n            key = cost / new\n            if best_key is None or key < best_key:\n                best_key, best, best_rows = key, c, rows\n        if best is None:\n            break                           # stuck -> infeasible (scores 0)\n        selected.append(best)\n        covered |= best_rows\n        remaining -= best_rows\n    return {"selected_columns": sorted(selected)}\n'

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
