"""SpecEvo example for CO-Bench :: Equitable partitioning problem  (category: Graph & set).

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

TASK = "Equitable partitioning problem"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    data = kwargs["data"]\n    n = len(data)\n    A = len(data[0]) if n else 0\n    G = 8\n    gcount = [[0] * A for _ in range(G)]  # per-group attribute-1 counts\n    gsize = [0] * G\n    assign = [0] * n\n    for i in range(n):\n        if i < G:            # seed each group once so exactly 8 groups are used\n            g = i\n        else:\n            ones = [a for a in range(A) if data[i][a] == 1]\n            if ones:\n                g = min(range(G), key=lambda g: (sum(gcount[g][a] for a in ones), gsize[g]))\n            else:\n                g = min(range(G), key=lambda g: gsize[g])\n        assign[i] = g + 1\n        gsize[g] += 1\n        for a in range(A):\n            if data[i][a] == 1:\n                gcount[g][a] += 1\n    return {"assignment": assign}\n'

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
