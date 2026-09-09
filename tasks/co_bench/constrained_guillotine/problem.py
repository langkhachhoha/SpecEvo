"""SpecEvo example for CO-Bench :: Constrained guillotine cutting  (category: Cutting).

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

TASK = "Constrained guillotine cutting"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    m = kwargs["m"]\n    L = kwargs["stock_length"]\n    W = kwargs["stock_width"]\n    pts = kwargs["piece_types"]\n    # A uniform grid of ONE identical piece type is always guillotine-separable.\n    best = None\n    for ti, p in enumerate(pts):\n        pl, pw, mx, val = p["length"], p["width"], p["max"], p["value"]\n        if pl <= 0 or pw <= 0 or mx <= 0:\n            continue\n        nx = L // pl\n        ny = W // pw\n        cap = nx * ny\n        if cap <= 0:\n            continue\n        cnt = min(cap, mx)\n        tot = cnt * val\n        if best is None or tot > best[0]:\n            best = (tot, ti, pl, pw, nx, cnt, val)\n    if best is None:\n        return {"total_value": 0, "placements": []}\n    _, ti, pl, pw, nx, cnt, val = best\n    placements = []\n    c = 0\n    tv = 0\n    iy = 0\n    while c < cnt:\n        for ix in range(nx):\n            if c >= cnt:\n                break\n            placements.append((ti + 1, ix * pl, iy * pw, pl, pw, 0))\n            c += 1\n            tv += val\n        iy += 1\n    return {"total_value": tv, "placements": placements}\n'

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
