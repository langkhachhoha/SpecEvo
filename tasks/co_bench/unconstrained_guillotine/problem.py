"""SpecEvo example for CO-Bench :: Unconstrained guillotine cutting  (category: Cutting).

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

TASK = "Unconstrained guillotine cutting"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    SW = kwargs["stock_width"]\n    SH = kwargs["stock_height"]\n    pieces = kwargs["pieces"]\n    rot = kwargs["allow_rotation"]\n    placements = []\n    cur_x = 0\n    cur_y = 0\n    shelf_h = 0\n    # shelf (next-fit) packing, most valuable pieces first; each piece used once.\n    for pid in sorted(pieces.keys(), key=lambda k: -pieces[k]["value"]):\n        p = pieces[pid]\n        opts = [(0, p["l"], p["w"])]\n        if rot:\n            opts.append((1, p["w"], p["l"]))\n        done = False\n        for orient, pw, ph in opts:  # try current shelf\n            if cur_x + pw <= SW and cur_y + ph <= SH:\n                placements.append({"piece_id": pid, "x": cur_x, "y": cur_y, "orientation": orient})\n                cur_x += pw\n                shelf_h = max(shelf_h, ph)\n                done = True\n                break\n        if done:\n            continue\n        ny = cur_y + shelf_h  # open a new shelf\n        for orient, pw, ph in opts:\n            if pw <= SW and ny + ph <= SH:\n                cur_y = ny\n                cur_x = 0\n                shelf_h = ph\n                placements.append({"piece_id": pid, "x": 0, "y": cur_y, "orientation": orient})\n                cur_x = pw\n                break\n    return {"placements": placements}\n'

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
