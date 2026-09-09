"""SpecEvo example for CO-Bench :: Packing unequal rectangles and squares area  (category: Packing).

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

TASK = "Packing unequal rectangles and squares area"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    import math\n    n = kwargs["n"]\n    cx = kwargs["cx"]\n    cy = kwargs["cy"]\n    R = kwargs["R"]\n    items = kwargs["items"]\n    rotation = kwargs["rotation"]\n    placements = [(-1, -1, 0)] * n\n    placed = []  # (xmin, xmax, ymin, ymax)\n\n    def find(hL, hW):\n        g = 60\n        cand = []\n        for a in range(g + 1):\n            x = cx - R + 2 * R * a / g\n            for b in range(g + 1):\n                y = cy - R + 2 * R * b / g\n                if abs(x + 1) < 1e-6 and abs(y + 1) < 1e-6:\n                    continue\n                # all four corners inside the circle\n                if math.hypot(abs(x - cx) + hL, abs(y - cy) + hW) > R - 1e-6:\n                    continue\n                cand.append((x, y))\n        cand.sort(key=lambda p: (p[1], p[0]))\n        for (x, y) in cand:\n            xmin, xmax = x - hL, x + hL\n            ymin, ymax = y - hW, y + hW\n            ok = True\n            for (oxmin, oxmax, oymin, oymax) in placed:\n                if not (xmax <= oxmin + 1e-6 or xmin >= oxmax - 1e-6 or\n                        ymax <= oymin + 1e-6 or ymin >= oymax - 1e-6):\n                    ok = False\n                    break\n            if ok:\n                return (x, y, xmin, xmax, ymin, ymax)\n        return None\n\n    def try_item(i):\n        L, W = items[i]\n        opts = [(0, L / 2.0, W / 2.0)]\n        if rotation:\n            opts.append((90, W / 2.0, L / 2.0))\n        for theta, hL, hW in opts:\n            res = find(hL, hW)\n            if res is not None:\n                x, y, xmin, xmax, ymin, ymax = res\n                placements[i] = (x, y, theta)\n                placed.append((xmin, xmax, ymin, ymax))\n                return True\n        return False\n\n    # maximise packed area: largest-area items first.\n    for i in sorted(range(n), key=lambda k: -(items[k][0] * items[k][1])):\n        try_item(i)\n    return {"placements": placements}\n'

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
