"""SpecEvo example for CO-Bench :: Packing unequal circles  (category: Packing).

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

TASK = "Packing unequal circles"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    n = kwargs["n"]\n    cx = kwargs["cx"]\n    cy = kwargs["cy"]\n    R = kwargs["R"]\n    radii = kwargs["radii"]\n    coords = [(-1, -1)] * n\n    placed = []  # (x, y, r)\n\n    def find(r):\n        lim = R - r\n        if lim < -1e-12:\n            return None\n        if lim <= 1e-12:\n            cand = [(cx, cy)]\n        else:\n            g = 70\n            cand = []\n            for a in range(g + 1):\n                x = cx - lim + 2 * lim * a / g\n                for b in range(g + 1):\n                    y = cy - lim + 2 * lim * b / g\n                    if (x - cx) ** 2 + (y - cy) ** 2 <= lim * lim + 1e-9:\n                        cand.append((x, y))\n            cand.sort(key=lambda p: (p[1], p[0]))\n        for (x, y) in cand:\n            if abs(x + 1) < 1e-6 and abs(y + 1) < 1e-6:\n                continue\n            ok = True\n            for (px, py, pr) in placed:\n                if (x - px) ** 2 + (y - py) ** 2 < (r + pr) ** 2 - 1e-9:\n                    ok = False\n                    break\n            if ok:\n                return (x, y)\n        return None\n\n    # maximise number: must pack a prefix of the (increasing-radius) list.\n    for i in range(n):\n        pos = find(radii[i])\n        if pos is None:\n            break\n        coords[i] = (pos[0], pos[1])\n        placed.append((pos[0], pos[1], radii[i]))\n    return {"coords": coords}\n'

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
