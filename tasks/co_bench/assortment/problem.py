"""SpecEvo example for CO-Bench :: Assortment problem  (category: Cutting).

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

TASK = "Assortment problem"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    m = kwargs["m"]\n    stocks = kwargs["stocks"]\n    pieces = kwargs["pieces"]\n    need = [p["min"] for p in pieces]\n    maxc = [p["max"] for p in pieces]\n\n    def piece_fits(p, s):\n        L, W = p["length"], p["width"]\n        SL, SW = s["length"], s["width"]\n        return (L <= SL + 1e-9 and W <= SW + 1e-9) or (W <= SL + 1e-9 and L <= SW + 1e-9)\n\n    # pick smallest-area stock type that can hold every piece type (with rotation)\n    cand = [(s["length"] * s["width"], si) for si, s in enumerate(stocks)\n            if all(piece_fits(p, s) for p in pieces)]\n    if not cand:\n        cand = [(-(s["length"] * s["width"]), si) for si, s in enumerate(stocks)]\n    cand.sort()\n    sidx = cand[0][1]\n    S = stocks[sidx]\n    SL, SW = S["length"], S["width"]\n\n    instances = []  # {\'pl\':[...], \'x\',\'y\',\'h\'}\n\n    def try_place(inst, p, ti):\n        L, W = p["length"], p["width"]\n        for orient, pw, ph in ((0, L, W), (1, W, L)):\n            if pw > SL + 1e-9 or ph > SW + 1e-9:\n                continue\n            if inst["x"] + pw <= SL + 1e-9 and inst["y"] + ph <= SW + 1e-9:\n                inst["pl"].append({"piece": ti + 1, "x": inst["x"], "y": inst["y"], "orientation": orient})\n                inst["x"] += pw\n                inst["h"] = max(inst["h"], ph)\n                return True\n            ny = inst["y"] + inst["h"]\n            if pw <= SL + 1e-9 and ny + ph <= SW + 1e-9:\n                inst["y"] = ny\n                inst["x"] = pw\n                inst["h"] = ph\n                inst["pl"].append({"piece": ti + 1, "x": 0, "y": ny, "orientation": orient})\n                return True\n        return False\n\n    counts = [0] * m\n\n    def place(ti):\n        p = pieces[ti]\n        for inst in instances:\n            if try_place(inst, p, ti):\n                counts[ti] += 1\n                return True\n        inst = {"pl": [], "x": 0, "y": 0, "h": 0}\n        instances.append(inst)\n        if try_place(inst, p, ti):\n            counts[ti] += 1\n            return True\n        instances.pop()  # piece cannot fit even an empty stock\n        return False\n\n    for ti in range(m):  # required minimum counts\n        for _ in range(need[ti]):\n            place(ti)\n\n    changed = True  # top up open stocks to cut waste (never exceed max)\n    while changed:\n        changed = False\n        for ti in range(m):\n            if counts[ti] >= maxc[ti]:\n                continue\n            p = pieces[ti]\n            for inst in instances:\n                if try_place(inst, p, ti):\n                    counts[ti] += 1\n                    changed = True\n                    break\n\n    placements = {}\n    for i, inst in enumerate(instances, 1):\n        placements[i] = {"stock_type": sidx + 1, "placements": inst["pl"]}\n    if not placements:\n        placements = {1: {"stock_type": sidx + 1, "placements": []}}\n    return {"objective": 0.0, "placements": placements}\n'

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
