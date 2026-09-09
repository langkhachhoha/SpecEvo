"""SpecEvo example for CO-Bench :: Container loading with weight restrictions  (category: Packing).

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

TASK = "Container loading with weight restrictions"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    L, W, H = kwargs["container"]\n    box_types = kwargs["box_types"]\n\n    def oriented(box, orient):\n        if orient == 1:\n            if box["length_flag"] != 1:\n                return None\n            return box["width"], box["height"], box["length"], box["lb1"]\n        if orient == 2:\n            if box["width_flag"] != 1:\n                return None\n            return box["length"], box["height"], box["width"], box["lb2"]\n        if box["height_flag"] != 1:\n            return None\n        return box["length"], box["width"], box["height"], box["lb3"]\n\n    best = None\n    for ti, box in enumerate(box_types):\n        vol = box["length"] * box["width"] * box["height"]\n        weight = box["weight"]\n        count = box["count"]\n        for orient in (1, 2, 3):\n            od = oriented(box, orient)\n            if od is None:\n                continue\n            dx, dy, dz, lb = od\n            if dx <= 0 or dy <= 0 or dz <= 0:\n                continue\n            nx = int(L // dx)\n            ny = int(W // dy)\n            nzc = int(H // dz)\n            if nx <= 0 or ny <= 0 or nzc <= 0:\n                continue\n            cap = dx * dy * lb\n            if weight > 0:\n                nz_load = int(cap // weight) + 1  # (k-1)*weight <= cap\n            else:\n                nz_load = nzc\n            nz = min(nzc, nz_load)\n            if nz <= 0:\n                continue\n            per_col = nx * ny\n            ncols = min(per_col, count // nz) if nz > 0 else 0\n            total = ncols * nz\n            if total <= 0:\n                continue\n            vtot = total * vol\n            if best is None or vtot > best[0]:\n                best = (vtot, ti, orient, dx, dy, dz, nx, ny, nz, ncols)\n    if best is None:\n        return {"instance": 1, "util": 0.0, "m": 0, "placements": []}\n    _, ti, orient, dx, dy, dz, nx, ny, nz, ncols = best\n    placements = []\n    col = 0\n    for iy in range(ny):\n        for ix in range(nx):\n            if col >= ncols:\n                break\n            for iz in range(nz):\n                placements.append({\n                    "box_type": ti + 1, "orientation": orient,\n                    "x": float(ix * dx), "y": float(iy * dy), "z": float(iz * dz),\n                })\n            col += 1\n        if col >= ncols:\n            break\n    return {"instance": 1, "util": 0.0, "m": len(placements), "placements": placements}\n'

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
