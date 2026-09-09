"""SpecEvo example for CO-Bench :: Container loading  (category: Packing).

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

TASK = "Container loading"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    L, W, H = kwargs["container"]\n    box_types = kwargs["box_types"]\n    best = None\n    for bt, info in box_types.items():\n        dims = info["dims"]\n        flags = info["flags"]\n        count = info["count"]\n        for v in range(3):\n            if flags[v] != 1:\n                continue\n            horz = [i for i in range(3) if i != v]\n            vert = dims[v]\n            for hswap in (0, 1):\n                h1 = dims[horz[0]]\n                h2 = dims[horz[1]]\n                if hswap:\n                    h1, h2 = h2, h1\n                if h1 <= 0 or h2 <= 0 or vert <= 0:\n                    continue\n                nx = int(L // h1)\n                ny = int(W // h2)\n                nz = int(H // vert)\n                cap = nx * ny * nz\n                if cap <= 0:\n                    continue\n                nboxes = min(cap, count)\n                vol = nboxes * h1 * h2 * vert\n                if best is None or vol > best[0]:\n                    best = (vol, bt, v, hswap, h1, h2, vert, nx, ny, nz, nboxes)\n    if best is None:\n        return {"placements": []}\n    _, bt, v, hswap, h1, h2, vert, nx, ny, nz, nboxes = best\n    placements = []\n    c = 0\n    for iz in range(nz):\n        for iy in range(ny):\n            for ix in range(nx):\n                if c >= nboxes:\n                    break\n                placements.append({\n                    "box_type": bt, "container_id": 0,\n                    "x": ix * h1, "y": iy * h2, "z": iz * vert,\n                    "v": v, "hswap": hswap,\n                })\n                c += 1\n    return {"placements": placements}\n'

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
