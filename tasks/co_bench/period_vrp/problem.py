"""SpecEvo example for CO-Bench :: Vehicle routing: period routing  (category: Routing).

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

TASK = "Vehicle routing: period routing"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    import math\n    depot = kwargs["depot"]\n    customers = kwargs["customers"]\n    vpd = kwargs["vehicles_per_day"]\n    cap = kwargs["vehicle_capacity"]\n    P = kwargs["period_length"]\n    by_id = {c["id"]: c for c in customers}\n\n    def dist(a, b):\n        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])\n\n    budget = [vpd[d] * cap for d in range(P)]\n    load = [0.0] * P\n    selected = {}\n    # choose the candidate schedule that keeps daily load most balanced (heaviest first)\n    for c in sorted(customers, key=lambda c: -c["demand"]):\n        scheds = c["schedules"]\n        if not scheds:\n            selected[c["id"]] = [0] * P\n            continue\n        best, best_key = None, None\n        for sc in scheds:\n            days = [d for d in range(P) if sc[d] == 1]\n            key = max(((load[d] + c["demand"]) / budget[d] if budget[d] > 0 else 9e9)\n                      for d in days) if days else 0.0\n            if best_key is None or key < best_key:\n                best_key, best = key, sc\n        selected[c["id"]] = list(best)\n        for d in range(P):\n            if best[d] == 1:\n                load[d] += c["demand"]\n\n    tours = {}\n    for d in range(1, P + 1):\n        today = sorted((cid for cid in selected if selected[cid][d - 1] == 1),\n                       key=lambda cid: -by_id[cid]["demand"])\n        V = vpd[d - 1]\n        bins = [[0.0, []] for _ in range(max(V, 1))]  # never exceed V tours\n        for cid in today:\n            dem = by_id[cid]["demand"]\n            fit = [b for b in bins if b[0] + dem <= cap + 1e-9]\n            b = max(fit, key=lambda b: cap - b[0]) if fit else min(bins, key=lambda b: b[0])\n            b[0] += dem\n            b[1].append(cid)\n        day_tours = []\n        for _, cids in bins:\n            if not cids:\n                continue\n            route, remaining, prev = [], set(cids), depot  # nearest-neighbour order\n            while remaining:\n                nxt = min(remaining, key=lambda cid: dist(prev, by_id[cid]))\n                route.append(nxt)\n                remaining.discard(nxt)\n                prev = by_id[nxt]\n            day_tours.append([0] + route + [0])\n        tours[d] = day_tours\n    return {"selected_schedules": selected, "tours": tours}\n'

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
