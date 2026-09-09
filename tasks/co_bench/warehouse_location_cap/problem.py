"""SpecEvo example for CO-Bench :: Capacitated warehouse location  (category: Facility location).

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

TASK = "Capacitated warehouse location"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    m = kwargs["m"]\n    n = kwargs["n"]\n    wh = kwargs["warehouses"]\n    cust = kwargs["customers"]\n    D = sum(c["demand"] for c in cust)\n    # open warehouses cheapest-fixed-cost first until capacity covers total demand\n    order = sorted(range(m), key=lambda i: (wh[i]["fixed_cost"], -wh[i]["capacity"]))\n    opened, capsum = [], 0.0\n    for i in order:\n        opened.append(i)\n        capsum += wh[i]["capacity"]\n        if capsum >= D - 1e-9:\n            break\n    remcap = [wh[i]["capacity"] for i in range(m)]\n    assignments = [[0.0] * m for _ in range(n)]\n    for j in range(n):\n        need = cust[j]["demand"]\n        costs = cust[j]["costs"]\n        for i in sorted(opened, key=lambda i: costs[i]):\n            if need <= 1e-12:\n                break\n            take = min(need, remcap[i])\n            if take > 0:\n                assignments[j][i] += take\n                remcap[i] -= take\n                need -= take\n        if need > 1e-9:  # spill to any remaining capacity (keeps feasibility)\n            for i in range(m):\n                if need <= 1e-12:\n                    break\n                take = min(need, remcap[i])\n                if take > 0:\n                    assignments[j][i] += take\n                    remcap[i] -= take\n                    need -= take\n    used = [sum(assignments[j][i] for j in range(n)) for i in range(m)]\n    warehouse_open = [1 if used[i] > 1e-12 else 0 for i in range(m)]\n    total = sum(wh[i]["fixed_cost"] for i in range(m) if warehouse_open[i])\n    for j in range(n):\n        d = cust[j]["demand"]\n        if d > 0:\n            total += sum(assignments[j][i] / d * cust[j]["costs"][i] for i in range(m))\n    return {"total_cost": total, "warehouse_open": warehouse_open, "assignments": assignments}\n'

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
