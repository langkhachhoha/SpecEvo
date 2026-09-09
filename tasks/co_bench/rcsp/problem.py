"""SpecEvo example for CO-Bench :: Resource constrained shortest path  (category: Routing).

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

TASK = "Resource constrained shortest path"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    import heapq\n    n = kwargs["n"]\n    K = kwargs["K"]\n    lb = kwargs["lower_bounds"]\n    ub = kwargs["upper_bounds"]\n    vres = kwargs["vertex_resources"]\n    graph = kwargs["graph"]\n\n    start_res = tuple(vres[0][k] for k in range(K))\n    if any(start_res[k] > ub[k] + 1e-6 for k in range(K)):\n        return {"total_cost": 0.0, "path": [1, n]}\n\n    use_dom = all(lb[k] <= 1e-9 for k in range(K))  # dominance is only sound w/o >= bounds\n    labels = [(0.0, start_res, 1, -1)]  # (cost, res, vertex, parent)\n    node_labels = {i: [] for i in range(1, n + 1)}\n    node_labels[1].append(0)\n    pq = [(0.0, 0)]\n    CAP = 200000\n    best = None\n\n    def dominated(v, cost, res):\n        for li in node_labels[v]:\n            lc, lr, _, _ = labels[li]\n            if lc <= cost + 1e-9 and all(lr[k] <= res[k] + 1e-9 for k in range(K)):\n                return True\n        return False\n\n    while pq and len(labels) < CAP:\n        c, li = heapq.heappop(pq)\n        cost, res, u, par = labels[li]\n        if c > cost + 1e-9:\n            continue\n        if u == n and all(res[k] >= lb[k] - 1e-6 for k in range(K)):\n            best = li\n            break\n        for (v, ac, ar) in graph.get(u, []):\n            nres = tuple(res[k] + ar[k] + vres[v - 1][k] for k in range(K))\n            if any(nres[k] > ub[k] + 1e-6 for k in range(K)):\n                continue\n            if use_dom and dominated(v, cost + ac, nres):\n                continue\n            idx = len(labels)\n            labels.append((cost + ac, nres, v, li))\n            node_labels[v].append(idx)\n            heapq.heappush(pq, (cost + ac, idx))\n            if len(labels) >= CAP:\n                break\n\n    if best is None:\n        return {"total_cost": 0.0, "path": [1, n]}\n    path = []\n    li = best\n    while li != -1:\n        path.append(labels[li][2])\n        li = labels[li][3]\n    path.reverse()\n    return {"total_cost": labels[best][0], "path": path}\n'

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
