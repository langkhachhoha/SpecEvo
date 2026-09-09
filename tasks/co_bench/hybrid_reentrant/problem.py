"""SpecEvo example for CO-Bench :: Hybrid Reentrant Shop Scheduling  (category: Scheduling).

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

TASK = "Hybrid Reentrant Shop Scheduling"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    import heapq\n    n = kwargs["n_jobs"]\n    m = kwargs["n_machines"]\n    init_time = kwargs["init_time"]\n    setup = kwargs["setup_times"]\n    proc = kwargs["processing_times"]\n\n    def makespan(perm):\n        op1 = [0] * (n + 1)\n        assign = [0] * (n + 1)\n        heap = [(0, mid) for mid in range(1, m + 1)]\n        heapq.heapify(heap)\n        for job in range(1, n + 1):\n            av, mid = heapq.heappop(heap)\n            op1[job] = av + init_time\n            assign[job] = mid\n            heapq.heappush(heap, (op1[job], mid))\n        op2 = [0] * (n + 1)\n        cur = 0\n        for job in perm:\n            st = max(op1[job], cur)\n            op2[job] = st + setup[job - 1]\n            cur = op2[job]\n        by_m = {mid: [] for mid in range(1, m + 1)}\n        for job in range(1, n + 1):\n            by_m[assign[job]].append(job)\n        op3 = [0] * (n + 1)\n        for mid in range(1, m + 1):\n            cmt = 0\n            for job in sorted(by_m[mid]):\n                st = max(cmt, op2[job])\n                cmt = st + proc[job - 1]\n                op3[job] = cmt\n        return max(op3)\n\n    ids = list(range(1, n + 1))\n    cands = [\n        ids[:],                                          # natural\n        sorted(ids, key=lambda j: setup[j - 1]),          # shortest setup first\n        sorted(ids, key=lambda j: -setup[j - 1]),         # longest setup first\n        sorted(ids, key=lambda j: proc[j - 1]),           # shortest processing first\n    ]\n    best = min(cands, key=makespan)\n    return {"permutation": best, "batch_assignment": [(j % m) + 1 for j in range(n)]}\n'

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
