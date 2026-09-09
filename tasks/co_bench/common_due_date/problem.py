"""SpecEvo example for CO-Bench :: Common due date scheduling  (category: Scheduling).

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

TASK = "Common due date scheduling"

PROBLEM_DESCRIPTION = ce.problem_description(TASK)

FUNCTION_SIGNATURE = ce.solve_template(TASK)

SEED_PROGRAM = 'def solve(**kwargs):\n    jobs = kwargs["jobs"]\n    h = kwargs.get("h", 0.6)\n    n = len(jobs)\n    total_p = sum(p for p, a, b in jobs)\n    d = int(total_p * h)\n\n    def penalty(perm):\n        c = 0\n        tot = 0\n        for idx in perm:\n            p, a, b = jobs[idx - 1]\n            c += p\n            if c < d:\n                tot += a * (d - c)\n            elif c > d:\n                tot += b * (c - d)\n        return tot\n\n    ids = list(range(1, n + 1))\n    cands = []\n    cands.append(ids[:])                                            # identity\n    cands.append(sorted(ids, key=lambda i: jobs[i - 1][0]))         # SPT\n    cands.append(sorted(ids, key=lambda i: -jobs[i - 1][0]))        # LPT\n    # V-shape around the due date: early jobs sorted by p/a desc, tardy by p/b asc\n    early = sorted(ids, key=lambda i: -(jobs[i - 1][0] / max(jobs[i - 1][1], 1e-9)))\n    tardy = sorted(ids, key=lambda i: (jobs[i - 1][0] / max(jobs[i - 1][2], 1e-9)))\n    used, seq, load = set(), [], 0\n    for i in early:  # fill up to the due date, then the rest\n        if load + jobs[i - 1][0] <= d:\n            seq.append(i)\n            used.add(i)\n            load += jobs[i - 1][0]\n    seq += [i for i in tardy if i not in used]\n    cands.append(seq)\n    best = min(cands, key=penalty)\n    return {"schedule": best}\n'

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
