#!/usr/bin/env python3
"""Verify every benchmark task is loadable and correctly specified.

Imports each ``tasks/<...>/problem.py`` in its own subprocess and checks the
contract the two runners rely on: ``PROBLEM_DESCRIPTION``, ``FUNCTION_SIGNATURE``
and a callable ``score_fn``. Makes no API calls and costs nothing, so it is the
fastest way to confirm a fresh checkout is complete.

    python scripts/check_tasks.py                 # every suite
    python scripts/check_tasks.py --suite math    # math | adrs | co_bench | lsr_synth
    python scripts/check_tasks.py --verbose       # list every task, not just failures
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = REPO_ROOT / "tasks"

# suite -> (label, glob patterns relative to tasks/)
SUITES: dict[str, tuple[str, list[str]]] = {
    "math": (
        "Mathematical discovery",
        [
            "circle_packing", "circle_packing_rect", "heilbronn_triangle",
            "heilbronn_convex_13", "minmax_distance_2", "minmax_distance_3",
            "signal_processing",
        ],
    ),
    "adrs": ("Systems optimization (ADRS)", ["ADRS/*"]),
    "co_bench": ("CO-Bench", ["co_bench/*"]),
    "lsr_synth": ("LSR-Synth", ["llm_srbench/*/*"]),
}

PROBE = r"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("problem", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(__import__("pathlib").Path(sys.argv[1]).parent))
spec.loader.exec_module(mod)
missing = [a for a in ("PROBLEM_DESCRIPTION", "FUNCTION_SIGNATURE", "score_fn")
           if not hasattr(mod, a)]
if missing:
    raise SystemExit("missing attributes: " + ", ".join(missing))
if not callable(mod.score_fn):
    raise SystemExit("score_fn is not callable")
if not str(mod.PROBLEM_DESCRIPTION).strip():
    raise SystemExit("PROBLEM_DESCRIPTION is empty")
"""


def discover(suites: list[str]) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for key in suites:
        _, patterns = SUITES[key]
        for pattern in patterns:
            for path in sorted(TASKS.glob(pattern)):
                if (path / "problem.py").is_file():
                    found.append((key, path))
    return found


def check(path: Path, timeout: float) -> tuple[Path, bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE, str(path / "problem.py")],
            capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return path, False, f"timed out after {timeout:.0f}s"
    if proc.returncode == 0:
        return path, True, ""
    detail = (proc.stderr or proc.stdout).strip().splitlines()
    return path, False, detail[-1] if detail else f"exit {proc.returncode}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=sorted(SUITES), action="append",
                    help="Limit to one suite (repeatable). Default: all.")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="Per-task import timeout in seconds (default: 120).")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel probes (default: 8).")
    ap.add_argument("--verbose", action="store_true", help="List passing tasks too.")
    args = ap.parse_args()

    suites = args.suite or sorted(SUITES)
    tasks = discover(suites)
    if not tasks:
        print("No tasks found — is tasks/ populated?", file=sys.stderr)
        return 2

    print(f"Checking {len(tasks)} tasks across {len(suites)} suite(s)\n")
    failures: list[tuple[Path, str]] = []
    per_suite: dict[str, list[bool]] = {k: [] for k in suites}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(check, p, args.timeout): (k, p) for k, p in tasks}
        for fut in concurrent.futures.as_completed(futures):
            key, _ = futures[fut]
            path, ok, err = fut.result()
            per_suite[key].append(ok)
            rel = path.relative_to(REPO_ROOT)
            if not ok:
                failures.append((rel, err))
                print(f"  FAIL  {rel}\n        {err}")
            elif args.verbose:
                print(f"  ok    {rel}")

    print()
    for key in suites:
        results = per_suite[key]
        label, _ = SUITES[key]
        print(f"  {label:32s} {sum(results):3d} / {len(results):3d}")

    total_ok = sum(sum(v) for v in per_suite.values())
    print(f"\n  {'TOTAL':32s} {total_ok:3d} / {len(tasks):3d}")

    if failures:
        print(f"\n{len(failures)} task(s) failed. Missing benchmark data is the usual "
              f"cause — run: bash scripts/download_benchmark_data.sh")
        return 1
    print("\nAll tasks load correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
