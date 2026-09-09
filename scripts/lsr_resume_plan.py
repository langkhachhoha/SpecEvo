#!/usr/bin/env python3
"""Work out how to resume one interrupted baseline problem.

``specevo-baselines-run --iterations N`` runs N *additional* iterations on top of a
checkpoint, so resuming correctly means knowing how much search a problem has
already had. Reading that off the checkpoint's name is not safe: when the runner
is asked to shut down early it still writes a final checkpoint numbered by the
*requested* budget, so a run that managed 10 of 30 iterations leaves behind a
directory called ``checkpoint_30``. Trusting that name silently truncates the
experiment to a third of its budget.

The program database inside the checkpoint does not lie — it holds one record per
evaluated program — so progress is measured as

    completed = min(label + 1, number of program records)

and the smaller of the two wins. On a clean run the two agree exactly; on an
early-shutdown checkpoint the record count exposes the inflated label.

Prints shell-assignable ``KEY=VALUE`` lines::

    LSR_CKPT=/abs/path/to/checkpoint_30   (empty when there is nothing to resume)
    LSR_COMPLETED=12
    LSR_REMAINING=18

Usage::

    python scripts/lsr_resume_plan.py --run-dir <problem dir> --iterations 500
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for d in run_dir.glob("**/checkpoints/checkpoint_*"):
        if not d.is_dir():
            continue
        m = re.search(r"checkpoint_(\d+)$", d.name)
        if m:
            found.append((int(m.group(1)), d))
    return sorted(found)


def _program_count(ckpt: Path) -> int | None:
    progs = ckpt / "programs"
    if progs.is_dir():
        n = len(list(progs.glob("*.json")))
        if n:
            return n
    # Older layouts kept the whole database in one file.
    for name in ("programs.json", "database.json"):
        f = ckpt / name
        if f.is_file():
            try:
                data = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                return len(data)
    return None


def plan(run_dir: Path, iterations: int) -> dict:
    cks = _checkpoints(run_dir)
    if not cks:
        return {"LSR_CKPT": "", "LSR_COMPLETED": 0, "LSR_REMAINING": iterations}

    label, path = cks[-1]
    by_label = label + 1
    counted = _program_count(path)
    completed = by_label if counted is None else min(by_label, counted)
    completed = max(completed, 0)
    return {
        "LSR_CKPT": str(path.resolve()),
        "LSR_COMPLETED": completed,
        "LSR_REMAINING": max(iterations - completed, 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iterations", type=int, required=True)
    args = ap.parse_args()

    for key, value in plan(Path(args.run_dir), args.iterations).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
