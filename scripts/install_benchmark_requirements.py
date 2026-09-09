#!/usr/bin/env python3
"""Install Python requirements for one or more SpecEvo Baselines benchmarks.

This helper supports native (non-Docker) benchmark runs. It locates nearby
requirements.txt files for a benchmark directory or evaluator path and installs
them into the current Python environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def _candidate_requirements(path: Path) -> list[Path]:
    """Return requirement files that can affect a benchmark/evaluator path."""
    if path.is_file():
        base_dir = path.parent
    else:
        base_dir = path

    candidates: list[Path] = []

    # Direct benchmark layout:
    #   benchmark/requirements.txt
    #   benchmark/evaluator.py
    candidates.append(base_dir / "requirements.txt")

    # Container-compatible layout used natively:
    #   benchmark/evaluator/evaluator.py
    #   benchmark/evaluator/requirements.txt
    candidates.append(base_dir / "evaluator" / "requirements.txt")

    # If the input itself is benchmark/evaluator/evaluator.py or benchmark/evaluator/.
    if base_dir.name == "evaluator":
        candidates.append(base_dir / "requirements.txt")
        candidates.append(base_dir.parent / "requirements.txt")

    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists() and candidate not in seen:
            result.append(candidate)
            seen.add(candidate)
    return result


def _installer_command(req: Path) -> list[str]:
    """Pick the best available installer for the current interpreter.

    ``uv sync`` creates venvs without pip, so ``python -m pip`` blows up in
    CI. Prefer ``uv pip install`` when ``uv`` is on PATH (it targets the
    active venv by default and matches how the rest of the repo installs
    things); otherwise fall back to ``python -m pip``.
    """
    uv = shutil.which("uv")
    if uv:
        env_python = os.environ.get("UV_PYTHON") or sys.executable
        return [uv, "pip", "install", "--python", env_python, "-r", str(req)]
    return [sys.executable, "-m", "pip", "install", "-r", str(req)]


def _install(requirements: list[Path], dry_run: bool) -> None:
    if not requirements:
        print("No requirements.txt files found.")
        return

    for req in requirements:
        print(f"Installing {req.relative_to(_repo_root())}")
        if dry_run:
            continue
        subprocess.run(_installer_command(req), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install requirements for native SpecEvo Baselines benchmark runs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Benchmark directories or evaluator files to inspect.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print requirement files without installing them.",
    )
    args = parser.parse_args()

    all_requirements: list[Path] = []
    seen: set[Path] = set()
    for raw_path in args.paths:
        path = _as_path(raw_path)
        if not path.exists():
            print(f"Path not found: {raw_path}", file=sys.stderr)
            return 1
        for req in _candidate_requirements(path):
            if req not in seen:
                all_requirements.append(req)
                seen.add(req)

    _install(all_requirements, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
