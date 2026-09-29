#!/usr/bin/env python3
"""Record the authoritative ID / OOD result for one finished LSR-Synth problem.

Called after each problem's search
finishes. It finds whatever best program that run left on disk, **re-evaluates
it from scratch** with the shared engine, and appends one JSON line to the run's
``results.jsonl``.

Re-evaluating rather than trusting the search's own metric dict is deliberate:
it pins the reported numbers to the program that is actually on disk (the thing a
reader can inspect), and it means a run that was interrupted and resumed reports
the same way as one that ran straight through.

Usage::

    python scripts/lsr_finalize.py --domain chem_react --problem crk0 \\
        --method evox --run-dir outputs/.../problems/crk0 \\
        --results outputs/.../results.jsonl

Exit status: 0 if a program was found and scored, 1 if the run left no program
(the line is still written, with ``status = "no_program"``, so the aggregate
knows the problem was attempted).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "llm_srbench"))

import lsr_eval as L  # noqa: E402


def _newest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = []
    for d in run_dir.glob("**/checkpoints/checkpoint_*"):
        m = re.search(r"checkpoint_(\d+)$", d.name)
        if m and d.is_dir():
            ckpts.append((int(m.group(1)), d))
    if not ckpts:
        return None
    return max(ckpts)[1]


def find_best_program(run_dir: Path) -> tuple[Path | None, dict]:
    """Locate the best program a run produced, plus whatever it recorded about it.

    Probes both runners' layouts in order of authority rather than branching on
    the method name, so a run that died between writing a checkpoint and writing
    its final ``best/`` directory still yields its best-so-far program.
    """
    info: dict = {}

    # SpecEvo / SpecEvo: <run_dir>/best_program.py + summary.json
    specevo_best = run_dir / "best_program.py"
    specevo_summary = run_dir / "summary.json"
    if specevo_summary.exists():
        try:
            s = json.loads(specevo_summary.read_text())
            info = {
                "search_best_score": s.get("best_score"),
                "search_metrics": s.get("best_metrics"),
                "evaluations": s.get("total_evaluations"),
                "cost_usd": s.get("total_cost"),
                "prompt_tokens": s.get("total_prompt_tokens"),
                "completion_tokens": s.get("total_completion_tokens"),
                "init_usage": s.get("init_usage"),
                "runtime_seconds": s.get("runtime_seconds"),
                "archive_size": s.get("archive_size"),
            }
        except Exception as exc:  # noqa: BLE001
            info = {"summary_read_error": str(exc)}
    if specevo_best.exists() and specevo_best.read_text().strip():
        return specevo_best, info

    # Baselines: <run_dir>/run/best/best_program.py + best_program_info.json
    for best_dir in sorted(run_dir.glob("**/best")):
        prog = best_dir / "best_program.py"
        if prog.exists() and prog.read_text().strip():
            meta = best_dir / "best_program_info.json"
            if meta.exists():
                try:
                    m = json.loads(meta.read_text())
                    info.update(
                        {
                            "search_metrics": m.get("metrics"),
                            "found_at_iteration": m.get("iteration"),
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
            return prog, info

    # Baselines, interrupted before the final save: newest checkpoint.
    ckpt = _newest_checkpoint(run_dir)
    if ckpt is not None:
        prog = ckpt / "best_program.py"
        if prog.exists() and prog.read_text().strip():
            meta = ckpt / "best_program_info.json"
            if meta.exists():
                try:
                    m = json.loads(meta.read_text())
                    info.update(
                        {
                            "search_metrics": m.get("metrics"),
                            "found_at_iteration": m.get("iteration"),
                            "from_checkpoint": ckpt.name,
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
            return prog, info

    return None, info


def iterations_completed(run_dir: Path) -> int | None:
    """How many search iterations a baseline run actually got through.

    SpecEvo reports ``total_evaluations`` in its summary; specevo_baselines does not
    record a total anywhere except the newest checkpoint's ``current_iteration``,
    which is what a resumed run counts from.
    """
    ckpt = _newest_checkpoint(run_dir)
    if ckpt is None:
        return None
    meta = ckpt / "best_program_info.json"
    if meta.exists():
        try:
            n = json.loads(meta.read_text()).get("current_iteration")
            if isinstance(n, int):
                return n + 1
        except Exception:  # noqa: BLE001
            pass
    m = re.search(r"checkpoint_(\d+)$", ckpt.name)
    return int(m.group(1)) + 1 if m else None


def read_cost(run_dir: Path, info: dict) -> float | None:
    """Total USD spent on this problem, across every attempt it took.

    Both runners reset their in-process cost counters when they start, so a
    problem that was interrupted and resumed would under-report if we simply read
    the final tally. We therefore sum the per-call ``cost_log.jsonl`` (which is
    appended to across resumes) and add the cost of any archived SpecEvo attempt,
    falling back to the runner's own total only when no call log exists.
    """
    total = 0.0
    found = False

    for log in sorted(run_dir.glob("**/cost_log.jsonl")):
        for line in log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec.get("cost_usd"), (int, float)):
                total += float(rec["cost_usd"])
                found = True
    if found:
        return total

    # SpecEvo restarts an interrupted problem; the abandoned attempt's spend is
    # real and belongs in the total.
    for summary in sorted(run_dir.glob("prev_attempt_*/summary.json")):
        try:
            v = json.loads(summary.read_text()).get("total_cost")
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, (int, float)):
            total += float(v)
            found = True

    if isinstance(info.get("cost_usd"), (int, float)):
        return total + float(info["cost_usd"])

    for totals in sorted(run_dir.glob("**/cost_log.totals.json")):
        try:
            t = json.loads(totals.read_text())
        except Exception:  # noqa: BLE001
            continue
        for key in ("total_cost_usd", "total_cost", "cost_usd", "total"):
            if isinstance(t.get(key), (int, float)):
                return total + float(t[key])
    return total if found else None


def read_tokens(run_dir: Path, info: dict) -> tuple[int | None, int | None]:
    """``(input_tokens, output_tokens)`` for this problem, across every attempt.

    Same precedence as :func:`read_cost`, for the same reason: the per-call
    ``cost_log.jsonl`` a baseline appends to survives resumes, so it wins;
    SpecEvo keeps no call log and reports its totals in ``summary.json``.
    """
    prompt = 0
    completion = 0
    found = False

    for log in sorted(run_dir.glob("**/cost_log.jsonl")):
        for line in log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec.get("prompt_tokens"), (int, float)):
                prompt += int(rec["prompt_tokens"])
                found = True
            if isinstance(rec.get("completion_tokens"), (int, float)):
                completion += int(rec["completion_tokens"])
                found = True
    if found:
        return prompt, completion

    for summary in sorted(run_dir.glob("prev_attempt_*/summary.json")):
        try:
            s = json.loads(summary.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(s.get("total_prompt_tokens"), int):
            prompt += s["total_prompt_tokens"]
            found = True
        if isinstance(s.get("total_completion_tokens"), int):
            completion += s["total_completion_tokens"]
            found = True

    if isinstance(info.get("prompt_tokens"), int) or isinstance(info.get("completion_tokens"), int):
        return prompt + int(info.get("prompt_tokens") or 0), completion + int(
            info.get("completion_tokens") or 0
        )

    for totals in sorted(run_dir.glob("**/cost_log.totals.json")):
        try:
            t = json.loads(totals.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(t.get("total_prompt_tokens"), int) or isinstance(
            t.get("total_completion_tokens"), int
        ):
            return prompt + int(t.get("total_prompt_tokens") or 0), completion + int(
                t.get("total_completion_tokens") or 0
            )
    return (prompt, completion) if found else (None, None)


def build_record(
    *, domain: str, problem: str, method: str, seed: str, run_dir: Path, iterations: int | None
) -> tuple[dict, bool]:
    meta = L.problem_meta(domain, problem)
    prog_path, info = find_best_program(run_dir)
    prompt_tokens, completion_tokens = read_tokens(run_dir, info)
    record: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": method,
        "domain": domain,
        "problem": problem,
        "instance_id": meta["instance_id"],
        "seed": seed,
        "iterations_requested": iterations,
        "gt_expression": meta["gt_expression"],
        "symbols": meta["symbols"],
        "n_train": meta["n_train"],
        "n_id_test": meta["n_id_test"],
        "n_ood_test": meta["n_ood_test"],
        "run_dir": str(run_dir),
        "cost_usd": read_cost(run_dir, info),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "init_usage": info.get("init_usage"),
        "evaluations": info.get("evaluations") or iterations_completed(run_dir),
        "runtime_seconds": info.get("runtime_seconds"),
        "found_at_iteration": info.get("found_at_iteration"),
        "search_metrics": info.get("search_metrics"),
    }

    if prog_path is None:
        record["status"] = "no_program"
        record["error"] = "the search left no best program on disk"
        return record, False

    source = prog_path.read_text()
    result = L.evaluate_source(domain, problem, source)
    record["best_program_path"] = str(prog_path)
    record["best_program"] = source
    record["final"] = result
    record["status"] = "ok" if result.get("valid") else "invalid_program"
    return record, bool(result.get("valid"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", default="1")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--results", required=True, help="results.jsonl to append to")
    ap.add_argument("--allow-failed-program", action="store_true",
                    help="Exit successfully after recording an invalid or missing program.")
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    record, ok = build_record(
        domain=args.domain,
        problem=args.problem,
        method=args.method,
        seed=args.seed,
        run_dir=run_dir,
        iterations=args.iterations,
    )

    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a") as f:
        f.write(json.dumps(record) + "\n")

    fin = record.get("final") or {}
    if ok:
        def fmt(v):  # noqa: ANN001
            return "n/a" if v is None else f"{v:.4g}"

        print(
            f"[finalize] {args.domain}/{args.problem} ({args.method}): "
            f"train NMSE={fmt(fin.get('train_nmse'))}  "
            f"ID NMSE={fmt(fin.get('id_nmse'))}  OOD NMSE={fmt(fin.get('ood_nmse'))}  "
            f"ID Acc(0.1)={fin.get('id_acc_0.1')}  OOD Acc(0.1)={fin.get('ood_acc_0.1')}  "
            f"cost=${record['cost_usd'] if record['cost_usd'] is not None else 0:.4f}"
        )
    else:
        print(
            f"[finalize] {args.domain}/{args.problem} ({args.method}): "
            f"{record['status']} — {record.get('error') or fin.get('error')}",
            file=sys.stderr,
        )
    return 0 if ok or args.allow_failed_program else 1


if __name__ == "__main__":
    raise SystemExit(main())
