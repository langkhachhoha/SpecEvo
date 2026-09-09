#!/usr/bin/env python3
"""Aggregate LSR-Synth per-problem results into the paper's ID / OOD table.

Reads every ``results.jsonl`` under the paths given (a run directory, a tree of
run directories, or the files themselves) and reports, per (method, domain):

    NMSE   median and mean on the in-domain (ID) and out-of-domain (OOD) test
           splits — the paper's Table 1 reports the *median*
    Acc0.1 percentage of problems whose maximum point-wise relative error on the
           split stays within 10% (the paper's Acc_tau with tau = 0.1)
    R2     mean coefficient of determination per split
    cost   total USD and total search evaluations
    tokens total input/output tokens (JSON/CSV only, not the table), plus
           SpecEvo's initialization-phase share of them

``within0.1`` is also shown: the mean *fraction of points* inside the 10%
tolerance. Acc0.1 is all-or-nothing per problem, so on problems whose target
passes through zero (matsci at zero strain, for instance) it can read 0 for an
otherwise excellent equation; ``within0.1`` is the graded companion and the two
should be read together.

Usage::

    python scripts/lsr_summarize.py outputs/lsr_synth/specevo/chem_react/seed1
    python scripts/lsr_summarize.py outputs/lsr_synth            # every run
    python scripts/lsr_summarize.py outputs/lsr_synth --csv table.csv
    python scripts/lsr_summarize.py <run_dir> --progress         # per-problem view
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

DOMAIN_ORDER = ["chem_react", "bio_pop_growth", "phys_osc", "matsci"]
DOMAIN_LABEL = {
    "chem_react": "Chemistry",
    "bio_pop_growth": "Biology",
    "phys_osc": "Physics",
    "matsci": "Material Sci",
}


def find_result_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("**/results.jsonl")))
        else:
            print(f"warning: {p} does not exist", file=sys.stderr)
    return out


def load_records(files: list[Path]) -> list[dict]:
    """Latest record per (method, domain, problem, seed) — a re-run supersedes."""
    latest: dict[tuple, dict] = {}
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: unparsable line in {f}", file=sys.stderr)
                continue
            key = (rec.get("method"), rec.get("domain"), rec.get("problem"), rec.get("seed"))
            latest[key] = rec
    return list(latest.values())


def _get(rec: dict, key: str) -> Optional[float]:
    final = rec.get("final") or {}
    v = final.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return None if not math.isfinite(float(v)) else float(v)


def _median(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _mean(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        groups.setdefault((rec.get("method", "?"), rec.get("domain", "?")), []).append(rec)

    rows: list[dict] = []
    for (method, domain), recs in groups.items():
        n = len(recs)
        # A problem the search could not solve at all still counts in the
        # denominator — dropping it would flatter the method.
        scored = [r for r in recs if (r.get("final") or {}).get("valid")]
        id_nmse = [v for r in scored if (v := _get(r, "id_nmse")) is not None]
        ood_nmse = [v for r in scored if (v := _get(r, "ood_nmse")) is not None]
        id_r2 = [v for r in scored if (v := _get(r, "id_r2")) is not None]
        ood_r2 = [v for r in scored if (v := _get(r, "ood_r2")) is not None]
        id_acc = [_get(r, "id_acc_0.1") or 0.0 for r in recs]
        ood_acc = [_get(r, "ood_acc_0.1") or 0.0 for r in recs]
        id_within = [_get(r, "id_frac_within") or 0.0 for r in recs]
        ood_within = [_get(r, "ood_frac_within") or 0.0 for r in recs]
        costs = [r["cost_usd"] for r in recs if isinstance(r.get("cost_usd"), (int, float))]
        evals = [r["evaluations"] for r in recs if isinstance(r.get("evaluations"), (int, float))]
        in_tok = [r["prompt_tokens"] for r in recs if isinstance(r.get("prompt_tokens"), int)]
        out_tok = [r["completion_tokens"] for r in recs if isinstance(r.get("completion_tokens"), int)]
        # SpecEvo's init phase (diverse seeds + variants); baselines have none.
        init_in = [
            (r.get("init_usage") or {}).get("prompt_tokens")
            for r in recs
            if isinstance((r.get("init_usage") or {}).get("prompt_tokens"), int)
        ]
        init_out = [
            (r.get("init_usage") or {}).get("completion_tokens")
            for r in recs
            if isinstance((r.get("init_usage") or {}).get("completion_tokens"), int)
        ]

        rows.append(
            {
                "method": method,
                "domain": domain,
                "n_problems": n,
                "n_scored": len(scored),
                "n_no_equation": n - len(scored),
                "id_nmse_median": _median(id_nmse),
                "id_nmse_mean": _mean(id_nmse),
                "ood_nmse_median": _median(ood_nmse),
                "ood_nmse_mean": _mean(ood_nmse),
                "id_acc_0.1_pct": 100.0 * _mean(id_acc) if id_acc else None,
                "ood_acc_0.1_pct": 100.0 * _mean(ood_acc) if ood_acc else None,
                "id_within_0.1_pct": 100.0 * _mean(id_within) if id_within else None,
                "ood_within_0.1_pct": 100.0 * _mean(ood_within) if ood_within else None,
                # R² is unbounded below, so one equation that diverges off the
                # training range drives the mean to -1e9 and tells you nothing
                # about the other nine problems. The median is what the table
                # shows; the mean is kept here for anyone who wants it.
                "id_r2_median": _median(id_r2),
                "ood_r2_median": _median(ood_r2),
                "id_r2_mean": _mean(id_r2),
                "ood_r2_mean": _mean(ood_r2),
                "ood_r2_worst": min(ood_r2) if ood_r2 else None,
                "total_cost_usd": sum(costs) if costs else None,
                "total_evaluations": int(sum(evals)) if evals else None,
                # Token totals are JSON/CSV-only — the fixed-width table is
                # already at the width of a terminal.
                "total_prompt_tokens": int(sum(in_tok)) if in_tok else None,
                "total_completion_tokens": int(sum(out_tok)) if out_tok else None,
                "init_prompt_tokens": int(sum(init_in)) if init_in else None,
                "init_completion_tokens": int(sum(init_out)) if init_out else None,
            }
        )

    rows.sort(
        key=lambda r: (
            r["method"],
            DOMAIN_ORDER.index(r["domain"]) if r["domain"] in DOMAIN_ORDER else 99,
        )
    )
    return rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _f(v: Optional[float], spec: str = ".3g") -> str:
    return "  n/a" if v is None else format(v, spec)


def render_table(rows: list[dict]) -> str:
    head = (
        f"{'method':<18} {'domain':<13} {'n':>3} {'fail':>4} {'ID NMSE':>10} {'OOD NMSE':>10} "
        f"{'ID Acc':>7} {'OOD Acc':>7} {'ID w/in':>7} {'OOD w/in':>8} "
        f"{'ID R2':>8} {'OOD R2':>8} {'cost $':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['method']:<18} {DOMAIN_LABEL.get(r['domain'], r['domain']):<13} "
            f"{r['n_problems']:>3} {r['n_no_equation']:>4} "
            f"{_f(r['id_nmse_median'], '10.3e')} {_f(r['ood_nmse_median'], '10.3e')} "
            f"{_f(r['id_acc_0.1_pct'], '6.1f')}% {_f(r['ood_acc_0.1_pct'], '6.1f')}% "
            f"{_f(r['id_within_0.1_pct'], '6.1f')}% {_f(r['ood_within_0.1_pct'], '7.1f')}% "
            f"{_f(r['id_r2_median'], '8.4f')} {_f(r['ood_r2_median'], '8.4f')} "
            f"{_f(r['total_cost_usd'], '8.3f')}"
        )
    lines.append("")
    lines.append("n = problems attempted, fail = those that produced no usable equation "
                 "(still counted in n, and in Acc/w/in as 0).")
    lines.append("NMSE and R2 columns are MEDIANS over the problems that produced an "
                 "equation (as in the paper's Table 1). R2 is unbounded below, so one")
    lines.append("equation that diverges off the training range would swamp a mean; "
                 "`ood_r2_worst` in the JSON/CSV shows that tail.")
    lines.append("Acc = Acc(0.1) as a % of problems; w/in = mean % of test points inside "
                 "the 10% tolerance. Read the two together.")
    return "\n".join(lines)


def render_progress(paths: list[str], records: list[dict]) -> str:
    lines: list[str] = []
    for raw in paths:
        root = Path(raw)
        if not root.is_dir():
            continue
        pdirs = sorted((root / "problems").glob("*")) if (root / "problems").is_dir() else []
        if not pdirs:
            continue
        done = [d for d in pdirs if (d / ".done").exists()]
        lines.append(f"{root}: {len(done)}/{len(pdirs)} problems finished")
        by_problem = {r.get("problem"): r for r in records}
        for d in pdirs:
            if not d.is_dir():
                continue
            rec = by_problem.get(d.name)
            if (d / ".done").exists() and rec:
                fin = rec.get("final") or {}
                lines.append(
                    f"    {d.name:<10} done    ID NMSE={_f(_get(rec, 'id_nmse'), '.3e')}"
                    f"  OOD NMSE={_f(_get(rec, 'ood_nmse'), '.3e')}"
                    f"  ID Acc={fin.get('id_acc_0.1', 0):.0f}"
                    f"  OOD Acc={fin.get('ood_acc_0.1', 0):.0f}"
                    f"  {'' if fin.get('valid') else '(' + str(rec.get('status')) + ')'}"
                )
            elif (d / ".done").exists():
                lines.append(f"    {d.name:<10} done    (no result line found)")
            else:
                ckpts = sorted((d / "run" / "checkpoints").glob("checkpoint_*")) \
                    if (d / "run" / "checkpoints").is_dir() else []
                where = f"checkpoint_{max(int(c.name.split('_')[-1]) for c in ckpts)}" if ckpts else "no checkpoint yet"
                lines.append(f"    {d.name:<10} pending ({where})")
    return "\n".join(lines) if lines else "no run directories with problems/ found"


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Run directories, trees of them, or results.jsonl files.")
    ap.add_argument("--csv", default=None, help="Also write the aggregate as CSV.")
    ap.add_argument("--json", default=None, help="Also write the aggregate as JSON.")
    ap.add_argument("--progress", action="store_true", help="Show per-problem progress instead of the table.")
    ap.add_argument("--quiet", action="store_true", help="Write files but print nothing.")
    args = ap.parse_args()

    files = find_result_files(args.paths)
    records = load_records(files)
    rows = aggregate(records)

    if args.json:
        payload: dict[str, Any] = {
            "n_records": len(records),
            "result_files": [str(f) for f in files],
            "rows": rows,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
    if args.csv:
        write_csv(rows, Path(args.csv))

    if args.quiet:
        return 0
    if args.progress:
        print(render_progress(args.paths, records))
        if rows:
            print()
            print(render_table(rows))
        return 0
    if not records:
        print("No results yet (no results.jsonl records found).")
        return 0
    print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
