#!/usr/bin/env python3
"""Symbolic accuracy for LSR-Synth runs — the LLM-SRBench GPT-4o equivalence judge.

The paper (LLM-SRBench, Shojaee & Nguyen et al., 2025, §2.3 / App. B.2) scores an
equation discovery method on two axes. ``scripts/lsr_summarize.py`` covers the
data-fidelity axis (NMSE, Acc0.1) from the numbers already in ``results.jsonl``.
This script covers the other one: **symbolic accuracy** — the fraction of
problems whose discovered hypothesis is *mathematically equivalent* to the
ground-truth equation, judged by GPT-4o.

The judge is asked the paper's question (App. B, Fig. 11):

    Given the ground truth expression A and the hypothesis B, determine if there
    exist any constant parameter values that would make the hypothesis
    equivalent to the given ground truth expression.

which is why the discovered *program* can be handed over as-is: its ``params[i]``
are free constants by construction. Following the paper we pre-process by

1. stripping comments and docstrings from the program — the judge should see the
   mathematics, not the model's prose about it (App. B.2, step 1);
2. replacing the ground truth's fitted constants with placeholder symbols, so
   equivalence is decided on structure and not on whether a coefficient came out
   as 0.1899 or 0.19 (step 2, ``--gt-constants``).

Every judgement (prompt inputs, verdict, reasoning, tokens) is appended to
``judgments.jsonl`` in the output directory and reused on the next run, so an
interrupted sweep resumes for free and a re-run costs nothing. Calls go out in
parallel (``--workers``).

Denominator note: a problem that produced no usable equation still counts, as
symbolic accuracy is *per problem attempted* — dropping it would flatter the
method. ``--expect-full`` additionally flags domains that are not yet complete
against the dataset's problem count.

Usage::

    python scripts/lsr_symbolic_accuracy.py outputs/lsr_synth
    python scripts/lsr_symbolic_accuracy.py outputs/lsr_synth --methods specevo,openevolve_native
    python scripts/lsr_symbolic_accuracy.py outputs/lsr_synth --workers 16 --csv sa.csv
    python scripts/lsr_symbolic_accuracy.py outputs/lsr_synth --dry-run   # prompts only, no API
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lsr_summarize import (  # noqa: E402  (path set above)
    DOMAIN_LABEL,
    DOMAIN_ORDER,
    find_result_files,
    load_records,
)

PROBLEMS_JSON = REPO_ROOT / "benchmarks" / "llm_srbench" / "data" / "problems.json"

# gpt-4o list price, USD per 1M tokens. Only used for the reported estimate.
PRICE_IN_PER_M = 2.50
PRICE_OUT_PER_M = 10.00


# --------------------------------------------------------------------------- #
# Environment / client
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overriding what is already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_endpoint(api_key: Optional[str], base_url: Optional[str], model: Optional[str]):
    """Return (api_key, base_url, model), filling the blanks from the environment.

    The repo's .env holds an OpenRouter key under OPENAI_API_KEY (see
    so a key starting with ``sk-or-`` is routed
    to OpenRouter and the model gets its ``openai/`` prefix.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(
            "no API key found. Put OPENAI_API_KEY=... in .env, export it, or pass --api-key."
        )
    if not base_url:
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if not base_url and key.startswith("sk-or-"):
        base_url = "https://openrouter.ai/api/v1"
    via_openrouter = bool(base_url and "openrouter" in base_url)
    if not model:
        model = "openai/gpt-4o" if via_openrouter else "gpt-4o"
    elif via_openrouter and "/" not in model:
        model = f"openai/{model}"
    return key, base_url, model


def make_client(api_key: str, base_url: Optional[str]):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


# --------------------------------------------------------------------------- #
# Hypothesis / ground-truth pre-processing
# --------------------------------------------------------------------------- #
def strip_program(source: str) -> str:
    """Drop comments and docstrings, keep the mathematics (paper App. B.2, step 1)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Not parseable (a truncated program, say) — fall back to a line filter.
        return "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        ).strip()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree)).strip()


# A number, optionally carrying the ``_z`` / ``_w`` suffix left behind by the
# upstream dataset dump where a parameter name (k_z) was substituted by its value.
# The lookbehind keeps digits that belong to an identifier — phys_osc ships its
# parameters as names already (F0, omega0), and those must survive intact.
_NUMBER = re.compile(r"(?<![\w.])\d+\.?\d*(?:[eE][+-]?\d+)?(?:_[A-Za-z]\w*)?")


def normalize_ground_truth(expr: str, mode: str = "symbol") -> str:
    """Replace the ground truth's fitted constants with placeholder symbols.

    ``symbol`` (default) gives every constant its own name — equivalence is then
    decided purely on structure, which is what the paper's "after removing
    parameters and constants" asks for. ``shared`` reuses one name per distinct
    value, mirroring how the paper's Table 4 writes a single ``k`` where the
    generator sampled a single coefficient. ``raw`` leaves the expression alone.

    Exponents are never touched: ``A(t)**2`` and ``A(t)**c`` are different
    hypotheses, and Table 4 keeps its powers literal.
    """
    if mode == "raw":
        return expr

    counter = itertools.count()
    by_value: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        before = expr[: m.start()].rstrip()
        if before.endswith("**"):  # an exponent, part of the structure
            return m.group(0)
        if mode == "shared":
            return by_value.setdefault(m.group(0), f"c{len(by_value)}")
        return f"c{next(counter)}"

    return _NUMBER.sub(repl, expr)


def hypothesis_of(rec: dict) -> tuple[Optional[str], str]:
    """Return (hypothesis text, kind) for a record, or (None, kind) if it has none."""
    program = rec.get("best_program")
    if not program:
        path = rec.get("best_program_path")
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.is_file():
                program = p.read_text()
    if program and program.strip():
        return strip_program(program), "program"
    for key in ("best_expression", "expression", "hypothesis"):
        value = rec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), "expression"
    return None, "program"


# --------------------------------------------------------------------------- #
# The judge
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an expert mathematician and scientist evaluating whether two "
    "mathematical expressions describe the same relationship. You reason "
    "carefully about algebraic equivalence and answer in JSON."
)

USER_TEMPLATE = """Question: Given the ground truth mathematical expression A and the hypothesis B, \
determine if there exist any constant parameter values that would make the hypothesis equivalent to \
the given ground truth expression.
Let's think step by step. Explain your reasoning and then provide the final answer as:
{{ "reasoning": "Brief step-by-step analysis", "answer": "Yes/No" }}

(A): '{gt}'

(B): Hypothesis as {kind}
{hypothesis}
"""


def build_prompt(gt: str, hypothesis: str, kind: str) -> str:
    return USER_TEMPLATE.format(
        gt=gt, kind="Program" if kind == "program" else "Expression", hypothesis=hypothesis
    )


_ANSWER_RE = re.compile(r'"answer"\s*:\s*"?\s*(yes|no)', re.IGNORECASE)


def parse_verdict(text: str) -> tuple[Optional[bool], str]:
    """Pull (answer, reasoning) out of the model's reply, JSON or prose."""
    reasoning = ""
    blob = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, re.DOTALL)
    if fence:
        blob = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", blob, re.DOTALL)
        if brace:
            blob = brace.group(0)
    try:
        payload = json.loads(blob)
        answer = str(payload.get("answer", "")).strip().lower()
        reasoning = str(payload.get("reasoning", "")).strip()
        if answer.startswith("yes"):
            return True, reasoning
        if answer.startswith("no"):
            return False, reasoning
    except (json.JSONDecodeError, AttributeError):
        pass

    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).lower() == "yes", reasoning or text.strip()[:500]
    tail = text.strip().lower()[-200:]
    if "answer: yes" in tail:
        return True, reasoning or text.strip()[:500]
    if "answer: no" in tail:
        return False, reasoning or text.strip()[:500]
    return None, text.strip()[:500]


def call_judge(client, model: str, prompt: str, temperature: float, max_retries: int) -> dict:
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
            if attempt == 0:  # JSON mode first; drop it if the endpoint refuses
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            answer, reasoning = parse_verdict(text)
            if answer is None:
                last_error = "unparsable verdict"
                continue
            return {
                "answer": answer,
                "reasoning": reasoning,
                "raw": text,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "error": None,
            }
        except Exception as exc:  # network, rate limit, refused response_format, ...
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))
    return {
        "answer": None,
        "reasoning": "",
        "raw": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": last_error or "no answer",
    }


# --------------------------------------------------------------------------- #
# Task assembly
# --------------------------------------------------------------------------- #
def load_gt_index() -> dict[str, str]:
    """instance_id and problem id -> ground-truth expression, from the dataset."""
    if not PROBLEMS_JSON.is_file():
        return {}
    data = json.loads(PROBLEMS_JSON.read_text())
    index: dict[str, str] = {}
    for domain in data.get("domains", {}).values():
        for prob in domain.get("problems", []):
            gt = prob.get("gt_expression")
            if not gt:
                continue
            index[prob.get("instance_id", "")] = gt
            index[prob.get("id", "")] = gt
    index.pop("", None)
    return index


def domain_sizes() -> dict[str, int]:
    if not PROBLEMS_JSON.is_file():
        return {}
    data = json.loads(PROBLEMS_JSON.read_text())
    return {
        name: int(d.get("n_available") or len(d.get("problems", [])))
        for name, d in data.get("domains", {}).items()
    }


def make_tasks(records: list[dict], gt_index: dict[str, str], gt_mode: str) -> list[dict]:
    tasks = []
    for rec in records:
        gt = rec.get("gt_expression") or gt_index.get(rec.get("instance_id", "")) \
            or gt_index.get(rec.get("problem", ""))
        hypothesis, kind = hypothesis_of(rec)
        gt_norm = normalize_ground_truth(gt, gt_mode) if gt else None
        tasks.append(
            {
                "method": rec.get("method", "?"),
                "domain": rec.get("domain", "?"),
                "problem": rec.get("problem", "?"),
                "seed": rec.get("seed"),
                "instance_id": rec.get("instance_id"),
                "status": rec.get("status"),
                "gt_expression": gt,
                "gt_normalized": gt_norm,
                "hypothesis": hypothesis,
                "hypothesis_kind": kind,
            }
        )
    tasks.sort(key=lambda t: (t["method"], DOMAIN_ORDER.index(t["domain"])
                              if t["domain"] in DOMAIN_ORDER else 99, t["problem"]))
    return tasks


def cache_key(task: dict, model: str, temperature: float) -> str:
    payload = json.dumps(
        {
            "gt": task["gt_normalized"],
            "hyp": task["hypothesis"],
            "kind": task["hypothesis_kind"],
            "model": model,
            "temperature": temperature,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def load_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    cache: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Only completed judgements are reusable; failures should be retried.
        if rec.get("key") and rec.get("answer") is not None:
            cache[rec["key"]] = rec
    return cache


# --------------------------------------------------------------------------- #
# Aggregation / rendering
# --------------------------------------------------------------------------- #
def aggregate(judged: list[dict], sizes: dict[str, int]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for j in judged:
        groups.setdefault((j["method"], j["domain"]), []).append(j)

    rows = []
    for (method, domain), items in groups.items():
        n = len(items)
        correct = sum(1 for j in items if j.get("answer") is True)
        no_eq = sum(1 for j in items if not j.get("hypothesis"))
        errors = sum(1 for j in items if j.get("answer") is None and j.get("hypothesis"))
        rows.append(
            {
                "method": method,
                "domain": domain,
                "n_problems": n,
                "n_available": sizes.get(domain),
                "n_correct": correct,
                "n_no_equation": no_eq,
                "n_judge_failed": errors,
                "symbolic_accuracy_pct": 100.0 * correct / n if n else None,
                "problems_correct": sorted(j["problem"] for j in items if j.get("answer") is True),
            }
        )
    rows.sort(key=lambda r: (r["method"], DOMAIN_ORDER.index(r["domain"])
                             if r["domain"] in DOMAIN_ORDER else 99))
    return rows


def overall_rows(rows: list[dict]) -> list[dict]:
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    out = []
    for method, rs in sorted(by_method.items()):
        n = sum(r["n_problems"] for r in rs)
        c = sum(r["n_correct"] for r in rs)
        out.append(
            {
                "method": method,
                "n_problems": n,
                "n_correct": c,
                "symbolic_accuracy_pct": 100.0 * c / n if n else None,
                "domain_mean_pct": (
                    sum(r["symbolic_accuracy_pct"] or 0.0 for r in rs) / len(rs) if rs else None
                ),
            }
        )
    return out


def render_table(rows: list[dict], totals: list[dict], expect_full: bool) -> str:
    head = (
        f"{'method':<20} {'domain':<14} {'n':>4} {'full':>5} {'correct':>8} "
        f"{'SA %':>7} {'no-eq':>6} {'judge-err':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        avail = r["n_available"]
        flag = ""
        if avail and r["n_problems"] < avail:
            flag = "*"
        lines.append(
            f"{r['method']:<20} {DOMAIN_LABEL.get(r['domain'], r['domain']):<14} "
            f"{r['n_problems']:>4} {(str(avail) + flag) if avail else '  n/a':>5} "
            f"{r['n_correct']:>8} "
            f"{(r['symbolic_accuracy_pct'] if r['symbolic_accuracy_pct'] is not None else 0):>6.2f}% "
            f"{r['n_no_equation']:>6} {r['n_judge_failed']:>10}"
        )
    lines.append("")
    lines.append(f"{'method':<20} {'ALL DOMAINS':<14} {'n':>4} {'correct':>8} {'SA %':>7}  "
                 f"{'macro-avg over domains':>22}")
    lines.append("-" * len(head))
    for t in totals:
        lines.append(
            f"{t['method']:<20} {'pooled':<14} {t['n_problems']:>4} {t['n_correct']:>8} "
            f"{(t['symbolic_accuracy_pct'] or 0):>6.2f}%  {(t['domain_mean_pct'] or 0):>21.2f}%"
        )
    lines.append("")
    lines.append(
        "SA = symbolic accuracy: % of problems attempted whose discovered equation the judge "
        "called equivalent to the ground truth."
    )
    lines.append(
        "Problems with no usable equation (no-eq) count in the denominator as failures, as in "
        "the paper. judge-err = calls that never returned a parsable verdict; rerun to retry them."
    )
    if expect_full and any(r["n_available"] and r["n_problems"] < r["n_available"] for r in rows):
        lines.append(
            "* this domain is not complete yet — the run covers fewer problems than the dataset "
            "holds, so the number is not the full-dataset figure."
        )
    return "\n".join(lines)


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        return
    fields = [k for k in rows[0] if k != "problems_correct"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", default=["outputs/lsr_synth"],
                    help="Run directories, trees of them, or results.jsonl files "
                         "(default: outputs/lsr_synth).")
    ap.add_argument("--methods", default=None, help="Comma-separated method filter.")
    ap.add_argument("--domains", default=None, help="Comma-separated domain filter.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Judge at most N problems per (method, domain) — smoke tests only.")
    ap.add_argument("--model", default=None,
                    help="Judge model (default: gpt-4o, or openai/gpt-4o via OpenRouter).")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint.")
    ap.add_argument("--api-key", default=None, help="Overrides OPENAI_API_KEY / .env.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Judge temperature (default 0.0 — the verdict should be stable).")
    ap.add_argument("--workers", type=int, default=8, help="Parallel judge calls (default 8).")
    ap.add_argument("--max-retries", type=int, default=4, help="Retries per call (default 4).")
    ap.add_argument("--gt-constants", choices=["symbol", "shared", "raw"], default="symbol",
                    help="How to placeholder the ground truth's fitted constants (default symbol).")
    ap.add_argument("--out-dir", default="outputs/symbolic_accuracy",
                    help="Where judgments.jsonl / summary land (default outputs/symbolic_accuracy).")
    ap.add_argument("--csv", default=None, help="Also write the per-domain table as CSV.")
    ap.add_argument("--json", default=None, help="Also write the summary JSON here.")
    ap.add_argument("--no-cache", action="store_true", help="Ignore judgments.jsonl and re-judge.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print one prompt and the task counts, call no API.")
    ap.add_argument("--expect-full", action="store_true",
                    help="Flag domains that cover fewer problems than the dataset holds.")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    files = find_result_files(args.paths)
    records = load_records(files)
    if args.methods:
        wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
        records = [r for r in records if r.get("method") in wanted]
    if args.domains:
        wanted = {d.strip() for d in args.domains.split(",") if d.strip()}
        records = [r for r in records if r.get("domain") in wanted]
    if not records:
        print("No results found (no results.jsonl records matched).")
        return 0

    tasks = make_tasks(records, load_gt_index(), args.gt_constants)
    if args.limit:
        kept, seen = [], {}
        for t in tasks:
            key = (t["method"], t["domain"])
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= args.limit:
                kept.append(t)
        tasks = kept

    missing_gt = [t for t in tasks if not t["gt_normalized"]]
    if missing_gt:
        print(f"warning: {len(missing_gt)} record(s) have no ground-truth expression; "
              f"they are counted as failures", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "judgments.jsonl"
    cache = {} if args.no_cache else load_cache(cache_path)

    for t in tasks:
        t["key"] = cache_key(t, args.model or "gpt-4o", args.temperature)

    todo = [t for t in tasks
            if t["hypothesis"] and t["gt_normalized"] and t["key"] not in cache]

    print(f"{len(records)} records | {len(tasks)} problems to score | "
          f"{len(tasks) - len(todo) - sum(1 for t in tasks if not t['hypothesis'] or not t['gt_normalized'])}"
          f" cached | {len(todo)} to judge")

    if args.dry_run:
        if todo:
            t = todo[0]
            print("\n--- example judge prompt "
                  f"({t['method']}/{t['domain']}/{t['problem']}) ---\n")
            print(build_prompt(t["gt_normalized"], t["hypothesis"], t["hypothesis_kind"]))
        return 0

    judged: list[dict] = []
    lock = threading.Lock()
    done = 0

    if todo:
        api_key, base_url, model = resolve_endpoint(args.api_key, args.base_url, args.model)
        # The cache key was built with the pre-resolution model name; keep it stable
        # so "gpt-4o" and "openai/gpt-4o" share a cache entry.
        client = make_client(api_key, base_url)
        print(f"judge: {model}" + (f" @ {base_url}" if base_url else "")
              + f" | temperature={args.temperature} | workers={args.workers}")

        cache_file = cache_path.open("a")

        def run(task: dict) -> dict:
            nonlocal done
            prompt = build_prompt(task["gt_normalized"], task["hypothesis"],
                                  task["hypothesis_kind"])
            result = call_judge(client, model, prompt, args.temperature, args.max_retries)
            row = {**task, **result, "model": model}
            with lock:
                cache_file.write(json.dumps(row) + "\n")
                cache_file.flush()
                done += 1
                verdict = ("YES" if result["answer"] else "no") if result["answer"] is not None \
                    else "ERR"
                print(f"  [{done}/{len(todo)}] {task['method']}/{task['domain']}/"
                      f"{task['problem']}: {verdict}"
                      + (f"  ({result['error']})" if result["error"] else ""))
            return row

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for row in pool.map(run, todo):
                cache[row["key"]] = row
        cache_file.close()

    prompt_tokens = completion_tokens = 0
    for t in tasks:
        hit = cache.get(t["key"])
        if hit is not None:
            prompt_tokens += hit.get("prompt_tokens") or 0
            completion_tokens += hit.get("completion_tokens") or 0
            judged.append({**t, "answer": hit.get("answer"), "reasoning": hit.get("reasoning", ""),
                           "error": hit.get("error")})
        else:
            reason = "no equation produced" if not t["hypothesis"] else (
                "no ground-truth expression" if not t["gt_normalized"] else "judge failed")
            judged.append({**t, "answer": None, "reasoning": "", "error": reason})

    sizes = domain_sizes()
    rows = aggregate(judged, sizes)
    totals = overall_rows(rows)

    payload = {
        "model": args.model or "gpt-4o",
        "temperature": args.temperature,
        "gt_constants": args.gt_constants,
        "n_records": len(records),
        "result_files": [str(f) for f in files],
        "judge_prompt_tokens": prompt_tokens,
        "judge_completion_tokens": completion_tokens,
        "judge_cost_usd_estimate": round(
            prompt_tokens / 1e6 * PRICE_IN_PER_M + completion_tokens / 1e6 * PRICE_OUT_PER_M, 4
        ),
        "rows": rows,
        "totals": totals,
        "per_problem": [
            {k: j[k] for k in ("method", "domain", "problem", "seed", "answer", "error")}
            for j in judged
        ],
    }
    summary_path = Path(args.json) if args.json else out_dir / "symbolic_accuracy.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    if args.csv:
        write_csv(rows, Path(args.csv))

    print()
    print(render_table(rows, totals, args.expect_full))
    print()
    print(f"judge tokens: {prompt_tokens} in / {completion_tokens} out  "
          f"(~${payload['judge_cost_usd_estimate']:.4f} at gpt-4o list price)")
    print(f"per-judgement records: {cache_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
