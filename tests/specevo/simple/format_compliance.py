"""Format-compliance stress test.

Runs the mutation LLM across a matrix of (paradigm hint × temperature) for
several distinct problems and tallies:

  - direct hits (## Description + ## Code both extracted cleanly)
  - fallback-summary hits (code OK, description missing/short)
  - hard failures (no code at all)

Also samples paragraph length (we want a real descriptive paragraph, not
one sentence) and checks for common boilerplate openers that we expect
to have been suppressed by the prompt.

Usage:

    OPENAI_API_KEY=... python tests/simple/format_compliance.py

Cost: O(20-30 LLM calls @ gpt-4o-mini) ~$0.01-0.03 per run.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

import litellm

litellm.suppress_debug_info = True

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / ".env"
if ENV.exists():
    for raw in ENV.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)
sys.path.insert(0, str(ROOT))

from specevo.simple.parser import (  # noqa: E402
    OUTPUT_FORMAT_INSTRUCTION,
    OutputParser,
    fallback_summarize,
)

MUTATION_MODEL = "openrouter/openai/gpt-4o-mini"

PROBLEMS = [
    (
        "coin-change",
        textwrap.dedent(
            """\
            Write a Python function `solve(nums: list[int], target: int) -> int` returning
            the minimum number of coins from `nums` summing to exactly `target`, or -1 if
            impossible. Each coin can be used unlimited times.
            """
        ).strip(),
    ),
    (
        "longest-path-DAG",
        textwrap.dedent(
            """\
            Write `solve(n: int, edges: list[tuple[int,int,int]]) -> int` returning the
            longest-path weight in a DAG with `n` nodes and weighted edges
            `(u, v, weight)`.
            """
        ).strip(),
    ),
    (
        "tsp-approx",
        textwrap.dedent(
            """\
            Write `solve(dist: list[list[float]]) -> float` approximating the TSP cost
            for a symmetric distance matrix. Beat naive nearest-neighbor when possible.
            """
        ).strip(),
    ),
]

PARADIGMS = [
    "dynamic-programming",
    "bfs",
    "branch-and-bound",
    "greedy",
    "simulated-annealing",
]

TEMPERATURES = [0.4, 0.9]


BOILERPLATE_PREFIXES = (
    "this solution",
    "this code",
    "this function",
    "this algorithm",
    "this implementation",
    "the solution",
    "the code",
    "the function",
    "the algorithm",
    "the implementation",
)


def starts_with_boilerplate(text: str) -> bool:
    if not text:
        return False
    head = text.lstrip().lower()[:30]
    return any(head.startswith(p) for p in BOILERPLATE_PREFIXES)


async def call_mutation(prompt: str, *, temperature: float = 0.7) -> str:
    resp = await litellm.acompletion(
        model=MUTATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=800,
        temperature=temperature,
    )
    return resp.choices[0].message.content


async def call_summary(prompt: str) -> str:
    return await call_mutation(prompt, temperature=0.3)


def make_prompt(problem: str, paradigm: str) -> str:
    return (
        f"{problem}\n\n"
        f"Implement the solution using the `{paradigm}` paradigm.\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTION}"
    )


async def one_case(problem_name: str, problem: str, paradigm: str, temperature: float) -> dict:
    parser = OutputParser()
    raw = await call_mutation(make_prompt(problem, paradigm), temperature=temperature)
    parsed = parser.parse(raw)

    outcome = {
        "problem": problem_name,
        "paradigm": paradigm,
        "temperature": temperature,
        "raw_len": len(raw),
        "has_description": parsed.has_description,
        "has_code": parsed.has_code,
        "description_len": len(parsed.description),
        "code_len": len(parsed.code),
        "raw_head": raw[:120].replace("\n", " | "),
        "direct_hit": False,
        "fallback_used": False,
        "hard_failure": False,
        "boilerplate_opener": False,
        "description_preview": "",
    }

    if not parsed.has_code:
        outcome["hard_failure"] = True
        outcome["description_preview"] = raw[:200]
        return outcome

    if parser.needs_fallback_summary(parsed):
        summary = await fallback_summarize(parsed.code, completion_fn=call_summary)
        outcome["fallback_used"] = True
        outcome["description_preview"] = summary.strip()[:200]
        outcome["description_len"] = len(summary.strip())
        outcome["boilerplate_opener"] = starts_with_boilerplate(summary)
    else:
        outcome["direct_hit"] = True
        outcome["description_preview"] = parsed.description[:200]
        outcome["boilerplate_opener"] = starts_with_boilerplate(parsed.description)

    return outcome


async def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; aborting.")
        return 1

    # Build the case matrix.
    cases: list[tuple[str, str, str, float]] = []
    for pname, problem in PROBLEMS:
        for paradigm in PARADIGMS:
            for temperature in TEMPERATURES:
                cases.append((pname, problem, paradigm, temperature))

    print(f"Running {len(cases)} cases ({len(PROBLEMS)} problems × {len(PARADIGMS)} paradigms × {len(TEMPERATURES)} temps)\n")

    # Bounded concurrency to be polite to OpenRouter.
    sem = asyncio.Semaphore(4)

    async def gated(case):
        async with sem:
            return await one_case(*case)

    results = await asyncio.gather(*[gated(c) for c in cases])

    direct = sum(1 for r in results if r["direct_hit"])
    fallback = sum(1 for r in results if r["fallback_used"])
    hard = sum(1 for r in results if r["hard_failure"])
    boiler = sum(1 for r in results if r["boilerplate_opener"])
    described = direct + fallback
    avg_desc_len = (
        sum(r["description_len"] for r in results if not r["hard_failure"]) / max(1, described)
    )
    total = len(results)

    print("=" * 72)
    print(f"Cases:                  {total}")
    print(f"Direct hits:            {direct} ({direct / total:.0%})")
    print(f"Fallback summary:       {fallback} ({fallback / total:.0%})")
    print(f"Hard failures:          {hard} ({hard / total:.0%})")
    print(f"Boilerplate openers:    {boiler}/{described} ({boiler / max(1, described):.0%})")
    print(f"Avg description chars:  {avg_desc_len:.0f}")
    print("=" * 72)

    # Per-paradigm breakdown
    print("\nPer-paradigm breakdown:")
    for paradigm in PARADIGMS:
        bucket = [r for r in results if r["paradigm"] == paradigm]
        d = sum(1 for r in bucket if r["direct_hit"])
        f = sum(1 for r in bucket if r["fallback_used"])
        h = sum(1 for r in bucket if r["hard_failure"])
        b = sum(1 for r in bucket if r["boilerplate_opener"])
        print(f"  {paradigm:24s}  direct={d}  fallback={f}  hard={h}  boiler={b}/{len(bucket)}")

    # Per-temperature breakdown
    print("\nPer-temperature breakdown:")
    for temp in TEMPERATURES:
        bucket = [r for r in results if r["temperature"] == temp]
        d = sum(1 for r in bucket if r["direct_hit"])
        f = sum(1 for r in bucket if r["fallback_used"])
        h = sum(1 for r in bucket if r["hard_failure"])
        print(f"  T={temp}  direct={d}  fallback={f}  hard={h}  ({len(bucket)} cases)")

    # Show one example per outcome category for QA.
    print("\nSample outcomes:")
    for label, predicate in [
        ("DIRECT", lambda r: r["direct_hit"]),
        ("FALLBACK", lambda r: r["fallback_used"]),
        ("HARD-FAIL", lambda r: r["hard_failure"]),
    ]:
        sample = next((r for r in results if predicate(r)), None)
        if sample:
            print(f"\n[{label}] {sample['problem']} / {sample['paradigm']} / T={sample['temperature']}")
            print(f"  description: {sample['description_preview']!r}")

    # Dump full results for later analysis.
    out_path = ROOT / "outputs" / "simple_evo_format_compliance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out_path}")

    # Exit non-zero if more than 5% hard failures.
    threshold = 0.05
    if hard / total > threshold:
        print(f"\nHard-failure rate {hard / total:.1%} > {threshold:.0%} threshold")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
