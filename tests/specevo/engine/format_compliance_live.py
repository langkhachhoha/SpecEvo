"""Live format-compliance audit for the actual SpecEvo prompts.

Unlike ``tests/simple/format_compliance.py`` (which exercises the bare
``OUTPUT_FORMAT_INSTRUCTION`` over standalone prompts), this script
constructs the *same* mutate / crossover / repair / Navigator prompts
that the SpecEvo orchestrator emits at runtime — by feeding tiny fixture
parents through ``specevo.engine.prompts``. It then runs each prompt
through the real mutation/Navigator models and reports:

  - direct hit / fallback / hard failure
  - whether ``OutputParser`` extracted description + code
  - description length stats

Usage::

    OPENAI_API_KEY=sk-or-v1-... python tests/specevo/format_compliance_live.py

Cost ≈ $0.02 (5 prompt shapes × 2 temperatures = 10 mutation calls + a
handful of Navigator + repair calls).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
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

# Mirror OpenAI key into OpenRouter slot if needed.
key = os.environ.get("OPENAI_API_KEY", "")
if key.startswith("sk-or-") and not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = key


from specevo.engine.prompts import (  # noqa: E402
    build_crossover_prompt,
    build_diverse_seed_prompt,
    build_init_variant_prompt,
    build_mutate_prompt,
    build_reframe_prompt,
    build_navigator_variant_prompt,
    build_surgical_exploit_prompt,
    build_synthesis_prompt,
)
from specevo.simple.parser import OutputParser  # noqa: E402

SPECULATOR_MODEL = os.getenv("SPECEVO_SPECULATOR_MODEL", "openrouter/openai/gpt-4o-mini")
NAVIGATOR_MODEL = os.getenv("SPECEVO_NAVIGATOR_MODEL", "openrouter/openai/gpt-4o-mini")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROBLEM = (
    "Implement `solve(coins: list[int], target: int) -> int` returning the "
    "minimum number of coins from `coins` summing to exactly `target`, or "
    "-1 if impossible (each coin may be used unlimited times)."
)
SIGNATURE = "def solve(coins: list[int], target: int) -> int:"

PARENT_DP = (
    "def solve(coins, target):\n"
    "    INF = float('inf')\n"
    "    dp = [0] + [INF] * target\n"
    "    for v in range(1, target + 1):\n"
    "        for c in coins:\n"
    "            if c <= v and dp[v - c] + 1 < dp[v]:\n"
    "                dp[v] = dp[v - c] + 1\n"
    "    return -1 if dp[target] == INF else dp[target]\n"
)

PARENT_BFS = (
    "from collections import deque\n"
    "def solve(coins, target):\n"
    "    if target == 0: return 0\n"
    "    seen = {0}\n"
    "    q = deque([(0, 0)])\n"
    "    while q:\n"
    "        s, depth = q.popleft()\n"
    "        for c in coins:\n"
    "            ns = s + c\n"
    "            if ns == target: return depth + 1\n"
    "            if ns < target and ns not in seen:\n"
    "                seen.add(ns)\n"
    "                q.append((ns, depth + 1))\n"
    "    return -1\n"
)

INSP_PAIRS_HEALTHY: list[tuple[str, float]] = [
    ("Bottom-up dynamic programming over a dp array indexed by remaining target; "
     "fills each cell with the minimum of (dp[v - c] + 1) across the coin set.", 0.95),
    ("Breadth-first search over partial sums, tracking visited states in a set "
     "so each reachable sum is expanded only once.", 0.82),
    ("Greedy descent that sorts coins descending and subtracts the largest fit "
     "before backtracking on dead ends.", 0.55),
]


RECENT_TRIALS_FIXTURE = [
    "[trial #1, early] ✓ score=0.95 Δ=+0.10 :: bottom-up DP with INF sentinel",
    "[trial #2, mid] ✗ score=0.82 Δ=-0.13 :: BFS expansion with visited set",
    "[trial #3, mid] ✓ score=0.97 Δ=+0.02 :: branch-and-bound with admissible bound",
]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


async def call(
    model: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int | None = 1200,
) -> str:
    """If ``max_tokens`` is None we omit the field entirely — required by
    reasoning-heavy models (GPT-5, o1) used for Navigator interventions, which
    otherwise burn the whole quota on internal reasoning and return an
    empty content block."""
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = await litellm.acompletion(**kwargs)
    return resp.choices[0].message.content or ""


def starts_with_boilerplate(text: str) -> bool:
    head = (text or "").lstrip().lower()[:30]
    return head.startswith(
        (
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
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def build_cases() -> list[dict]:
    """Construct every prompt shape the orchestrator actually emits."""
    cases: list[dict] = []

    # 1) Mutate × 2 temperatures
    mutate_prompt = build_mutate_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        parent_code=PARENT_DP,
        parent_score=0.95,
        inspirations=INSP_PAIRS_HEALTHY,
        advisor=None,
    )
    for t in (0.4, 0.9):
        cases.append({"shape": "mutate", "temperature": t, "model": SPECULATOR_MODEL, "prompt": mutate_prompt})

    # 2) Crossover × 2 temperatures
    cross_prompt = build_crossover_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        parent_a_code=PARENT_DP,
        parent_a_score=0.95,
        parent_b_code=PARENT_BFS,
        parent_b_score=0.82,
        inspirations=INSP_PAIRS_HEALTHY[:2],
        advisor=None,
    )
    for t in (0.4, 0.9):
        cases.append({"shape": "crossover", "temperature": t, "model": SPECULATOR_MODEL, "prompt": cross_prompt})

    # 3) Navigator × 3 stages (orchestrator picks one per call, but we
    #    audit all three because the templates differ). Reasoning-heavy
    #    Navigator models (e.g. GPT-5) burn most of a 1200-token budget on
    #    internal thinking, so we mirror the orchestrator's higher cap.
    # Three Navigator modes; the orchestrator dispatches based on
    # stagnation level but we audit all three so the templates can be
    # regression-tested.
    mode_builders = {
        "synthesis": build_synthesis_prompt,
        "reframe": build_reframe_prompt,
        "surgical": build_surgical_exploit_prompt,
    }
    for stage, builder in mode_builders.items():
        para_prompt = builder(
            problem_description=PROBLEM,
            function_signature=SIGNATURE,
            n_evaluations=42,
            n_cells=3,
            stagnation=0.5,
            anchors=[
                (
                    "def make_change(amount, coins):\n"
                    "    INF = float('inf')\n"
                    "    dp = [0] + [INF] * amount\n"
                    "    for a in range(1, amount + 1):\n"
                    "        for c in coins:\n"
                    "            if c <= a and dp[a - c] + 1 < dp[a]:\n"
                    "                dp[a] = dp[a - c] + 1\n"
                    "    return -1 if dp[amount] == INF else dp[amount]\n",
                    "Bottom-up dynamic programming over remaining-target dp array, "
                    "initialised to infinity; minimum-over-coins relaxation.",
                    0.95,
                ),
                (
                    "from collections import deque\n"
                    "def make_change(amount, coins):\n"
                    "    seen = {0}\n"
                    "    frontier = deque([(0, 0)])\n"
                    "    while frontier:\n"
                    "        total, depth = frontier.popleft()\n"
                    "        if total == amount:\n"
                    "            return depth\n"
                    "        for c in coins:\n"
                    "            nxt = total + c\n"
                    "            if nxt <= amount and nxt not in seen:\n"
                    "                seen.add(nxt)\n"
                    "                frontier.append((nxt, depth + 1))\n"
                    "    return -1\n",
                    "Breadth-first search over partial sums; visited set keyed on "
                    "running total, depth tracked alongside.",
                    0.82,
                ),
                (
                    "def make_change(amount, coins):\n"
                    "    coins = sorted(coins, reverse=True)\n"
                    "    def go(rem, i):\n"
                    "        if rem == 0:\n"
                    "            return 0\n"
                    "        if i == len(coins):\n"
                    "            return -1\n"
                    "        best = -1\n"
                    "        for k in range(rem // coins[i], -1, -1):\n"
                    "            sub = go(rem - k * coins[i], i + 1)\n"
                    "            if sub >= 0:\n"
                    "                best = sub + k if best < 0 else min(best, sub + k)\n"
                    "        return best\n"
                    "    return go(amount, 0)\n",
                    "Greedy descent sorting coins descending, backtracking when a "
                    "cell becomes infeasible.",
                    0.55,
                ),
            ],
            inspirations=[
                (
                    "Meet-in-the-middle: enumerate combinations of the first half "
                    "of coin types into a hash, then look up complements.",
                    0.71,
                ),
                (
                    "Integer linear programming relaxation followed by rounding.",
                    0.40,
                ),
            ],
            recent_trials=RECENT_TRIALS_FIXTURE,
        )
        cases.append(
            {
                "shape": f"navigator-{stage}",
                "temperature": 0.7,
                "model": NAVIGATOR_MODEL,
                "prompt": para_prompt,
                # Navigator calls intentionally do NOT cap tokens — see the
                # ``call`` helper. Reasoning-heavy models need the full
                # provider-side default to produce any visible content.
                "max_tokens": None,
            }
        )

    # 5) Bootstrap phase 1 — frontier "diverse seed" prompt. We exercise
    #    two sub-shapes: cold-start (no existing seeds) and warm
    #    (with one seed already in the context).
    diverse_cold = build_diverse_seed_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        existing_seeds=[],
    )
    cases.append({
        "shape": "diverse-seed-cold",
        "temperature": 0.8,
        "model": NAVIGATOR_MODEL,
        "prompt": diverse_cold,
        "max_tokens": None,
    })
    diverse_warm = build_diverse_seed_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        existing_seeds=[(PARENT_DP, 0.95)],
    )
    cases.append({
        "shape": "diverse-seed-warm",
        "temperature": 0.8,
        "model": NAVIGATOR_MODEL,
        "prompt": diverse_warm,
        "max_tokens": None,
    })

    # 6) Bootstrap phase 2 — mutation "init variant" prompt.
    init_var_prompt = build_init_variant_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        inspirations=[(PARENT_DP, 0.95), (PARENT_BFS, 0.82)],
    )
    cases.append({
        "shape": "init-variant",
        "temperature": 0.9,
        "model": SPECULATOR_MODEL,
        "prompt": init_var_prompt,
    })

    # 7) Navigator fanout — mutation "Navigator variant" prompt.
    pv_prompt = build_navigator_variant_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        base_code=PARENT_DP,
        base_score=0.95,
    )
    cases.append({
        "shape": "navigator-variant",
        "temperature": 0.8,
        "model": SPECULATOR_MODEL,
        "prompt": pv_prompt,
    })

    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_case(case: dict, parser: OutputParser) -> dict:
    raw = await call(
        case["model"],
        case["prompt"],
        temperature=case["temperature"],
        max_tokens=case.get("max_tokens", 1200),
    )
    parsed = parser.parse(raw)
    needs_fallback = parser.needs_fallback_summary(parsed)
    direct_hit = parsed.has_code and not needs_fallback
    return {
        "shape": case["shape"],
        "temperature": case["temperature"],
        "model": case["model"],
        "prompt_chars": len(case["prompt"]),
        "raw_chars": len(raw),
        "has_description": parsed.has_description,
        "has_code": parsed.has_code,
        "description_chars": len(parsed.description),
        "code_chars": len(parsed.code),
        "direct_hit": direct_hit,
        "fallback_needed": needs_fallback,
        "hard_failure": not parsed.has_code,
        "boilerplate_opener": starts_with_boilerplate(parsed.description),
        "description_preview": parsed.description[:240].replace("\n", " "),
        "code_preview": parsed.code[:120].replace("\n", " | "),
        "raw_head": raw[:120].replace("\n", " | "),
    }


async def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; aborting.", file=sys.stderr)
        return 1

    cases = build_cases()
    print(f"Running {len(cases)} live SpecEvo prompt-shape cases")
    print(f"  speculator_model = {SPECULATOR_MODEL}")
    print(f"  navigator_model = {NAVIGATOR_MODEL}")
    print()

    parser = OutputParser()
    sem = asyncio.Semaphore(4)

    async def gated(c: dict) -> dict:
        async with sem:
            try:
                return await run_case(c, parser)
            except Exception as e:
                return {
                    "shape": c["shape"],
                    "temperature": c["temperature"],
                    "model": c["model"],
                    "prompt_chars": len(c["prompt"]),
                    "raw_chars": 0,
                    "has_description": False,
                    "has_code": False,
                    "description_chars": 0,
                    "code_chars": 0,
                    "direct_hit": False,
                    "fallback_needed": False,
                    "hard_failure": True,
                    "boilerplate_opener": False,
                    "description_preview": "",
                    "code_preview": "",
                    "raw_head": f"<ERROR: {e}>"[:200],
                }

    results = await asyncio.gather(*[gated(c) for c in cases])

    # ----- aggregate -----
    n = len(results)
    direct = sum(r["direct_hit"] for r in results)
    fallback = sum(r["fallback_needed"] for r in results)
    hard = sum(r["hard_failure"] for r in results)
    boiler = sum(r["boilerplate_opener"] for r in results)

    print("=" * 76)
    print(f"Total cases:          {n}")
    print(f"Direct hits:          {direct} ({direct / n:.0%})")
    print(f"Needed fallback:      {fallback} ({fallback / n:.0%})")
    print(f"Hard failures:        {hard} ({hard / n:.0%})")
    print(f"Boilerplate openers:  {boiler} ({boiler / max(1, n):.0%})")
    print("=" * 76)

    # Per-shape breakdown.
    print("\nPer-prompt-shape breakdown:")
    shapes = sorted({r["shape"] for r in results})
    for s in shapes:
        bucket = [r for r in results if r["shape"] == s]
        d = sum(r["direct_hit"] for r in bucket)
        f = sum(r["fallback_needed"] for r in bucket)
        h = sum(r["hard_failure"] for r in bucket)
        avg = sum(r["description_chars"] for r in bucket) / max(1, len(bucket))
        print(f"  {s:18s}  direct={d}/{len(bucket)}  fallback={f}  hard={h}  avg_desc_chars={avg:.0f}")

    # Sample one of each outcome.
    print("\nSamples:")
    for label, pred in (
        ("DIRECT", lambda r: r["direct_hit"]),
        ("FALLBACK", lambda r: r["fallback_needed"]),
        ("HARD-FAIL", lambda r: r["hard_failure"]),
    ):
        sample = next((r for r in results if pred(r)), None)
        if sample:
            print(f"\n[{label}] shape={sample['shape']} T={sample['temperature']}")
            print(f"  raw_head: {sample['raw_head']!r}")
            if sample["description_preview"]:
                print(f"  description: {sample['description_preview']!r}")

    # Persist.
    out_path = ROOT / "outputs" / "specevo_format_compliance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out_path}")

    # Exit non-zero if any prompt shape consistently breaks (>10% hard
    # failure across all shapes or >25% within one shape).
    if hard / n > 0.10:
        print(f"\nOverall hard-failure rate {hard / n:.1%} exceeds 10% threshold")
        return 2
    for s in shapes:
        bucket = [r for r in results if r["shape"] == s]
        per_shape_hard = sum(r["hard_failure"] for r in bucket) / len(bucket)
        if per_shape_hard > 0.25:
            print(f"\nShape {s!r} hard-failure rate {per_shape_hard:.1%} exceeds 25%")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
