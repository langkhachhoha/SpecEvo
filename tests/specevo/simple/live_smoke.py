"""Live smoke test for SIMPLE-EVO description-embedding + format compliance.

Run manually:

    OPENAI_API_KEY=... python tests/simple/live_smoke.py

NOT a pytest test; it makes real API calls. Costs a few cents total
(text-embedding-3-small is ~$0.02 / 1M tokens, gpt-4o-mini is cheap too).
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path

import litellm

litellm.suppress_debug_info = True

# Load .env if present at repo root.
ROOT = Path(__file__).resolve().parents[3]  # specevo_baselines/
ENV = ROOT / ".env"
if ENV.exists():
    for raw in ENV.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

# Make the source tree importable when run as a script.
sys.path.insert(0, str(ROOT))

from specevo.simple.embedder import DescriptionEmbedder, EmbedderConfig, cosine  # noqa: E402
from specevo.simple.parser import (  # noqa: E402
    OUTPUT_FORMAT_INSTRUCTION,
    OutputParser,
    fallback_summarize,
)
from specevo.simple.pool import Pool, PoolConfig, Program  # noqa: E402

MUTATION_MODEL = "openrouter/openai/gpt-4o-mini"


PROBLEM = textwrap.dedent(
    """\
    Write a Python function `solve(nums: list[int], target: int) -> int` that
    returns the minimum number of coins from `nums` summing to exactly
    `target`, or -1 if impossible. You may use each coin unlimited times.
    """
).strip()


def make_prompt(paradigm_hint: str) -> str:
    return (
        f"{PROBLEM}\n\n"
        f"Implement a {paradigm_hint} solution.\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTION}"
    )


async def call_mutation(prompt: str) -> str:
    resp = await litellm.acompletion(
        model=MUTATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=800,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def banner(s: str) -> None:
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


async def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; aborting.")
        return 1

    parser = OutputParser()
    embedder = DescriptionEmbedder(EmbedderConfig())
    pool = Pool(PoolConfig(K=20, niche_cosine_threshold=0.95, family_cosine_threshold=0.80, max_per_family=4))

    banner("1) Generating solutions with different paradigm hints (incl. 2 DP variants to test family detection)")
    paradigms = [
        "dynamic programming (bottom-up tabulation)",
        "dynamic programming (top-down memoization)",  # same family as above
        "BFS over partial sums (treat each sum as a graph node)",
        "greedy descent with backtracking",
    ]

    raw_outputs: list[tuple[str, str]] = []
    for hint in paradigms:
        prompt = make_prompt(hint)
        raw = await call_mutation(prompt)
        raw_outputs.append((hint, raw))
        print(f"\n--- raw output for paradigm: {hint} ---")
        print(raw[:600])
        if len(raw) > 600:
            print(f"... <{len(raw) - 600} more chars>")

    banner("2) Parsing outputs (description + code extraction)")
    parsed_items: list[tuple[str, str, str, bool]] = []
    for hint, raw in raw_outputs:
        parsed = parser.parse(raw)
        needs_fb = parser.needs_fallback_summary(parsed)
        print(f"\n[{hint}]")
        print(f"  has_description: {parsed.has_description} (len={len(parsed.description)})")
        print(f"  has_code:        {parsed.has_code} (len={len(parsed.code)})")
        print(f"  needs_fallback:  {needs_fb}")
        if parsed.has_description:
            print(f"  description: {parsed.description[:200]}")
        if needs_fb and parsed.has_code:
            print("  ... invoking fallback summarizer")
            summary = await fallback_summarize(parsed.code, completion_fn=call_mutation)
            parsed = type(parsed)(
                description=summary.strip(),
                code=parsed.code,
                description_was_fallback=True,
            )
            print(f"  summarized: {parsed.description[:200]}")
        parsed_items.append((hint, parsed.description, parsed.code, parsed.description_was_fallback))

    banner("3) Embedding descriptions + admitting to pool")
    descs = [d for _, d, _, _ in parsed_items]
    embeddings = embedder.embed_batch(descs)
    print(f"Embedded {len(descs)} descriptions, dim={len(embeddings[0]) if embeddings else 0}")

    for (hint, desc, code, was_fb), emb in zip(parsed_items, embeddings, strict=True):
        prog = Program(
            code=code,
            description=desc,
            score=0.5,  # placeholder
            embedding=emb,
            source="init",
            created_at_eval=0,
        )
        ok, reason = pool.add(prog)
        print(f"  add({hint[:30]:<30}) -> ok={ok}  reason={reason}  fallback={was_fb}")

    banner("4) Pairwise cosine of admitted descriptions")
    progs = pool.programs()
    for i in range(len(progs)):
        for j in range(i + 1, len(progs)):
            c = cosine(progs[i].embedding, progs[j].embedding)
            print(f"  cos({i}={progs[i].description[:30]!r}..., {j}={progs[j].description[:30]!r}...) = {c:.3f}")

    banner("5) Family clustering result")
    print(f"  pool size: {len(pool)}  families: {pool.num_families()}")
    for fid, members in pool.families().items():
        print(f"  family {fid}: {[m.description[:50] for m in members]}")

    banner("6) Representatives per phase")
    for phase in ("early", "mid", "late"):
        reps = pool.representatives(phase, n=2)  # type: ignore[arg-type]
        print(f"  phase={phase}: {[r.description[:60] for r in reps]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
