#!/usr/bin/env python3
"""Live smoke test for SpecEvo's analysis-augmented prompts.

Calls one speculator-model completion per prompt variant (5 mutate +
crossover variants) and asserts that the returned text contains all
three sections in order: ``## Analysis``, ``## Description``, ``## Code``.

Run with:

    uv run python scripts/smoke_specevo_prompts.py

Requires an ``OPENROUTER_API_KEY`` (or ``OPENAI_API_KEY`` starting with
``sk-or-``) in the environment or .env file. Costs roughly $0.01-0.02
across the full sweep.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)
    key = os.environ.get("OPENAI_API_KEY", "")
    if key.startswith("sk-or-") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = key


_load_env()


from specevo.engine.prompts import (  # noqa: E402
    CROSSOVER_PROMPTS,
    MUTATE_PROMPTS,
    build_crossover_prompt,
    build_mutate_prompt,
    build_targeted_mutate_prompt,
)
from specevo.clients import LM  # noqa: E402
from specevo.simple.parser import OutputParser  # noqa: E402


MODEL = os.environ.get(
    "SMOKE_MUTATION_MODEL", "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
)

PROBLEM = (
    "Maximise the sum of f(0)+f(1)+f(2) where f is the function the "
    "solver writes. Integer inputs only."
)
SIGNATURE = "def solve(x: int) -> int:"
PARENT_CODE = "def solve(x: int) -> int:\n    return x + 1\n"
PARENT_CODE_B = "def solve(x: int) -> int:\n    return 2 * x\n"


_HDR_ANALYSIS = re.compile(r"^\s*##\s*Analysis\s*$", re.IGNORECASE | re.MULTILINE)
_HDR_DESC = re.compile(r"^\s*##\s*Description\s*$", re.IGNORECASE | re.MULTILINE)
_HDR_CODE = re.compile(r"^\s*##\s*Code\s*$", re.IGNORECASE | re.MULTILINE)


async def _call_one(lm: LM, prompt: str) -> str:
    res = await lm.acompletion(prompt, temperature=0.6, max_tokens=1200)
    return res.text or ""


def _check_sections(text: str, label: str) -> dict:
    """Verify the three-section contract on a single LLM response."""
    out: dict = {"label": label}
    m_a = _HDR_ANALYSIS.search(text)
    m_d = _HDR_DESC.search(text)
    m_c = _HDR_CODE.search(text)
    out["analysis_pos"] = m_a.start() if m_a else None
    out["description_pos"] = m_d.start() if m_d else None
    out["code_pos"] = m_c.start() if m_c else None

    parser = OutputParser()
    parsed = parser.parse(text)
    out["parsed_desc_len"] = len(parsed.description)
    out["parsed_desc_preview"] = parsed.description[:120].replace("\n", " ")
    out["has_code"] = parsed.has_code

    issues: list[str] = []
    if m_a is None:
        issues.append("missing ## Analysis")
    if m_d is None:
        issues.append("missing ## Description")
    if m_c is None and not parsed.has_code:
        issues.append("missing ## Code AND no fenced block")
    if (
        m_a is not None
        and m_d is not None
        and m_a.start() > m_d.start()
    ):
        issues.append("## Analysis appears AFTER ## Description (wrong order)")
    # Bullets leaking into description.
    desc_block = parsed.description
    if desc_block:
        # The contract says no headings / bullets in description.
        if re.search(r"^\s*[-*]\s", desc_block, re.MULTILINE):
            issues.append("description contains bullet points (leak from analysis)")
        if re.search(r"^\s*#{1,6}\s", desc_block, re.MULTILINE):
            issues.append("description contains a markdown heading (leak from analysis)")
        if re.search(r"^\s*\d+\.\s+\*\*", desc_block, re.MULTILINE):
            issues.append("description contains numbered-bold list (leak from analysis)")
    out["issues"] = issues
    return out


async def _run() -> int:
    print(f"[smoke] model = {MODEL}")
    if not (
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    ):
        print("ERROR: no OPENROUTER_API_KEY / OPENAI_API_KEY found", file=sys.stderr)
        return 2

    lm = LM(MODEL)
    cases: list[tuple[str, str]] = []

    for label, tmpl in MUTATE_PROMPTS.items():
        prompt = build_mutate_prompt(
            problem_description=PROBLEM,
            function_signature=SIGNATURE,
            parent_code=PARENT_CODE,
            parent_score=3.0,
            inspirations=[("doubles the input", 6.0)],
            template=tmpl,
        )
        cases.append((f"mutate/{label}", prompt))

    for label, tmpl in CROSSOVER_PROMPTS.items():
        prompt = build_crossover_prompt(
            problem_description=PROBLEM,
            function_signature=SIGNATURE,
            parent_a_code=PARENT_CODE,
            parent_a_score=3.0,
            parent_b_code=PARENT_CODE_B,
            parent_b_score=6.0,
            inspirations=[],
            template=tmpl,
        )
        cases.append((f"crossover/{label}", prompt))

    cases.append(
        (
            "mutate/targeted",
            build_targeted_mutate_prompt(
                problem_description=PROBLEM,
                function_signature=SIGNATURE,
                parent_code=PARENT_CODE,
                parent_score=3.0,
                analysis=(
                    "Components: identity-plus-one only. Strengths: trivial, "
                    "no errors. Weaknesses: linear growth caps the sum at "
                    "f(0)+f(1)+f(2)=1+2+3=6. Suggested changes: 1) cube the "
                    "input; 2) use 2**x; 3) factorial."
                ),
                inspirations=[],
            ),
        )
    )

    results: list[dict] = []
    for label, prompt in cases:
        print(f"[smoke] calling {label} ...")
        try:
            text = await _call_one(lm, prompt)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"label": label, "issues": [f"call failed: {e}"]})
            continue
        r = _check_sections(text, label)
        results.append(r)
        status = "OK" if not r["issues"] else "FAIL"
        print(
            f"  {status} | analysis@{r['analysis_pos']} desc@{r['description_pos']} "
            f"code@{r['code_pos']} | desc_len={r['parsed_desc_len']}"
        )
        if r["parsed_desc_preview"]:
            print(f"    desc preview: {r['parsed_desc_preview']!r}")
        for issue in r["issues"]:
            print(f"    - {issue}")

    print()
    print(f"[smoke] summary: {sum(1 for r in results if not r['issues'])}/{len(results)} passed")
    return 0 if all(not r["issues"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
