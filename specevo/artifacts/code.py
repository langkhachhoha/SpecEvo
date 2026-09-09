"""Code artifact adapter for SpecEvo's existing public API."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..clients.base import ClientSpec, client_name
from ..config import LeviConfig
from ..core import Program
from ..equilibrium.prompts import (
    CODE_REPAIR_PROMPT,
    PARADIGM_SHIFT_PROMPTS,
    STRATEGY_SUMMARY_PROMPT,
    VARIANT_GENERATION_PROMPT,
    get_budget_stage,
)
from ..prompts import OutputMode, ProgramWithScore, PromptBuilder
from ..utils import ResilientProcessPool, evaluate_code, extract_code, extract_fn_name
from .base import ArtifactAdapter

DIVERSITY_SEED_PROMPT = """# {problem_title}

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Your Task: ALGORITHMIC DIVERSITY

You MUST design a solution using a **FUNDAMENTALLY DIFFERENT ALGORITHM** than the existing seeds.

**DO NOT:**
- Make minor variations or parameter tweaks to existing approaches
- Use the same core algorithm with different constants
- Reorder or refactor existing logic

**DO:**
- Analyze what algorithmic paradigm each existing seed uses
- Identify what aspects of the problem they exploit (or ignore)
- Design from first principles using a completely different strategy
- Think about what information in the problem they are NOT using
- Consider entirely different ways to model or decompose the problem

The goal is to explore different regions of the algorithm design space. A population of diverse algorithms will outperform a population of similar ones.

## Existing Seeds (analyze their algorithms, then do something DIFFERENT):
{existing_seeds}

## Output
Output ONLY the complete Python code in a ```python block.
"""


def _trajectory_body(
    *,
    best_score: float | None,
    evals_since_best: int | None,
    stagnation: float | None,
    top_failures: Sequence[str] | None,
) -> str:
    """Render only the bullet body of the search-trajectory context.

    Returns "" when nothing meaningful is available. Caller adds the section
    header (so this can be passed to either PromptBuilder.add_section, which
    adds its own '## Title', or to a manual template).
    """
    parts: list[str] = []
    if best_score is not None:
        parts.append(f"- Current best score: {best_score:.17g}")
    if evals_since_best is not None:
        parts.append(f"- Evaluations since the last NEW BEST: {evals_since_best}")
    if stagnation is not None:
        parts.append(f"- Stagnation depth s(t): {stagnation:.2f} (0=improving, 1=stuck)")
    if top_failures:
        bullets = "\n".join(f"  - {f}" for f in top_failures[:3])
        parts.append("- Recurring failure modes:\n" + bullets)
    return "\n".join(parts)


def _build_trajectory_block(
    *,
    best_score: float | None,
    evals_since_best: int | None,
    stagnation: float | None,
    top_failures: Sequence[str] | None,
) -> str:
    """Markdown section with header. Used by build_paradigm_shift_prompt
    which splices the block into a manual template.
    """
    body = _trajectory_body(
        best_score=best_score,
        evals_since_best=evals_since_best,
        stagnation=stagnation,
        top_failures=top_failures,
    )
    if not body:
        return ""
    return "\n## Search Trajectory\n" + body + "\n"


def apply_diff(original: str, diff_response: str) -> str | None:
    """Apply SEARCH/REPLACE diff blocks to original code."""
    result = original

    pattern = r"<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE"
    matches = re.findall(pattern, diff_response, re.DOTALL)

    if not matches:
        return extract_code(diff_response)

    for search, replace in matches:
        search = search.strip()
        replace = replace.strip()
        if search in result:
            result = result.replace(search, replace, 1)
        else:
            return None

    return result


class CodeAdapter(ArtifactAdapter):
    """Adapter for SpecEvo's existing code-evolution behavior."""

    artifact_type = "code"

    def __init__(self, config: LeviConfig):
        self.config = config
        self.fn_name = extract_fn_name(config.function_signature)

    def make_program(self, content: str, metadata: dict[str, Any] | None = None) -> Program:
        return Program(content=content, metadata=metadata or {})

    def snapshot_content(self, elite_data: Mapping[str, Any]) -> str:
        content = elite_data.get("content")
        if isinstance(content, str):
            return content

        legacy_code = elite_data.get("code")
        if isinstance(legacy_code, str):
            return legacy_code

        raise KeyError("content")

    async def evaluate(
        self,
        executor: ResilientProcessPool,
        content: str,
        *,
        inputs: list[Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return await executor.run(
            evaluate_code,
            content,
            self.config.score_fn,
            self.config.inputs if inputs is None else inputs,
            self.fn_name,
            timeout=self.config.pipeline.eval_timeout if timeout is None else timeout,
        )

    def build_mutation_prompt_from_template(
        self,
        parents: Sequence[ProgramWithScore],
        template: str,
        *,
        meta_advice: str | None = None,
        feedback: Sequence[str] | None = None,
        best_score: float | None = None,
        evals_since_best: int | None = None,
        stagnation: float | None = None,
        top_failures: Sequence[str] | None = None,
    ) -> str:
        """Build mutation prompt from a full-template string (prompt-bank mode).

        Supported placeholders (missing ones render as empty string):
          {problem_description}, {function_signature}, {parents_block},
          {search_trajectory_block}, {feedback_block}, {advisor_block}
        """

        class _SafeFmt(dict):
            def __missing__(self, key):  # type: ignore[override]
                return ""

        parent_blocks: list[str] = []
        for i, p in enumerate(parents):
            label = f"v{i + 1}"
            parent_blocks.append(f"## {label}\nScore: {p.score}\n```python\n{p.program.content}\n```")
        parents_block = "\n\n".join(parent_blocks)

        trajectory_body = _trajectory_body(
            best_score=best_score,
            evals_since_best=evals_since_best,
            stagnation=stagnation,
            top_failures=top_failures,
        )
        search_trajectory_block = f"## Search Trajectory\n{trajectory_body}" if trajectory_body else ""

        feedback_block = ""
        if feedback:
            bullets = "\n".join(f"- {f}" for f in feedback)
            feedback_block = f"## Feedback\n{bullets}"

        advisor_block = f"## Meta-Advice\n{meta_advice}" if meta_advice else ""

        values = _SafeFmt(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            parents_block=parents_block,
            search_trajectory_block=search_trajectory_block,
            feedback_block=feedback_block,
            advisor_block=advisor_block,
        )
        return template.format_map(values)

    def build_mutation_prompt(
        self,
        parents: Sequence[ProgramWithScore],
        *,
        meta_advice: str | None = None,
        model: ClientSpec | None = None,
        use_diff: bool = False,
        best_score: float | None = None,
        evals_since_best: int | None = None,
        stagnation: float | None = None,
        top_failures: Sequence[str] | None = None,
    ) -> str:
        builder = PromptBuilder()
        builder.add_section("Problem", self.config.problem_description, priority=10)
        builder.add_section("Signature", f"```python\n{self.config.function_signature}\n```", priority=20)
        builder.add_parents(list(parents), priority=30)

        # SAL Cơ chế A.2 — inject lightweight trajectory context so the
        # mutation model can see how far it is from the running best score.
        # We pass only the bullet body (header is added by PromptBuilder).
        trajectory_body = _trajectory_body(
            best_score=best_score,
            evals_since_best=evals_since_best,
            stagnation=stagnation,
            top_failures=top_failures,
        )
        if trajectory_body:
            builder.add_section("Search Trajectory", trajectory_body, priority=40)

        mutation_overrides = self.config.prompt_overrides.get("mutation", {})
        model_key = client_name(model) if model is not None else None
        if model_key and model_key in mutation_overrides:
            builder.set_custom_output(mutation_overrides[model_key])
        else:
            builder.set_output_mode(OutputMode.DIFF if use_diff else OutputMode.FULL)

        if meta_advice:
            builder.add_section("Meta-Advice", meta_advice, priority=100)

        return builder.build()

    def extract_candidate(
        self,
        response_text: str,
        *,
        parent_content: str | None = None,
        use_diff: bool = False,
    ) -> str | None:
        if use_diff:
            if parent_content is None:
                raise ValueError("parent_content is required when use_diff=True")
            return apply_diff(parent_content, response_text)
        return extract_code(response_text)

    def build_diversity_prompt(self, existing_candidates: Sequence[tuple[str, float]]) -> str:
        existing_seeds_text = "\n\n---\n\n".join(
            [
                f"### Seed {idx + 1} (Score: {score:.17g}):\n```python\n{content}\n```"
                for idx, (content, score) in enumerate(existing_candidates)
            ]
        )
        prompt_template = self.config.init.diversity_prompt or DIVERSITY_SEED_PROMPT
        return prompt_template.format(
            problem_title="Algorithm Optimization",
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            existing_seeds=existing_seeds_text,
        )

    def build_init_variant_prompt(self, parents: Sequence[ProgramWithScore]) -> str:
        builder = PromptBuilder()
        builder.add_section("Problem", self.config.problem_description, priority=10)
        builder.add_section("Signature", f"```python\n{self.config.function_signature}\n```", priority=20)
        builder.add_parents(list(parents), priority=30)
        builder.set_output_mode(OutputMode.FULL)
        return builder.build()

    def build_paradigm_shift_prompt(
        self,
        representatives: Sequence[tuple[int, Any]],
        *,
        n_evaluations: int,
        budget_progress: float = 0.0,
        stagnation: float | None = None,
        best_score: float | None = None,
        evals_since_best: int | None = None,
        top_failures: Sequence[str] | None = None,
        sal_thresholds: tuple[float, float] | None = None,
        strategy_log_block: str = "",
    ) -> str:
        """Build the early/mid/late paradigm-shift prompt.

        ``strategy_log_block`` is a markdown section (or empty string) listing
        past PE attempts; it is rendered just below the representative
        solutions so the heavy model can see what has and hasn't worked.
        """
        if sal_thresholds is not None:
            stage = get_budget_stage(
                budget_progress,
                stagnation=stagnation,
                mid_threshold=sal_thresholds[0],
                late_threshold=sal_thresholds[1],
            )
        else:
            stage = get_budget_stage(budget_progress, stagnation=stagnation)

        rep_text_parts = []
        for idx, (cluster_id, elite) in enumerate(representatives):
            score = elite.result.primary_score
            content = elite.program.content
            rep_text_parts.append(
                f"### Region {idx + 1} (Cluster {cluster_id}, Score: {score:.17g})\n```python\n{content}\n```"
            )

        representative_solutions = "\n\n".join(rep_text_parts)
        trajectory_block = _build_trajectory_block(
            best_score=best_score,
            evals_since_best=evals_since_best,
            stagnation=stagnation,
            top_failures=top_failures,
        )

        override = self.config.prompt_overrides.get("paradigm_shift")
        if override:
            return f"""# Algorithmic Paradigm Shift Challenge

## Problem
{self.config.problem_description}

## Function Signature
```python
{self.config.function_signature}
```

## Current Best Solutions ({len(representatives)} regions, {n_evaluations} evaluations)

{representative_solutions}
{strategy_log_block}{trajectory_block}
## Your Task
{override}

Output ONLY complete, runnable Python code in a ```python block.
"""

        template = PARADIGM_SHIFT_PROMPTS[stage]
        rendered = template.format(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            n_evaluations=n_evaluations,
            n_regions=len(representatives),
            representative_solutions=representative_solutions,
            strategy_log_block=strategy_log_block,
        )
        if trajectory_block:
            # Insert trajectory block right before the final Output section so
            # the model sees it as additional context, not as an instruction.
            split_token = "## Output"
            if split_token in rendered:
                head, _, tail = rendered.partition(split_token)
                rendered = f"{head}{trajectory_block}\n{split_token}{tail}"
            else:
                rendered = f"{rendered}\n{trajectory_block}"
        return rendered

    def build_variant_prompt(self, base_content: str, base_score: float) -> str:
        return VARIANT_GENERATION_PROMPT.format(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            base_code=base_content,
            base_score=base_score,
        )

    # ------------------------------------------------------------------
    # Strategy Log — light-model post-mortem summariser
    # ------------------------------------------------------------------

    def build_strategy_summary_prompt(self, code: str) -> str:
        """Render the IDEA/QUALITY strategy summarisation prompt.

        Passes the problem statement and function signature so the
        summariser can phrase "what works / where this loses points" in
        terms of *this* problem, not generic algorithmics.
        """
        return STRATEGY_SUMMARY_PROMPT.format(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            code=code,
        )

    # ------------------------------------------------------------------
    # Code Error Repair — one-shot light-model fix prompt
    # ------------------------------------------------------------------

    def build_code_repair_prompt(
        self,
        broken_code: str,
        *,
        error_msg: str,
        parent_score: float,
    ) -> str:
        return CODE_REPAIR_PROMPT.format(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            broken_code=broken_code,
            error_msg=(error_msg or "").strip() or "(unknown error)",
            parent_score=parent_score,
        )
