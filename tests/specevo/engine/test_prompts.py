"""Tests for the SpecEvo prompt templates and the :class:`PromptSampler`.

These are pure string-construction tests — no network. They verify
that:

* All five operator templates (3 mutate + 2 crossover) render without
  KeyError and contain their required section headers.
* The :class:`PromptSampler` returns a label that matches one of the
  registered templates, and exercises every template across enough
  draws.
* The targeted-mutate template injects the analysis text verbatim.
* All three Navigator intervention mode builders render and carry the right
  ``MOVE: ...`` requirement plus their distinguishing section names.
"""

from __future__ import annotations

import random

from specevo.engine.prompts import (
    CROSSOVER_PROMPTS,
    MUTATE_PROMPTS,
    PromptSampler,
    build_crossover_prompt,
    build_mutate_prompt,
    build_reframe_prompt,
    build_surgical_exploit_prompt,
    build_synthesis_prompt,
    build_targeted_mutate_prompt,
)


PROBLEM = "Maximise solve(0) + solve(1) + solve(2)."
SIGNATURE = "def solve(x):"
PARENT_CODE = "def solve(x):\n    return x\n"


def test_all_three_mutate_templates_render() -> None:
    for label, template in MUTATE_PROMPTS.items():
        prompt = build_mutate_prompt(
            problem_description=PROBLEM,
            function_signature=SIGNATURE,
            parent_code=PARENT_CODE,
            parent_score=1.23,
            inspirations=[("desc1", 1.0)],
            advisor="be careful",
            template=template,
        )
        # Common required pieces.
        assert "def solve(x):" in prompt
        assert "Score: 1.2300" in prompt
        assert "desc1" in prompt
        assert "be careful" in prompt
        # Each variant should mention its distinguishing keyword.
        if label == "general":
            assert "Components" in prompt
        elif label == "focused_fix":
            assert "Tightest constraint" in prompt
        elif label == "mechanism_swap":
            assert "Mechanism" in prompt


def test_all_two_crossover_templates_render() -> None:
    for label, template in CROSSOVER_PROMPTS.items():
        prompt = build_crossover_prompt(
            problem_description=PROBLEM,
            function_signature=SIGNATURE,
            parent_a_code=PARENT_CODE,
            parent_a_score=1.0,
            parent_b_code=PARENT_CODE,
            parent_b_score=2.0,
            inspirations=[("descX", 0.5)],
            template=template,
        )
        assert "def solve(x):" in prompt
        assert "1.0000" in prompt and "2.0000" in prompt
        if label == "structural":
            assert "Structural Hybrid" in prompt or "structural hybrid" in prompt.lower()
        elif label == "component_swap":
            assert "Component Swap" in prompt or "component swap" in prompt.lower()


def test_targeted_mutate_injects_analysis_verbatim() -> None:
    analysis = "ANALYSIS-MARKER: cooling decays too fast at step ~5000"
    prompt = build_targeted_mutate_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        parent_code=PARENT_CODE,
        parent_score=2.0,
        analysis=analysis,
        inspirations=[],
    )
    assert "ANALYSIS-MARKER" in prompt
    assert "Targeted Mutate" in prompt
    assert "exactly ONE structural change" in prompt


def test_prompt_sampler_returns_valid_labels() -> None:
    sampler = PromptSampler()
    rng = random.Random(0)
    mutate_labels = {sampler.pick_mutate(rng)[0] for _ in range(60)}
    crossover_labels = {sampler.pick_crossover(rng)[0] for _ in range(60)}
    # With 60 draws on a uniform 3-way / 2-way, every variant should
    # be hit at least once.
    assert mutate_labels == {"general", "focused_fix", "mechanism_swap"}
    assert crossover_labels == {"structural", "component_swap"}


def test_paradigm_synthesis_prompt_render() -> None:
    prompt = build_synthesis_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        n_evaluations=100,
        n_cells=20,
        anchors=[
            (PARENT_CODE, "anchor1", 2.0),
            (PARENT_CODE, "anchor2", 1.9),
            (PARENT_CODE, "anchor3", 1.85),
        ],
        inspirations=[("desc", 1.0)],
        stagnation=0.3,
    )
    assert "MOVE: SYNTHESIS" in prompt
    assert "Component table" in prompt
    assert "anchor1" in prompt


def test_navigator_intervention_prompt_render() -> None:
    prompt = build_reframe_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        n_evaluations=100,
        n_cells=20,
        anchors=[(PARENT_CODE, "anchorA", 2.0), (PARENT_CODE, "anchorB", 1.5)],
        inspirations=[],
        stagnation=0.55,
    )
    assert "MOVE: SHIFT" in prompt
    assert "fundamentally different" in prompt
    assert "Paradigm name" in prompt


def test_prompts_keep_analysis_separate_from_description() -> None:
    """The structured analysis sections (Components, Strengths, etc.)
    must go into a ``## Analysis`` block — NOT into ``## Description``.
    The archive embeds ``## Description``, so if the bullets leak there
    the embedding loses its semantic value. This pins the contract.
    """
    prompt = build_mutate_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        parent_code=PARENT_CODE,
        parent_score=1.0,
        inspirations=[],
        template=MUTATE_PROMPTS["general"],
    )
    # The hint that orders Analysis → Description → Code MUST be present.
    assert "## Analysis" in prompt
    assert "## Description" in prompt
    assert "## Code" in prompt
    # The model is explicitly told the analysis is not used for embedding.
    assert "ignores it for archiving" in prompt or "NOT used to embed" in prompt
    # The structured headers must be in the Analysis instructions, not
    # in the Description instructions. We check the inverse: the
    # OUTPUT_FORMAT_INSTRUCTION block's "no bullet points" constraint
    # is still in the prompt and the Output sections reminder names
    # exactly THREE sections.
    assert "THREE sections" in prompt
    assert "no bullet points" in prompt


def test_paradigm_surgical_prompt_clamps_to_one_anchor() -> None:
    """Surgical mode must render only ONE anchor even if more are passed."""
    prompt = build_surgical_exploit_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        n_evaluations=200,
        n_cells=30,
        anchors=[
            (PARENT_CODE, "champion", 2.5),
            (PARENT_CODE, "runner_up", 2.4),
            (PARENT_CODE, "third", 2.3),
        ],
        inspirations=[("top_desc1", 2.2), ("top_desc2", 2.1)],
        stagnation=0.85,
    )
    assert "MOVE: SURGICAL" in prompt
    assert "champion" in prompt
    # Other anchors should NOT appear as anchor blocks.
    assert "runner_up" not in prompt
    assert "third" not in prompt
    # But inspirations (description-only) must be present.
    assert "top_desc1" in prompt
