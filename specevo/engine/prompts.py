"""SpecEvo prompt templates.

Operator prompts used by the mutation / Navigator workers, plus thin
wrappers around the framework's :mod:`specevo.equilibrium.prompts` and
:mod:`specevo.artifacts.code` strings for the bootstrap phase.

Compared to the previous version, three things changed:

1. **Mutate / crossover prompts ask for explicit analysis first.** The
   model writes a short ``# Analysis`` section (strengths, weaknesses,
   chosen target) *before* the code, so it commits to a hypothesis
   rather than randomly rewriting whatever it sees.

2. **Multiple prompt templates per operator.** Three mutate templates
   (general improvement, focused weakness fix, mechanism swap) and two
   crossover templates (structural hybrid, targeted component swap)
   are exposed via :class:`PromptSampler`. Every call draws one
   template uniformly at random — no learned weights — so the
   mutation model sees prompt-level diversity even when the parent
   pool is narrow. The :data:`TARGETED_MUTATE_PROMPT` is a separate
   template only used when a cached LLM-generated analysis is
   available.

3. **Three Navigator intervention modes.** ``build_synthesis_prompt`` (2-3
   anchors, hybridise close contenders), ``build_reframe_prompt``
   (2 anchors, propose a genuinely new Navigator class), and
   ``build_surgical_exploit_prompt`` (1 anchor = best, top descriptions
   as inspiration, deep micro-improvement of the current champion).
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..artifacts.code import DIVERSITY_SEED_PROMPT
from ..equilibrium.prompts import VARIANT_GENERATION_PROMPT
from ..simple.parser import OUTPUT_FORMAT_INSTRUCTION

__all__ = [
    "MUTATE_PROMPTS",
    "CROSSOVER_PROMPTS",
    "TARGETED_MUTATE_PROMPT",
    "PromptSampler",
    "build_mutate_prompt",
    "build_crossover_prompt",
    "build_targeted_mutate_prompt",
    "build_analysis_prompt",
    "build_synthesis_prompt",
    "build_reframe_prompt",
    "build_surgical_exploit_prompt",
    "build_diverse_seed_prompt",
    "build_init_variant_prompt",
    "build_navigator_variant_prompt",
    "build_advisor_prompt",
    "error_signature",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _inspiration_block(inspirations: Sequence[tuple[str, float]]) -> str:
    """Render inspirations as ``description + score`` only (no code)."""
    if not inspirations:
        return "(no inspirations available yet)"
    parts: list[str] = []
    for i, (desc, score) in enumerate(inspirations, start=1):
        d = (desc or "").strip().replace("\n", " ")
        if len(d) > 400:
            d = d[:400].rstrip() + "…"
        parts.append(f"{i}. (score={score:.3f}) {d}")
    return "\n".join(parts)


def _advisor_block(advisor: str | None) -> str:
    """Format the optional meta-advisor lesson block."""
    if not advisor:
        return ""
    return f"## Lessons learnt so far\n{advisor.strip()}\n\n"


# Reminder injected immediately AFTER OUTPUT_FORMAT_INSTRUCTION in every
# analysis-augmented prompt. The base instruction says "output must
# contain `## Description` and `## Code` in order" — we extend that
# contract with an extra `## Analysis` section that comes FIRST and is
# ignored by the parser (so the embedded description stays a clean
# 2-4 sentence paragraph). Without this reminder the model tends to
# either fold the analysis bullets into `## Description` (which then
# corrupts the embedding used for cell assignment) or skip them
# entirely.
ANALYSIS_OUTPUT_ORDER_HINT = """\

## Output sections (final reminder)
Your output MUST contain THREE sections, in this exact order:

1. ``## Analysis`` — the structured reasoning requested above. This
   section is read by the model on subsequent passes but is NOT used
   to embed the program into the search archive.
2. ``## Description`` — the short paragraph defined by the output
   format spec above (2-4 sentences, ≤ 80 words, plain prose, no
   bullet points). This IS what the archive embeds, so do not paste
   the analysis bullets here.
3. ``## Code`` — the fenced ```python``` program block.

Do not merge sections. Do not skip ``## Description``.
"""


def _anchor_block(anchors: Sequence[tuple[str, str, float]]) -> str:
    """Render anchor representatives with full code + description."""
    if not anchors:
        return "(no anchors available yet)"
    parts: list[str] = []
    for i, (code, desc, score) in enumerate(anchors, start=1):
        d = (desc or "").strip().replace("\n", " ")
        if len(d) > 400:
            d = d[:400].rstrip() + "…"
        parts.append(
            f"#### Anchor {i} (score={score:.4f})\n"
            f"_Description_: {d or '(no description)'}\n"
            f"```python\n{code}\n```"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Mutate prompt templates — three variants, all ask for explicit analysis.
# ---------------------------------------------------------------------------


MUTATE_PROMPT_GENERAL = """\
# Mutate — Improvement Pass

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Parent solution
Score: {parent_score:.4f}
```python
{parent_code}
```

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{advisor_block}\
## Your task

Write an **improved** version of the parent. This is a mutation pass:
preserve the parent`s useful structure, but make a concrete change that
is likely to improve the score.

Before the final code, write a short ``## Analysis`` section. The search
system ignores it for archiving, but may reuse it in future passes. It
must contain exactly four sub-sections:

1. **Components.** List the 3-5 main components/phases of the parent
   program, using concrete code names when possible.
2. **Strengths.** State which components are working well and should be
   preserved.
3. **Weaknesses.** Identify the most likely bottleneck. Cite the exact
   routine, loop, variable, constant, update rule, repair step, or
   acceptance rule.
4. **Plan.** In one sentence, state the concrete mutation you will make
   and why it should improve the score.

Then write the improved program. Keep what works, change what does not.
The new code must contain at least one real algorithmic improvement, not
only renaming, reformatting, comment changes, seed changes, or isolated
constant retuning. Treat the inspirations as high-level ideas only — you
do not have their code, so do not copy or assume hidden details.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors (matching parentheses, quotes, indentation)
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


MUTATE_PROMPT_FOCUSED_FIX = """\
# Mutate — Focused Weakness Fix

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Parent solution
Score: {parent_score:.4f}
```python
{parent_code}
```

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{advisor_block}\
## Your task

The parent is already mostly working. Your job is to pick exactly **ONE**
weakness and fix it surgically. Preserve the parent`s algorithmic
identity and leave all unrelated components untouched.

First, write a short ``## Analysis`` section with exactly three
sub-sections:

1. **Tightest constraint.** Identify the single most limiting factor in
   the parent`s score. Be specific: cite the exact routine, loop,
   variable, constant, update rule, repair step, or acceptance rule.
   Prefer a real algorithmic bottleneck over seed changes, formatting,
   or isolated constant tuning.
2. **Fix.** State the minimum code change that addresses this constraint
   and why it should improve the score. One sentence only.
3. **Preserved.** State which top-level routines/components remain
   unchanged.

Then output the complete program implementing exactly that one surgical
change. The change must affect the algorithm`s behavior in a meaningful
way. Do not retune unrelated constants, rename variables, reformat code,
change comments, change only the random seed, or add extra improvements.
Do not add new helper functions unless the helper is directly required by
the surgical fix.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


MUTATE_PROMPT_MECHANISM_SWAP = """\
# Mutate — Mechanism Swap

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Parent solution
Score: {parent_score:.4f}
```python
{parent_code}
```

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{advisor_block}\
## Your task

The parent contains one or more **interchangeable mechanisms**: local
slots where another algorithmic choice can fit without changing the
overall algorithm. Identify one such slot and **swap only that
mechanism** for a stronger alternative.

Examples of valid swaps:
- random uniform initialisation → quasi-random / stratified initialisation
- fixed step size → adaptive step size or line search
- greedy acceptance → Metropolis or score-aware acceptance
- fixed restart schedule → stagnation-triggered restart
- simple repair rule → slack-aware repair rule

First, write a short ``## Analysis`` section with exactly three
sub-sections:

1. **Mechanism identified.** Name the exact slot being replaced. Cite
   the function, loop, block, variable, or update rule.
2. **Old vs new.** State: “Currently uses X; replacing it with Y because
   Z.” The replacement must fit the same role and preserve the parent`s
   overall algorithm.
3. **Risk.** One sentence on what could go wrong with the swap and how
   the new code guards against it.

Then write the complete program. Keep the signature, I/O contract, and
parent structure stable. Only the selected mechanism should change. Do
not add unrelated improvements, retune unrelated constants, rename
variables, reformat code, or rewrite the algorithm into a new paradigm.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


# Targeted mutation — only fired when an LLM-generated analysis already
# exists for this parent (see :meth:`SpecEvoOrchestrator._analyze_parent`).
TARGETED_MUTATE_PROMPT = """\
# Targeted Mutate (analysis-guided)

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Parent solution
Score: {parent_score:.4f}
```python
{parent_code}
```

## Analysis of this parent (produced earlier by a review pass)
{analysis}

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{advisor_block}\
## Your task

Read the analysis above and choose exactly ONE suggested change — the one
most likely to improve the score on this problem. Do not invent a new
direction; select from the existing analysis and implement it cleanly.

First, write a short ``## Analysis`` section with exactly three
sub-sections:

1. **Chosen bottleneck.** Quote or precisely restate the bottleneck from
   the analysis above that you are targeting.
2. **Implementation plan.** In one sentence, describe the exact
   structural code change you will make.
3. **What stays unchanged.** List the main routines, constants, or
   control-flow blocks that will remain unchanged.

Then write the complete improved program. Make exactly ONE structural change.
Do not rewrite the algorithm, add unrelated improvements, retune unrelated
constants, rename variables, reformat code, or change only the random seed.
The rest of the parent is assumed to be working and should remain stable.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


# ---------------------------------------------------------------------------
# Crossover prompt templates — two variants.
# ---------------------------------------------------------------------------


CROSSOVER_PROMPT_STRUCTURAL = """\
# Crossover — Structural Hybrid

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Parent A
Score: {parent_a_score:.4f}
```python
{parent_a_code}
```

## Parent B
Score: {parent_b_score:.4f}
```python
{parent_b_code}
```

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{advisor_block}\
## Your task

Produce a **structural hybrid** from the two parents. Select the
strongest compatible mechanism from each parent and re-integrate them
into one coherent program. Do not simply paste a block from A into B;
every routine in the output must fit the same data flow, objective, and
constraint logic.

First, write a short ``## Analysis`` section with exactly three
sub-sections:

1. **Component table.** Give a 3-row mapping:
   - Initialisation: from A/B/hybrid — reason
   - Optimisation core: from A/B/hybrid — reason
   - Constraint handling: from A/B/hybrid — reason
2. **Compatibility note.** One sentence explaining how you reconciled
   interface or representation differences between the selected
   components.
3. **Expected improvement.** One sentence explaining why this hybrid
   should plausibly outperform both parents.

Then write the complete program. It must be self-contained: every helper
function used in the output must be defined in the output. Do not
reference functions, variables, or hidden state from either parent unless
you re-define them. Preserve the required signature and I/O contract.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


CROSSOVER_PROMPT_COMPONENT_SWAP = """\
# Crossover — Targeted Component Swap

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Base parent (we keep this skeleton)
Score: {parent_a_score:.4f}
```python
{parent_a_code}
```

## Donor parent (we steal ONE component from this)
Score: {parent_b_score:.4f}
```python
{parent_b_code}
```

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{advisor_block}\
## Your task

Treat the base parent as the **skeleton**: preserve its overall control
flow, data flow, and termination logic. Identify exactly **ONE**
component in the donor parent that is clearly better than the base
parent`s equivalent, and transplant only that component. Adapt the
surrounding glue minimally so it fits.

First, write a short ``## Analysis`` section with exactly three
sub-sections:

1. **Donor component chosen.** Name the donor routine, block, update
   rule, repair step, constant schedule, or data structure being
   transplanted, and explain why it is better than the base equivalent.
2. **Glue work.** State the minimal changes needed to integrate it into
   the base parent, such as variable names, data layout, helper calls,
   or call sites.
3. **What is NOT changed.** Confirm which parts of the base parent remain
   unchanged, especially initialisation, main optimisation loop,
   termination condition, and I/O contract. If one of these must change,
   explain the exact reason.

Then write the complete resulting program. Keep the base parent
recognisable. The output must contain one real donor-derived component,
but the base parent must still determine the program`s overall
structure. Do not import unrelated mechanisms from the donor, do not
rewrite the base algorithm, and do not make extra improvements beyond
the single transplant.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


# ---------------------------------------------------------------------------
# Prompt sampler — uniform random over the variant templates above.
# ---------------------------------------------------------------------------


@dataclass
class PromptSampler:
    """Picks one mutate / crossover prompt variant uniformly at random.

    No learning, no Thompson sampling — the variants are roughly
    equally useful and the mutation model benefits from prompt-level
    diversity regardless of which template wins on a given parent.

    ``single_prompt`` is the operator-prompt ablation (A6): when set,
    the sampler stops drawing at random and always returns the single
    simplest template per operator — ``MUTATE_PROMPT_GENERAL`` for
    mutate and ``CROSSOVER_PROMPT_STRUCTURAL`` for crossover. This
    isolates the contribution of prompt-template diversity from the
    rest of the search.
    """

    mutate_templates: list[str] = field(
        default_factory=lambda: [
            MUTATE_PROMPT_GENERAL,
            MUTATE_PROMPT_FOCUSED_FIX,
            MUTATE_PROMPT_MECHANISM_SWAP,
        ]
    )
    crossover_templates: list[str] = field(
        default_factory=lambda: [
            CROSSOVER_PROMPT_STRUCTURAL,
            CROSSOVER_PROMPT_COMPONENT_SWAP,
        ]
    )
    single_prompt: bool = False

    def pick_mutate(self, rng: random.Random) -> tuple[str, str]:
        """Return (label, template) for one mutate variant.

        Under ``single_prompt`` (A6) this always returns the simplest
        general-improvement template, ignoring ``rng``."""
        if self.single_prompt:
            return "general", MUTATE_PROMPT_GENERAL
        labels = ["general", "focused_fix", "mechanism_swap"]
        idx = rng.randrange(len(self.mutate_templates))
        return labels[idx], self.mutate_templates[idx]

    def pick_crossover(self, rng: random.Random) -> tuple[str, str]:
        """Return (label, template) for one crossover variant.

        Under ``single_prompt`` (A6) this always returns the simplest
        structural-hybrid template, ignoring ``rng``."""
        if self.single_prompt:
            return "structural", CROSSOVER_PROMPT_STRUCTURAL
        labels = ["structural", "component_swap"]
        idx = rng.randrange(len(self.crossover_templates))
        return labels[idx], self.crossover_templates[idx]


# Public aliases so callers can introspect / extend the variant set.
MUTATE_PROMPTS = {
    "general": MUTATE_PROMPT_GENERAL,
    "focused_fix": MUTATE_PROMPT_FOCUSED_FIX,
    "mechanism_swap": MUTATE_PROMPT_MECHANISM_SWAP,
}
CROSSOVER_PROMPTS = {
    "structural": CROSSOVER_PROMPT_STRUCTURAL,
    "component_swap": CROSSOVER_PROMPT_COMPONENT_SWAP,
}


def build_mutate_prompt(
    *,
    problem_description: str,
    function_signature: str,
    parent_code: str,
    parent_score: float,
    inspirations: Sequence[tuple[str, float]],
    advisor: str | None = None,
    template: str | None = None,
) -> str:
    tmpl = template if template is not None else MUTATE_PROMPT_GENERAL
    return tmpl.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_code=parent_code,
        parent_score=parent_score,
        inspirations_block=_inspiration_block(inspirations),
        advisor_block=_advisor_block(advisor),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )


def build_crossover_prompt(
    *,
    problem_description: str,
    function_signature: str,
    parent_a_code: str,
    parent_a_score: float,
    parent_b_code: str,
    parent_b_score: float,
    inspirations: Sequence[tuple[str, float]],
    advisor: str | None = None,
    template: str | None = None,
) -> str:
    tmpl = template if template is not None else CROSSOVER_PROMPT_STRUCTURAL
    return tmpl.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_a_code=parent_a_code,
        parent_a_score=parent_a_score,
        parent_b_code=parent_b_code,
        parent_b_score=parent_b_score,
        inspirations_block=_inspiration_block(inspirations),
        advisor_block=_advisor_block(advisor),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )


def build_targeted_mutate_prompt(
    *,
    problem_description: str,
    function_signature: str,
    parent_code: str,
    parent_score: float,
    analysis: str,
    inspirations: Sequence[tuple[str, float]],
    advisor: str | None = None,
) -> str:
    return TARGETED_MUTATE_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_code=parent_code,
        parent_score=parent_score,
        analysis=analysis.strip(),
        inspirations_block=_inspiration_block(inspirations),
        advisor_block=_advisor_block(advisor),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )


# ---------------------------------------------------------------------------
# Analysis prompt — produced by the mutation model and cached per parent.
# ---------------------------------------------------------------------------


ANALYSIS_PROMPT = """\
# Code Review — Bottleneck Identification

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Program under review (score={parent_score:.4f})
_Description_: {parent_description}
```python
{parent_code}
```

## Your task

Review the program and identify why it likely does not score higher.
Write exactly **three** short sections using the headings below. Write
analysis text only — do NOT write code, pseudocode, code blocks, or
extra sections.

Be specific, score-oriented, and implementation-oriented. Cite concrete
code elements: function names, loops, variables, update rules, repair
steps, acceptance rules, data structures, and magic constants. If line
numbers are unavailable, refer to the nearest function/block name and
the relevant variable or constant.

### Algorithm summary
In 1-2 sentences, describe the algorithmic class, its main search or
construction strategy, and the key data structures it relies on.

### Top 3 bottlenecks, ranked by expected impact
List exactly three distinct bottlenecks. For each one, include:
- **Component:** the exact routine, block, variable, constant, update
  rule, repair step, acceptance rule, or data structure involved.
- **Why it hurts:** how this mechanism limits score, feasibility,
  exploration, exploitation, runtime, or objective alignment.
- **Impact:** High / Medium / Low expected improvement if fixed.

Prefer real algorithmic bottlenecks over cosmetic issues or isolated
constant tuning. The three bottlenecks must differ in kind.

### Suggested changes
Give exactly three actionable changes, one per bottleneck and in the
same order. Each change must state what to replace, add, or remove, and
where. Each should be specific enough to implement directly in a
mutation pass without further clarification.

Avoid vague advice such as "improve optimization", "tune parameters",
"make it faster", or "use a better heuristic". Keep the whole review
under 250 words.
"""


def build_analysis_prompt(
    *,
    problem_description: str,
    function_signature: str,
    parent_code: str,
    parent_score: float,
    parent_description: str,
) -> str:
    desc = (parent_description or "").strip() or "(no description)"
    return ANALYSIS_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_code=parent_code,
        parent_score=parent_score,
        parent_description=desc,
    )


# ---------------------------------------------------------------------------
# Bootstrap-phase prompt builders — mirror the framework's Diversifier.
# ---------------------------------------------------------------------------


INIT_VARIANT_PROMPT = """\
# Init Variant

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Inspirations (existing diverse seeds — code + score)
{inspirations_block}

## Your task
Write an improved, complete, self-contained implementation of the function. Use the inspirations to identify useful patterns, 
but produce a meaningfully different seed rather than copying one. Aim for a variant that is both feasible and competitive.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. The function signature must match exactly what is specified
4. Ensure there are no syntax errors (matching parentheses, quotes, indentation)

{format_instruction}
"""


def _seed_block(seeds: Sequence[tuple[str, float]]) -> str:
    if not seeds:
        return "(no seeds available yet)"
    parts: list[str] = []
    for idx, (code, score) in enumerate(seeds, start=1):
        parts.append(
            f"### Seed {idx} (score={score:.4f})\n```python\n{code}\n```"
        )
    return "\n\n".join(parts)


def build_diverse_seed_prompt(
    *,
    problem_description: str,
    function_signature: str,
    existing_seeds: Sequence[tuple[str, float]],
) -> str:
    existing_seeds_text = "\n\n---\n\n".join(
        f"### Seed {i + 1} (Score: {score:.17g}):\n```python\n{code}\n```"
        for i, (code, score) in enumerate(existing_seeds)
    )
    rendered = DIVERSITY_SEED_PROMPT.format(
        problem_title="Algorithm Optimization",
        problem_description=problem_description,
        function_signature=function_signature,
        existing_seeds=existing_seeds_text,
    )
    return rendered + "\n\n" + OUTPUT_FORMAT_INSTRUCTION


def build_init_variant_prompt(
    *,
    problem_description: str,
    function_signature: str,
    inspirations: Sequence[tuple[str, float]],
) -> str:
    return INIT_VARIANT_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        inspirations_block=_seed_block(inspirations),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
    )


# ---------------------------------------------------------------------------
# Meta-advisor ("Advisor")
#
# The Advisor periodically writes a short prescriptive note that future
# mutation prompts include verbatim. Credit is assigned by **behavioural
# niche (archive cell)**, never by which prompt-template produced a program:
# the template is a uniform random draw and carries no actionable signal.
# Each cycle the Advisor reads, per niche — LEADING (top-score incumbents),
# IMPROVING (niches whose cell incumbent advanced this window: what just
# paid off), SATURATED (niches still attracting attempts but whose frontier
# has not moved: mined out), UNDER-EXPLORED (occupied niches with few recent
# attempts) — plus an *accumulated* failure-knowledge base (every error seen
# so far, grouped by a domain-agnostic signature and counted by recurrence)
# that feeds AVOID. There is no score-delta threshold: "improving vs
# saturated" is decided purely by whether a niche's frontier moved.
# ---------------------------------------------------------------------------


ADVISOR_PROMPT = """\
# Lessons-Learned Advisor

You are reviewing the search trajectory on this optimisation problem and
writing a short prescriptive note that the mutation model will read
verbatim before SOME future attempts (roughly one in three). Your job is
NOT to restate the problem or repeat constraints the grader already
enforces — it is to amplify what is working, name behavioural niches that
have *saturated* (still attract attempts but no longer improve), point at
the next concrete thing to try, and call out failure modes that are
actually costing evaluations.

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Current state
- Best score so far: {best_score}
- Evaluations completed: {n_evaluations}
- Accept rate (last window): {accept_rate}
- Stagnation level: {stagnation_level} (0=fresh, 1=plateaued)

## Leaders — current niche champions (description + score)
{leaders_block}

## IMPROVING niches — frontier advanced this window (what is actually paying off)
{improving_block}

## SATURATED niches — many recent attempts but frontier stuck (mined out)
{saturated_block}

## Under-explored niches — few recent attempts (room to push)
{under_explored_block}

## Accumulated failure knowledge — every error seen so far, by recurrence
{error_knowledge_block}

## Previous advice (carried over so you can refine, not repeat)
{previous_advice_block}

## Your task
Write the new advice block using EXACTLY these four short sections, in
this order, with these literal headers and no other markdown:

WORKING: <1-2 sentences naming the concrete approach / structure / mechanism
that the IMPROVING niches and leaders share — read their descriptions and
describe the algorithm itself, not any meta-process. If nothing is clearly
working yet, say so plainly.>

SATURATED: <1-2 sentences naming any behavioural niche / strategy family
that keeps attracting attempts but no longer improves — i.e. the SATURATED
niches above. Future prompts should de-emphasise this direction. If no
clear saturation, write "none".>

TRY NEXT: <2-3 short imperative suggestions, ordered by priority. Be
code-shaped (mention specific data structures, algorithms, numerical
ranges, library calls). Build on WORKING, push into the under-explored
niches, and EXPLICITLY move away from SATURATED — do NOT propose more of
the same.>

AVOID: <1-2 anti-patterns drawn from the accumulated failure knowledge
above, prioritising the most recurrent ones. Skip generic defensive advice
(constraint checks that the grader already enforces, type validations,
etc.) unless they recur heavily here.>

Total length: under 140 words. No preamble. No extra headers. No bullet
characters — write each section as a single short paragraph after its
header.
"""


def error_signature(msg: str) -> str:
    """Domain-agnostic recurrence key for an error message.

    Lower-cases, strips digits and punctuation, collapses whitespace and
    keeps the leading words, so that e.g. ``"Overlap between circles 0 and
    2"`` and ``"Overlap between circles 3 and 5"`` map to the same signature
    and are counted as one recurring failure mode. There is no hard-coded,
    benchmark-specific keyword table — the recurring text *is* the taxonomy,
    which keeps the Advisor portable across problems.
    """
    s = (msg or "").lower().strip()
    s = re.sub(r"[0-9]+(\.[0-9]+)?", "", s)   # drop numbers (indices, sizes)
    s = re.sub(r"[^a-z\s]", " ", s)            # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "unknown error"
    return " ".join(s.split()[:8])


def _niche_block(
    niches: Sequence[tuple[str, float]], empty: str = "(none in this window)"
) -> str:
    """Render ``[(description, score), ...]`` niches, one per line."""
    if not niches:
        return empty
    parts: list[str] = []
    for desc, score in niches:
        d = (desc or "").strip().replace("\n", " ")
        if len(d) > 280:
            d = d[:280].rstrip() + "…"
        parts.append(f"- (score={score:.4f}) {d}")
    return "\n".join(parts)


def _saturated_block(niches: Sequence[tuple[str, float, int]]) -> str:
    """Render busy-but-stale niches with their recent-attempt count."""
    if not niches:
        return "(none — no niche is both stuck and busy)"
    parts: list[str] = []
    for desc, score, attempts in niches:
        d = (desc or "").strip().replace("\n", " ")
        if len(d) > 240:
            d = d[:240].rstrip() + "…"
        parts.append(f"- ({attempts} recent attempts, score={score:.4f}) {d}")
    return "\n".join(parts)


def _error_knowledge_block(error_knowledge: Sequence[tuple[str, int, str]]) -> str:
    """Render the accumulated failure knowledge, most-recurrent first.

    ``error_knowledge`` is ``[(signature, count, example_message), ...]``
    accumulated across the *whole* run (not just the current window)."""
    if not error_knowledge:
        return "(no failures recorded yet)"
    rows = sorted(error_knowledge, key=lambda t: -t[1])[:8]
    lines: list[str] = []
    for _sig, count, example in rows:
        clean = (example or "").strip().replace("\n", " ")
        if len(clean) > 160:
            clean = clean[:160].rstrip() + "…"
        lines.append(f"- ×{count}  {clean}")
    return "\n".join(lines)


def build_advisor_prompt(
    *,
    problem_description: str,
    function_signature: str,
    best_score: float,
    n_evaluations: int,
    accept_rate: float,
    stagnation_level: float,
    leaders: Sequence[tuple[str, float]] = (),
    improving: Sequence[tuple[str, float]] = (),
    saturated: Sequence[tuple[str, float, int]] = (),
    under_explored: Sequence[tuple[str, float]] = (),
    error_knowledge: Sequence[tuple[str, int, str]] = (),
    previous_advice: str | None = None,
) -> str:
    """Build the Advisor prompt from region-based trajectory signals.

    ``leaders``        -- ``[(description, score), ...]`` top niche champions.
    ``improving``      -- ``[(description, score), ...]`` niches whose frontier
                          (cell incumbent) advanced in the current window.
    ``saturated``      -- ``[(description, score, n_recent_attempts), ...]``
                          niches that keep attracting attempts but whose
                          frontier has not moved.
    ``under_explored`` -- ``[(description, score), ...]`` occupied niches with
                          the fewest recent attempts.
    ``error_knowledge``-- ``[(signature, count, example), ...]`` accumulated
                          across the whole run; rendered most-recurrent first.
    """
    if best_score == float("-inf") or not math.isfinite(best_score):
        best_score_str = "n/a"
    else:
        best_score_str = f"{best_score:.4f}"
    previous = (previous_advice or "").strip()
    previous_block = previous if previous else "(none yet — this is the first cycle)"
    return ADVISOR_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        best_score=best_score_str,
        n_evaluations=n_evaluations,
        accept_rate=f"{accept_rate:.2f}",
        stagnation_level=f"{stagnation_level:.2f}",
        leaders_block=_niche_block(leaders, empty="(archive empty)"),
        improving_block=_niche_block(improving),
        saturated_block=_saturated_block(saturated),
        under_explored_block=_niche_block(under_explored),
        error_knowledge_block=_error_knowledge_block(error_knowledge),
        previous_advice_block=previous_block,
    )


def build_navigator_variant_prompt(
    *,
    problem_description: str,
    function_signature: str,
    base_code: str,
    base_score: float,
) -> str:
    rendered = VARIANT_GENERATION_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        base_code=base_code,
        base_score=base_score,
    )
    return rendered + "\n\n" + OUTPUT_FORMAT_INSTRUCTION


# ---------------------------------------------------------------------------
# Navigator intervention prompts — three modes, dispatched by the orchestrator.
# ---------------------------------------------------------------------------
#
# Each mode receives a different ANCHOR configuration and asks the
# frontier model for a different kind of move:
#
#   • synthesis  — 2-3 anchors, hybridise close contenders.
#   • shift      — 2 anchors, design a fundamentally new Navigator.
#   • surgical   — 1 anchor = best, top descriptions as inspiration;
#                  deep tuning of the current champion.
#
# The orchestrator picks the mode based on stagnation level (low →
# synthesis, mid → surgical, high → shift) and supplies the right
# number of anchors. See ``SpecEvoConfig.navigator_*_max_stagnation`` and
# ``SpecEvoOrchestrator._pick_navigator_mode`` for the authoritative routing:
# deepest stagnation routes to ``reframe`` (a genuinely new Navigator is the
# most reliable way out of a plateau), the mid band to ``surgical``.


SYNTHESIS_PROMPT = """\
# Paradigm Synthesis Challenge

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Archive Snapshot
The archive has run {n_evaluations} evaluations and currently occupies
{n_cells} behavioural cells. Stagnation level is {stagnation:.2f}
(0 = just improved, 1 = stuck). The search is **mildly stalled**:
several anchors are close in score but no single mutation has
combined their strengths.

### Top anchors (close-in-score, full code)
{anchor_block}

### Other paradigm inspirations (description + score only)
{inspiration_block}
{strategy_log_block}
## Your Task

Your job is **synthesis**, not invention. Read the anchors and write
ONE new program that combines 2-3 concrete mechanisms drawn from
different anchors into a structurally coherent whole, beating each of
them individually.

First, write a ``## Analysis`` section. Its first line MUST be
``MOVE: SYNTHESIS``. After that line, include exactly these three
sub-sections:

1. **Component table.** A mapping like:
   - Initialisation: from Anchor X (reason …)
   - Optimisation core: from Anchor Y (reason …)
   - Constraint repair: hybrid (reason …)
2. **Coherence note.** A sentence describing how the borrowed
   components share data — variable layout, units, call ordering.
   This is the hard part of synthesis: avoid Frankenstein code.
3. **Why this should beat all anchors.** One sentence per anchor:
   "beats anchor X because …".

Then write the program. Avoid any strategy whose Strategy-Log entry
has delta ≤ 0 — that approach has already failed. Do NOT just retune
constants in one anchor (the mutation worker is already doing that).

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. Every helper function must be defined here (don't reference functions
   from the anchors by name unless you copy their definition)

{format_instruction}
{analysis_output_order_hint}\
"""


REFRAME_PROMPT = """\
# Paradigm Shift Challenge — Genuinely New Approach

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Archive Snapshot
The archive has run {n_evaluations} evaluations and currently occupies
{n_cells} behavioural cells. Stagnation level is {stagnation:.2f} —
the search is **moderately stalled**, suggesting the current paradigm
family has been mined out.

### Strongest paradigms currently in the archive (full code)
{anchor_block}

### Other paradigm inspirations (description + score only)
{inspiration_block}
{strategy_log_block}
## Your Task

Design a **fundamentally different algorithmic approach** — a
paradigm class that does NOT appear in any anchor or inspiration
above. The new program's internal data structures, control flow, and
termination condition must all reflect the new paradigm.

Concrete forbidden moves:
- Re-running the same algorithm with new constants.
- Stitching a sub-routine from one anchor onto another (that is
  synthesis, not a shift).
- Renaming variables in an existing anchor.

First, write a ``## Analysis`` section. Its first line MUST be
``MOVE: SHIFT``. After that line, include exactly these four
sub-sections:

1. **Paradigm name.** The textbook name of the algorithm class you
   are proposing (e.g. "Lloyd relaxation", "Power diagram packing",
   "Lagrangian relaxation with subgradient ascent", "Branch-and-cut
   over a conflict graph").
2. **Why this paradigm fits the problem.** Two sentences. Cite the
   specific problem feature that the paradigm exploits.
3. **Why current anchors miss it.** One sentence: what assumption
   the anchors share that the new paradigm drops.
4. **Risk.** One sentence on the most likely implementation pitfall
   and how your code avoids it.

Then write the complete, runnable program. Avoid any strategy whose
Strategy-Log entry has delta ≤ 0 — that approach has been tried.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. Implement the paradigm yourself — do not assume any non-stdlib
   exotic library is available beyond numpy / scipy.

{format_instruction}
{analysis_output_order_hint}\
"""


SURGICAL_EXPLOIT_PROMPT = """\
# Surgical Exploit Challenge — Tune the Champion

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Archive Snapshot
The archive has run {n_evaluations} evaluations and currently occupies
{n_cells} behavioural cells. Stagnation level is {stagnation:.2f} —
the search is **deeply stalled**. The same family of solutions has
dominated for many evaluations, and previous paradigm attempts have
not produced improvements (see Strategy Log).

### Current champion (the ONLY anchor you target)
{anchor_block}

### Top-ranked paradigm descriptions (for context only — code withheld)
{inspiration_block}
{strategy_log_block}
## Your Task

A new paradigm will NOT help here — previous paradigm trials confirm
that. What WILL help is a **precise structural improvement** to the
champion. Be the careful surgeon, not the wild inventor.

First, write a ``## Analysis`` section. Its first line MUST be:
``MOVE: SURGICAL``.After that line, include exactly these four sub-sections:

1. **Tightest constraint.** Identify the single mechanism most likely
   limiting the champion`s score right now. Cite the exact function,
   loop, variable, update rule, repair step, acceptance rule, objective
   calculation, or magic constant. Prefer a bottleneck where the current
   algorithm wastes useful signal, accepts poor moves, loses feasibility,
   converges too early, or leaves exploitable slack.

2. **Structural fix.** Propose exactly ONE local structural change that
   directly addresses this constraint. The fix must change the algorithm`s
   behaviour in a meaningful way, such as adding score-aware acceptance,
   improving repair, using existing slack/objective information, adding a
   targeted local polish, or replacing a weak update rule. Do not merely
   rename variables, reformat code, change comments, change the random
   seed, or retune an isolated constant.

3. **Preservation list.** List the routines, constants, and control-flow
   blocks that will remain unchanged. The champion's overall algorithm
   class, I/O contract, and working components must stay intact.

4. **Expected delta.** Briefly explain why this specific fix should
   improve the score, and name the main risk. State how the implementation
   guards against that risk, e.g. fallback to the old state, feasibility
   check, bounded step size, deterministic randomness, or NaN/Inf guard.

Then write the complete program. Implement exactly ONE structural, local
fix. Do not add unrelated improvements. Do not introduce a fundamentally
different paradigm. If the champion already computes a score, feasibility,
slack, loss, or quality signal internally, prefer using that signal to
guide the fix rather than adding a blind heuristic.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. Keep the champion's overall control flow and naming intact

{format_instruction}
{analysis_output_order_hint}\
"""


def _strategy_log_block(recent_trials: Sequence[str]) -> str:
    if not recent_trials:
        return ""
    return (
        "\n## Strategy Log (recent Navigator attempts)\n"
        + "\n".join(f"- {line}" for line in recent_trials)
        + "\n"
    )


def build_synthesis_prompt(
    *,
    problem_description: str,
    function_signature: str,
    n_evaluations: int,
    n_cells: int,
    anchors: Sequence[tuple[str, str, float]],
    inspirations: Sequence[tuple[str, float]] = (),
    recent_trials: Sequence[str] = (),
    stagnation: float = 0.0,
) -> str:
    """Synthesis mode: 2-3 anchors close in score, hybridise them."""
    return SYNTHESIS_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        n_evaluations=n_evaluations,
        n_cells=n_cells,
        stagnation=stagnation,
        anchor_block=_anchor_block(anchors),
        inspiration_block=_inspiration_block(inspirations),
        strategy_log_block=_strategy_log_block(recent_trials),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )


def build_reframe_prompt(
    *,
    problem_description: str,
    function_signature: str,
    n_evaluations: int,
    n_cells: int,
    anchors: Sequence[tuple[str, str, float]],
    inspirations: Sequence[tuple[str, float]] = (),
    recent_trials: Sequence[str] = (),
    stagnation: float = 0.0,
) -> str:
    """Shift mode: 2 anchors, propose a genuinely new Navigator class."""
    return REFRAME_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        n_evaluations=n_evaluations,
        n_cells=n_cells,
        stagnation=stagnation,
        anchor_block=_anchor_block(anchors),
        inspiration_block=_inspiration_block(inspirations),
        strategy_log_block=_strategy_log_block(recent_trials),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )


def build_surgical_exploit_prompt(
    *,
    problem_description: str,
    function_signature: str,
    n_evaluations: int,
    n_cells: int,
    anchors: Sequence[tuple[str, str, float]],
    inspirations: Sequence[tuple[str, float]] = (),
    recent_trials: Sequence[str] = (),
    stagnation: float = 0.0,
) -> str:
    """Surgical mode: 1 anchor (the champion), top descriptions as
    inspiration only. Frontier writes a structural local fix."""
    return SURGICAL_EXPLOIT_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        n_evaluations=n_evaluations,
        n_cells=n_cells,
        stagnation=stagnation,
        anchor_block=_anchor_block(anchors[:1]),
        inspiration_block=_inspiration_block(inspirations),
        strategy_log_block=_strategy_log_block(recent_trials),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )
