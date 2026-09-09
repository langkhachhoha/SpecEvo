"""Tests for the SpecEvo meta-advisor ("Advisor") prompt + error signatures.

The Advisor assigns credit by **behavioural niche (archive cell)**, never by
which prompt-template produced a program. On each trigger it receives:

* LEADING niches — the top-score cell incumbents (description + score).
* IMPROVING niches — cells whose frontier advanced in the current window.
* SATURATED niches — cells that keep attracting attempts but whose frontier
  has not moved, with their recent-attempt count.
* UNDER-EXPLORED niches — occupied cells with few recent attempts.
* An *accumulated* failure-knowledge base: every error seen so far, grouped
  by a domain-agnostic :func:`error_signature` and counted by recurrence.

These tests pin (a) that all signal streams flow through the prompt verbatim,
(b) that the four-section ``WORKING / SATURATED / TRY NEXT / AVOID`` contract
is announced in order, (c) that ``error_signature`` collapses the same failure
with different numbers into one key, (d) that ``errors_only`` ablation parity
is achievable by passing empty success-side sequences, and (e) that long
descriptions / error examples are truncated.
"""

from __future__ import annotations

from specevo.engine.prompts import (
    build_advisor_prompt,
    error_signature,
)


PROBLEM = "Maximise some objective f."
SIGNATURE = "def solve(x): ..."


# ---------------------------------------------------------------------------
# Error signature (domain-agnostic recurrence key)
# ---------------------------------------------------------------------------


def test_error_signature_collapses_same_failure_with_different_numbers() -> None:
    """The same failure mode with different indices/sizes must map to one
    signature so it aggregates into a single counted entry."""
    a = error_signature("Overlap between circles 0 and 2")
    b = error_signature("Overlap between circles 3 and 5")
    assert a == b
    # And a structurally different failure must NOT collide with it.
    c = error_signature("operands could not be broadcast together with shapes (21,) (21,2)")
    assert c != a


def test_error_signature_handles_empty_and_numeric_only() -> None:
    assert error_signature("") == "unknown error"
    # A message that is only digits/punctuation reduces to the empty key.
    assert error_signature("12345 -- 67.8") == "unknown error"


def test_error_signature_is_bounded() -> None:
    """Signatures keep only the leading words so they stay compact keys."""
    sig = error_signature("alpha beta gamma delta epsilon zeta eta theta iota kappa")
    assert len(sig.split()) <= 8


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_prompt_announces_four_section_contract() -> None:
    """The model must be told to emit WORKING / SATURATED / TRY NEXT / AVOID
    in that order — this is the structural contract the operators consume
    verbatim."""
    p = build_advisor_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=1.23,
        n_evaluations=100,
        accept_rate=0.4,
        stagnation_level=0.2,
        leaders=[("hex lattice", 1.23)],
        improving=[("hex lattice", 1.23)],
        saturated=[],
        under_explored=[],
        error_knowledge=[("overlap between circles and", 2, "Overlap between circles 0 and 2")],
        previous_advice=None,
    )
    assert "WORKING:" in p
    assert "SATURATED:" in p
    assert "TRY NEXT:" in p
    assert "AVOID:" in p
    assert (
        p.index("WORKING:")
        < p.index("SATURATED:")
        < p.index("TRY NEXT:")
        < p.index("AVOID:")
    )
    # Credit is by niche, not by template: the prompt must not solicit
    # operator names or a score-delta threshold.
    assert "Cite the operator" not in p
    assert "mutate_focused_fix" not in p


def test_prompt_includes_all_region_signal_streams() -> None:
    p = build_advisor_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=2.5,
        n_evaluations=200,
        accept_rate=0.3,
        stagnation_level=0.6,
        leaders=[
            ("hexagonal lattice with Adam relaxation", 2.50),
            ("simulated annealing with aspect-ratio sweeps", 2.45),
        ],
        improving=[("greedy insertion then local polish", 2.50)],
        saturated=[("pure gradient descent on coordinates", 2.40, 7)],
        under_explored=[("power-diagram packing", 2.10)],
        error_knowledge=[
            ("overlap between circles and", 20, "Overlap between circles 1 and 2"),
            ("operands could not be broadcast", 3, "operands could not be broadcast together with shapes (21,) (21,2)"),
        ],
        previous_advice="prior wisdom about relaxation",
    )
    # Leaders flow through with score formatting.
    assert "hexagonal lattice" in p
    assert "2.5000" in p
    # Improving niche content flows through.
    assert "greedy insertion then local polish" in p
    # Saturated niche flows through WITH its attempt count.
    assert "pure gradient descent on coordinates" in p
    assert "7 recent attempts" in p
    # Under-explored niche flows through.
    assert "power-diagram packing" in p
    # Accumulated error knowledge: example + recurrence count, most-recurrent
    # first.
    assert "×20" in p
    assert "×3" in p
    assert p.index("×20") < p.index("×3")
    # Previous advice carried over.
    assert "prior wisdom" in p
    # Section ordering: leaders < improving < saturated < under-explored <
    # error knowledge.
    assert (
        p.index("## Leaders")
        < p.index("## IMPROVING niches")
        < p.index("## SATURATED niches")
        < p.index("## Under-explored niches")
        < p.index("## Accumulated failure knowledge")
    )


def test_prompt_handles_empty_signals_gracefully() -> None:
    """First trigger of a fresh run: archive empty, no improving/saturated/
    under-explored niches, no errors. Prompt must render with placeholders."""
    p = build_advisor_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=float("-inf"),
        n_evaluations=0,
        accept_rate=0.0,
        stagnation_level=0.0,
        leaders=[],
        improving=[],
        saturated=[],
        under_explored=[],
        error_knowledge=[],
        previous_advice=None,
    )
    assert "n/a" in p  # best score
    assert "(archive empty)" in p  # leaders
    # improving + under_explored both render the same "(none in this window)".
    assert p.count("(none in this window)") >= 2
    assert "(none — no niche is both stuck and busy)" in p  # saturated
    assert "(no failures recorded yet)" in p  # error knowledge
    assert "(none yet" in p  # previous advice


def test_prompt_errors_only_mode_parity() -> None:
    """The ``advisor_mode='errors_only'`` ablation is implemented at the
    orchestrator layer by passing empty success-side (region) sequences to
    this builder. Verify the prompt still renders the empty-state placeholders
    while the accumulated failure knowledge still flows through."""
    p = build_advisor_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=2.5,
        n_evaluations=200,
        accept_rate=0.3,
        stagnation_level=0.6,
        leaders=[],
        improving=[],
        saturated=[],
        under_explored=[],
        error_knowledge=[("overlap between circles and", 4, "Overlap between circles 1 and 2")],
        previous_advice=None,
    )
    assert "(archive empty)" in p
    assert p.count("(none in this window)") >= 2
    assert "(none — no niche is both stuck and busy)" in p
    # Errors still flow.
    assert "×4" in p
    assert "Overlap between circles" in p


def test_prompt_truncates_long_descriptions() -> None:
    long_desc = "x" * 1000
    p = build_advisor_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=1.0,
        n_evaluations=10,
        accept_rate=0.1,
        stagnation_level=0.0,
        leaders=[(long_desc, 1.0)],
        improving=[],
        saturated=[],
        under_explored=[],
        error_knowledge=[],
        previous_advice=None,
    )
    assert "…" in p
    assert "x" * 500 not in p


def test_prompt_truncates_long_error_examples() -> None:
    long_msg = "Overlap " + "x" * 1000
    p = build_advisor_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=1.0,
        n_evaluations=10,
        accept_rate=0.1,
        stagnation_level=0.0,
        leaders=[],
        improving=[],
        saturated=[],
        under_explored=[],
        error_knowledge=[("overlap x", 1, long_msg)],
        previous_advice=None,
    )
    assert "…" in p
    assert "x" * 500 not in p
