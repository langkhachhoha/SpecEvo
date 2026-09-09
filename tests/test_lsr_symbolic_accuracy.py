"""Unit guards for the LSR-Synth symbolic-accuracy judge (scripts/lsr_symbolic_accuracy.py).

Everything here is offline: the pre-processing that decides what the judge sees,
the verdict parser that decides what comes back, and the aggregation that turns
verdicts into the paper's per-domain percentage. The API call itself is not
exercised — ``--dry-run`` on a real run directory covers that path by hand.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import lsr_symbolic_accuracy as sa  # noqa: E402


# --------------------------------------------------------------------------- #
# Ground-truth normalisation
# --------------------------------------------------------------------------- #
def test_fitted_constants_become_placeholders():
    gt = "-0.18997742423620262*A(t)**2 + 0.7497988950401423*A(t)"
    assert sa.normalize_ground_truth(gt) == "-c0*A(t)**2 + c1*A(t)"


def test_dataset_parameter_suffix_is_one_constant():
    """The upstream dump writes k_z as '0.189..._z'; that is a single parameter."""
    gt = "0.18997742423620262_z*A(t)**2"
    assert sa.normalize_ground_truth(gt) == "c0*A(t)**2"


def test_exponents_survive():
    """A(t)**2 and A(t)**c are different hypotheses — powers are structure."""
    gt = "-1.5*A(t)**2 + 2.0*P(t)**0.333333333333333"
    assert sa.normalize_ground_truth(gt) == "-c0*A(t)**2 + c1*P(t)**0.333333333333333"


def test_digits_inside_identifiers_survive():
    """phys_osc ships symbolic parameters (F0, omega0) — they must not be mangled."""
    gt = "F0*sin(t) - omega0**2*x(t)*exp(-Abs(x(t)))"
    assert sa.normalize_ground_truth(gt) == gt


def test_shared_mode_reuses_one_name_per_value():
    gt = "-0.77*A(t)**2 - 0.77*A(t) + 0.5*cos(t)"
    assert sa.normalize_ground_truth(gt, "shared") == "-c0*A(t)**2 - c0*A(t) + c1*cos(t)"


def test_raw_mode_is_a_passthrough():
    gt = "-0.77*A(t)**2"
    assert sa.normalize_ground_truth(gt, "raw") == gt


# --------------------------------------------------------------------------- #
# Program pre-processing
# --------------------------------------------------------------------------- #
def test_comments_and_docstrings_are_stripped_but_maths_is_kept():
    program = '''
import numpy as np

def equation(t, A, params):
    """Mathematical function for the rate of change of concentration."""
    # First-order decay
    term1 = -params[0] * A
    return term1 + params[1] * A ** 2
'''
    out = sa.strip_program(program)
    assert "First-order decay" not in out
    assert "Mathematical function" not in out
    assert "-params[0] * A" in out
    assert "params[1] * A ** 2" in out


def test_unparseable_program_falls_back_to_a_line_filter():
    out = sa.strip_program("# a comment\ndef equation(  :\n    return 1\n")
    assert "a comment" not in out
    assert "return 1" in out


def test_hypothesis_falls_back_to_an_expression_field():
    rec = {"best_program": "", "best_expression": "c0*x + c1"}
    assert sa.hypothesis_of(rec) == ("c0*x + c1", "expression")


def test_missing_hypothesis_is_reported_as_none():
    hypothesis, _ = sa.hypothesis_of({"best_program": "", "best_program_path": "/nope/none.py"})
    assert hypothesis is None


# --------------------------------------------------------------------------- #
# Verdict parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"reasoning": "same terms", "answer": "Yes"}', True),
        ('{"reasoning": "extra terms", "answer": "No"}', False),
        ('```json\n{"reasoning": "r", "answer": "yes"}\n```', True),
        ('Let me think.\n{"reasoning": "r", "answer": "No"}', False),
        ("Reasoning: the forms differ.\nAnswer: No", False),
        ("nothing useful here", None),
    ],
)
def test_parse_verdict(text, expected):
    assert sa.parse_verdict(text)[0] is expected


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_symbolic_accuracy_counts_failures_in_the_denominator():
    judged = [
        {"method": "specevo", "domain": "chem_react", "problem": "crk0",
         "answer": True, "hypothesis": "x"},
        {"method": "specevo", "domain": "chem_react", "problem": "crk1",
         "answer": False, "hypothesis": "x"},
        # No equation at all: a failure, not a problem to drop.
        {"method": "specevo", "domain": "chem_react", "problem": "crk2",
         "answer": None, "hypothesis": None},
        {"method": "specevo", "domain": "chem_react", "problem": "crk3",
         "answer": False, "hypothesis": "x"},
    ]
    (row,) = sa.aggregate(judged, {"chem_react": 36})
    assert row["n_problems"] == 4
    assert row["n_correct"] == 1
    assert row["n_no_equation"] == 1
    assert row["symbolic_accuracy_pct"] == 25.0
    assert row["n_available"] == 36


def test_totals_pool_problems_and_average_domains():
    rows = [
        {"method": "m", "domain": "chem_react", "n_problems": 10, "n_correct": 1,
         "symbolic_accuracy_pct": 10.0},
        {"method": "m", "domain": "matsci", "n_problems": 30, "n_correct": 9,
         "symbolic_accuracy_pct": 30.0},
    ]
    (total,) = sa.overall_rows(rows)
    assert total["n_problems"] == 40
    assert total["n_correct"] == 10
    assert total["symbolic_accuracy_pct"] == 25.0
    assert total["domain_mean_pct"] == 20.0
