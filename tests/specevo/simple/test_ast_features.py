"""Tests for the 14-d AST count features used by ClusterArchive."""

from __future__ import annotations

from specevo.simple.ast_features import N_AST_FEATURES, compute_ast_features


def test_shape_and_dtype() -> None:
    sig = compute_ast_features("x = 1\n")
    assert sig.shape == (N_AST_FEATURES,)
    assert sig.dtype.name == "float32"


def test_parse_failure_returns_zero_vector() -> None:
    sig = compute_ast_features("def !@#$ broken")
    assert sig.shape == (N_AST_FEATURES,)
    assert (sig == 0).all()


def test_empty_string_returns_zero_vector() -> None:
    sig = compute_ast_features("")
    assert (sig == 0).all()


def test_loop_count_distinguishes_loop_vs_comprehension() -> None:
    loop_code = (
        "def f(x):\n"
        "    total = 0\n"
        "    for i in range(len(x)):\n"
        "        total += x[i]\n"
        "    return total\n"
    )
    comp_code = "def f(x):\n    return sum(v for v in x)\n"
    loop_sig = compute_ast_features(loop_code)
    comp_sig = compute_ast_features(comp_code)
    # loop has 1 For node, comp has 0
    # index 2 is loop_count in the feature layout
    assert loop_sig[2] > comp_sig[2]
    # index 6 is comprehension_count; reversed
    assert comp_sig[6] > loop_sig[6]
