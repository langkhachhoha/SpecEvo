"""Tests for the unified **Adaptive Island Expansion** mechanism.

Replaces the previous two mechanisms (`StagnationRescue` and
`AdaptiveCvtConfig`). The contract is:

* When a Punctuated Equilibrium candidate fails the standard
  ``pool.add()`` admission test AND stagnation s(t) is high enough,
  open a brand-new cell at the candidate's own (normalised) behaviour
  vector instead of discarding the candidate or evicting the incumbent.
* Bounded by ``max_per_run`` (per-run cap on expansions) and
  ``max_total_centroids`` (hard ceiling on archive size).
"""

from __future__ import annotations

import numpy as np
import pytest

from specevo.behavior.extractor import BehaviorExtractor
from specevo.config.models import (
    AdaptiveIslandConfig,
    BudgetConfig,
    SalConfig,
)
from specevo.core import EvaluationResult, Program
from specevo.equilibrium.equilibrium import PunctuatedEquilibrium
from specevo.pipeline.state import PipelineState
from specevo.pool import CVTMAPElitesPool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pool(n_centroids: int = 4) -> CVTMAPElitesPool:
    """Pool with a fixed 4-centroid square in 2-D normalised space."""
    extractor = BehaviorExtractor(ast_features=["loop_count", "branch_count"])
    pool = CVTMAPElitesPool(
        behavior_extractor=extractor,
        n_centroids=n_centroids,
        data_driven_centroids=False,
    )
    pool._centroids = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    pool._n_centroids = 4
    pool._mins = np.zeros(2)
    pool._maxs = np.ones(2)
    pool._ranges = np.ones(2)
    return pool


def _make_pe(cfg: AdaptiveIslandConfig) -> PunctuatedEquilibrium:
    """Build a minimum-viable PE without going through evolve_code."""

    class _Cfg:
        adaptive_island = cfg
        sal = SalConfig(enabled=True)

    pe = PunctuatedEquilibrium.__new__(PunctuatedEquilibrium)
    pe.config = _Cfg()
    pe.pool = _make_pool()
    pe.state = PipelineState(BudgetConfig(evaluations=200))
    pe._is_bundle = False
    return pe


def _candidate(score: float) -> tuple[Program, EvaluationResult]:
    program = Program(content=f"def f(): return {score}", metadata={})
    result = EvaluationResult(scores={"score": score}, is_valid=True)
    return program, result


# ---------------------------------------------------------------------------
# pool.add_as_new_cell — the primitive that island expansion sits on
# ---------------------------------------------------------------------------


class TestAddAsNewCell:
    def test_appends_new_centroid_and_elite(self):
        pool = _make_pool()
        prog, res = _candidate(0.42)
        behavior = pool._extractor.extract(prog, res.scores)
        new_idx = pool.add_as_new_cell(prog, res, behavior)
        assert new_idx is not None
        assert new_idx == 4  # appended after the 4 existing centroids
        assert pool._n_centroids == 5
        elite = pool.get_elite(new_idx)
        assert elite is not None
        assert elite.result.primary_score == pytest.approx(0.42)
        # The new centroid sits at the candidate's normalised behaviour vector.
        expected = pool._behavior_to_normalized_vector(behavior)
        np.testing.assert_allclose(pool._centroids[new_idx], expected)

    def test_invalid_result_returns_none(self):
        pool = _make_pool()
        prog = Program(content="x = 1")
        bad_result = EvaluationResult(scores={"score": 0.0}, is_valid=False)
        behavior = pool._extractor.extract(prog, bad_result.scores)
        assert pool.add_as_new_cell(prog, bad_result, behavior) is None
        assert pool._n_centroids == 4

    def test_updates_best_score(self):
        pool = _make_pool()
        pool._best_score = 0.5
        prog, res = _candidate(0.9)
        behavior = pool._extractor.extract(prog, res.scores)
        pool.add_as_new_cell(prog, res, behavior)
        assert pool._best_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# PE._try_island_expansion — gating, counter, log
# ---------------------------------------------------------------------------


class TestIslandExpansionGate:
    def test_disabled_never_expands(self):
        pe = _make_pe(AdaptiveIslandConfig(enabled=False))
        prog, res = _candidate(0.5)
        expanded, _ = pe._try_island_expansion(prog, res, stagnation=1.0)
        assert not expanded
        assert pe.pool._n_centroids == 4

    def test_below_threshold_no_expansion(self):
        pe = _make_pe(
            AdaptiveIslandConfig(enabled=True, stagnation_threshold=0.7, max_per_run=10)
        )
        prog, res = _candidate(0.5)
        expanded, _ = pe._try_island_expansion(prog, res, stagnation=0.5)
        assert not expanded
        assert pe.pool._n_centroids == 4

    def test_above_threshold_opens_island(self):
        pe = _make_pe(
            AdaptiveIslandConfig(enabled=True, stagnation_threshold=0.7, max_per_run=10)
        )
        prog, res = _candidate(0.5)
        expanded, cell_idx = pe._try_island_expansion(prog, res, stagnation=0.8)
        assert expanded
        assert cell_idx == 4
        assert pe.pool._n_centroids == 5
        assert pe.state.island_expansion_count == 1

    def test_respects_max_per_run(self):
        cfg = AdaptiveIslandConfig(enabled=True, stagnation_threshold=0.0, max_per_run=2)
        pe = _make_pe(cfg)
        admitted = 0
        for i in range(5):
            prog, res = _candidate(float(i) * 0.1)
            expanded, _ = pe._try_island_expansion(prog, res, stagnation=1.0)
            if expanded:
                admitted += 1
        assert admitted == 2
        assert pe.state.island_expansion_count == 2

    def test_respects_max_total_centroids(self):
        cfg = AdaptiveIslandConfig(
            enabled=True, stagnation_threshold=0.0, max_per_run=10, max_total_centroids=6
        )
        pe = _make_pe(cfg)
        admitted = 0
        for i in range(5):
            prog, res = _candidate(float(i) * 0.1)
            expanded, _ = pe._try_island_expansion(prog, res, stagnation=1.0)
            if expanded:
                admitted += 1
        # Start at 4 centroids, hard cap 6 → only 2 expansions allowed.
        assert admitted == 2
        assert pe.pool._n_centroids == 6

    def test_bundle_mode_is_skipped(self):
        cfg = AdaptiveIslandConfig(enabled=True, stagnation_threshold=0.0, max_per_run=10)
        pe = _make_pe(cfg)
        pe._is_bundle = True
        prog, res = _candidate(0.5)
        expanded, _ = pe._try_island_expansion(prog, res, stagnation=1.0)
        assert not expanded


# ---------------------------------------------------------------------------
# Integration sanity — under high stagnation, distinct candidates open
# distinct islands at distinct positions.
# ---------------------------------------------------------------------------


class TestIslandIntegrationSanity:
    def test_multiple_candidates_open_distinct_cells(self):
        cfg = AdaptiveIslandConfig(
            enabled=True, stagnation_threshold=0.0, max_per_run=10, max_total_centroids=100
        )
        pe = _make_pe(cfg)
        progs = [
            Program(content="def f():\n    x = 1\n    return x"),
            Program(content="def g():\n    for i in range(10):\n        pass\n    return i"),
            Program(content="def h():\n    if 1:\n        return 0\n    return 1"),
        ]
        results = [
            EvaluationResult(scores={"score": float(i)}, is_valid=True) for i in range(3)
        ]
        cells = []
        for p, r in zip(progs, results):
            expanded, idx = pe._try_island_expansion(p, r, stagnation=1.0)
            assert expanded
            cells.append(idx)
        # Three sequential expansions must produce three distinct cell indices
        # (the index is the new tail of the centroid table after each append).
        assert len(set(cells)) == 3
        assert pe.state.island_expansion_count == 3
