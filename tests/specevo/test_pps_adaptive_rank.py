"""Tests for the two contributions kept from the early refactor:

1. PPS — Posterior-Plateau Stagnation
2. AdaptiveRankSampler — parameter-free Zipfian rank sampling

The HLS / Strategic-Blueprint contribution was retired in favour of a
phase-staged paradigm-shift prompt plus Strategy History; see
``test_strategy_log.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from specevo.config.models import BudgetConfig
from specevo.core import EvaluationResult, Program
from specevo.pipeline.state import PipelineState
from specevo.pool.cvt_map_elites import AdaptiveRankSampler, Elite


def _make_state(budget: BudgetConfig | None = None) -> PipelineState:
    return PipelineState(budget or BudgetConfig(evaluations=200))


def _mock_elites(scores: list[float]) -> dict[int, Elite]:
    elites = {}
    for i, s in enumerate(scores):
        elites[i] = Elite(
            program=Program(content=f"# {i}"),
            result=EvaluationResult(scores={"score": s}, is_valid=True),
            behavior={},  # type: ignore[arg-type]
        )
    return elites


class TestPPS:
    def test_alpha_is_zero_at_t_zero_so_signal_is_plateau(self):
        state = _make_state(BudgetConfig(evaluations=100))
        state.budget_tracker.eval_count = 40
        state.eval_count_at_last_best = 0  # plateau p=40/80 = 0.5
        # b ≈ 40/100 = 0.4, α = 0.16 — still mostly plateau-driven.
        s = state.stagnation_depth(tau=80)
        assert 0.40 < s < 0.55

    def test_record_new_best_resets_plateau_and_seeds_hazard(self):
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 5.0
        state.budget_tracker.eval_count = 100
        state.eval_count_at_last_best = 50  # plateau in progress

        state.record_new_best()

        assert state.eval_count_at_last_best == 100
        assert len(state.new_best_history) == 1
        eval_count, cost = state.new_best_history[-1]
        assert eval_count == 100
        assert cost == pytest.approx(5.0)

    def test_hazard_estimate_uses_window_since_first_history_entry(self):
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 9.0  # b = 0.9
        state.budget_tracker.eval_count = 100
        state.eval_count_at_last_best = 20  # plateau p = 1
        # 5 NEW BEST events, the first at cost=1.0, last at cost=5.0.
        for i in range(5):
            state.new_best_history.append((10 * (i + 1), 1.0 + i))
        s = state.stagnation_depth(tau=80)
        # B_W = 9 - 1 = 8, λ̂ = 6/8 = 0.75; B_rem = 1; exp(-0.75) ≈ 0.47.
        # posterior = 1.0 * 0.47 = 0.47.  α = 0.81 → s = 0.19·1 + 0.81·0.47 ≈ 0.57.
        assert 0.45 < s < 0.65

    def test_pps_clamped_to_0_1(self):
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 100.0  # overshoot
        state.budget_tracker.eval_count = 1000
        state.eval_count_at_last_best = 0
        s = state.stagnation_depth(tau=10)
        assert 0.0 <= s <= 1.0

    def test_no_budget_falls_back_to_pure_plateau(self):
        state = _make_state(BudgetConfig(target_score=10.0))
        state.budget_tracker.eval_count = 100
        state.eval_count_at_last_best = 60  # p = 40/80 = 0.5
        s = state.stagnation_depth(tau=80)
        assert 0.49 < s < 0.51

    def test_freshly_improving_returns_zero_regardless_of_budget(self):
        """Behavioural guarantee: if every recent eval is a NEW BEST, PPS
        cannot panic-trigger no matter how late in the run we are."""
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 9.5  # b ≈ 1.0
        state.budget_tracker.eval_count = 200
        state.eval_count_at_last_best = 200  # plateau = 0
        for i in range(20):
            state.new_best_history.append((i * 10, 0.4 * i))
        s = state.stagnation_depth(tau=80)
        assert s == 0.0


class TestAdaptiveRankSampler:
    def test_low_stagnation_prefers_top_ranked(self):
        np.random.seed(0)
        sampler = AdaptiveRankSampler()
        elites = _mock_elites([10 - i for i in range(10)])

        counts = np.zeros(10)
        for _ in range(3000):
            cells = sampler.select_cells(elites, n=1, context={"stagnation": 0.0})
            counts[cells[0]] += 1
        # Top-ranked cell should dominate (Zipfian with β=2).
        assert counts[0] > counts[3] > counts[9]

    def test_high_stagnation_approaches_uniform(self):
        np.random.seed(1)
        sampler = AdaptiveRankSampler()
        elites = _mock_elites([10 - i for i in range(10)])

        counts = np.zeros(10)
        for _ in range(3000):
            cells = sampler.select_cells(elites, n=1, context={"stagnation": 1.0})
            counts[cells[0]] += 1
        # With β=0.2 the worst rank should still get a meaningful share.
        assert counts[9] / counts[0] > 0.4

    def test_no_replacement(self):
        np.random.seed(2)
        sampler = AdaptiveRankSampler()
        elites = _mock_elites([10 - i for i in range(10)])
        chosen = sampler.select_cells(elites, n=5, context={"stagnation": 0.5})
        assert len(set(chosen)) == 5

    def test_default_context_does_not_crash(self):
        sampler = AdaptiveRankSampler()
        elites = _mock_elites([1.0, 2.0, 3.0])
        cells = sampler.select_cells(elites, n=2, context=None)
        assert len(cells) == 2

    def test_beta_floor(self):
        sampler = AdaptiveRankSampler(beta_max=2.0, beta_min=0.5)
        assert sampler._compute_beta(stagnation=1.0) == 0.5
        assert sampler._compute_beta(stagnation=0.0) == 2.0
