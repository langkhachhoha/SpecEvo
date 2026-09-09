"""Tests for SAL (Stagnation-Adaptive Loop) primitives.

These tests deliberately avoid touching the LLM clients so they run without
litellm installed. They cover:
  - stagnation depth math on synthetic score history
  - Thompson Beta-Bernoulli sampler weighting + update
  - PE prompt staging (Cơ chế A) on the equilibrium.prompts helper
  - farthest-elite contrast selection (Cơ chế B) on the pool
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from specevo.behavior import BehaviorExtractor
from specevo.config import BudgetConfig, SalConfig
from specevo.core import EvaluationResult, Program
from specevo.equilibrium.prompts import get_budget_stage
from specevo.pipeline.state import PipelineState, ScoreHistoryEntry
from specevo.pool import CVTMAPElitesPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(budget: BudgetConfig | None = None) -> PipelineState:
    """Default: no budget caps → stagnation_depth() reflects only the plateau term.

    Tests that exercise the new budget-pressure branch pass an explicit
    ``BudgetConfig`` with the relevant limits set.
    """
    return PipelineState(budget if budget is not None else BudgetConfig())


def _append_score(state: PipelineState, score: float, accepted: bool = True) -> None:
    """Append a synthetic eval to score_history and keep the
    `eval_count_at_last_best` cache aligned with what the real consumer does.
    """
    eval_no = (state.score_history[-1].eval_number if state.score_history else 0) + 1
    prev_best = state.best_score_so_far
    best = max(score, prev_best if prev_best != float("-inf") else score)
    is_new_best = score > (prev_best if prev_best != float("-inf") else float("-inf"))
    if is_new_best:
        state.best_score_so_far = score
    # Mirror BudgetTracker counter so stagnation cache makes sense.
    state.budget_tracker.eval_count = eval_no
    if is_new_best:
        state.eval_count_at_last_best = eval_no
    state.score_history.append(
        ScoreHistoryEntry(
            eval_number=eval_no,
            score=score,
            best_score=best,
            timestamp=time.time(),
            accepted=accepted,
            sampler="adaptive_rank",
            archive_size=10,
            cell_index=0,
        )
    )


# ---------------------------------------------------------------------------
# Stagnation depth
# ---------------------------------------------------------------------------


class TestStagnationDepth:
    def test_empty_history_returns_zero(self):
        state = _make_state()
        assert state.stagnation_depth(tau=80) == 0.0

    def test_freshly_improving_returns_zero(self):
        state = _make_state()
        for v in [1.0, 1.5, 2.0, 2.5]:
            _append_score(state, v)
        # Every entry is a new best → n_since_best = 0.
        assert state.stagnation_depth(tau=80) == 0.0
        assert state.evals_since_best() == 0

    def test_partial_plateau_is_in_zero_one(self):
        state = _make_state()
        _append_score(state, 1.0)
        _append_score(state, 2.0)  # last NEW BEST
        for _ in range(40):
            _append_score(state, 1.5)
        s = state.stagnation_depth(tau=80)
        # 40 evals since best, tau=80 → s should be 0.5.
        assert 0.49 < s < 0.51

    def test_saturates_at_one(self):
        state = _make_state()
        _append_score(state, 2.0)
        for _ in range(500):
            _append_score(state, 1.0)
        assert state.stagnation_depth(tau=80) == 1.0

    def test_evals_since_best_counts_correctly(self):
        state = _make_state()
        _append_score(state, 1.0)
        _append_score(state, 2.0)  # NEW BEST at eval 2
        _append_score(state, 1.5)
        _append_score(state, 1.5)
        _append_score(state, 1.5)
        assert state.evals_since_best() == 3

    def test_pps_returns_zero_when_improving_despite_budget(self):
        """PPS: when plateau term is zero (every eval is a NEW BEST), the
        signal stays at zero regardless of budget consumed.

        This is a *deliberate* departure from the legacy max(plateau,
        budget_ratios) behaviour: panic-triggering PE while the search is
        producing NEW BESTs is wasteful. PPS bakes that intuition into the
        formula via the multiplicative ``p · exp(-λ̂·B_rem)`` term.
        """
        state = _make_state(BudgetConfig(evaluations=100))
        for v in [1.0, 1.5, 2.0]:
            _append_score(state, v)  # always NEW BEST → p(t) = 0
        s = state.stagnation_depth(tau=80)
        assert s < 1e-9

    def test_pps_plateau_wins_when_budget_low(self):
        """Early-run with caps barely consumed: α(t)=b² ≈ 0 → s ≈ p(t)."""
        state = _make_state(BudgetConfig(evaluations=10_000))
        _append_score(state, 2.0)  # NEW BEST
        for _ in range(40):
            _append_score(state, 1.5)
        s = state.stagnation_depth(tau=80)
        # b ≈ 41/10000 → α ≈ 0 → s ≈ p ≈ 0.5.
        assert 0.49 < s < 0.51

    def test_pps_amplifies_late_run_plateau_without_progress(self):
        """Late run, plateau active, no NEW BEST in window → posterior_stuck
        ≈ p, blended weight stays close to p so PE / Hard-PE still fire."""
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 8.0  # b = 0.8 → α = 0.64
        # Force plateau without populating new_best_history.
        state.budget_tracker.eval_count = 200
        state.eval_count_at_last_best = 100  # p = min(1, 100/80) = 1.0
        s = state.stagnation_depth(tau=80)
        # B_rem = 2, λ̂ ≈ 1/8 (Laplace), exp(-2/8) ≈ 0.78 → posterior_stuck
        # ≈ 0.78. Blend: 0.36·1.0 + 0.64·0.78 ≈ 0.86.
        assert 0.80 < s < 0.92

    def test_pps_protects_against_panic_when_frequent_new_best(self):
        """Many NEW BEST events in history → high λ̂ → posterior_stuck → 0
        even if p(t) is positive. Result: blended s stays under p."""
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 8.0  # b = 0.8
        state.budget_tracker.eval_count = 200
        state.eval_count_at_last_best = 100  # plateau p = 1.0
        # 12 NEW BEST events spread over the consumed budget → high hazard.
        for i in range(12):
            state.new_best_history.append((10 * i, 0.6 * i))
        s = state.stagnation_depth(tau=80)
        assert s < 0.5

    def test_no_budget_keeps_legacy_zero(self):
        """Without budget limits, signal stays pure plateau."""
        state = _make_state(BudgetConfig())
        for v in [1.0, 1.5, 2.0]:
            _append_score(state, v)  # always NEW BEST
        assert state.stagnation_depth(tau=80) == 0.0

    def test_end_of_run_floor(self):
        """When ≥95% of budget consumed AND plateau saturated, s must be
        at least the plateau term (PPS safety floor)."""
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 9.8  # b = 0.98
        state.budget_tracker.eval_count = 300
        state.eval_count_at_last_best = 100  # p = 1.0
        s = state.stagnation_depth(tau=80)
        assert s >= 0.95


# ---------------------------------------------------------------------------
# Cơ chế A — get_budget_stage routing
# ---------------------------------------------------------------------------


class TestPromptStaging:
    def test_legacy_path_returns_early(self):
        # When stagnation is None we keep historical behaviour.
        assert get_budget_stage(0.0) == "early"
        assert get_budget_stage(0.9) == "early"

    def test_low_stagnation_returns_early(self):
        assert get_budget_stage(0.5, stagnation=0.1) == "early"

    def test_mid_stagnation_returns_mid(self):
        assert get_budget_stage(0.5, stagnation=0.5) == "mid"

    def test_high_stagnation_returns_late(self):
        assert get_budget_stage(0.5, stagnation=0.85) == "late"

    def test_custom_thresholds_respected(self):
        assert get_budget_stage(0.0, stagnation=0.2, mid_threshold=0.1, late_threshold=0.5) == "mid"
        assert get_budget_stage(0.0, stagnation=0.6, mid_threshold=0.1, late_threshold=0.5) == "late"


# ---------------------------------------------------------------------------
# Cơ chế D — Thompson Beta-Bernoulli bandit
# ---------------------------------------------------------------------------


def _fresh_pool() -> CVTMAPElitesPool:
    """A pool with a couple of bandit arms; no actual archive needed.

    With AdaptiveRankSampler the arm space is (sampler, model[, prompt_id,
    llm_temperature]); sampler-internal temperatures are no longer
    arm dimensions. We register two model arms on the same sampler so the
    bandit has something meaningful to discriminate over.
    """
    extractor = BehaviorExtractor(ast_features=["loop_count", "branch_count"])
    pool = CVTMAPElitesPool(behavior_extractor=extractor, n_centroids=4)
    pool.register_sampler_model_pair("adaptive_rank", "model_a", weight=1.0)
    pool.register_sampler_model_pair("adaptive_rank", "model_b", weight=1.0)
    return pool


class TestThompsonBandit:
    def test_legacy_call_is_deterministic(self):
        pool = _fresh_pool()
        # No stagnation = legacy weighted-roulette path.
        np.random.seed(0)
        name, model, _, _ = pool.get_weighted_sampler_config(stagnation=None)
        assert (name, model) in {
            ("adaptive_rank", "model_a"),
            ("adaptive_rank", "model_b"),
        }

    def test_update_bandit_increments_alpha_on_accept(self):
        pool = _fresh_pool()
        pool.update_bandit("adaptive_rank", "model_a", accepted=True, is_new_best=True)
        stats = pool.get_bandit_stats()
        a = next(s for s in stats if s["model"] == "model_a")
        assert a["alpha"] == pytest.approx(2.0)
        assert a["beta"] == pytest.approx(1.0)
        assert a["new_best_count"] == 1

    def test_update_bandit_increments_beta_on_reject(self):
        pool = _fresh_pool()
        pool.update_bandit("adaptive_rank", "model_b", accepted=False)
        stats = pool.get_bandit_stats()
        b = next(s for s in stats if s["model"] == "model_b")
        assert b["alpha"] == pytest.approx(1.0)
        assert b["beta"] == pytest.approx(2.0)

    def test_high_stagnation_biases_toward_high_alpha_arm(self):
        pool = _fresh_pool()
        # Make arm A look really good: many accepts, a couple of NEW BESTs.
        for _ in range(30):
            pool.update_bandit("adaptive_rank", "model_a", accepted=True, is_new_best=False)
        pool.update_bandit("adaptive_rank", "model_a", accepted=True, is_new_best=True)
        pool.update_bandit("adaptive_rank", "model_a", accepted=True, is_new_best=True)
        # Arm B: a lot of rejects.
        for _ in range(20):
            pool.update_bandit("adaptive_rank", "model_b", accepted=False)

        np.random.seed(42)
        a_wins = 0
        n_draws = 400
        for _ in range(n_draws):
            _, model, _, _ = pool.get_weighted_sampler_config(stagnation=0.9)
            if model == "model_a":
                a_wins += 1
        # Under high stagnation we expect strong commitment to arm A.
        assert a_wins / n_draws > 0.85, f"Expected >85% commitment to arm A, got {a_wins/n_draws:.2f}"

    def test_unknown_arm_is_silently_skipped(self):
        pool = _fresh_pool()
        # Should not raise.
        pool.update_bandit("nonexistent_sampler", "no_such_model", accepted=True)
        stats = pool.get_bandit_stats()
        # All arms still have prior values.
        for s in stats:
            assert s["alpha"] == pytest.approx(1.0)
            assert s["beta"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cơ chế B — contrastive farthest-elite selection
# ---------------------------------------------------------------------------


class TestFarthestElite:
    def _populate(self, pool: CVTMAPElitesPool):
        # Add 3 fake elites with manually-set behaviors in normalized space.
        # We bypass the centroid logic by inserting elites directly.
        from specevo.pool.cvt_map_elites import Elite

        progs = [Program(content=f"def f(): return {i}", metadata={}) for i in range(3)]
        # Synthetic behaviors that we know are spread out.
        behaviors = [
            {"loop_count": 0.1, "branch_count": 0.1},
            {"loop_count": 0.5, "branch_count": 0.5},
            {"loop_count": 0.9, "branch_count": 0.9},
        ]
        scores = [1.0, 2.0, 1.5]
        for i, (p, b, sc) in enumerate(zip(progs, behaviors, scores)):
            elite = Elite(
                program=p,
                result=EvaluationResult(scores={"score": sc}, is_valid=True),
                behavior=b,  # FeatureVector is dict-like
            )
            pool._elites[i] = elite

    def test_returns_none_when_archive_small(self):
        extractor = BehaviorExtractor(ast_features=["loop_count", "branch_count"])
        pool = CVTMAPElitesPool(behavior_extractor=extractor, n_centroids=4)
        assert pool.select_diverse_elite_from(0) is None

    def test_returns_far_elite_when_anchor_low(self):
        extractor = BehaviorExtractor(ast_features=["loop_count", "branch_count"])
        pool = CVTMAPElitesPool(behavior_extractor=extractor, n_centroids=4)
        self._populate(pool)
        # Anchored at cell 0 (behavior 0.1/0.1), the farthest is cell 2.
        far = pool.select_diverse_elite_from(0)
        assert far is not None
        assert far.result.primary_score == pytest.approx(1.5)

    def test_best_elite(self):
        extractor = BehaviorExtractor(ast_features=["loop_count", "branch_count"])
        pool = CVTMAPElitesPool(behavior_extractor=extractor, n_centroids=4)
        self._populate(pool)
        best = pool.best_elite()
        assert best is not None
        assert best.result.primary_score == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# SalConfig defaults
# ---------------------------------------------------------------------------


class TestSalConfigDefaults:
    def test_enabled_by_default(self):
        c = SalConfig()
        assert c.enabled is True

    def test_thresholds_ordered(self):
        c = SalConfig()
        assert 0.0 < c.pe_staging_mid_threshold < c.pe_staging_late_threshold < 1.0

    def test_w_min_is_a_floor(self):
        c = SalConfig()
        assert 0.0 < c.bandit_w_min < 0.5


# ---------------------------------------------------------------------------
# Cơ chế C — dual-mode meta-advice helper
# ---------------------------------------------------------------------------


class TestMetaAdviceOffensiveExtras:
    def test_gather_extras_when_stagnant(self):
        from specevo.pipeline.consumer import _format_metrics_for_llm, _gather_sal_meta_extras

        state = _make_state()
        _append_score(state, 2.0)  # NEW BEST
        for _ in range(30):
            _append_score(state, 1.5, accepted=False)

        extras = _gather_sal_meta_extras(state)
        assert extras["best_score"] == pytest.approx(2.0)
        assert extras["evals_since_best"] == 30
        # Most-recent score history shows accepts under the same sampler.
        assert isinstance(extras["per_sampler_accepts"], dict)

    def test_format_metrics_appends_offensive_block(self):
        from specevo.pipeline.consumer import _format_metrics_for_llm

        block = _format_metrics_for_llm(
            metrics={"acceptances": 2, "rejections": 5, "errors": 1, "top_errors": []},
            previous_advice="",
            progress_pct=50.0,
            sal_extras={
                "best_score": 2.6,
                "evals_since_best": 80,
                "stagnation": 0.95,
                "per_sampler_accepts": {"adaptive_rank": 5},
            },
        )
        assert "Search Trajectory" in block
        assert "Best score" in block or "best score" in block
        assert "0.95" in block

    def test_format_metrics_omits_offensive_block_by_default(self):
        from specevo.pipeline.consumer import _format_metrics_for_llm

        block = _format_metrics_for_llm(
            metrics={"acceptances": 0, "rejections": 0, "errors": 0, "top_errors": []},
            previous_advice="",
            progress_pct=10.0,
            sal_extras=None,
        )
        assert "Search Trajectory" not in block


# ---------------------------------------------------------------------------
# Cơ chế E — Hard-PE gate
# ---------------------------------------------------------------------------


class TestHardPEGate:
    """Tests for PunctuatedEquilibrium._should_fire_hard_pe.

    We avoid constructing a full PE (it needs an artifact adapter and an
    executor) and just call the gate predicate via duck-typing.
    """

    def _gate(
        self,
        *,
        sal_enabled=True,
        sal_e=True,
        hard_pe_count=0,
        consecutive_pe_no_best=0,
        s_value=0.0,
        threshold=0.8,
        max_per_run=2,
    ):
        from specevo.equilibrium.equilibrium import PunctuatedEquilibrium

        class _StubConfig:
            class _Sal:
                enabled = sal_enabled
                enable_e_hard_pe = sal_e
                tau = 80
                hard_pe_threshold = threshold
                hard_pe_max_per_run = max_per_run

            sal = _Sal()

        class _StubState:
            def __init__(self):
                self.hard_pe_count = hard_pe_count
                self.consecutive_pe_no_best = consecutive_pe_no_best

            def stagnation_depth(self, tau):
                return s_value

        pe = PunctuatedEquilibrium.__new__(PunctuatedEquilibrium)
        pe.config = _StubConfig()
        pe.state = _StubState()
        return pe._should_fire_hard_pe()

    def test_disabled_when_sal_off(self):
        assert self._gate(sal_enabled=False) is False

    def test_disabled_when_mechanism_off(self):
        assert self._gate(sal_e=False, consecutive_pe_no_best=3, s_value=0.95) is False

    def test_blocked_by_max_per_run(self):
        assert (
            self._gate(hard_pe_count=2, consecutive_pe_no_best=3, s_value=0.95, max_per_run=2)
            is False
        )

    def test_blocked_when_pe_just_helped(self):
        # Need ≥2 consecutive PE-no-best before firing.
        assert self._gate(consecutive_pe_no_best=1, s_value=0.95) is False

    def test_blocked_when_stagnation_below_threshold(self):
        assert (
            self._gate(consecutive_pe_no_best=3, s_value=0.5, threshold=0.8)
            is False
        )

    def test_fires_when_all_conditions_met(self):
        assert self._gate(consecutive_pe_no_best=2, s_value=0.9) is True

    def test_fires_at_boundary(self):
        # s exactly at threshold should fire (>=).
        assert self._gate(consecutive_pe_no_best=2, s_value=0.8, threshold=0.8) is True
