"""Tests for the Code Error Repair pipeline.

The light-model repair call itself is not exercised here (it requires the
network); we test the pure-Python pieces: error-buffer push/dedupe,
rank-by-parent-score selection, ``fire_repair_if_due`` gating, and the
adapter prompt builder.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from specevo.config.models import BudgetConfig, CodeRepairConfig
from specevo.pipeline.state import ErrorRecord, PipelineState


def _make_state() -> PipelineState:
    return PipelineState(BudgetConfig(evaluations=100))


def _push(state, code: str, parent_score: float, msg: str = "TypeError") -> None:
    state.push_error_record(
        ErrorRecord(
            code=code,
            parent_score=parent_score,
            error_msg=msg,
            parent_cell=0,
            source="main_loop",
        )
    )


class TestErrorBuffer:
    def test_push_and_dedupe(self):
        state = _make_state()
        _push(state, "def f(): return 1", 0.5)
        _push(state, "def f(): return 1", 0.6)  # duplicate by code prefix
        _push(state, "def g(): return 2", 0.3)
        assert len(state.error_buffer) == 2

    def test_buffer_respects_maxlen(self):
        state = _make_state()
        # Default buffer_size in PipelineState is 64; push 70 unique codes.
        for i in range(70):
            _push(state, f"# {i}\ndef f(): return {i}", float(i))
        assert len(state.error_buffer) == state.error_buffer.maxlen
        assert state.error_buffer.maxlen == 64


class TestRankPick:
    def test_high_parent_score_is_preferred(self):
        np.random.seed(0)
        state = _make_state()
        # Three records with widely-separated parent scores.
        _push(state, "code-A", 9.0)
        _push(state, "code-B", 4.0)
        _push(state, "code-C", 1.0)

        # Re-push the same records on every iteration since sample_error_for_repair
        # removes the chosen one (one-shot semantics).
        counts: Counter = Counter()
        for _ in range(800):
            state = _make_state()
            _push(state, "code-A", 9.0)
            _push(state, "code-B", 4.0)
            _push(state, "code-C", 1.0)
            rec = state.sample_error_for_repair(beta=1.5)
            assert rec is not None
            counts[rec.code] += 1
        assert counts["code-A"] > counts["code-B"] > counts["code-C"]

    def test_pop_removes_record(self):
        state = _make_state()
        _push(state, "x", 1.0)
        _push(state, "y", 2.0)
        rec = state.sample_error_for_repair()
        assert rec is not None
        assert len(state.error_buffer) == 1
        remaining = state.error_buffer[0].code
        assert remaining != rec.code

    def test_empty_buffer_returns_none(self):
        state = _make_state()
        assert state.sample_error_for_repair() is None


class TestFireRepairGate:
    def test_disabled_returns_none(self):
        state = _make_state()
        _push(state, "code", 1.0)
        cfg = CodeRepairConfig(enabled=False)
        assert state.fire_repair_if_due(cfg) is None

    def test_respects_max_per_run(self):
        state = _make_state()
        _push(state, "code", 1.0)
        cfg = CodeRepairConfig(enabled=True, repair_every_n=1, max_per_run=1)
        # First fire is fine; the second is gated.
        assert state.fire_repair_if_due(cfg) is not None
        _push(state, "code-2", 1.0)
        # eval_count didn't advance, so the cooldown gate ALSO fires here.
        state.budget_tracker.eval_count = 50
        assert state.fire_repair_if_due(cfg) is None

    def test_respects_repair_every_n_cooldown(self):
        state = _make_state()
        cfg = CodeRepairConfig(enabled=True, repair_every_n=10, max_per_run=100)
        # First fire — last_repair_at_eval was -inf, so this passes.
        _push(state, "first", 5.0)
        state.budget_tracker.eval_count = 0
        assert state.fire_repair_if_due(cfg) is not None
        # Push more, but only 5 evals have passed — cooldown should block.
        _push(state, "second", 3.0)
        state.budget_tracker.eval_count = 5
        assert state.fire_repair_if_due(cfg) is None
        # Wait until eval_count - last_repair_at_eval >= 10.
        state.budget_tracker.eval_count = 10
        assert state.fire_repair_if_due(cfg) is not None

    def test_increments_counter(self):
        state = _make_state()
        _push(state, "x", 1.0)
        cfg = CodeRepairConfig(enabled=True, repair_every_n=1)
        state.fire_repair_if_due(cfg)
        assert state.repair_attempt_count == 1


class TestRepairPromptBuilder:
    def test_repair_prompt_includes_error_and_code(self):
        from specevo.artifacts.code import CodeAdapter

        class _StubConfig:
            problem_description = "Sum a list."
            function_signature = "def sumlist(xs): ..."
            prompt_overrides: dict = {}
            init = type("I", (), {"diversity_prompt": None})()

            class _Pipeline:
                eval_timeout = 30.0

            pipeline = _Pipeline()

        adapter = CodeAdapter.__new__(CodeAdapter)
        adapter.config = _StubConfig()
        adapter.fn_name = "sumlist"

        prompt = adapter.build_code_repair_prompt(
            "def sumlist(xs): return suml(xs)",
            error_msg="NameError: name 'suml' is not defined",
            parent_score=0.5,  # exact-representable float so format() matches verbatim
        )
        assert "Sum a list." in prompt
        assert "NameError" in prompt
        assert "def sumlist(xs): return suml(xs)" in prompt
        assert "0.5" in prompt
