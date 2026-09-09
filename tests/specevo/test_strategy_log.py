"""Tests for the Strategy History post-mortem log.

The light summariser path itself is not exercised here (it requires the
network); we test the pure-Python pieces: record formatting, deque cap,
``format_strategy_log``, and prompt-block injection on the code adapter.
"""

from __future__ import annotations

import math

from specevo.config.models import BudgetConfig
from specevo.pipeline.state import PipelineState, StrategyRecord


def _make_state() -> PipelineState:
    return PipelineState(BudgetConfig(evaluations=100))


class TestStrategyRecord:
    def test_format_line_accepted_positive_delta(self):
        rec = StrategyRecord(
            pe_event_id=3,
            stage="mid",
            summary="Greedy bin packing with first-fit-decreasing.",
            best_before=0.82,
            paradigm_score=0.89,
            delta_score=0.07,
            accepted=True,
        )
        line = rec.format_line()
        # Header now uses '### PE #N [stage] — Δ=…, score=…, accepted/rejected'
        # so the heavy prompt can render multi-line structured summaries
        # nicely indented underneath.
        assert "### PE #3 [mid]" in line
        assert "+0.07" in line
        assert "accepted" in line
        assert "first-fit-decreasing" in line

    def test_format_line_rejected_with_nan_score(self):
        rec = StrategyRecord(
            pe_event_id=1,
            stage="early",
            summary="",
            best_before=0.5,
            paradigm_score=math.nan,
            delta_score=0.0,
            accepted=False,
        )
        line = rec.format_line()
        assert "rejected" in line
        assert "score=n/a" in line
        assert "(no summary)" in line

    def test_format_line_multiline_summary_is_indented(self):
        rec = StrategyRecord(
            pe_event_id=7,
            stage="late",
            summary=(
                "ALGORITHM:\nGreedy first-fit decreasing\n\nKEY TACTICS:\n"
                "- Sort items by weight descending"
            ),
            best_before=0.5,
            paradigm_score=0.55,
            delta_score=0.05,
            accepted=True,
        )
        out = rec.format_line()
        # The body lines should be indented under the header.
        body_lines = [ln for ln in out.splitlines() if "Sort items" in ln]
        assert body_lines, "summary body missing"
        assert body_lines[0].startswith("  "), "body must be indented"


class TestStrategyHistory:
    def test_empty_log_renders_empty_string(self):
        state = _make_state()
        assert state.format_strategy_log() == ""

    def test_append_and_format(self):
        state = _make_state()
        state.append_strategy_record(
            StrategyRecord(
                pe_event_id=1,
                stage="early",
                summary="Random restart hill climber.",
                best_before=0.0,
                paradigm_score=0.4,
                delta_score=0.4,
                accepted=True,
            )
        )
        state.append_strategy_record(
            StrategyRecord(
                pe_event_id=2,
                stage="mid",
                summary="Simulated annealing on a global energy.",
                best_before=0.4,
                paradigm_score=0.42,
                delta_score=0.0,
                accepted=False,
            )
        )
        rendered = state.format_strategy_log()
        assert "## Strategy Log" in rendered
        assert "PE #1" in rendered
        assert "PE #2" in rendered
        assert "Δ ≤ 0" in rendered  # the guidance line includes the symbol

    def test_max_entries_truncation(self):
        state = _make_state()
        for i in range(20):
            state.append_strategy_record(
                StrategyRecord(
                    pe_event_id=i,
                    stage="late",
                    summary=f"strategy {i}",
                    best_before=float(i),
                    paradigm_score=float(i),
                    delta_score=0.0,
                    accepted=False,
                )
            )
        # The deque caps at 12; the prompt block keeps the most-recent 5 only.
        assert len(state.strategy_history) == 12
        rendered = state.format_strategy_log(max_entries=5)
        # The 5 most recent are PE #15..#19 (since deque holds #8..#19).
        for i in range(15, 20):
            assert f"PE #{i}" in rendered
        for i in range(10, 15):
            assert f"PE #{i}" not in rendered


class TestPromptInjection:
    def test_paradigm_shift_prompt_includes_strategy_block(self):
        """The CodeAdapter must splice the strategy log into the prompt
        verbatim so the heavy model actually sees it."""
        from specevo.artifacts.code import CodeAdapter
        from specevo.core import EvaluationResult, Program
        from specevo.pool.cvt_map_elites import Elite

        # Bare-minimum LeviConfig is heavy to construct; build the adapter
        # with a tiny stand-in config that has only the fields the prompt
        # builder reads.
        class _StubConfig:
            problem_description = "Solve X."
            function_signature = "def solve(x): ..."
            prompt_overrides: dict = {}
            init = type("I", (), {"diversity_prompt": None})()

            class _Pipeline:
                eval_timeout = 30.0

            pipeline = _Pipeline()

            def __getattr__(self, name):
                return None

        cfg = _StubConfig()
        adapter = CodeAdapter.__new__(CodeAdapter)
        adapter.config = cfg
        adapter.fn_name = "solve"

        elite = Elite(
            program=Program(content="def solve(x): return x"),
            result=EvaluationResult(scores={"score": 1.0}, is_valid=True),
            behavior={},  # type: ignore[arg-type]
        )
        log_block = "\n## Strategy Log (already tried in this run)\n- PE #1 ...\n"
        prompt = adapter.build_paradigm_shift_prompt(
            [(0, elite)],
            n_evaluations=10,
            stagnation=0.0,
            strategy_log_block=log_block,
        )
        assert "## Strategy Log" in prompt
