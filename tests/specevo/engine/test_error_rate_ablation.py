"""Tests for the error-rate ablation instrumentation.

The instrumentation answers "does the Advisor lower the fraction of
candidates that fail to evaluate?" by closing a measurement window every N
post-init evaluations. Three properties matter for the numbers to be
comparable across ablation arms, and all three are pinned here:

* **Origin.** Windows (and the ``post_init_budget_evals`` cap) start at the
  end of the bootstrap. The ~105 init evaluations are a fixed prelude that
  is identical across arms and would otherwise dilute window 1.
* **One bucket, one exclusion.** Every failed attempt counts toward the
  single error rate; a failed LLM call generated no candidate and so leaves
  *both* sides of the ratio instead of registering as a success.
* **Window count.** A budget of ``k × interval`` yields exactly ``k``
  windows of exactly ``interval`` evaluations. Workers already in flight
  when the budget trips still finish, and that overshoot must not open a
  ragged extra window.
* **Inertness.** With ``ablation_window_evals=0`` (the default, i.e. every
  run launched from specevo.yml) no counter moves and no file is written.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from specevo.engine.orchestrator import (
    SpecEvoConfig,
    SpecEvoOrchestrator,
    format_error_rate_table,
)
from specevo.simple import Program


INTERVAL = 5
BUDGET = 20  # → exactly 4 windows
N_INIT = 11  # a bootstrap that is deliberately not a multiple of INTERVAL


def _config(tmp_path: Path, **kwargs) -> SpecEvoConfig:
    return SpecEvoConfig(
        problem_description="Maximise f.",
        function_signature="def solve(x): ...",
        score_fn=lambda *a, **k: {"score": 0.0},
        fn_name="solve",
        output_dir=tmp_path,
        **kwargs,
    )


def _armed(tmp_path: Path, **kwargs) -> SpecEvoOrchestrator:
    """An orchestrator past its bootstrap, with instrumentation armed.

    Mirrors what :meth:`SpecEvoOrchestrator.run` does once
    ``_bootstrap_population`` returns.
    """
    cfg = _config(
        tmp_path,
        post_init_budget_evals=BUDGET,
        ablation_window_evals=INTERVAL,
        **kwargs,
    )
    orch = SpecEvoOrchestrator(cfg)
    orch.start_time = time.time()
    _feed(orch, N_INIT, error_every=2)  # init evals, instrumentation not armed
    orch._post_init_eval_count = orch.monitor.eval_count
    cfg.budget_evals = orch._post_init_eval_count + BUDGET
    orch._window_mark = {
        "eval_count": orch._post_init_eval_count,
        "errors": orch._error_total,
        "excluded": orch._llm_failure_total,
    }
    return orch


def _feed(
    orch: SpecEvoOrchestrator,
    n: int,
    *,
    error_every: int = 0,
    error_msg: str = "ZeroDivisionError: division by zero",
) -> None:
    """Push ``n`` evaluations through the same two counters the real
    ``_record_reject`` / ``_admit`` paths use."""
    for i in range(n):
        err = error_msg if error_every and i % error_every == 0 else None
        orch.monitor.record_eval(score=0.5, accepted=err is None)
        orch._ablation_after_eval(err)


def test_init_evaluations_do_not_open_windows(tmp_path: Path) -> None:
    cfg = _config(tmp_path, ablation_window_evals=INTERVAL)
    orch = SpecEvoOrchestrator(cfg)
    orch.start_time = time.time()
    _feed(orch, 3 * INTERVAL, error_every=2)
    assert orch._error_rate_windows == []


def test_windows_start_at_the_post_init_origin(tmp_path: Path) -> None:
    orch = _armed(tmp_path)
    _feed(orch, INTERVAL)
    (window,) = orch._error_rate_windows
    assert window["eval_start"] == N_INIT
    assert window["eval_end"] == N_INIT + INTERVAL
    assert window["post_init_evals"] == INTERVAL


def test_errors_are_scoped_to_their_own_window(tmp_path: Path) -> None:
    """Init-phase errors must not leak into window 1, and each window
    reports only the errors raised inside it."""
    orch = _armed(tmp_path)
    _feed(orch, INTERVAL)  # no errors
    _feed(orch, INTERVAL, error_every=1)  # every eval fails
    first, second = orch._error_rate_windows
    assert (first["n_errors"], first["error_rate"]) == (0, 0.0)
    assert (second["n_errors"], second["error_rate"]) == (INTERVAL, 1.0)


@pytest.mark.parametrize(
    "error_msg",
    [
        "parse_miss (no code in output)",
        "executor error: pool died",
        "non-dict result: NoneType",
        "Overlap between circles 0 and 2",
        "ZeroDivisionError: division by zero",
    ],
)
def test_every_failure_shape_counts_as_one_error(error_msg: str) -> None:
    """One bucket: it does not matter *how* the attempt failed."""
    assert SpecEvoOrchestrator._is_llm_failure(error_msg) is False


@pytest.mark.parametrize(
    "error_msg",
    ["LLM error: 429 rate limited", "worker exception: RuntimeError"],
)
def test_llm_call_failures_are_the_only_exclusion(error_msg: str) -> None:
    assert SpecEvoOrchestrator._is_llm_failure(error_msg) is True


def test_llm_failures_leave_both_sides_of_the_ratio(tmp_path: Path) -> None:
    """A failed LLM call produced no candidate, so it is neither an error
    nor a success — counting it either way would bias the comparison."""
    orch = _armed(tmp_path)
    # Of INTERVAL evaluations, every other one is a dead LLM call and the
    # rest are genuine failures: the rate must read 100%, not 50%.
    _feed(orch, INTERVAL, error_every=2, error_msg="LLM error: 429")
    (window,) = orch._error_rate_windows
    n_excluded = window["n_excluded"]
    assert n_excluded > 0
    assert window["n_evaluations"] == INTERVAL
    assert window["n_scored"] == INTERVAL - n_excluded
    assert window["n_errors"] == 0
    assert window["error_rate"] == 0.0


def test_window_of_only_llm_failures_reports_no_rate(tmp_path: Path) -> None:
    """Nothing was scored, so the rate is undefined rather than 0%."""
    orch = _armed(tmp_path)
    _feed(orch, INTERVAL, error_every=1, error_msg="LLM error: timeout")
    (window,) = orch._error_rate_windows
    assert window["n_scored"] == 0
    assert window["error_rate"] is None


def test_budget_overshoot_does_not_open_a_ragged_window(tmp_path: Path) -> None:
    """In-flight workers land a few evaluations past the budget; those
    belong to the last full window, not to a new one."""
    orch = _armed(tmp_path)
    _feed(orch, BUDGET + 3)
    orch._flush_ablation_windows()
    windows = orch._error_rate_windows
    assert len(windows) == BUDGET // INTERVAL
    assert [w["n_evaluations"] for w in windows] == [INTERVAL] * len(windows)
    assert not any(w["final_partial"] for w in windows)


def test_trailing_window_is_flushed_when_the_run_stops_early(tmp_path: Path) -> None:
    """A run cut short (dollar cap, crash) still reports its partial tail so
    the evaluations it did spend are not silently dropped."""
    orch = _armed(tmp_path)
    _feed(orch, INTERVAL + 2)
    orch._flush_ablation_windows()
    full, partial = orch._error_rate_windows
    assert full["n_evaluations"] == INTERVAL and not full["final_partial"]
    assert partial["n_evaluations"] == 2 and partial["final_partial"]


def test_checkpoint_dumps_the_whole_population_with_code(tmp_path: Path) -> None:
    orch = _armed(tmp_path, ablation_checkpoint_population=True)
    for k in range(3):
        orch.archive.add(Program(
            code=f"def solve(x): return {k}",
            description=f"variant {k}",
            score=float(k),
            embedding=np.zeros(4),
        ))
    _feed(orch, 2 * INTERVAL)

    paths = sorted((tmp_path / "checkpoints").glob("checkpoint_*.json"))
    assert [p.name for p in paths] == ["checkpoint_01.json", "checkpoint_02.json"]
    payload = json.loads(paths[0].read_text())
    assert payload["checkpoint"] == 1
    assert payload["population_size"] == 3
    # Sorted best-first, and the code itself is present (that is the point
    # of the dump — it is what gets read back for post-hoc analysis).
    assert [p["score"] for p in payload["population"]] == [2.0, 1.0, 0.0]
    assert payload["population"][0]["code"] == "def solve(x): return 2"


def test_report_and_table_agree_with_the_windows(tmp_path: Path) -> None:
    orch = _armed(tmp_path)
    _feed(orch, BUDGET, error_every=2)
    orch._flush_ablation_windows()
    orch._report_error_rate()

    report = json.loads((tmp_path / "error_rate_report.json").read_text())
    assert report["n_windows"] == BUDGET // INTERVAL
    assert report["total_evaluations"] == BUDGET
    assert report["total_scored"] == BUDGET  # no LLM failures in this run
    assert report["total_excluded_llm_failures"] == 0
    assert report["total_errors"] == sum(w["n_errors"] for w in orch._error_rate_windows)
    assert report["overall_error_rate"] == report["total_errors"] / BUDGET
    assert report["post_init_eval_count"] == N_INIT

    table = format_error_rate_table(orch._error_rate_windows, interval=INTERVAL)
    assert f"{report['total_errors']}/{BUDGET}" in table


def test_instrumentation_off_is_completely_inert(tmp_path: Path) -> None:
    """The default path — every run launched from specevo.yml."""
    cfg = _config(tmp_path)
    assert cfg.ablation_window_evals == 0
    assert cfg.post_init_budget_evals is None
    assert cfg.ablation_checkpoint_population is False
    assert cfg.align_advisor_to_post_init is False

    orch = SpecEvoOrchestrator(cfg)
    orch.start_time = time.time()
    orch._window_mark = {"eval_count": 0, "errors": 0, "excluded": 0}  # even if armed
    _feed(orch, 4 * INTERVAL, error_every=1)
    orch._flush_ablation_windows()

    assert orch._error_total == 0
    assert orch._llm_failure_total == 0
    assert orch._error_rate_windows == []
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "error_rate_report.json").exists()
    assert format_error_rate_table([], interval=INTERVAL) == ""
