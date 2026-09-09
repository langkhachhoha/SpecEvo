"""Tests for the SpecEvo monitor (plateau + accept-rate only)."""

from __future__ import annotations

from specevo.simple.monitor import Monitor, MonitorConfig


def test_plateau_grows_until_new_best() -> None:
    m = Monitor()
    for _ in range(5):
        m.record_eval(score=0.1, accepted=False)
    assert m.plateau_steps == 5
    m.record_eval(score=1.0, accepted=True)
    assert m.plateau_steps == 0
    assert m.best_score == 1.0


def test_stagnation_level_caps_at_one() -> None:
    m = Monitor(MonitorConfig(plateau_max=10))
    for _ in range(25):
        m.record_eval(score=-1.0, accepted=False)
    assert m.stagnation_level() == 1.0


def test_local_stagnation_drives_signal_when_global_is_low() -> None:
    """Admits without a new best should still pull stagnation up
    (local signal). And the converse: admits should *reset* local
    stagnation back to 0."""
    m = Monitor(MonitorConfig(plateau_max=1000, admit_gap_max=10))
    # First admit, sets best.
    m.record_eval(score=5.0, accepted=True)
    # 5 rejects in a row — local_stagnation should climb to 0.5.
    for _ in range(5):
        m.record_eval(score=-1.0, accepted=False)
    assert m.global_stagnation() < 0.1  # still close to best
    assert abs(m.local_stagnation() - 0.5) < 1e-6
    assert abs(m.stagnation_level() - 0.5) < 1e-6
    # An admit (even one that doesn't beat best) must reset local.
    m.record_eval(score=4.0, accepted=True)
    assert m.local_stagnation() == 0.0
    # Now drive local saturated.
    for _ in range(15):
        m.record_eval(score=-1.0, accepted=False)
    assert m.local_stagnation() == 1.0


def test_global_stagnation_unchanged_by_non_best_admits() -> None:
    """Only NEW BEST events should reset the global timer — an admit
    that didn't beat best must NOT reset it."""
    m = Monitor(MonitorConfig(plateau_max=20, admit_gap_max=100))
    m.record_eval(score=10.0, accepted=True)  # first best
    # Lots of non-best admits.
    for _ in range(10):
        m.record_eval(score=5.0, accepted=True)
    assert m.plateau_steps == 10  # global timer still ticking
    assert abs(m.global_stagnation() - 0.5) < 1e-6


def test_accept_rate_window() -> None:
    m = Monitor(MonitorConfig(accept_window_size=10))
    for _ in range(5):
        m.record_eval(score=0.1, accepted=True)
    for _ in range(5):
        m.record_eval(score=0.1, accepted=False)
    assert abs(m.acceptance_rate() - 0.5) < 1e-6


def test_only_accepted_new_best_updates_best_score() -> None:
    m = Monitor()
    # Even though the score is "good", it was not accepted into the
    # archive, so it must NOT count as a new best.
    m.record_eval(score=10.0, accepted=False)
    assert m.best_score == float("-inf")
    m.record_eval(score=0.5, accepted=True)
    assert m.best_score == 0.5


def test_snapshot_contains_basic_fields() -> None:
    m = Monitor()
    m.record_eval(score=0.5, accepted=True)
    snap = m.snapshot()
    for key in (
        "eval_count", "best_score",
        "plateau_steps", "admit_gap",
        "global_stagnation", "local_stagnation", "stagnation_level",
        "accept_rate",
    ):
        assert key in snap, key
