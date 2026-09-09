"""Token accounting in the LSR-Synth finaliser (scripts/lsr_finalize.py).

``read_tokens`` mirrors ``read_cost``: baselines append one record per LLM
call to ``cost_log.jsonl`` (which survives resumes), while SpecEvo keeps no
call log and reports run totals in ``summary.json``. These tests pin that
precedence, and the resume case where an abandoned SpecEvo attempt's tokens
still belong in the total.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import lsr_finalize as lf  # noqa: E402


def _write_call_log(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_read_tokens_sums_a_baseline_call_log(tmp_path: Path):
    _write_call_log(
        tmp_path / "cost_log.jsonl",
        [
            {"prompt_tokens": 100, "completion_tokens": 10, "cost_usd": 0.01},
            {"prompt_tokens": 250, "completion_tokens": 30, "cost_usd": 0.02},
        ],
    )
    assert lf.read_tokens(tmp_path, {}) == (350, 40)


def test_read_tokens_tolerates_junk_lines(tmp_path: Path):
    log = tmp_path / "cost_log.jsonl"
    log.write_text(
        '{"prompt_tokens": 5, "completion_tokens": 1}\n'
        "\n"
        "not json at all\n"
        '{"prompt_tokens": 7}\n'
    )
    assert lf.read_tokens(tmp_path, {}) == (12, 1)


def test_read_tokens_falls_back_to_specevo_summary(tmp_path: Path):
    """SpecEvo writes no call log — its summary.json totals are the source."""
    info = {"prompt_tokens": 9000, "completion_tokens": 1200}
    assert lf.read_tokens(tmp_path, info) == (9000, 1200)


def test_read_tokens_adds_abandoned_specevo_attempts(tmp_path: Path):
    prev = tmp_path / "prev_attempt_123"
    prev.mkdir()
    (prev / "summary.json").write_text(
        json.dumps({"total_prompt_tokens": 400, "total_completion_tokens": 50})
    )
    info = {"prompt_tokens": 1000, "completion_tokens": 100}
    assert lf.read_tokens(tmp_path, info) == (1400, 150)


def test_call_log_wins_over_summary(tmp_path: Path):
    """A resumed baseline's log is authoritative; the in-process total is not."""
    _write_call_log(tmp_path / "cost_log.jsonl", [{"prompt_tokens": 3, "completion_tokens": 2}])
    assert lf.read_tokens(tmp_path, {"prompt_tokens": 999, "completion_tokens": 999}) == (3, 2)


def test_read_tokens_uses_totals_file_when_nothing_else_exists(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "cost_log.totals.json").write_text(
        json.dumps({"total_prompt_tokens": 77, "total_completion_tokens": 11})
    )
    assert lf.read_tokens(tmp_path, {}) == (77, 11)


def test_read_tokens_none_when_unknown(tmp_path: Path):
    assert lf.read_tokens(tmp_path, {}) == (None, None)


def test_find_best_program_lifts_specevo_token_totals(tmp_path: Path):
    (tmp_path / "best_program.py").write_text("def f(x):\n    return x\n")
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "best_score": 1.0,
                "total_evaluations": 500,
                "total_cost": 1.25,
                "total_prompt_tokens": 123456,
                "total_completion_tokens": 7890,
                "init_usage": {
                    "llm_calls": 105,
                    "prompt_tokens": 40000,
                    "completion_tokens": 2000,
                },
            }
        )
    )
    prog, info = lf.find_best_program(tmp_path)
    assert prog is not None
    assert info["prompt_tokens"] == 123456
    assert info["completion_tokens"] == 7890
    assert info["init_usage"]["prompt_tokens"] == 40000
