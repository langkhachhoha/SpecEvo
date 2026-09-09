"""Tests for ``eval_code_log.jsonl`` — every candidate's source, failures included.

``snap.json`` and the archive only ever hold programs that survived. This log
is the complementary record of what the models actually wrote: one entry per
candidate, whether it was admitted, rejected, crashed, or never parsed at all.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from specevo.engine.evallog import EvalCodeLog
from specevo.engine.orchestrator import SpecEvoConfig, SpecEvoOrchestrator
from specevo.simple import EmbedderConfig

from .test_orchestrator import (  # reuse the no-network fakes
    _EXTRA_MUTATION_RESPONSE,
    _MUTATION_RESPONSES,
    _PARADIGM_RESPONSE,
    SEED,
    _FakeLM,
    _hash_embed,
    _score_fn,
)

_NO_CODE_RESPONSE = """## Description
I thought about it and decided to explain rather than write any code.
"""

_BROKEN_RESPONSE = """## Description
A variant that raises as soon as it is called.

## Code
```python
def solve(x):
    raise ValueError("boom")
```
"""


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _config(tmp_path: Path, **kwargs) -> SpecEvoConfig:
    defaults = dict(
        problem_description="Maximise solve(0)+solve(1)+solve(2).",
        function_signature="def solve(x):",
        score_fn=_score_fn,
        fn_name="solve",
        seed_program=SEED,
        budget_evals=6,
        n_workers=2,
        n_eval_processes=2,
        eval_timeout=5.0,
        navigator_interval=3,
        output_dir=tmp_path / "specevo_run",
        embedder_config=EmbedderConfig(model="fake/embed", dim=64),
    )
    defaults.update(kwargs)
    return SpecEvoConfig(**defaults)


def _run(cfg: SpecEvoConfig, mutation_responses: list[str]):
    orch = SpecEvoOrchestrator(cfg)
    orch.speculator_lm = _FakeLM("fake/mutation", mutation_responses)
    orch.navigator_lm = _FakeLM("fake/Navigator", [_PARADIGM_RESPONSE])
    return asyncio.run(orch.run())


# ---------------------------------------------------------------------------
# EvalCodeLog unit tests
# ---------------------------------------------------------------------------


def test_evallog_classifies_each_outcome(tmp_path: Path):
    log = EvalCodeLog(tmp_path / "eval_code_log.jsonl", run={"method": "specevo"})
    log.record(source="mutate_general", code="def solve(x): return x", score=3.0)
    log.record(source="mutate_general", code="def solve(x): boom", error="NameError: boom")
    log.record(source="mutate_general", code=None, error="parse_miss (no code in output)")

    statuses = [r["status"] for r in _records(log.path)]
    assert statuses == ["ok", "eval_failed", "no_code"]

    log.finalize({"best_score": 3.0})
    folded = json.loads((tmp_path / "eval_code_log.json").read_text())
    assert folded["n_records"] == 3
    assert folded["by_status"] == {"ok": 1, "eval_failed": 1, "no_code": 1}
    assert folded["summary"]["best_score"] == 3.0
    assert folded["run"]["method"] == "specevo"


def test_evallog_never_raises_on_unserialisable_input(tmp_path: Path):
    log = EvalCodeLog(tmp_path / "eval_code_log.jsonl")
    log.record(source="x", code="pass", metrics={"obj": object()}, score=float("-inf"))
    record = _records(log.path)[0]
    assert record["score"] is None  # -inf never reaches the file


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_run_keeps_the_code_of_every_candidate(tmp_path: Path, monkeypatch):
    from specevo.simple import embedder as embedder_module

    monkeypatch.setattr(embedder_module.DescriptionEmbedder, "embed", _hash_embed)

    cfg = _config(tmp_path, save_eval_code=True)
    result = _run(cfg, _MUTATION_RESPONSES + [_EXTRA_MUTATION_RESPONSE])

    log_path = Path(result.output_dir) / "eval_code_log.jsonl"
    records = _records(log_path)
    assert len(records) >= result.total_evaluations
    assert any(r["status"] == "ok" and "def solve" in r["code"] for r in records)
    assert all(r["source"] for r in records)

    folded = json.loads((Path(result.output_dir) / "eval_code_log.json").read_text())
    assert folded["n_records"] == len(records)
    assert folded["summary"]["best_score"] == result.best_score


def test_failed_and_unparseable_candidates_are_kept_too(tmp_path: Path, monkeypatch):
    """The whole point: a candidate that never scored still leaves its text."""
    from specevo.simple import embedder as embedder_module

    monkeypatch.setattr(embedder_module.DescriptionEmbedder, "embed", _hash_embed)

    cfg = _config(tmp_path, save_eval_code=True, budget_evals=8, navigator_interval=0)
    result = _run(cfg, [_BROKEN_RESPONSE, _NO_CODE_RESPONSE] * 8)

    records = _records(Path(result.output_dir) / "eval_code_log.jsonl")
    by_status = {r["status"] for r in records}
    assert "eval_failed" in by_status, "a crashing candidate left no record"
    assert "no_code" in by_status, "a response with no code block left no record"

    crashed = next(r for r in records if r["status"] == "eval_failed")
    assert "raise ValueError" in crashed["code"]
    assert crashed["error"]

    missed = next(r for r in records if r["status"] == "no_code")
    assert missed["code"] is None
    # The raw response is kept, so "why was there no code" stays answerable.
    assert missed["llm_response"] is None or "decided to explain" in missed["llm_response"]


def test_off_by_default(tmp_path: Path, monkeypatch):
    from specevo.simple import embedder as embedder_module

    monkeypatch.setattr(embedder_module.DescriptionEmbedder, "embed", _hash_embed)

    cfg = _config(tmp_path)
    result = _run(cfg, _MUTATION_RESPONSES + [_EXTRA_MUTATION_RESPONSE])
    assert not (Path(result.output_dir) / "eval_code_log.jsonl").exists()
