"""Tests for the ``snap.json`` search trace.

``snap.json`` is the run's timeline — every Navigator call (tagged with
which of the three modes was routed to) and every new best (tagged with
which component produced it). It is written incrementally, so the two
things worth pinning are: the events land with the right shape, and the
file on disk is always a complete, strictly-valid JSON document even
mid-run.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from specevo.engine.orchestrator import SpecEvoConfig, SpecEvoOrchestrator
from specevo.engine.snaplog import NAVIGATOR_MODES, SnapLog, classify_producer
from specevo.simple import EmbedderConfig
from specevo.simple.archive import Program
from specevo.utils.resilient_pool import ResilientProcessPool

from .test_orchestrator import (  # reuse the no-network fakes
    _EXTRA_MUTATION_RESPONSE,
    _MUTATION_RESPONSES,
    _PARADIGM_RESPONSE,
    SEED,
    _FakeLM,
    _hash_embed,
    _score_fn,
)


def _strict_load(path: Path) -> dict:
    """Parse *path*, refusing the ``Infinity`` / ``NaN`` JSON extensions.

    Scores start at ``-inf`` and problem metrics can be ``nan``; the
    trace must never leak either into the file, or downstream parsers
    (jq, JS, strict Python) choke on it.
    """
    def _reject(token: str) -> None:
        raise AssertionError(f"non-JSON constant {token!r} in {path}")

    return json.loads(path.read_text(), parse_constant=_reject)


# ---------------------------------------------------------------------------
# SnapLog unit tests
# ---------------------------------------------------------------------------


def test_classify_producer_covers_every_source_label():
    assert classify_producer("init") == "init"
    assert classify_producer("Navigator") == "navigator"
    assert classify_producer("navigator_reframe") == "navigator"
    assert classify_producer("navigator_variant") == "navigator_variant"
    # Operator labels the main loop synthesises at runtime.
    assert classify_producer("mutate_general") == "speculator"
    assert classify_producer("mutate_targeted") == "speculator"
    assert classify_producer("crossover_structural") == "speculator"


def test_snaplog_writes_on_construction_and_after_each_event(tmp_path: Path):
    path = tmp_path / "snap.json"
    log = SnapLog(path, run={"method": "specevo"})

    # File exists before a single event — a run killed during bootstrap
    # still leaves a parseable trace.
    doc = _strict_load(path)
    assert doc["run"]["method"] == "specevo"
    assert doc["events"] == []
    assert doc["summary"]["navigator"]["calls"] == 0

    event = log.navigator_call(
        trigger=1, mode="reframe", forced=False, at_eval=42,
        stagnation=0.9, global_stagnation=0.9, local_stagnation=0.4,
        prev_best=float("-inf"),  # nothing admitted yet
        occupied_cells=7, archive_size=12,
        anchors=[{"cell_id": 3, "score": 1.5, "description": "greedy"}],
        n_inspirations=2, cost_usd=0.5,
    )
    # Flushed immediately, while the frontier call is still outstanding.
    doc = _strict_load(path)
    assert doc["events"][0]["outcome"] == "pending"
    assert doc["events"][0]["mode"] == "reframe"
    assert doc["events"][0]["prev_best"] is None  # -inf sanitised away

    log.navigator_result(
        event, outcome="accepted", eval_index=43, score=2.0,
        delta_vs_prev_best=0.5, is_new_best=True,
        description="a new Navigator", code="def solve(x): return x",
    )
    log.navigator_fanout(event, {"n_variants": 4, "n_accepted": 2, "n_new_best": 1})
    log.new_best(
        at_eval=43, source="Navigator", navigator_mode="reframe", model="gpt-5",
        score=2.0, prev_best=1.5, evals_since_prev_best=11,
        stagnation_before=0.9, cell_id=3, description="a new Navigator",
        code="def solve(x): return x", metrics={"score": 2.0}, cost_usd=0.5,
    )
    log.new_best(
        at_eval=50, source="mutate_targeted", navigator_mode=None, model="qwen",
        score=2.75, prev_best=2.0, evals_since_prev_best=7,
        stagnation_before=0.1, cell_id=4, description="tuned constants",
        code="def solve(x): return x + 1", metrics={}, cost_usd=0.6,
    )

    doc = _strict_load(path)
    nav = doc["summary"]["navigator"]
    assert nav["calls"] == 1
    assert nav["by_mode"]["reframe"] == 1
    assert nav["accepted_by_mode"]["reframe"] == 1
    assert nav["new_best_by_mode"]["reframe"] == 1
    assert nav["fanout_variants"] == 4 and nav["fanout_accepted"] == 2

    nb = doc["summary"]["new_best"]
    assert nb["count"] == 2
    assert nb["by_producer"] == {
        "init": 0, "speculator": 1, "navigator": 1, "navigator_variant": 0,
    }
    assert nb["by_source"]["mutate_targeted"] == 1
    assert nb["score_gain_by_producer"]["navigator"] == 0.5
    assert nb["score_gain_by_producer"]["speculator"] == 0.75
    assert nb["best_score"] == 2.75

    # Full code + description are kept, not just summaries.
    best_events = [e for e in doc["events"] if e["event"] == "new_best"]
    assert best_events[0]["code"] == "def solve(x): return x"
    assert best_events[0]["navigator_mode"] == "reframe"
    assert best_events[1]["navigator_mode"] is None


def test_snaplog_sanitises_non_finite_metrics(tmp_path: Path):
    path = tmp_path / "snap.json"
    log = SnapLog(path)
    log.new_best(
        at_eval=1, source="init", navigator_mode=None, model="m",
        score=1.0, prev_best=float("-inf"), evals_since_prev_best=1,
        stagnation_before=0.0, cell_id=0, description="d", code="c",
        metrics={"nan_metric": math.nan, "inf_metric": math.inf, "ok": 3},
        cost_usd=0.0,
    )
    doc = _strict_load(path)  # would raise on NaN / Infinity
    ev = doc["events"][0]
    assert ev["metrics"] == {"nan_metric": None, "inf_metric": None, "ok": 3}
    # First-ever best has no previous score to improve on.
    assert ev["prev_best"] is None and ev["improvement"] is None


def test_snaplog_clips_pathological_output(tmp_path: Path):
    path = tmp_path / "snap.json"
    log = SnapLog(path)
    log.new_best(
        at_eval=1, source="init", navigator_mode=None, model="m",
        score=1.0, prev_best=0.0, evals_since_prev_best=1,
        stagnation_before=0.0, cell_id=0, description="d",
        code="x" * (SnapLog.MAX_CODE_CHARS + 5_000), metrics={}, cost_usd=0.0,
    )
    code = _strict_load(path)["events"][0]["code"]
    assert len(code) < SnapLog.MAX_CODE_CHARS + 200
    assert "truncated" in code


def test_snaplog_never_raises_on_a_bad_write(tmp_path: Path):
    """Instrumentation must not be able to kill a paid run."""
    log = SnapLog(tmp_path / "snap.json")
    log.path = tmp_path / "no" / "such" / "dir" / "snap.json"
    # Every entry point swallows its own failure.
    log.new_best(
        at_eval=1, source="init", navigator_mode=None, model="m", score=1.0,
        prev_best=0.0, evals_since_prev_best=1, stagnation_before=0.0,
        cell_id=0, description="d", code="c", metrics={}, cost_usd=0.0,
    )
    log.finalize({"best_score": 1.0})
    assert log.summary()["new_best"]["count"] == 1


# ---------------------------------------------------------------------------
# End-to-end: a real (faked-LLM) run produces a usable trace
# ---------------------------------------------------------------------------


def test_run_writes_snap_trace(tmp_path: Path, monkeypatch):
    from specevo.simple import embedder as embedder_module

    monkeypatch.setattr(embedder_module.DescriptionEmbedder, "embed", _hash_embed)

    cfg = SpecEvoConfig(
        problem_description="Maximise solve(0)+solve(1)+solve(2).",
        function_signature="def solve(x):",
        score_fn=_score_fn,
        fn_name="solve",
        seed_program=SEED,
        budget_evals=8,
        n_workers=2,
        n_eval_processes=2,
        eval_timeout=5.0,
        # Keep the bootstrap tiny: the default 5 seeds × 20 variants would
        # burn ~100 evaluations before the main loop ever starts.
        n_diverse_seeds=1,
        n_variants_per_seed=2,
        navigator_interval=999,  # Navigator is covered by its own test below
        output_dir=tmp_path / "specevo_run",
        embedder_config=EmbedderConfig(model="fake/embed", dim=64),
    )
    orch = SpecEvoOrchestrator(cfg)
    orch.speculator_lm = _FakeLM("fake/mutation", _MUTATION_RESPONSES + [_EXTRA_MUTATION_RESPONSE])
    orch.navigator_lm = _FakeLM("fake/Navigator", [_PARADIGM_RESPONSE])

    result = asyncio.run(orch.run())

    snap_path = Path(result.output_dir) / "snap.json"
    assert snap_path.exists()
    doc = _strict_load(snap_path)

    assert doc["run"]["method"] == "specevo"
    assert doc["final"]["total_evaluations"] == result.total_evaluations
    assert doc["final"]["best_score"] == result.best_score

    # The seed alone guarantees at least one new best, and every new-best
    # event carries the code that achieved it.
    bests = [e for e in doc["events"] if e["event"] == "new_best"]
    assert bests, "a run that admits a seed must log at least one new best"
    assert [e["score"] for e in bests] == sorted(e["score"] for e in bests)
    for e in bests:
        assert e["producer"] in ("init", "speculator", "navigator", "navigator_variant")
        assert e["code"] and e["description"] is not None
        assert e["at_eval"] >= 1
    assert doc["summary"]["new_best"]["count"] == len(bests)
    assert doc["summary"]["new_best"]["best_score"] == result.best_score
    # The trace dates each improvement to an evaluation index — the thing
    # snapshot.json structurally cannot tell you.
    assert all(e["at_eval"] <= result.total_evaluations for e in bests)


def _seed_archive(orch: SpecEvoOrchestrator, n: int) -> None:
    """Fill the archive directly so a Navigator intervention has anchors to read."""
    rng = np.random.default_rng(0)
    for i in range(n):
        emb = rng.normal(size=64).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        orch.archive.add(Program(
            code=f"def solve(x):\n    return x + {i}\n",
            description=f"Navigator {i % 3} variant {i}",
            score=float(2.0 - 0.01 * i),
            embedding=emb,
            source="init",
        ))


@pytest.mark.parametrize("mode", NAVIGATOR_MODES)
def test_navigator_call_is_traced_with_its_mode(mode: str, tmp_path: Path, monkeypatch):
    """One frontier call → one ``navigator`` event carrying its mode.

    Driven through ``_navigator_intervention`` directly rather than through
    ``run()``: the PE cron only fires from the main loop after the
    bootstrap phase, which makes "did the Navigator wake" a timing race
    in a test. ``navigator_force_mode`` pins the routed mode so each of
    the three shows up in the trace.
    """
    from specevo.simple import embedder as embedder_module

    monkeypatch.setattr(embedder_module.DescriptionEmbedder, "embed", _hash_embed)

    cfg = SpecEvoConfig(
        problem_description="Maximise solve(0)+solve(1)+solve(2).",
        function_signature="def solve(x):",
        score_fn=_score_fn,
        fn_name="solve",
        budget_evals=100,
        n_workers=1,
        n_eval_processes=1,
        # Generous: the evaluator pool spawns (not forks) a fresh
        # interpreter per candidate, which can take several seconds on a
        # loaded machine. A tight timeout here turns into a flaky
        # ``eval_error`` outcome that has nothing to do with the trace.
        eval_timeout=60.0,
        navigator_interval=999,
        navigator_min_archive_size=1,
        navigator_force_mode=mode,
        n_navigator_variants=2,
        output_dir=tmp_path / f"specevo_nav_{mode}",
        embedder_config=EmbedderConfig(model="fake/embed", dim=64),
    )
    orch = SpecEvoOrchestrator(cfg)
    orch.speculator_lm = _FakeLM("fake/mutation", [_EXTRA_MUTATION_RESPONSE])
    orch.navigator_lm = _FakeLM("fake/Navigator", [_PARADIGM_RESPONSE])
    _seed_archive(orch, 6)

    async def _drive() -> None:
        orch.start_time = time.time()
        orch._eval_processes = ResilientProcessPool(max_workers=1)
        orch.pe_trigger_count = 1
        try:
            await orch._navigator_intervention()
        finally:
            orch._eval_processes.shutdown()

    asyncio.run(_drive())

    doc = _strict_load(Path(cfg.output_dir) / "snap.json")
    navs = [e for e in doc["events"] if e["event"] == "navigator"]
    assert len(navs) == 1
    ev = navs[0]
    assert ev["mode"] == mode
    assert ev["mode_forced"] is True
    assert ev["outcome"] in ("accepted", "rejected"), ev["outcome"]
    assert ev["code"] and "2 ** x" in ev["code"]  # the frontier's program
    assert ev["description"]
    assert ev["stagnation"] is not None
    assert ev["n_anchors"] == len(ev["anchors"]) >= 1
    # Cheap-variant fanout is attributed to the Navigator call that spawned it.
    assert ev["fanout"]["n_variants"] == 2
    assert ev["fanout"]["n_accepted"] + ev["fanout"]["n_failed"] <= 2

    summary = doc["summary"]["navigator"]
    assert summary["calls"] == 1
    assert summary["by_mode"][mode] == 1
    assert summary["fanout_variants"] == 2

    # The frontier seed beats every archive entry (2**0+2**1+2**2 = 7 > 2.0),
    # so it must also register as a new best attributed to the Navigator.
    bests = [e for e in doc["events"] if e["event"] == "new_best"]
    assert bests, "the frontier seed outscores the whole seeded archive"
    assert bests[0]["producer"] == "navigator"
    assert bests[0]["navigator_mode"] == mode
    assert doc["summary"]["new_best"]["by_producer"]["navigator"] >= 1
