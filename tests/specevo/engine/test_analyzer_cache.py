"""Tests for the SpecEvo analyzer cache (Đề xuất 1, accumulating policy).

Each ``_refresh_analyses`` call:

1. Evicts cache entries whose ``id()`` no longer maps to a program
   currently in the archive (the program got kicked out by a better
   cell incumbent, or dropped during a re-cluster).
2. Walks the archive in score-descending order and picks the first
   ``analyzer_top_k`` programs that do NOT yet have a cache entry.
   Programs that are already cached are skipped — their analysis is
   not re-generated.

Behavioural consequence with a frozen top-K: refresh #1 covers ranks
1-3, refresh #2 covers ranks 4-6, refresh #N covers ranks 3(N-1)+1…3N,
until the archive is fully cached, after which refreshes are no-ops
unless cell churn introduces a fresh program.

We do not call any LLMs here — we monkey-patch ``_analyze_parent`` so
each "analysis" is just a marker string, then assert which programs
got analysed across successive refreshes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from specevo.engine.orchestrator import SpecEvoConfig, SpecEvoOrchestrator
from specevo.simple import EmbedderConfig
from specevo.simple.archive import Program


def _stub_orch(top_k: int = 3) -> SpecEvoOrchestrator:
    cfg = SpecEvoConfig(
        problem_description="test",
        function_signature="def solve(x):",
        score_fn=lambda fn, _i=None: {"score": 0.0},
        fn_name="solve",
        budget_evals=1,
        n_workers=1,
        n_eval_processes=1,
        eval_timeout=1.0,
        navigator_interval=999,
        analyzer_top_k=top_k,
        analyzer_interval=999,
        output_dir="/tmp/specevo_test_analyzer_cache",
        embedder_config=EmbedderConfig(model="fake/embed", dim=8),
    )
    return SpecEvoOrchestrator(cfg)


def _make_program(score: float, idx: int) -> Program:
    rng = np.random.default_rng(idx)
    emb = rng.normal(size=8).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-9
    return Program(
        code=f"def solve(x):\n    return x + {idx}\n",
        description=f"program {idx} score {score}",
        score=float(score),
        embedding=emb,
    )


def _patch_analyze(orch: SpecEvoOrchestrator, calls: list[int]) -> None:
    """Patch ``_analyze_parent`` to record ``id(parent)`` and seed the
    cache with a deterministic marker. Side-stepped from the real
    LLM-driven implementation entirely."""

    async def fake(parent: Program) -> str:
        async with orch._analysis_lock:
            orch._analysis_cache[id(parent)] = f"ANALYSIS-{parent.score:.3f}"
        calls.append(id(parent))
        return orch._analysis_cache[id(parent)]

    orch._analyze_parent = fake  # type: ignore[method-assign]


def test_refresh_analyses_picks_top_k_when_cache_empty() -> None:
    """First refresh on a populated archive must analyse exactly the
    top-K programs by score."""
    orch = _stub_orch(top_k=3)
    progs = [_make_program(score=2.5 - 0.1 * i, idx=i) for i in range(8)]
    for p in progs:
        orch.archive.add(p)

    calls: list[int] = []
    _patch_analyze(orch, calls)

    asyncio.run(orch._refresh_analyses())

    expected_ids = {id(p) for p in progs[:3]}  # ranks 1, 2, 3
    assert set(calls) == expected_ids
    assert set(orch._analysis_cache.keys()) == expected_ids


def test_refresh_analyses_skips_cached_and_descends() -> None:
    """With a frozen archive, refresh #2 must analyse the NEXT
    uncached ranks (4-6) and leave ranks 1-3 alone."""
    orch = _stub_orch(top_k=3)
    progs = [_make_program(score=2.5 - 0.1 * i, idx=i) for i in range(8)]
    for p in progs:
        orch.archive.add(p)

    calls: list[int] = []
    _patch_analyze(orch, calls)

    asyncio.run(orch._refresh_analyses())  # ranks 1-3
    calls.clear()
    asyncio.run(orch._refresh_analyses())  # ranks 4-6

    expected_ids = {id(p) for p in progs[3:6]}
    assert set(calls) == expected_ids
    # ranks 1-6 are now all cached.
    assert set(orch._analysis_cache.keys()) == {id(p) for p in progs[:6]}


def test_refresh_analyses_becomes_noop_when_archive_fully_cached() -> None:
    """After enough refreshes to cover the archive, further refreshes
    do nothing — no new LLM calls are issued."""
    orch = _stub_orch(top_k=3)
    progs = [_make_program(score=1.0 - 0.05 * i, idx=i) for i in range(6)]
    for p in progs:
        orch.archive.add(p)

    calls: list[int] = []
    _patch_analyze(orch, calls)

    asyncio.run(orch._refresh_analyses())  # ranks 1-3
    asyncio.run(orch._refresh_analyses())  # ranks 4-6 — archive fully cached
    calls.clear()
    asyncio.run(orch._refresh_analyses())  # should be a no-op

    assert calls == []
    # Cache still contains every program in the archive.
    assert set(orch._analysis_cache.keys()) == {id(p) for p in progs}


def test_refresh_analyses_evicts_only_when_program_leaves_archive() -> None:
    """A cached analysis for program P must survive across refreshes
    as long as P is still in the archive. The instant P leaves (cell
    incumbent kicked it out, or any other reason it disappears from
    ``archive.programs()``), the cache entry is evicted on the next
    refresh and the new resident gets analysed.

    We simulate cell churn by directly editing the archive's program
    list — the real eviction path runs inside ``ClusterArchive.add``
    and depends on the live PCA/AST clustering state, which is too
    stateful to construct deterministically in unit tests. The
    accumulating-cache predicate we want to verify is independent of
    *why* a program left the archive; it only checks
    ``id(p) in live_ids``."""
    orch = _stub_orch(top_k=3)
    initial = [_make_program(score=2.5 - 0.1 * i, idx=i) for i in range(5)]
    for p in initial:
        orch.archive.add(p)

    calls: list[int] = []
    _patch_analyze(orch, calls)

    asyncio.run(orch._refresh_analyses())  # ranks 1-3
    assert set(orch._analysis_cache.keys()) == {id(p) for p in initial[:3]}

    # Cell churn: rank-2 leaves the archive, a higher-scoring program
    # appears in the lineup. We bypass ``add()`` to make the swap
    # deterministic regardless of the current clustering basis.
    rank2 = initial[1]
    replacement = _make_program(score=rank2.score + 1.0, idx=99)
    orch.archive._programs[:] = [
        p for p in orch.archive._programs if p is not rank2
    ] + [replacement]

    live_ids = {id(p) for p in orch.archive.programs()}
    assert id(rank2) not in live_ids
    assert id(replacement) in live_ids

    calls.clear()
    asyncio.run(orch._refresh_analyses())

    # rank2's cache entry got evicted (program no longer live).
    assert id(rank2) not in orch._analysis_cache
    # The replacement was analysed (it's now in the top-K and uncached).
    assert id(replacement) in orch._analysis_cache
    assert id(replacement) in calls
    # Ranks 1 and 3 of the original lineup were NOT re-analysed.
    assert id(initial[0]) not in calls
    assert id(initial[2]) not in calls


def test_refresh_analyses_handles_empty_archive() -> None:
    orch = _stub_orch(top_k=3)
    calls: list[int] = []
    _patch_analyze(orch, calls)
    asyncio.run(orch._refresh_analyses())
    assert calls == []
    assert orch._analysis_cache == {}
