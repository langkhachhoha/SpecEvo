"""Tests for the 3-mode Navigator intervention dispatch in SpecEvoOrchestrator.

We do not call any LLMs here — we only verify the routing logic
inside :meth:`SpecEvoOrchestrator._pick_navigator_mode` and the anchor
configuration inside :meth:`_build_navigator_prompt_for_mode`.
"""

from __future__ import annotations

import numpy as np

from specevo.engine.orchestrator import SpecEvoConfig, SpecEvoOrchestrator
from specevo.simple import EmbedderConfig
from specevo.simple.archive import Program


def _stub_orch() -> SpecEvoOrchestrator:
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
        output_dir="/tmp/specevo_test_navigator_modes",
        embedder_config=EmbedderConfig(model="fake/embed", dim=8),
    )
    return SpecEvoOrchestrator(cfg)


def _add(orch: SpecEvoOrchestrator, n: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        emb = rng.normal(size=8).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        p = Program(
            code=f"def solve(x):\n    return x + {i}\n",
            description=f"Navigator {i % 4} variant {i}",
            score=float(2.0 - 0.01 * i),
            embedding=emb,
        )
        orch.archive.add(p)


def test_mode_dispatch_by_stagnation() -> None:
    """Mode order is synthesis (low) → surgical (mid) → shift (high).

    The high-stagnation case is routed to ``reframe`` rather than
    ``surgical``: when the search is deeply stuck, jumping to a fresh
    Navigator is more likely to break the plateau than sanding the same
    local optimum further. This matches the empirical pattern from
    earlier runs where ``surgical`` calls at stagnation=1.0 produced
    only same-score admits while a single ``reframe`` call broke through.
    """
    orch = _stub_orch()
    cfg = orch.config

    # synthesis at low stagnation.
    orch.monitor.eval_count = 10
    orch.monitor.last_best_eval = 10
    orch.monitor.last_admit_eval = 10
    assert orch._pick_navigator_mode() == "synthesis"

    # surgical at mid stagnation: push the admit gap so local stagnation
    # is mid-range. admit_gap_max default is 20; we want stagnation
    # around 0.5-0.6 → admit_gap=12.
    orch.monitor.eval_count = 22
    orch.monitor.last_admit_eval = 10
    orch.monitor.last_best_eval = 22
    s = orch.monitor.stagnation_level()
    assert cfg.navigator_synthesis_max_stagnation < s <= cfg.navigator_surgical_max_stagnation
    assert orch._pick_navigator_mode() == "surgical"

    # shift at high stagnation: admit_gap >> admit_gap_max.
    orch.monitor.eval_count = 200
    orch.monitor.last_admit_eval = 10
    orch.monitor.last_best_eval = 10
    assert orch.monitor.stagnation_level() > cfg.navigator_surgical_max_stagnation
    assert orch._pick_navigator_mode() == "reframe"


def test_synthesis_mode_surfaces_top_3_anchors() -> None:
    orch = _stub_orch()
    _add(orch, n=10)
    prompt, anchors, _insps = orch._build_navigator_prompt_for_mode("synthesis")
    assert "MOVE: SYNTHESIS" in prompt
    assert len(anchors) == orch.config.navigator_synthesis_n_anchors


def test_shift_mode_surfaces_top_2_anchors() -> None:
    orch = _stub_orch()
    _add(orch, n=10)
    prompt, anchors, _insps = orch._build_navigator_prompt_for_mode("reframe")
    assert "MOVE: SHIFT" in prompt
    assert len(anchors) == orch.config.navigator_reframe_n_anchors


def test_surgical_mode_surfaces_only_champion() -> None:
    orch = _stub_orch()
    _add(orch, n=10)
    prompt, anchors, insps = orch._build_navigator_prompt_for_mode("surgical")
    assert "MOVE: SURGICAL" in prompt
    # Exactly one anchor regardless of how many programs are in the
    # archive — surgical mode focuses on the champion only.
    assert len(anchors) == 1
    # Inspirations (description-only) should be non-empty when the
    # archive has more programs.
    assert len(insps) >= 1
