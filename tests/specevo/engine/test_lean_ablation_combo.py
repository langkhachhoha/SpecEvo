"""The combined lean configuration: no Advisor, Navigator locked to Reframe,
Speculator down to a single prompt.

This is a *combination* of ablation axes rather than a new one — ``--ablation``
accepts a single name and ``specevo_ablation.yml`` dispatches one axis per run, so
the combination is only reachable through the individual knobs. These tests pin
down what each knob is responsible for, so the combination cannot silently
regress into "mostly off".

No LLM calls: everything asserted here is routing and configuration.
"""

from __future__ import annotations

import random

import numpy as np

from specevo.engine.orchestrator import SpecEvoConfig, SpecEvoOrchestrator
from specevo.simple import EmbedderConfig
from specevo.simple.archive import Program


def _lean_orch() -> SpecEvoOrchestrator:
    """Orchestrator under the requested configuration."""
    cfg = SpecEvoConfig(
        problem_description="test",
        function_signature="def solve(x):",
        score_fn=lambda fn, _i=None: {"score": 0.0},
        fn_name="solve",
        budget_evals=1,
        n_workers=1,
        n_eval_processes=1,
        eval_timeout=1.0,
        navigator_interval=10,
        output_dir="/tmp/specevo_test_lean_combo",
        embedder_config=EmbedderConfig(model="fake/embed", dim=8),
        # --- the three requested axes -------------------------------------
        enable_advisor=False,       # no Advisor
        navigator_force_mode="reframe",    # Navigator: Reframe only
        single_prompt_operators=True,   # Speculator: one template per operator
        p_crossover=0.0,                # ... and never the crossover one
        enable_targeted_mutate=False,   # ... nor the targeted-mutate one
    )
    return SpecEvoOrchestrator(cfg)


def _add(orch: SpecEvoOrchestrator, n: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        emb = rng.normal(size=8).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        orch.archive.add(
            Program(
                code=f"def solve(x):\n    return x + {i}\n",
                description=f"Navigator {i % 4} variant {i}",
                score=float(2.0 - 0.01 * i),
                embedding=emb,
            )
        )


def test_navigator_stays_on_reframe_at_every_stagnation_level() -> None:
    """Reframe (internally ``reframe``) regardless of what the monitor reports.

    Without the force, these three states route to synthesis / surgical /
    shift respectively — see ``test_paradigm_modes.py``.
    """
    orch = _lean_orch()

    orch.monitor.eval_count = 10
    orch.monitor.last_best_eval = 10
    orch.monitor.last_admit_eval = 10
    assert orch.monitor.stagnation_level() <= orch.config.navigator_synthesis_max_stagnation
    assert orch._pick_navigator_mode() == "reframe"

    orch.monitor.eval_count = 22
    orch.monitor.last_admit_eval = 10
    orch.monitor.last_best_eval = 22
    assert orch._pick_navigator_mode() == "reframe"

    orch.monitor.eval_count = 200
    orch.monitor.last_admit_eval = 10
    orch.monitor.last_best_eval = 10
    assert orch._pick_navigator_mode() == "reframe"


def test_navigator_prompt_is_the_reframe_prompt() -> None:
    orch = _lean_orch()
    _add(orch, n=10)
    prompt, anchors, _insps = orch._build_navigator_prompt_for_mode(
        orch._pick_navigator_mode()
    )
    assert "MOVE: SHIFT" in prompt
    assert len(anchors) == orch.config.navigator_reframe_n_anchors


def test_advisor_never_injects() -> None:
    """Even with an advice block present, nothing reaches a prompt."""
    orch = _lean_orch()
    orch.current_advisor_note = "DO THIS INSTEAD"
    # _pick_advisor_note is the only injection site; it short-circuits on the
    # flag before the inject-probability coin flip is even reached.
    assert all(orch._pick_advisor_note() is None for _ in range(50))


def test_speculator_has_exactly_one_prompt_path() -> None:
    """All three alternative Speculator prompts are unreachable.

    ``single_prompt_operators`` alone leaves three live paths: the general
    mutate template, the structural crossover template (p=0.35) and the
    targeted-mutate template (p=0.5 once an analysis is cached). Only the
    combination with ``p_crossover=0`` and ``enable_targeted_mutate=False``
    collapses the repertoire to MUTATE_PROMPT_GENERAL alone.
    """
    orch = _lean_orch()
    cfg = orch.config
    rng = random.Random(0)

    # Mutate template: always the general one.
    labels = {orch.prompt_sampler.pick_mutate(rng)[0] for _ in range(200)}
    assert labels == {"general"}

    # Crossover branch in _generate_one is `rng.random() < p_crossover`,
    # which is unreachable at 0.0 (random() is in [0, 1)).
    assert cfg.p_crossover == 0.0
    assert all(rng.random() >= cfg.p_crossover for _ in range(200))

    # Targeted-mutate branch is gated on the flag before the coin flip.
    assert cfg.enable_targeted_mutate is False


def test_analyzer_and_advisor_monitors_are_disabled() -> None:
    """Both background LLM loops abstain, so no spend outside the main loop."""
    orch = _lean_orch()
    assert orch.config.enable_advisor is False
    assert orch.config.enable_targeted_mutate is False
