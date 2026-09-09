"""End-to-end tests for RelayEvolve and the cheap/strong routing baselines.

The LLM is stubbed, so these run offline and cost nothing while still driving
the real controllers, the real OpenEvolve populations, the real evaluator
subprocess path and the real parallel loop.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path

import pytest

from specevo_baselines.config import _DB_CONFIG_BY_TYPE, Config, LLMModelConfig
from specevo_baselines.llm.base import LLMResponse
from specevo_baselines.search.relay.bank import Candidate, RelayBank, curate_seed_population
from specevo_baselines.search.relay.embedding import CandidateEmbedder
from specevo_baselines.search.relay.scheduler import GROW, GrowDeepenScheduler, TwoArmedBandit

INITIAL_PROGRAM = textwrap.dedent(
    """
    def solve():
        return 1.0
    """
).strip()

EVALUATOR = textwrap.dedent(
    """
    import importlib.util


    def evaluate(program_path):
        spec = importlib.util.spec_from_file_location("cand", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = float(module.solve())
        return {"combined_score": value, "validity": 1}
    """
).strip()


@pytest.fixture(
    params=[
        "relayevolve",
        "relay_all_cheap",
        "relay_all_strong",
        "relay_fixed_switch",
        "relay_random",
        "relay_bandit",
    ]
)
def search_type_all(request) -> str:
    return request.param


@pytest.fixture()
def benchmark(tmp_path: Path) -> Path:
    (tmp_path / "initial_program.py").write_text(INITIAL_PROGRAM + "\n")
    (tmp_path / "evaluator.py").write_text(EVALUATOR + "\n")
    return tmp_path


def _make_config(search_type: str, workers: int = 4) -> Config:
    config = Config()
    config.language = "python"
    config.search.type = search_type
    config.search.database = _DB_CONFIG_BY_TYPE[search_type]()
    config.search.database.random_seed = 7
    config.search.database.block_size = 2
    config.search.database.max_trajectories = 3
    config.search.database.trajectory_horizon = 2
    config.search.database.init_grow_blocks = 2
    config.search.database.bank_size = 4
    config.max_parallel_iterations = workers
    config.max_iterations = 12
    config.monitor.enabled = False
    config.human_feedback_enabled = False
    config.evaluator.timeout = 30
    config.evaluator.cascade_evaluation = False
    config.llm.models = [
        LLMModelConfig(name="stub-cheap", api_base="https://example.invalid/v1", api_key="k")
    ]
    config.llm.guide_models = [
        LLMModelConfig(name="stub-strong", api_base="https://example.invalid/v1", api_key="k")
    ]
    config.llm.evaluator_models = list(config.llm.models)
    return config


class _StubPool:
    """Stands in for an LLMPool; each tier improves the score at its own rate."""

    def __init__(self, name: str, step: float, counter: dict):
        self.models_cfg = [LLMModelConfig(name=name)]
        self.name = name
        self.step = step
        self.counter = counter

    async def generate(self, system_message, messages, **kwargs) -> LLMResponse:
        self.counter[self.name] = self.counter.get(self.name, 0) + 1
        value = 1.0 + self.step * self.counter[self.name]
        await asyncio.sleep(0)
        return LLMResponse(text=f"```python\ndef solve():\n    return {value:.6f}\n```")


def _run(
    search_type: str,
    benchmark: Path,
    tmp_path: Path,
    workers: int = 4,
    save_eval_code: bool = False,
):
    from specevo_baselines import Runner

    config = _make_config(search_type, workers=workers)
    config.search.database.save_eval_code = save_eval_code
    output_dir = tmp_path / f"out_{search_type}"
    runner = Runner(
        initial_program_path=str(benchmark / "initial_program.py"),
        evaluation_file=str(benchmark / "evaluator.py"),
        config=config,
        output_dir=str(output_dir),
    )

    counter: dict = {}
    original = Runner.run

    async def patched(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        return result

    # Swap the pools once the controller exists: Runner builds it inside run().
    from specevo_baselines.search.relay.tiered import TieredController

    real_init = TieredController.__init__

    def init(self, controller_input):
        real_init(self, controller_input)
        self.cheap_llms = _StubPool("cheap", 0.01, counter)
        self.strong_llms = _StubPool("strong", 0.05, counter)
        self.llms = self.cheap_llms

    TieredController.__init__ = init
    try:
        best = asyncio.run(runner.run(iterations=config.max_iterations))
    finally:
        TieredController.__init__ = real_init
    return best, output_dir, counter


@pytest.mark.parametrize(
    "search_type",
    [
        "relayevolve",
        "relay_all_cheap",
        "relay_all_strong",
        "relay_fixed_switch",
        "relay_random",
        "relay_bandit",
    ],
)
def test_methods_complete_and_report(search_type, benchmark, tmp_path, monkeypatch):
    monkeypatch.delenv("SPECEVO_MAX_COST_USD", raising=False)
    best, output_dir, counter = _run(search_type, benchmark, tmp_path)

    assert best is not None, f"{search_type} produced no program"
    assert best.metrics["combined_score"] > 1.0

    summary = json.loads((output_dir / "relay_summary.json").read_text())
    assert summary["iterations_used"] > 0
    assert sum(summary["llm_calls_by_tier"].values()) > 0

    progress = [
        json.loads(line)
        for line in (output_dir / "relay_progress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert progress, "no progress records written"
    assert all(record["tier"] in ("cheap", "strong") for record in progress)


def test_save_eval_code_keeps_every_generation(benchmark, tmp_path, monkeypatch):
    """--save-eval-code writes one record per generation, code included."""
    monkeypatch.delenv("SPECEVO_MAX_COST_USD", raising=False)
    _, output_dir, _ = _run("relayevolve", benchmark, tmp_path, save_eval_code=True)

    records = [
        json.loads(line)
        for line in (output_dir / "eval_code_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "relay_summary.json").read_text())
    assert len(records) >= summary["iterations_used"]
    assert all(record["tier"] in ("cheap", "strong") for record in records)
    assert all(record["status"] in ("ok", "evaluation_failed", "no_program") for record in records)
    # The point of the log: the source is there, not just the score.
    assert any(record["status"] == "ok" and "def solve" in (record["code"] or "") for record in records)

    folded = json.loads((output_dir / "eval_code_log.json").read_text())
    assert folded["n_records"] == len(records)
    assert folded["summary"]["iterations_used"] == summary["iterations_used"]


def test_eval_code_log_is_off_by_default(benchmark, tmp_path, monkeypatch):
    monkeypatch.delenv("SPECEVO_MAX_COST_USD", raising=False)
    _, output_dir, _ = _run("relay_all_cheap", benchmark, tmp_path)
    assert not (output_dir / "eval_code_log.jsonl").exists()


def test_retries_are_off_by_default_and_failures_spend_a_generation(
    benchmark, tmp_path, monkeypatch
):
    """One generation == one model call: an invalid program is not retried."""
    monkeypatch.delenv("SPECEVO_MAX_COST_USD", raising=False)

    from specevo_baselines import Runner
    from specevo_baselines.search.relay.tiered import TieredController

    config = _make_config("relay_all_cheap", workers=2)
    config.max_iterations = 6
    assert config.search.database.retry_times == 1

    runner = Runner(
        initial_program_path=str(benchmark / "initial_program.py"),
        evaluation_file=str(benchmark / "evaluator.py"),
        config=config,
        output_dir=str(tmp_path / "out_noretry"),
    )

    counter: dict = {}

    class _BrokenPool(_StubPool):
        async def generate(self, system_message, messages, **kwargs):
            self.counter[self.name] = self.counter.get(self.name, 0) + 1
            await asyncio.sleep(0)
            return LLMResponse(text="no code block here at all")

    real_init = TieredController.__init__

    def init(self, controller_input):
        real_init(self, controller_input)
        self.cheap_llms = _BrokenPool("cheap", 0.0, counter)
        self.strong_llms = _StubPool("strong", 0.05, counter)
        self.llms = self.cheap_llms

    TieredController.__init__ = init
    try:
        asyncio.run(runner.run(iterations=config.max_iterations))
    finally:
        TieredController.__init__ = real_init

    # Six generations, six calls — no generation was retried.
    assert counter["cheap"] == 6

    summary = json.loads((tmp_path / "out_noretry" / "relay_summary.json").read_text())
    assert summary["iterations_used"] == 6


def test_every_method_reports_why_it_stopped(search_type_all, benchmark, tmp_path, monkeypatch):
    """A run that ran out of money must not look like one that finished."""
    monkeypatch.delenv("SPECEVO_MAX_COST_USD", raising=False)
    _, output_dir, _ = _run(search_type_all, benchmark, tmp_path)
    summary = json.loads((output_dir / "relay_summary.json").read_text())

    assert summary["stop_reason"] in {
        "budget_exhausted",
        "generation_cap",
        "generations_ended_early",
        "interrupted",
    }
    assert summary["requested_iterations"] == 12
    # Nothing was interrupted and no budget was set, so these runs completed.
    assert summary["stop_reason"] == "generation_cap"
    assert summary["iterations_used"] == summary["requested_iterations"]


def test_all_cheap_never_calls_the_strong_model(benchmark, tmp_path, monkeypatch):
    monkeypatch.delenv("SPECEVO_MAX_COST_USD", raising=False)
    _, output_dir, counter = _run("relay_all_cheap", benchmark, tmp_path)
    assert counter.get("strong", 0) == 0
    assert counter.get("cheap", 0) > 0


def test_all_strong_never_calls_the_cheap_model(benchmark, tmp_path, monkeypatch):
    monkeypatch.delenv("SPECEVO_MAX_COST_USD", raising=False)
    _, output_dir, counter = _run("relay_all_strong", benchmark, tmp_path)
    assert counter.get("cheap", 0) == 0
    assert counter.get("strong", 0) > 0


def test_relayevolve_hands_off_from_cheap_to_strong(benchmark, tmp_path, monkeypatch):
    monkeypatch.delenv("SPECEVO_MAX_COST_USD", raising=False)
    _, output_dir, counter = _run("relayevolve", benchmark, tmp_path)

    summary = json.loads((output_dir / "relay_summary.json").read_text())
    assert summary["handoff_iteration"] is not None
    assert summary["cheap_iterations"] > 0
    assert summary["strong_iterations"] > 0
    assert summary["blocks"], "no Grow/Deepen blocks recorded"
    assert summary["seeds"], "handoff produced no seeds"
    # Both phases actually ran on their own model.
    assert counter.get("cheap", 0) > 0 and counter.get("strong", 0) > 0
    # Trajectories are separate populations, not one shared pool.
    assert len(summary["trajectories"]) >= 1


def test_fixed_switch_runs_cheap_before_strong(benchmark, tmp_path, monkeypatch):
    monkeypatch.delenv("SPECEVO_MAX_COST_USD", raising=False)
    _, output_dir, _ = _run("relay_fixed_switch", benchmark, tmp_path)
    progress = [
        json.loads(line)
        for line in (output_dir / "relay_progress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    tiers = [record["tier"] for record in progress]
    assert "cheap" in tiers and "strong" in tiers
    assert tiers.index("strong") > tiers.index("cheap")


# ---------------------------------------------------------------------------
# Unit-level checks of the relay objective and the schedulers
# ---------------------------------------------------------------------------


def _candidate(idx: int, score: float, body: str) -> Candidate:
    return Candidate(id=f"c{idx}", solution=body, score=score, text=f"cand {idx}", iteration=idx)


def test_relay_gain_is_non_negative_and_saturates():
    bank = RelayBank(CandidateEmbedder(dim=64), k=3, r=2)
    first, _ = bank.update_block(
        [
            _candidate(1, 1.0, "def a():\n    return 1"),
            _candidate(2, 2.0, "import math\ndef b():\n    return math.pi"),
            _candidate(3, 1.5, "class C:\n    def run(self):\n        return 3"),
        ]
    )
    # Re-offering an identical candidate cannot improve the bank.
    repeat, repeat_rel = bank.update_block([_candidate(1, 1.0, "def a():\n    return 1")])
    assert first > 0
    assert repeat == 0.0 and repeat_rel == 0.0


def test_curation_prefers_quality_and_coverage():
    bank = RelayBank(CandidateEmbedder(dim=64), k=2, r=1, lam=0.5)
    bank.update_block(
        [
            _candidate(1, 0.1, "def a():\n    return 1"),
            _candidate(2, 0.9, "def a():\n    return 2"),  # near-duplicate, high score
            _candidate(3, 0.8, "import numpy\nclass Z:\n    pass"),  # distinct, high score
        ]
    )
    seeds = curate_seed_population(bank, 2)
    assert len(seeds) == 2
    # The best candidate is always kept.
    assert max(s.score for s in seeds) == pytest.approx(0.9)
    # Coverage forces the second slot to the structurally different candidate.
    assert {s.id for s in seeds} == {"c2", "c3"}


def test_grow_deepen_respects_caps():
    scheduler = GrowDeepenScheduler(max_trajectories=2, trajectory_horizon=2, init_grow_blocks=1)
    live: list[int] = []
    actions = []
    for step in range(10):
        action = scheduler.select(live)
        if action is None:
            break
        if action == GROW:
            traj = len(live)
            live.append(traj)
        else:
            traj = int(action.split(":")[1])
        scheduler.observe(action, traj, reward=0.1, count_reward=step > 0)
        actions.append(action)
    assert len(live) == 2
    # 2 trajectories x 2 blocks each.
    assert len(actions) == 4


def test_two_armed_bandit_diversifies_within_a_batch():
    """A batch is dispatched before any reward returns; it must not collapse."""
    bandit = TwoArmedBandit(exploration_c=0.5)
    pending: dict[str, int] = {}
    batch = []
    for _ in range(4):
        arm = bandit.select(pending)
        pending[arm] = pending.get(arm, 0) + 1
        batch.append(arm)
    assert set(batch) == {"cheap", "strong"}


def test_two_armed_bandit_prefers_the_rewarding_arm():
    bandit = TwoArmedBandit(exploration_c=0.1)
    for _ in range(30):
        arm = bandit.select()
        bandit.observe(arm, 1.0 if arm == "strong" else 0.0)
    assert bandit.pulls["strong"] > bandit.pulls["cheap"]
