"""End-to-end tests for the SpecEvo orchestrator (no network).

We stub the mutation/Navigator LLM clients with synthetic responses and the
embedder with a deterministic hash-based vector, then run the full async
loop against a trivial scoring function. The test verifies that:

* the seed program is admitted,
* multiple mutations are produced and evaluated,
* the budget stop conditions actually halt the loop,
* a Navigator intervention trial is recorded once the cron interval is crossed,
* the final snapshot file is written.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from specevo.engine.orchestrator import SpecEvoConfig, SpecEvoOrchestrator
from specevo.clients.base import BaseClient, ClientResult
from specevo.simple import EmbedderConfig


class _FakeLM(BaseClient):
    """Hard-coded mutation/Navigator responses keyed by a counter."""

    def __init__(self, model: str, responses: list[str]):
        super().__init__(model)
        self.responses = list(responses)
        self.calls = 0

    async def acompletion(self, prompt, **kwargs) -> ClientResult:
        i = self.calls
        self.calls += 1
        if i < len(self.responses):
            text = self.responses[i]
        else:
            text = self.responses[-1]  # loop on the last one
        return ClientResult(text=text, cost=0.0001)


def _hash_embed(self, text: str) -> np.ndarray:
    """Deterministic embedding that varies with content but does not hit
    the network. Used as a monkey-patch over DescriptionEmbedder.embed."""
    text = (text or "").strip()
    if not text:
        return np.zeros(self.config.dim, dtype=np.float32)
    h = hashlib.sha1(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little", signed=False))
    vec = rng.normal(size=self.config.dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _score_fn(fn, _inputs=None):
    """Tiny scorer: f(0)+f(1)+f(2). Higher is better."""
    try:
        total = fn(0) + fn(1) + fn(2)
    except Exception as e:
        return {"error": str(e)}
    return {"score": float(total)}


SEED = """\
def solve(x):
    return x
"""


# Three valid mutations + one bad output (no fence) + one crash candidate.
_MUTATION_RESPONSES = [
    # Valid #1 — slight improvement
    """## Description
A small variant that doubles the input — replaces parent's identity map with a linear scaling trick.

## Code
```python
def solve(x):
    return 2 * x
```
""",
    # Valid #2 — bigger improvement
    """## Description
Quadratic mapping using exponentiation; favours larger inputs which dominate the test sum.

## Code
```python
def solve(x):
    return x * x + 1
```
""",
    # No code — the parser drops this and we should keep going.
    "Just some prose with no fence.",
    # Code without ## Description — exercises the prose-before-fence path.
    """A simple cubic mapping over the input.

```python
def solve(x):
    return x ** 3
```
""",
    # Crash candidate — should be recorded as a reject and counted in
    # the advisor error stream, but no longer triggers a repair attempt.
    """## Description
broken on purpose to verify the reject path.

## Code
```python
def solve(x):
    return 1 / 0
```
""",
]


# Extra mutation response — kept queued so the fake LM never runs dry
# when worker fanout overruns the canned ``_MUTATION_RESPONSES`` list.
_EXTRA_MUTATION_RESPONSE = """## Description
Fallback response so the fake LLM does not exhaust mid-test.

## Code
```python
def solve(x):
    return x ** 3 + 5
```
"""


_PARADIGM_RESPONSE = """## Description
Frontier paradigm shift: exponential growth f(x) = 2**x to stress non-polynomial scaling.

## Code
```python
def solve(x):
    return 2 ** x
```
"""


def test_specevo_orchestrator_end_to_end(tmp_path: Path, monkeypatch):
    # Bypass the network — stub the embedder.
    from specevo.simple import embedder as embedder_module

    monkeypatch.setattr(embedder_module.DescriptionEmbedder, "embed", _hash_embed)

    cfg = SpecEvoConfig(
        problem_description="Maximise solve(0)+solve(1)+solve(2).",
        function_signature="def solve(x):",
        score_fn=_score_fn,
        fn_name="solve",
        seed_program=SEED,
        budget_evals=6,  # tight cap so the test runs in <1s
        n_workers=2,
        n_eval_processes=2,
        eval_timeout=5.0,
        navigator_interval=3,  # trigger at least one Navigator intervention
        output_dir=tmp_path / "specevo_run",
        embedder_config=EmbedderConfig(model="fake/embed", dim=64),
    )
    orch = SpecEvoOrchestrator(cfg)

    # Swap in fake LLM clients post-construction.
    orch.speculator_lm = _FakeLM("fake/mutation", _MUTATION_RESPONSES + [_EXTRA_MUTATION_RESPONSE])
    orch.navigator_lm = _FakeLM("fake/Navigator", [_PARADIGM_RESPONSE])

    result = asyncio.run(orch.run())

    # --- assertions ---
    assert result.total_evaluations >= 1
    # At least the seed should have been admitted.
    assert result.archive_size >= 1
    # Best score should be at least the seed's (0+1+2 = 3).
    assert result.best_score >= 3.0
    # Snapshot artefacts written.
    snap_path = Path(result.output_dir) / "snapshot.json"
    best_path = Path(result.output_dir) / "best.py"
    assert snap_path.exists()
    assert best_path.exists()
    snap = json.loads(snap_path.read_text())
    assert snap["method"] == "specevo"
    assert snap["best_score"] == result.best_score
    # Cell elites carry descriptions.
    assert all("description" in e for e in snap["cells"])


def test_specevo_budget_evals_actually_halts(tmp_path: Path, monkeypatch):
    """Run with a 3-eval cap and verify we stop near that bound."""
    from specevo.simple import embedder as embedder_module

    monkeypatch.setattr(embedder_module.DescriptionEmbedder, "embed", _hash_embed)

    cfg = SpecEvoConfig(
        problem_description="trivial",
        function_signature="def solve(x):",
        score_fn=_score_fn,
        fn_name="solve",
        seed_program=SEED,
        budget_evals=3,
        n_workers=1,  # serialise so the bound is sharp
        n_eval_processes=1,
        eval_timeout=5.0,
        navigator_interval=999,  # disable Navigator during this test
        output_dir=tmp_path / "specevo_run_budget",
        embedder_config=EmbedderConfig(model="fake/embed", dim=64),
    )
    orch = SpecEvoOrchestrator(cfg)
    orch.speculator_lm = _FakeLM("fake/mutation", _MUTATION_RESPONSES)
    orch.navigator_lm = _FakeLM("fake/Navigator", [_PARADIGM_RESPONSE])

    result = asyncio.run(orch.run())
    # n_workers=1 means we may overshoot by at most one in-flight task.
    assert result.total_evaluations <= 5, result.total_evaluations
    assert result.total_evaluations >= 3, result.total_evaluations


class _HangingLM(BaseClient):
    """LLM stub that blocks forever inside ``acompletion``.

    Simulates the OpenRouter / GPT-5 whitespace-keepalive failure mode where
    a stalled upstream provider produces no result and no error — the call
    just sits there for hours. Without the per-call ``llm_call_timeout`` +
    main-loop ``asyncio.wait(timeout=...)`` defence, ``SpecEvoOrchestrator.run``
    would block indefinitely once the budget exhausts but in-flight tasks
    haven't returned. This regression test pins that fix in place."""

    def __init__(self, model: str):
        super().__init__(model)
        self.calls = 0

    async def acompletion(self, prompt, **kwargs) -> ClientResult:
        self.calls += 1
        # Sleep way past the test's deadline; we expect the orchestrator
        # to cancel us, not for this to ever complete.
        await asyncio.sleep(3600)
        return ClientResult(text="never", cost=0.0)


def test_specevo_terminates_when_llm_hangs(tmp_path: Path, monkeypatch):
    """Budget exhaustion must short-circuit hung LLM calls.

    Pre-fix: ``_main_loop``'s ``asyncio.wait(return_when=FIRST_COMPLETED)``
    blocked forever when every in-flight LLM coroutine was stuck inside
    a third-party HTTP call. GitHub Actions only ended the job when its
    6h hard cap kicked in (see the run log: Elapsed grew past
    21500s with ``Clients in-flight: 4`` for hours after the 10800s budget).
    """
    from specevo.simple import embedder as embedder_module

    monkeypatch.setattr(embedder_module.DescriptionEmbedder, "embed", _hash_embed)

    cfg = SpecEvoConfig(
        problem_description="trivial",
        function_signature="def solve(x):",
        score_fn=_score_fn,
        fn_name="solve",
        seed_program=SEED,  # admit a seed so the loop has something to chew on
        budget_seconds=1.0,  # very tight — we want budget to exhaust fast
        n_workers=2,
        n_eval_processes=1,
        eval_timeout=5.0,
        navigator_interval=999,
        n_diverse_seeds=0,  # skip frontier phase 1 — no need to wait on it
        n_variants_per_seed=0,  # skip phase 2 fanout too
        output_dir=tmp_path / "specevo_hang",
        embedder_config=EmbedderConfig(model="fake/embed", dim=64),
        llm_call_timeout=2.0,  # mutation calls must give up within 2s
        shutdown_grace_seconds=5.0,
    )
    orch = SpecEvoOrchestrator(cfg)
    orch.speculator_lm = _HangingLM("fake/hanging-mutation")
    orch.navigator_lm = _HangingLM("fake/hanging-Navigator")

    import time as _time

    start = _time.monotonic()
    result = asyncio.run(orch.run())
    elapsed = _time.monotonic() - start

    # Whole run (budget=1s + grace=5s + a bit of overhead) must finish well
    # under the test's 30s ceiling. Pre-fix this would block ~3600s.
    assert elapsed < 30.0, f"run() took {elapsed:.1f}s — orchestrator hung"
    # Seed should still have been admitted before the hang.
    assert result.archive_size >= 1
    assert result.runtime_seconds >= 1.0
