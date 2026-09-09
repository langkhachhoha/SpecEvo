"""Token accounting for SpecEvo/SpecEvo runs (offline, fast).

Baselines report input/output tokens through ``cost_log.totals.json``;
SpecEvo reports them through its own ``summary.json`` / ``snapshot.json``.
These are the two links worth pinning cheaply: the client pulling ``usage``
off a completion response, and the call log accumulating it. The
whole-run wiring on top of them is covered by the orchestrator's existing
end-to-end tests.
"""

from __future__ import annotations

from types import SimpleNamespace

from specevo.engine.orchestrator import _CallLog
from specevo.clients.base import ClientResult
from specevo.clients.lm import _extract_tokens


# ---------------------------------------------------------------------------
# Client-level extraction
# ---------------------------------------------------------------------------


def test_extract_tokens_openai_shape():
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45))
    assert _extract_tokens(resp) == (120, 45)


def test_extract_tokens_anthropic_shape():
    resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=9))
    assert _extract_tokens(resp) == (7, 9)


def test_extract_tokens_dict_usage():
    resp = SimpleNamespace(usage={"prompt_tokens": 3, "completion_tokens": 4})
    assert _extract_tokens(resp) == (3, 4)


def test_extract_tokens_missing_or_junk_is_zero():
    """Token accounting must never be able to break a search."""
    assert _extract_tokens(SimpleNamespace()) == (0, 0)
    assert _extract_tokens(SimpleNamespace(usage=None)) == (0, 0)
    assert _extract_tokens(SimpleNamespace(usage=SimpleNamespace())) == (0, 0)
    junk = SimpleNamespace(usage=SimpleNamespace(prompt_tokens="lots", completion_tokens=-5))
    assert _extract_tokens(junk) == (0, 0)


def test_client_result_defaults_to_zero_tokens():
    """CLI-backed clients report no usage; summing must still be safe."""
    r = ClientResult(text="hi", cost=0.0)
    assert (r.prompt_tokens, r.completion_tokens) == (0, 0)


# ---------------------------------------------------------------------------
# Call log
# ---------------------------------------------------------------------------


def test_call_log_accumulates_tokens():
    log = _CallLog()
    log.record(0.01, 100, 20)
    log.record(0.02, 50, 5)
    log.record(0.0)  # a backend that reports no usage
    assert log.calls == 3
    assert log.prompt_tokens == 150
    assert log.completion_tokens == 25
    assert log.snapshot() == {
        "llm_calls": 3,
        "cost_usd": 0.03,
        "prompt_tokens": 150,
        "completion_tokens": 25,
        "total_tokens": 175,
    }


def test_call_log_snapshot_is_a_frozen_copy():
    """This is what keeps the init-phase split from tracking the search."""
    log = _CallLog()
    log.record(0.01, 10, 2)
    snap = log.snapshot()
    log.record(0.01, 10, 2)
    assert snap["prompt_tokens"] == 10
