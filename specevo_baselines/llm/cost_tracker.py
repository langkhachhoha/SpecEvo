"""Per-LLM-call cost tracking, written to a JSONL sidecar.

Activated only when ``SPECEVO_COST_LOG`` is set in the environment.
Each successful LLM call appends one record. A ``totals.json`` sibling file
is rewritten on every call so a runner can read the running total cheaply.

This exists because specevo_baselines's OpenAI wrapper normally discards the
``usage`` object from completion responses, and on a shared OpenRouter key
polling ``/credits`` cannot isolate this run from concurrent activity.

For OpenRouter requests we set ``extra_body={"usage": {"include": True}}``
so the provider returns ``usage.cost`` (USD) — that field is the ground
truth and is what gets logged. For non-OpenRouter providers we log
tokens but ``cost_usd`` will be ``None`` (no provider-side dollar number
is exposed).

A **spend budget** can be set with ``SPECEVO_MAX_COST_USD``. Once the
running total reaches it, every hook registered via
``register_budget_hook`` fires exactly once — the discovery controllers
use that to request a graceful shutdown, so the run stops at the next
iteration boundary with its results intact. This is a *soft* cap for the
same reason specevo's ``--dollars`` is: calls already in flight still
complete, so the final total can overshoot slightly. It relies on
provider-reported ``usage.cost`` and therefore only bites on OpenRouter.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_TOTAL_COST = 0.0
_TOTAL_CALLS = 0
_TOTAL_PROMPT_TOKENS = 0
_TOTAL_COMPLETION_TOKENS = 0

# Fired once, when the running total first reaches the budget. Bound methods
# are held weakly: this list lives for the whole process, and a controller
# that has finished its run must still be collectable.
_BUDGET_HOOKS: List[Callable[[], Optional[Callable[[float, float], None]]]] = []
_BUDGET_FIRED = False


def is_openrouter(api_base: Optional[str]) -> bool:
    return bool(api_base) and "openrouter.ai" in api_base.lower()


def cost_log_path() -> Optional[Path]:
    p = os.environ.get("SPECEVO_COST_LOG")
    return Path(p) if p else None


def budget_usd() -> Optional[float]:
    """USD spend cap from ``SPECEVO_MAX_COST_USD``, or ``None`` if unset.

    Blank, unparseable and non-positive values all mean "no cap" so that a
    workflow can pass the input through verbatim without branching on it.
    """
    raw = (os.environ.get("SPECEVO_MAX_COST_USD") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def register_budget_hook(hook: Callable[[float, float], None]) -> None:
    """Call ``hook(total_cost, budget)`` when the spend budget is reached.

    Bound methods are kept as weak references, so registering does not pin
    the owner (a discovery controller) in memory after its run ends; dead
    entries are dropped on the next registration. Registering after the
    budget has already been hit invokes the hook immediately — otherwise a
    controller built late in a run would never learn that spending is over.
    """
    ref: Callable[[], Optional[Callable[[float, float], None]]]
    ref = weakref.WeakMethod(hook) if inspect.ismethod(hook) else (lambda h=hook: h)

    fire_now = False
    with _LOCK:
        _BUDGET_HOOKS[:] = [r for r in _BUDGET_HOOKS if r() is not None]
        if not any(r() == hook for r in _BUDGET_HOOKS):
            _BUDGET_HOOKS.append(ref)
        if _BUDGET_FIRED:
            fire_now = True
            total, budget = _TOTAL_COST, budget_usd() or 0.0
    if fire_now:
        _safe_hook(hook, total, budget)


def budget_exceeded() -> bool:
    """True once total spend has reached the configured budget."""
    budget = budget_usd()
    if budget is None:
        return False
    with _LOCK:
        return _TOTAL_COST >= budget


def _safe_hook(hook: Callable[[float, float], None], total: float, budget: float) -> None:
    try:
        hook(total, budget)
    except Exception:  # noqa: BLE001 - a broken hook must not kill the LLM call
        logger.debug("Budget hook raised", exc_info=True)


def inject_openrouter_usage(params: Dict[str, Any], api_base: Optional[str]) -> None:
    """Mutate ``params`` so OpenRouter returns ``usage.cost`` in the response.

    No-op for non-OpenRouter endpoints. Safe to call multiple times; if the
    caller already provided ``extra_body['usage']`` we leave it alone.
    """
    if not is_openrouter(api_base):
        return
    extra = params.setdefault("extra_body", {})
    if not isinstance(extra, dict):
        return
    usage = extra.setdefault("usage", {"include": True})
    if isinstance(usage, dict) and "include" not in usage:
        usage["include"] = True


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_usage_record(response: Any) -> Dict[str, Any]:
    usage = _get(response, "usage")
    if usage is None:
        return {}
    prompt = _get(usage, "prompt_tokens") or _get(usage, "input_tokens")
    completion = _get(usage, "completion_tokens") or _get(usage, "output_tokens")
    total = _get(usage, "total_tokens")
    cost = _get(usage, "cost")
    cost_details = _get(usage, "cost_details")
    prompt_details = _get(usage, "prompt_tokens_details")
    cached = _get(prompt_details, "cached_tokens") if prompt_details is not None else None
    rec: Dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cost_usd": float(cost) if cost is not None else None,
        "cached_tokens": cached,
    }
    if cost_details is not None:
        # Convert SDK objects to plain dict where possible.
        try:
            if hasattr(cost_details, "model_dump"):
                rec["cost_details"] = cost_details.model_dump()
            elif isinstance(cost_details, dict):
                rec["cost_details"] = cost_details
        except Exception:  # noqa: BLE001
            pass
    return rec


def record_call(
    response: Any,
    *,
    model: str,
    api_base: Optional[str],
    api_kind: str,
    call_site: str,
) -> None:
    """Append a record for one LLM response, and update the running budget.

    No-op unless ``SPECEVO_COST_LOG`` or ``SPECEVO_MAX_COST_USD`` is
    set — a budget alone is enough to keep the in-memory totals, so the cap
    works even when nobody asked for the JSONL sidecar.
    """
    log_path = cost_log_path()
    budget = budget_usd()
    if log_path is None and budget is None:
        return

    usage_rec = _extract_usage_record(response)
    record = {
        "ts": time.time(),
        "model": model,
        "api_base": api_base,
        "api_kind": api_kind,  # "chat.completions" | "responses"
        "call_site": call_site,  # e.g. "OpenAILLM._call_api"
        "response_id": _get(response, "id"),
        **usage_rec,
    }

    global _TOTAL_COST, _TOTAL_CALLS, _TOTAL_PROMPT_TOKENS, _TOTAL_COMPLETION_TOKENS
    global _BUDGET_FIRED
    with _LOCK:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        _TOTAL_CALLS += 1
        if record.get("cost_usd") is not None:
            _TOTAL_COST += float(record["cost_usd"])
        if record.get("prompt_tokens"):
            _TOTAL_PROMPT_TOKENS += int(record["prompt_tokens"])
        if record.get("completion_tokens"):
            _TOTAL_COMPLETION_TOKENS += int(record["completion_tokens"])

        if log_path is not None:
            totals = {
                "calls": _TOTAL_CALLS,
                "total_cost_usd": round(_TOTAL_COST, 6),
                "total_prompt_tokens": _TOTAL_PROMPT_TOKENS,
                "total_completion_tokens": _TOTAL_COMPLETION_TOKENS,
                "updated_ts": time.time(),
            }
            if budget is not None:
                totals["budget_usd"] = budget
            totals_path = log_path.with_suffix(".totals.json")
            tmp_path = totals_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(totals, indent=2))
            tmp_path.replace(totals_path)

        # Decide under the lock so exactly one call gets to fire the hooks,
        # but run them outside it — a hook that touches the tracker again
        # (or blocks) must not deadlock the LLM path.
        fire = budget is not None and not _BUDGET_FIRED and _TOTAL_COST >= budget
        if fire:
            _BUDGET_FIRED = True
            hooks = [h for h in (r() for r in _BUDGET_HOOKS) if h is not None]
            total_now = _TOTAL_COST

    if fire:
        logger.warning(
            "💸 Spend budget reached: $%.4f of $%.4f after %d LLM calls — "
            "requesting a graceful stop.",
            total_now,
            budget,
            _TOTAL_CALLS,
        )
        for hook in hooks:
            _safe_hook(hook, total_now, budget)


def get_totals_snapshot() -> Dict[str, Any]:
    """Return a copy of the in-memory running totals."""
    with _LOCK:
        return {
            "calls": _TOTAL_CALLS,
            "total_cost_usd": round(_TOTAL_COST, 6),
            "total_prompt_tokens": _TOTAL_PROMPT_TOKENS,
            "total_completion_tokens": _TOTAL_COMPLETION_TOKENS,
        }


def format_cost_suffix() -> str:
    """Return a `[cost=$X.XXXX, calls=N]` suffix when tracking is on, else ``""``.

    Designed to be appended to iteration log lines so each iteration line
    carries the cumulative LLM spend at that point — making score-vs-cost
    curves derivable from log scrape alone. Under a budget the cap is shown
    too (``cost=$0.1234/$5.00``) so progress towards the cut-off is visible;
    the plain form is unchanged when no budget is set, keeping existing log
    scrapes valid.
    """
    budget = budget_usd()
    if cost_log_path() is None and budget is None:
        return ""
    with _LOCK:
        spent = f"${_TOTAL_COST:.4f}" if budget is None else f"${_TOTAL_COST:.4f}/${budget:.2f}"
        return f" [cost={spent}, llm_calls={_TOTAL_CALLS}]"
